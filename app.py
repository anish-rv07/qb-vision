"""
QB Vision: analyze first-person quarterback footage from smart glasses.

Flow:
  1. The user uploads a clip of one play (plus optional play info).
  2. We upload the clip with the Gemini Files API and wait until it is ACTIVE.
  3. We ask the best available Gemini model (Pro first, Flash as a fallback) for a JSON analysis.
  4. We render the JSON as a readable report (cards, tables, lists).
"""

import html
import json
import os
import re
import tempfile
import time
from pathlib import Path

import gradio as gr
import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

# Network errors from the Gemini SDK. Newer SDK versions use the `httpx2` package,
# older ones use `httpx`, so we catch whichever is installed.
_http_modules = []
for _name in ("httpx2", "httpx"):
    try:
        _http_modules.append(__import__(_name))
    except ImportError:
        pass
TIMEOUT_ERRORS = tuple(m.TimeoutException for m in _http_modules)
NETWORK_ERRORS = tuple(m.TransportError for m in _http_modules) + (ConnectionError,)

# Free Hugging Face accounts host Gradio apps on "ZeroGPU" hardware, which refuses
# to start unless a Gradio event is wired to a function marked @spaces.GPU. This app
# needs no GPU (Gemini does the work), so we wire a do-nothing function to a hidden
# button that is never clicked: no GPU is ever requested and no quota is used.
# `spaces` is preinstalled on Hugging Face; locally it's usually missing, so we skip it.
try:
    import spaces

    @spaces.GPU(duration=1)
    def _zerogpu_noop():
        """Placeholder required by ZeroGPU. Never called."""
        return None
except ImportError:
    _zerogpu_noop = None

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv()  # Reads .env locally. On Hugging Face Spaces, use a Secret instead.

APP_DIR = Path(__file__).parent
PROMPT_PATH = APP_DIR / "analysis_prompt.txt"
EXAMPLES_DIR = APP_DIR / "examples"

VIDEO_FPS = 4                 # Frames per second Gemini samples (catches route breaks).
MAX_OUTPUT_TOKENS = 65536     # The response is long; Pro models also "think" inside this budget.
FILE_ACTIVE_TIMEOUT = 300     # Seconds to wait for the uploaded video to finish processing.
REQUEST_TIMEOUT = 150         # Seconds before one Gemini request counts as stuck (normal runs take ~60).
TOTAL_TIME_BUDGET = 420       # Stop trying more models after this many seconds in total.

# Spoken coaching (optional). Without ELEVENLABS_API_KEY the app works the same, just silent.
ELEVENLABS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")  # "George"
ELEVENLABS_MODEL = os.environ.get("ELEVENLABS_MODEL", "eleven_multilingual_v2")
# Known models, best first. The model list API can be out of date (it may list
# retired models and omit new previews), so we also try these and skip any that 404.
# Pro is tried first; Flash is the fallback when Pro is unavailable (e.g. free tier).
KNOWN_MODELS = ["gemini-3.1-pro-preview", "gemini-3.8-flash"]
VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".avi", ".mkv", ".mpeg", ".mpg", ".3gp"}
MAX_VIDEO_BYTES = 2 * 1024**3  # Files API limit is 2 GB per file.

MIME_TYPES = {
    ".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm",
    ".avi": "video/x-msvideo", ".mkv": "video/x-matroska", ".mpeg": "video/mpeg",
    ".mpg": "video/mpeg", ".3gp": "video/3gpp",
}


class FriendlyError(Exception):
    """An error whose message is safe and helpful to show the user directly."""


# ---------------------------------------------------------------------------
# Gemini helpers
# ---------------------------------------------------------------------------

_client = None
_working_model = None  # Remembered after the first successful call.


def get_client():
    """Create the Gemini client once, reading the key from the environment."""
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise FriendlyError(
                "GEMINI_API_KEY is not set. Add it to a .env file locally, "
                "or as a Secret in your Hugging Face Space settings."
            )
        # v1beta is required for preview models such as gemini-3.1-pro-preview.
        _client = genai.Client(
            api_key=api_key,
            http_options={"api_version": "v1beta", "timeout": REQUEST_TIMEOUT * 1000},
        )
    return _client


def candidate_models(client):
    """
    Return models to try, best first: Pro, then Flash, then Flash-Lite (newest first in each).
    The model that worked last time is tried first. Set GEMINI_MODEL to force one.
    """
    override = os.environ.get("GEMINI_MODEL")
    if override:
        return [override]

    names = set(KNOWN_MODELS)
    try:
        for m in client.models.list():
            name = (m.name or "").removeprefix("models/")
            if "generateContent" in (m.supported_actions or []):
                names.add(name)
    except Exception as e:  # Listing is a nice-to-have; don't fail the request over it.
        print(f"[QB Vision] Could not list models: {e}")

    ranked = []
    for name in names:
        # Only plain "gemini-X.Y-pro" / "-flash" / "-flash-lite" text models.
        match = re.fullmatch(r"gemini-(\d+(?:\.\d+)?)-(pro|flash)(-lite)?(-preview)?(-[\w-]+)?", name)
        if not match or any(bad in name for bad in
                            ("tts", "image", "audio", "computer", "embedding", "customtools")):
            continue
        # Pro first; Flash next; Flash-Lite (faster, less accurate) as the last resort.
        tier = 2 if match.group(2) == "pro" else (0 if match.group(3) else 1)
        version = float(match.group(1))
        is_plain = match.group(5) is None  # "-preview" yes, "-preview-customtools" no
        ranked.append(((tier, version, is_plain, -len(name)), name))
    models = [name for _, name in sorted(ranked, reverse=True)]
    if _working_model in models:
        models.remove(_working_model)
        models.insert(0, _working_model)
    return models


def load_prompt(play_info):
    """Load the prompt file and fill in {play_info}. Uses str.replace because the prompt has JSON braces."""
    try:
        template = PROMPT_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FriendlyError(f"Could not find the prompt file: {PROMPT_PATH.name}")
    play_info = (play_info or "").strip() or "None provided."
    return template.replace("{play_info}", play_info)


def check_video(video_path):
    """Basic checks on the uploaded file before we spend an API call on it."""
    if not video_path:
        raise FriendlyError("Please upload a video clip of one play first.")
    path = Path(video_path)
    if not path.exists():
        raise FriendlyError("The uploaded video could not be found. Please upload it again.")
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise FriendlyError(
            f"'{path.suffix}' is not a supported video type. Try an .mp4 or .mov file."
        )
    size = path.stat().st_size
    if size == 0:
        raise FriendlyError("That video file is empty. Please try a different clip.")
    if size > MAX_VIDEO_BYTES:
        raise FriendlyError("That video is over 2 GB. Please trim it to just the one play.")
    return path


def upload_and_wait(client, path):
    """Upload the video with the Files API and wait until Gemini finishes processing it."""
    mime = MIME_TYPES.get(path.suffix.lower(), "video/mp4")
    try:
        file = client.files.upload(file=str(path), config={"mime_type": mime})
    except errors.APIError as e:
        raise FriendlyError(describe_api_error(e, "uploading the video"))
    except Exception as e:
        raise FriendlyError(f"Uploading the video failed: {e}")

    start = time.time()
    while file.state == types.FileState.PROCESSING:
        if time.time() - start > FILE_ACTIVE_TIMEOUT:
            raise FriendlyError("Gemini took too long to process the video. Try a shorter clip.")
        time.sleep(2)
        file = client.files.get(name=file.name)

    if file.state != types.FileState.ACTIVE:
        raise FriendlyError(
            "Gemini could not process this video. It may be corrupted or in an "
            "unsupported format. Try re-exporting it as an .mp4."
        )
    return file


def describe_api_error(e, action):
    """Turn a Gemini API error into a plain-English message."""
    code = getattr(e, "code", None)
    if code == 429 and "limit: 0" in str(e):
        return ("This API key has no quota for this Gemini model (free tier limit is 0). "
                "Enable billing for your project in Google AI Studio, then try again.")
    if code == 429:
        return ("Gemini rate limit or quota reached. Wait a minute and try again, "
                "or check your quota in Google AI Studio.")
    if code in (401, 403):
        return "Gemini rejected the API key. Check that GEMINI_API_KEY is correct and enabled."
    if code == 404:
        return "The Gemini model or file was not found. Try setting GEMINI_MODEL to a model you can access."
    if code == 400:
        return f"Gemini could not accept the request while {action}: {e.message or e}"
    if code and code >= 500:
        return "Gemini is having trouble right now (server error). Please try again in a moment."
    return f"Gemini API error while {action}: {e}"


def parse_json(text):
    """Parse the model's JSON. Tolerates stray markdown fences or text around the object."""
    if not text:
        raise FriendlyError("Gemini returned an empty response. Please try again.")
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Last resort: grab everything between the first "{" and the last "}".
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            pass
    raise FriendlyError("Gemini's response wasn't valid JSON. Please try again; the raw text is below.")


class ModelStuck(Exception):
    """A model timed out or kept dropping the connection; try the next one."""


def generate_with_retry(client, model, parts):
    """Ask Gemini for the analysis. Retries once if the connection drops; gives up on timeouts."""
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    for attempt in range(2):
        try:
            return client.models.generate_content(
                model=model, contents=[types.Content(role="user", parts=parts)], config=config
            )
        except TIMEOUT_ERRORS:
            # Stuck for REQUEST_TIMEOUT seconds: don't wait again on the same model.
            raise ModelStuck(f"{model} timed out")
        except NETWORK_ERRORS as e:
            # Long video requests can get cut off by flaky networks; try once more.
            print(f"[QB Vision] Connection problem with {model} ({e!r}), attempt {attempt + 1}.")
    raise ModelStuck(f"{model} kept dropping the connection")


def analyze_with_gemini(video_path, play_info):
    """Run the full Gemini pipeline. Returns (parsed_dict, raw_text, model_name)."""
    global _working_model
    path = check_video(video_path)
    prompt = load_prompt(play_info)
    client = get_client()

    file = upload_and_wait(client, path)
    try:
        # The video part carries videoMetadata so Gemini samples at VIDEO_FPS.
        video_part = types.Part(
            file_data=types.FileData(file_uri=file.uri, mime_type=file.mime_type),
            video_metadata=types.VideoMetadata(fps=VIDEO_FPS),
        )
        response, model, last_error, busy_skipped = None, None, None, False
        started = time.time()
        candidates = candidate_models(client)
        for round_number in range(2):  # A second round only if every model was busy.
            for candidate in candidates:
                if time.time() - started > TOTAL_TIME_BUDGET:
                    break  # Don't keep the user waiting forever.
                try:
                    response = generate_with_retry(client, candidate, [video_part, types.Part(text=prompt)])
                    model = candidate
                    break
                except errors.APIError as e:
                    last_error = e
                    # Move on to the next model if this one is retired (404), has no quota on
                    # this key (429 "limit: 0", e.g. Pro on the free tier), or is overloaded (5xx).
                    no_quota = e.code == 429 and "limit: 0" in str(e)
                    busy = e.code is not None and e.code >= 500
                    if e.code == 404 or no_quota or busy:
                        print(f"[QB Vision] {candidate} unavailable ({e.code}), trying next model.")
                        busy_skipped = busy_skipped or busy
                        continue
                    raise FriendlyError(describe_api_error(e, "analyzing the play"))
                except ModelStuck as e:
                    print(f"[QB Vision] {e}, trying next model.")
                    busy_skipped = True
                    last_error = None
                    continue
            if time.time() - started > TOTAL_TIME_BUDGET:
                break
            if response is not None or not busy_skipped or round_number == 1:
                break
            print("[QB Vision] Every model was busy; waiting 10s and trying again.")
            time.sleep(10)

        if response is None:
            if busy_skipped:
                raise FriendlyError("Gemini's servers are overloaded right now. "
                                    "Please wait a minute and click Analyze play again.")
            raise FriendlyError("None of the Gemini models are available to this API key. "
                                "Set GEMINI_MODEL to a model you can use.")
        _working_model = model
        print(f"[QB Vision] Using model: {model}")

        raw = response.text or ""
        # If the response was cut off or blocked, say so instead of a confusing parse error.
        finish = response.candidates[0].finish_reason if response.candidates else None
        if not raw:
            reason = finish.name if finish else "no candidates returned"
            raise FriendlyError(f"Gemini returned no analysis ({reason}). Please try again.")
        try:
            data = parse_json(raw)
        except FriendlyError as e:
            if finish == types.FinishReason.MAX_TOKENS:
                e = FriendlyError("Gemini's response was cut off (too long). Try a shorter clip.")
            e.raw = raw  # Keep the raw text so we can show it.
            raise e
        if not isinstance(data, dict):
            err = FriendlyError("Gemini returned JSON in an unexpected shape.")
            err.raw = raw
            raise err
        return data, raw, model
    finally:
        # Clean up the uploaded file; Gemini would delete it after 48h anyway.
        try:
            client.files.delete(name=file.name)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Rendering helpers (JSON -> readable UI)
# ---------------------------------------------------------------------------

def text(value, empty="—"):
    """Turn any JSON value into display text."""
    if value is None or value == "" or value == []:
        return empty
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, list):
        return ", ".join(text(v) for v in value)
    if isinstance(value, dict):
        return "; ".join(f"{k}: {text(v)}" for k, v in value.items())
    return str(value)


def is_none(value):
    """True for empty values, including the AI writing the word "None" or "null"."""
    return value in (None, "", []) or (isinstance(value, str) and value.strip().lower() in ("none", "null", "n/a"))


def as_list(value):
    """Make sure we always iterate over a list, even if the model returned something else."""
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


GRADE_COLORS = {"A": "#22c55e", "B": "#a3e635", "C": "#facc15", "D": "#fb923c", "F": "#ef4444"}


def render_summary(data, model):
    """The summary card at the top: grade, coaching tip, coverage, confidence."""
    s = data.get("summary") or {}
    grade = text(s.get("decision_grade"), "?")
    color = GRADE_COLORS.get(grade[:1].upper(), "#8b9ab3")
    coverage = s.get("coverage") or {}
    if not isinstance(coverage, dict):
        coverage = {"post_snap": coverage}
    e = lambda v: html.escape(text(v))  # Escape model text before putting it in HTML.

    reason = s.get("confidence_reason")
    return f"""
<div class="qb-card">
  <div class="qb-score" style="--grade:{color}">
    <div class="qb-score-label">Grade</div>
    <div class="qb-score-value">{html.escape(grade)}</div>
  </div>
  <div class="qb-body">
    <div class="qb-label">Your #1 tip</div>
    <div class="qb-tip">{e(s.get("coaching_tip"))}</div>
    <div class="qb-row">
      <div><div class="qb-label">What the defense showed</div>{e(coverage.get("pre_snap"))}</div>
      <div><div class="qb-label">What the defense really did</div>{e(coverage.get("post_snap"))}</div>
      <div><div class="qb-label">How well the AI could see</div>{e(text(s.get("confidence")).capitalize())}
        {f'<div class="qb-small">{e(reason)}</div>' if reason else ""}</div>
    </div>
    <div class="qb-small">The grade is for your decision, not the result · AI model: {html.escape(model)}</div>
  </div>
</div>"""


def html_table(headers, rows, empty):
    """A simple, readable table. Cells are escaped unless already marked safe (badges)."""
    if not rows:
        return f'<div class="qb-empty">{empty}</div>'
    head = "".join(f"<th>{h}</th>" for h in headers)
    # data-label lets phones show each row as a small card with labels (see CSS).
    body = "".join(
        "<tr>" + "".join(f'<td data-label="{h}">{c}</td>' for h, c in zip(headers, row)) + "</tr>"
        for row in rows
    )
    return f'<table class="qb-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def who(item):
    """'R1' tag plus a plain description of the player."""
    e = lambda v: html.escape(text(v))
    return f'<span class="qb-tag">{e(item.get("id"))}</span> {e(item.get("description"))}'


def open_badge(value):
    """Green 'Yes' / red 'No' badge for 'open when you threw'."""
    if isinstance(value, str):
        value = {"true": True, "yes": True, "false": False, "no": False}.get(value.strip().lower(), value)
    if value is True:
        return '<span class="qb-badge qb-yes">Yes</span>'
    if value is False:
        return '<span class="qb-badge qb-no">No</span>'
    return html.escape(text(value))


def render_receivers(data):
    rows = []
    for r in as_list(data.get("receivers")):
        r = r if isinstance(r, dict) else {"description": r}
        e = lambda v: html.escape(text(v))
        where = e(r.get("position"))
        rows.append([who(r) + (f'<div class="qb-sub">{where}</div>' if r.get("position") else ""),
                     e(r.get("route")),
                     e(r.get("open_window")) if not is_none(r.get("open_window")) else "Never open",
                     open_badge(r.get("open_at_throw"))])
    return html_table(["Receiver", "What he ran", "When he was open", "Open when you threw?"],
                      rows, "No receivers were visible.")


def render_defenders(data):
    rows = []
    for d in as_list(data.get("defenders")):
        d = d if isinstance(d, dict) else {"description": d}
        e = lambda v: html.escape(text(v))
        rows.append([who(d), e(d.get("alignment")), e(d.get("behavior"))])
    return html_table(["Defender", "Where he lined up", "What he did"],
                      rows, "No defenders were visible.")


def render_eyes(data):
    eyes = data.get("qb_eyes") or {}
    if not isinstance(eyes, dict):
        return text(eyes)
    progression = as_list(eyes.get("progression"))
    steps = " → ".join(text(p) for p in progression) if progression else "—"
    stared = None if is_none(eyes.get("stared_down_target")) else eyes.get("stared_down_target")
    return (f"**Before the snap you looked at:** {text(eyes.get('pre_snap_scan'))}\n\n"
            f"**Where your eyes went, in order:** {steps}\n\n"
            f"**Locked onto one receiver?** {('Yes, ' + text(stared)) if stared else 'No, good job.'}")


def render_throw(data):
    t = data.get("throw") or {}
    if not isinstance(t, dict):
        return text(t)
    target = text(t.get("target"))
    if t.get("time"):
        target += f" (at {text(t.get('time'))})"
    better = None if is_none(t.get("better_option")) else t.get("better_option")
    return (f"**You threw to:** {target}\n\n"
            f"**Where the ball went:** {text(t.get('placement'))}\n\n"
            f"**Pressure on you:** {text(t.get('pressure'))}\n\n"
            f"**Better choice:** {text(better, 'None. That was the right read.')}")


def render_timeline(data):
    lines = []
    for item in as_list(data.get("timeline")):
        if isinstance(item, dict):
            when = item.get("time") or item.get("t") or item.get("timestamp")
            event = item.get("event") or item.get("description") or text(item)
            lines.append(f"- **{text(when)}**: {text(event)}" if when else f"- {text(event)}")
        else:
            lines.append(f"- {text(item)}")
    return "\n".join(lines) or "_No play-by-play returned._"


def render_couldnt_see(data):
    items = as_list(data.get("couldnt_see"))
    return "\n".join(f"- {text(i)}" for i in items) or "_Nothing. The camera saw the whole play._"


# ---------------------------------------------------------------------------
# Spoken coaching with ElevenLabs
# ---------------------------------------------------------------------------

def spoken_grade(grade):
    """'B+' -> 'B plus', so the voice reads it naturally."""
    grade = text(grade, "").strip()
    return grade.replace("+", " plus").replace("-", " minus") if grade else ""


def coaching_script(data):
    """
    A short, friendly message for the voice to read (about 15-25 seconds).
    Built from the analysis so it always matches what's on screen.
    """
    summary = data.get("summary") or {}
    eyes = data.get("qb_eyes") or {}
    throw = data.get("throw") or {}
    eyes = eyes if isinstance(eyes, dict) else {}
    throw = throw if isinstance(throw, dict) else {}

    lines = []
    grade = spoken_grade(summary.get("decision_grade"))
    if grade:
        lines.append(f"Your grade on this play: {grade}.")
    if not is_none(summary.get("coaching_tip")):
        lines.append(text(summary.get("coaching_tip")))
    if not is_none(eyes.get("stared_down_target")):
        lines.append("Watch your eyes. You locked onto one receiver, and the defense can read that.")
    if not is_none(throw.get("better_option")):
        lines.append(f"Better choice: {text(throw.get('better_option'))}")
    else:
        lines.append("You picked the right receiver. Keep it up.")
    return " ".join(line.strip() for line in lines)


def speak(script):
    """
    Turn the script into an MP3 with ElevenLabs. Returns (file path or None, note).
    Voice problems never break the analysis; they just show a short note.
    """
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        return None, ""  # Voice is optional: stay quiet when no key is set.
    if not script:
        return None, ""
    try:
        response = httpx.post(
            ELEVENLABS_URL.format(voice_id=ELEVENLABS_VOICE_ID),
            params={"output_format": "mp3_44100_128"},
            headers={"xi-api-key": api_key, "Content-Type": "application/json"},
            json={"text": script, "model_id": ELEVENLABS_MODEL},
            timeout=60,
        )
    except httpx.HTTPError as e:
        print(f"[QB Vision] ElevenLabs request failed: {e!r}")
        return None, "Couldn't reach the voice service. The written report is above."
    if response.status_code != 200:
        print(f"[QB Vision] ElevenLabs error {response.status_code}: {response.text[:300]}")
        if response.status_code in (401, 403):
            return None, "The ElevenLabs key was rejected. Check ELEVENLABS_API_KEY."
        if response.status_code == 429:
            return None, "The voice service is busy or out of credits. Try again later."
        return None, "The voice service had a problem. The written report is above."

    # Save the audio to a temp file that Gradio can play.
    with tempfile.NamedTemporaryFile(prefix="qb_voice_", suffix=".mp3", delete=False) as f:
        f.write(response.content)
        return f.name, ""


def voice_coach(raw_json):
    """
    Runs right after the analysis (so the report shows first, then the voice).
    Reads the analysis from the raw JSON box; does nothing if the analysis failed.
    """
    try:
        data = json.loads(raw_json) if raw_json else None
    except json.JSONDecodeError:
        data = None
    if not isinstance(data, dict) or "summary" not in data:
        return None, ""
    path, note = speak(coaching_script(data))
    return path, (f"_{note}_" if note else "")


# ---------------------------------------------------------------------------
# Main Gradio handler
# ---------------------------------------------------------------------------

def analyze(video_path, play_info, progress=gr.Progress()):
    """
    Called when the user clicks Analyze. Always returns values for every output,
    so a failure shows a friendly message instead of crashing the app.
    """
    empty = ("", "", "", "", "", "", "", "")  # summary, receivers, defenders, eyes, throw, timeline, couldnt_see, raw

    try:
        progress(0.1, desc="Uploading video to Gemini...")
        data, raw, model = analyze_with_gemini(video_path, play_info)
        progress(0.9, desc="Building report...")
    except FriendlyError as e:
        raw = getattr(e, "raw", "")
        return (f"⚠️ {e}",) + empty[:-1] + (raw,)
    except Exception as e:  # Anything unexpected: log it, show a short message.
        print(f"[QB Vision] Unexpected error: {e!r}")
        return (f"⚠️ Something went wrong: {e}",) + empty

    return (
        "✅ Analysis complete.",
        render_summary(data, model),
        render_receivers(data),
        render_defenders(data),
        render_eyes(data),
        render_throw(data),
        render_timeline(data),
        render_couldnt_see(data),
        json.dumps(data, indent=2),
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

# Game-night broadcast look: dark navy, turf green, condensed sports type.
# Light and dark mode get the same colors so the site looks identical on every computer.
NAVY, PANEL, PANEL_2, LINE = "#070b14", "#0e1626", "#142038", "#22314d"
TEXT, MUTED, TURF, TURF_LIGHT = "#e6ecf5", "#8b9ab3", "#22c55e", "#4ade80"


def _same_in_dark_mode(**settings):
    """Gradio themes have separate dark-mode values; copy each setting to both."""
    return {**settings, **{f"{key}_dark": value for key, value in settings.items()}}


THEME = gr.themes.Base(
    primary_hue="green",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Barlow"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
).set(**_same_in_dark_mode(
    body_background_fill=NAVY,
    body_text_color=TEXT,
    body_text_color_subdued=MUTED,
    background_fill_primary=PANEL,
    background_fill_secondary=PANEL_2,
    border_color_primary=LINE,
    border_color_accent=TURF,
    color_accent_soft=PANEL_2,
    block_background_fill=PANEL,
    block_border_color=LINE,
    block_label_background_fill=PANEL_2,
    block_label_text_color=MUTED,
    block_title_text_color=TEXT,
    block_info_text_color=MUTED,
    panel_background_fill=PANEL,
    panel_border_color=LINE,
    input_background_fill=NAVY,
    input_border_color=LINE,
    input_border_color_focus=TURF,
    input_placeholder_color=MUTED,
    button_primary_background_fill=TURF,
    button_primary_background_fill_hover=TURF_LIGHT,
    button_primary_text_color="#03140a",
    button_secondary_background_fill=PANEL_2,
    button_secondary_text_color=TEXT,
    table_even_background_fill=PANEL,
    table_odd_background_fill=PANEL_2,
    table_border_color=LINE,
    table_text_color=TEXT,
    code_background_fill=NAVY,
    accordion_text_color=TEXT,
    link_text_color=TURF_LIGHT,
    loader_color=TURF,
))

# Condensed display font for the header, section titles and the grade.
HEAD = ('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
        'family=Barlow+Condensed:ital,wght@0,600;0,700;0,800;1,800&display=swap">')

CSS = """
body, gradio-app { background: #070b14 !important; }
.gradio-container { max-width: 1280px !important; }

/* Broadcast header with a yard-line stripe underneath */
.qb-header { padding: 18px 4px 0; }
.qb-topline { display:flex; align-items:center; justify-content:space-between; gap:12px; flex-wrap:wrap; }
.qb-brand { font-family:'Barlow Condensed', sans-serif; font-weight:800; font-style:italic;
            font-size:56px; line-height:1; letter-spacing:.01em; color:#fff; text-transform:uppercase;
            display:flex; align-items:center; gap:14px; }
.qb-brand .qb-bar { width:10px; height:46px; background:#22c55e; transform:skewX(-12deg); }
.qb-brand em { color:#22c55e; font-style:italic; }
.qb-pill { font-family:'Barlow Condensed', sans-serif; font-weight:700; letter-spacing:.12em;
           text-transform:uppercase; font-size:14px; color:#4ade80; border:1px solid #22c55e;
           padding:6px 12px; border-radius:4px; background:rgba(34,197,94,.08); }
.qb-pill::before { content:""; display:inline-block; width:8px; height:8px; border-radius:50%;
                   background:#22c55e; margin-right:8px; box-shadow:0 0 8px #22c55e; }
.qb-tagline { font-family:'Barlow Condensed', sans-serif; font-weight:600; font-size:20px;
              letter-spacing:.14em; text-transform:uppercase; color:#8b9ab3; margin:8px 0 6px 24px; }
.qb-intro { color:#c3cddc; font-size:15px; margin:0 0 16px 24px; max-width:820px; }
.qb-yards { height:12px; border-radius:2px; margin-bottom:8px;
            background:
              repeating-linear-gradient(90deg, rgba(255,255,255,.55) 0 2px, transparent 2px 48px),
              linear-gradient(90deg, #14532d, #166534 50%, #14532d); }

/* Section titles */
.qb-section { font-family:'Barlow Condensed', sans-serif; font-weight:700; font-size:22px;
              letter-spacing:.08em; text-transform:uppercase; color:#fff;
              display:flex; align-items:center; gap:10px; margin:10px 0 2px; }
.qb-section span { width:6px; height:20px; background:#22c55e; transform:skewX(-12deg); }

/* Status line + primary button */
.qb-status p { font-family:'Barlow Condensed', sans-serif; font-size:18px; letter-spacing:.04em; color:#c3cddc; }
button.primary { font-family:'Barlow Condensed', sans-serif !important; font-weight:800 !important;
                 font-size:20px !important; letter-spacing:.1em; text-transform:uppercase; }

/* Summary card with a scoreboard-style grade */
.qb-card { display:flex; gap:22px; padding:20px; border-radius:10px; border:1px solid #22314d;
           background:linear-gradient(135deg, #0e1626 0%, #101c33 100%);
           box-shadow:inset 0 3px 0 #22c55e; }
.qb-score { min-width:110px; height:118px; border-radius:8px; background:#03060c;
            border:2px solid var(--grade); display:flex; flex-direction:column;
            align-items:center; justify-content:center;
            box-shadow:0 0 18px color-mix(in srgb, var(--grade) 35%, transparent),
                       inset 0 0 22px rgba(0,0,0,.8); }
.qb-score-label { font-family:'Barlow Condensed', sans-serif; font-weight:700; font-size:13px;
                  letter-spacing:.25em; text-transform:uppercase; color:#8b9ab3; }
.qb-score-value { font-family:'Barlow Condensed', sans-serif; font-weight:800; font-size:68px;
                  line-height:1; color:var(--grade);
                  text-shadow:0 0 14px color-mix(in srgb, var(--grade) 60%, transparent); }
.qb-body { flex:1; }
.qb-label { font-family:'Barlow Condensed', sans-serif; font-weight:700; font-size:13px;
            text-transform:uppercase; letter-spacing:.14em; color:#4ade80; margin-bottom:3px; }
.qb-tip { font-size:19px; font-weight:600; color:#fff; margin-bottom:16px; line-height:1.35; }
.qb-row { display:grid; grid-template-columns:repeat(auto-fit, minmax(180px, 1fr)); gap:16px;
          margin-bottom:10px; padding-top:12px; border-top:1px solid #22314d; color:#dbe3ee; }
.qb-small { font-size:12px; color:#8b9ab3; margin-top:4px; }

/* Readable receiver/defender tables */
.qb-table { width:100%; border-collapse:separate; border-spacing:0; border:1px solid #22314d;
            border-radius:10px; overflow:hidden; font-size:15px; }
.qb-table th, .qb-table td { border:none !important; }
.qb-table th { text-align:left; font-family:'Barlow Condensed', sans-serif; font-weight:700;
               font-size:14px; letter-spacing:.1em; text-transform:uppercase; color:#4ade80;
               background:#142038; padding:10px 12px; }
.qb-table td { padding:12px; color:#e6ecf5; border-top:1px solid #22314d !important; vertical-align:top;
               line-height:1.4; background:#0e1626; }
.qb-table tr:nth-child(even) td { background:#111b2e; }
.qb-tag { display:inline-block; font-family:'Barlow Condensed', sans-serif; font-weight:800;
          font-size:14px; color:#03140a; background:#22c55e; border-radius:4px; padding:1px 7px;
          margin-right:4px; }
.qb-sub { font-size:13px; color:#8b9ab3; margin-top:3px; }
.qb-badge { display:inline-block; font-weight:700; font-size:13px; padding:3px 10px; border-radius:999px; }
.qb-yes { background:rgba(34,197,94,.15); color:#4ade80; border:1px solid #22c55e; }
.qb-no { background:rgba(239,68,68,.12); color:#f87171; border:1px solid #ef4444; }
.qb-empty { color:#8b9ab3; padding:8px 0; }

@media (max-width: 640px) {
  /* Phones: each table row becomes a card with a label above every value */
  .qb-table, .qb-table tbody, .qb-table tr, .qb-table td { display:block; width:100%; }
  .qb-table { border:none; }
  .qb-table thead { display:none; }
  .qb-table tr { border:1px solid #22314d; border-radius:10px; margin-bottom:10px; overflow:hidden; }
  .qb-table td { border-top:none !important; padding:6px 12px; }
  .qb-table td:first-child { padding-top:12px; }
  .qb-table td:last-child { padding-bottom:12px; }
  .qb-table td:not(:first-child)::before { content:attr(data-label); display:block;
      font-family:'Barlow Condensed', sans-serif; font-weight:700; font-size:12px;
      letter-spacing:.1em; text-transform:uppercase; color:#4ade80; }
  .qb-brand { font-size:40px; }
  .qb-card { flex-direction:column; }
}
"""


def find_examples():
    """Any .mp4/.mov files in examples/ become clickable examples."""
    if not EXAMPLES_DIR.is_dir():
        return []
    clips = sorted(p for p in EXAMPLES_DIR.iterdir() if p.suffix.lower() in {".mp4", ".mov"})
    return [[str(p), ""] for p in clips]


def section(title):
    """A broadcast-style section title with a turf-green bar."""
    return gr.HTML(f'<div class="qb-section"><span></span>{html.escape(title)}</div>')


with gr.Blocks(title="QB Vision") as demo:
    gr.HTML(
        """
<div class="qb-header">
  <div class="qb-topline">
    <div class="qb-brand"><span class="qb-bar"></span>QB <em>Vision</em></div>
    <div class="qb-pill">Smart-glasses film</div>
  </div>
  <div class="qb-tagline">Film breakdown from the quarterback's eyes</div>
  <div class="qb-intro">Upload first-person footage of <b>one play</b> from a quarterback's smart
  glasses (real speed). Gemini breaks down the coverage, the receivers, where the QB's eyes
  went, and the throw.</div>
  <div class="qb-yards"></div>
</div>"""
    )

    with gr.Row():
        with gr.Column(scale=1):
            video_in = gr.Video(label="Play clip", sources=["upload"])
            play_info_in = gr.Textbox(
                label="Play info (optional)",
                placeholder="e.g. Shotgun trips right, Mesh concept. Offense in white jerseys.",
                lines=3,
            )
            analyze_btn = gr.Button("Analyze play", variant="primary")

            examples = find_examples()
            if examples:
                gr.Examples(examples=examples, inputs=[video_in, play_info_in],
                            label="No glasses footage? Try an example clip")

        with gr.Column(scale=2):
            status_out = gr.Markdown("Upload a clip and click **Analyze play**.", elem_classes="qb-status")
            summary_out = gr.HTML()
            voice_out = gr.Audio(
                label="Coach's voice", type="filepath", autoplay=True, interactive=False,
                visible=bool(os.environ.get("ELEVENLABS_API_KEY")),  # Hidden without a key.
                waveform_options=gr.WaveformOptions(waveform_color="#22314d",
                                                    waveform_progress_color="#22c55e"),
            )
            voice_note = gr.Markdown(elem_classes="qb-status")

            section("Your receivers")
            receivers_out = gr.HTML()
            section("The defense")
            defenders_out = gr.HTML()

            with gr.Row():
                with gr.Column():
                    section("Where your eyes went")
                    eyes_out = gr.Markdown()
                with gr.Column():
                    section("The throw")
                    throw_out = gr.Markdown()

            with gr.Row():
                with gr.Column():
                    section("Play-by-play")
                    timeline_out = gr.Markdown()
                with gr.Column():
                    section("What the camera missed")
                    couldnt_see_out = gr.Markdown()

            with gr.Accordion("Technical details (raw AI output)", open=False):
                raw_out = gr.Code(language="json")

    analyze_btn.click(
        analyze,
        inputs=[video_in, play_info_in],
        outputs=[status_out, summary_out, receivers_out, defenders_out, eyes_out,
                 throw_out, timeline_out, couldnt_see_out, raw_out],
    ).then(
        # After the report is on screen, generate the spoken coaching.
        voice_coach,
        inputs=[raw_out],
        outputs=[voice_out, voice_note],
    )

    # Hidden, never-clicked button that satisfies ZeroGPU's startup check (see top of file).
    if _zerogpu_noop is not None:
        gr.Button(visible=False).click(_zerogpu_noop, api_visibility="private")


if __name__ == "__main__":
    # Hosts like Render tell the app which port to use via $PORT and need it to listen
    # on all interfaces (0.0.0.0). Locally this falls back to http://127.0.0.1:7860.
    port = os.environ.get("PORT")
    demo.launch(
        theme=THEME,
        css=CSS,
        head=HEAD,
        server_name="0.0.0.0" if port else None,
        server_port=int(port) if port else None,
    )
