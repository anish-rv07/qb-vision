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
import time
from pathlib import Path

import gradio as gr
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


def as_list(value):
    """Make sure we always iterate over a list, even if the model returned something else."""
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


GRADE_COLORS = {"A": "#16a34a", "B": "#65a30d", "C": "#ca8a04", "D": "#ea580c", "F": "#dc2626"}


def render_summary(data, model):
    """The summary card at the top: grade, coaching tip, coverage, confidence."""
    s = data.get("summary") or {}
    grade = text(s.get("decision_grade"), "?")
    color = GRADE_COLORS.get(grade[:1].upper(), "#6b7280")
    coverage = s.get("coverage") or {}
    if not isinstance(coverage, dict):
        coverage = {"post_snap": coverage}
    e = lambda v: html.escape(text(v))  # Escape model text before putting it in HTML.

    reason = s.get("confidence_reason")
    return f"""
<div class="qb-card">
  <div class="qb-grade" style="background:{color}">{html.escape(grade)}</div>
  <div class="qb-body">
    <div class="qb-label">Coaching tip</div>
    <div class="qb-tip">{e(s.get("coaching_tip"))}</div>
    <div class="qb-row">
      <div><div class="qb-label">Coverage (pre-snap)</div>{e(coverage.get("pre_snap"))}</div>
      <div><div class="qb-label">Coverage (post-snap)</div>{e(coverage.get("post_snap"))}</div>
      <div><div class="qb-label">Confidence</div>{e(s.get("confidence"))}
        {f'<div class="qb-small">{e(reason)}</div>' if reason else ""}</div>
    </div>
    <div class="qb-small">Decision grade · analyzed by {html.escape(model)}</div>
  </div>
</div>"""


def render_receivers(data):
    rows = []
    for r in as_list(data.get("receivers")):
        r = r if isinstance(r, dict) else {"description": r}
        rows.append([text(r.get("id")), text(r.get("description")), text(r.get("position")),
                     text(r.get("route")), text(r.get("open_window")), text(r.get("open_at_throw"))])
    return rows


def render_defenders(data):
    rows = []
    for d in as_list(data.get("defenders")):
        d = d if isinstance(d, dict) else {"description": d}
        rows.append([text(d.get("id")), text(d.get("description")),
                     text(d.get("alignment")), text(d.get("behavior"))])
    return rows


def render_eyes(data):
    eyes = data.get("qb_eyes") or {}
    if not isinstance(eyes, dict):
        return text(eyes)
    progression = as_list(eyes.get("progression"))
    steps = " → ".join(text(p) for p in progression) if progression else "—"
    return (f"**Pre-snap scan:** {text(eyes.get('pre_snap_scan'))}\n\n"
            f"**Progression:** {steps}\n\n"
            f"**Stared down target:** {text(eyes.get('stared_down_target'), 'None')}")


def render_throw(data):
    t = data.get("throw") or {}
    if not isinstance(t, dict):
        return text(t)
    target = text(t.get("target"))
    if t.get("time"):
        target += f" (at {text(t.get('time'))})"
    return (f"**Target:** {target}\n\n"
            f"**Placement:** {text(t.get('placement'))}\n\n"
            f"**Pressure:** {text(t.get('pressure'))}\n\n"
            f"**Better option:** {text(t.get('better_option'), 'None. Good choice.')}")


def render_timeline(data):
    lines = []
    for item in as_list(data.get("timeline")):
        if isinstance(item, dict):
            when = item.get("time") or item.get("t") or item.get("timestamp")
            event = item.get("event") or item.get("description") or text(item)
            lines.append(f"- **{text(when)}**: {text(event)}" if when else f"- {text(event)}")
        else:
            lines.append(f"- {text(item)}")
    return "\n".join(lines) or "_No timeline returned._"


def render_couldnt_see(data):
    items = as_list(data.get("couldnt_see"))
    return "\n".join(f"- {text(i)}" for i in items) or "_Nothing flagged._"


# ---------------------------------------------------------------------------
# Main Gradio handler
# ---------------------------------------------------------------------------

def analyze(video_path, play_info, progress=gr.Progress()):
    """
    Called when the user clicks Analyze. Always returns values for every output,
    so a failure shows a friendly message instead of crashing the app.
    """
    empty = ("", [], [], "", "", "", "", "")  # summary, receivers, defenders, eyes, throw, timeline, couldnt_see, raw

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

CSS = """
.qb-card { display:flex; gap:20px; padding:18px; border-radius:12px;
           border:1px solid var(--border-color-primary); background:var(--background-fill-secondary); }
.qb-grade { min-width:84px; height:84px; border-radius:12px; color:white; font-size:44px;
            font-weight:700; display:flex; align-items:center; justify-content:center; }
.qb-body { flex:1; }
.qb-label { font-size:12px; text-transform:uppercase; letter-spacing:.05em; opacity:.65; margin-bottom:2px; }
.qb-tip { font-size:18px; font-weight:600; margin-bottom:14px; }
.qb-row { display:grid; grid-template-columns:repeat(auto-fit, minmax(180px, 1fr)); gap:14px; margin-bottom:10px; }
.qb-small { font-size:12px; opacity:.65; margin-top:4px; }
"""


def find_examples():
    """Any .mp4/.mov files in examples/ become clickable examples."""
    if not EXAMPLES_DIR.is_dir():
        return []
    clips = sorted(p for p in EXAMPLES_DIR.iterdir() if p.suffix.lower() in {".mp4", ".mov"})
    return [[str(p), ""] for p in clips]


with gr.Blocks(title="QB Vision") as demo:
    gr.Markdown(
        "# 🏈 QB Vision\n"
        "Upload first-person footage of **one play** from a quarterback's smart glasses "
        "(real speed). Gemini breaks down the coverage, the receivers, where the QB's eyes "
        "went, and the throw."
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
            status_out = gr.Markdown("Upload a clip and click **Analyze play**.")
            summary_out = gr.HTML()

            gr.Markdown("### Receivers")
            receivers_out = gr.Dataframe(
                headers=["ID", "Description", "Position", "Route", "Open window", "Open at throw"],
                interactive=False, wrap=True,
            )
            gr.Markdown("### Defenders")
            defenders_out = gr.Dataframe(
                headers=["ID", "Description", "Alignment", "Behavior"],
                interactive=False, wrap=True,
            )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 👀 QB eyes")
                    eyes_out = gr.Markdown()
                with gr.Column():
                    gr.Markdown("### 🎯 The throw")
                    throw_out = gr.Markdown()

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### ⏱️ Timeline")
                    timeline_out = gr.Markdown()
                with gr.Column():
                    gr.Markdown("### 🌫️ Couldn't see")
                    couldnt_see_out = gr.Markdown()

            with gr.Accordion("Raw JSON", open=False):
                raw_out = gr.Code(language="json")

    analyze_btn.click(
        analyze,
        inputs=[video_in, play_info_in],
        outputs=[status_out, summary_out, receivers_out, defenders_out, eyes_out,
                 throw_out, timeline_out, couldnt_see_out, raw_out],
    )

    # Hidden, never-clicked button that satisfies ZeroGPU's startup check (see top of file).
    if _zerogpu_noop is not None:
        gr.Button(visible=False).click(_zerogpu_noop, api_visibility="private")


if __name__ == "__main__":
    # Hosts like Render tell the app which port to use via $PORT and need it to listen
    # on all interfaces (0.0.0.0). Locally this falls back to http://127.0.0.1:7860.
    port = os.environ.get("PORT")
    demo.launch(
        css=CSS,
        server_name="0.0.0.0" if port else None,
        server_port=int(port) if port else None,
    )
