---
title: QB Vision
emoji: 🏈
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.32.0
python_version: "3.12"
app_file: app.py
pinned: false
short_description: QB smart-glasses play breakdown with Gemini
---

# QB Vision

Analyzes first-person football footage from smart glasses worn by a quarterback.
Upload a clip of one play and Gemini breaks down coverage, receivers, defenders,
where the QB's eyes went, and the throw.

## Run locally

```bash
pip install -r requirements.txt
cp .env.example .env   # then put your Gemini key in .env
python app.py
```

## Deploy to Render (free)

1. On https://render.com, sign up and connect your GitHub account.
2. Click **New → Blueprint**, pick this repo. Render reads `render.yaml`.
3. When asked, paste your Gemini key for `GEMINI_API_KEY`, then click **Apply**.
4. Wait for the build (a few minutes). Your app is at `https://qb-vision-XXXX.onrender.com`.

Free Render apps sleep after ~15 minutes idle; the next visit takes about a minute to wake.

### Spoken coaching (optional)

After each analysis, ElevenLabs reads a short coaching message (grade, tip, and the
better choice). Add `ELEVENLABS_API_KEY` in the Render service's **Environment** tab to turn
it on. Optional: `ELEVENLABS_VOICE_ID` picks a different voice (default: "George").

## Deploy to Hugging Face Spaces

1. Create a new Space with the **Gradio** SDK on **ZeroGPU** hardware (the free option
   for Gradio Spaces) and push these files. The app needs no GPU; see the comment at
   the top of `app.py` for the small placeholder ZeroGPU requires.
2. In the Space's **Settings → Variables and secrets**, add a secret named `GEMINI_API_KEY`.
3. Optional: add `GEMINI_MODEL` to force a specific model.

On Spaces, remove `gradio` from `requirements.txt` (it's preinstalled there and
controlled by `sdk_version` above).

## Example clips

Put `.mp4` or `.mov` files in `examples/` and they show up as clickable examples.
Spaces stores large binary files with Git LFS (`git lfs track "*.mp4" "*.mov"`).

## Files

- `app.py`: the Gradio app
- `analysis_prompt.txt`: the prompt sent to Gemini (`{play_info}` is filled in by the app)
