#!/usr/bin/env python3
"""
REYD Yiddish TTS — console/web app wrapping REYD-TTS/FastSpeech2.

Usage:
  python tts.py "אַ גוטן טאָג"
  python tts.py --file input.txt --output speech.wav
  python tts.py --web [--port 8080]
  python tts.py --setup

Web API:
  GET/POST /synthesize?text=...&orth=yivo_respelled&speaker=0&autoplay=1
  (POST may also send text as the plain-text request body instead of ?text=)
  → audio/wav, or an autoplaying HTML page if autoplay is set
"""

import argparse
import base64
import io
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
REPO_DIR = ROOT / "FastSpeech2"
FIGSHARE_MODELS_URL = "https://ndownloader.figshare.com/files/34363709"
FIGSHARE_TEXTGRIDS = {
    "yivo_respelled": "https://ndownloader.figshare.com/files/34363712",
    "yivo_original":  "https://ndownloader.figshare.com/files/34363715",
    "hasidic":        "https://ndownloader.figshare.com/files/34363718",
}

ORTHOGRAPHIES = ("yivo_respelled", "yivo_original", "hasidic")
SPEAKERS = {0: "male Lithuanian", 1: "female Lithuanian", 2: "male Polish"}


# ── setup ──────────────────────────────────────────────────────────────────────

def run(cmd, **kwargs):
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        sys.exit(f"Command failed: {' '.join(str(c) for c in cmd)}")
    return result


def setup():
    if not REPO_DIR.exists():
        print("Cloning REYD-TTS/FastSpeech2 ...")
        run(["git", "clone", "https://github.com/REYD-TTS/FastSpeech2.git", str(REPO_DIR)])
    else:
        print("Repo already present.")

    ckpt_marker = REPO_DIR / "output" / "ckpt" / "yivo_respelled" / "100000.pth.tar"
    if not ckpt_marker.exists():
        models_zip = ROOT / "pretrained_models.zip"
        if not models_zip.exists():
            print("Downloading pretrained models (~1 GB) ...")
            run(["curl", "-L", "--progress-bar", "-o", str(models_zip), FIGSHARE_MODELS_URL])
        else:
            print("Zip already downloaded, unpacking ...")
        run(["unzip", "-o", str(models_zip), "-d", str(REPO_DIR)])
        for orth in ORTHOGRAPHIES:
            ckpt_dir = REPO_DIR / "output" / "ckpt" / orth
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            src = REPO_DIR / orth / "100000.pth.tar"
            dst = ckpt_dir / "100000.pth.tar"
            if src.exists() and not dst.exists():
                src.rename(dst)
    else:
        print("Pretrained models already present.")

    for name in ("generator_universal", "generator_LJSpeech"):
        pth = REPO_DIR / "hifigan" / f"{name}.pth.tar"
        if not pth.exists():
            run(["unzip", "-o", str(pth.with_suffix(".tar.zip")), "-d", str(REPO_DIR / "hifigan")])

    for orth, url in FIGSHARE_TEXTGRIDS.items():
        stats_file = REPO_DIR / "preprocessed_data" / f"yiddish_textgrids_{orth}" / "stats.json"
        if stats_file.exists():
            print(f"Preprocessed data for {orth} already present.")
            continue
        tg_zip = ROOT / f"textgrids_{orth}.zip"
        if not tg_zip.exists():
            print(f"Downloading textgrids for {orth} (~613 MB) ...")
            run(["curl", "-L", "--progress-bar", "-o", str(tg_zip), url])
        else:
            print(f"Textgrids zip for {orth} already downloaded, unpacking ...")
        run(["unzip", "-o", str(tg_zip), "-d", str(REPO_DIR)])

    try:
        import torch  # noqa: F401
        print("torch already installed.")
    except ImportError:
        print("Installing torch ...")
        if platform.system() == "Linux" and platform.machine() not in ("x86_64", "AMD64"):
            run([sys.executable, "-m", "pip", "install", "-q", "torch"])
        elif platform.system() == "Linux":
            run([sys.executable, "-m", "pip", "install", "-q", "torch",
                 "--index-url", "https://download.pytorch.org/whl/cpu"])
        else:
            run([sys.executable, "-m", "pip", "install", "-q", "torch"])

    run([sys.executable, "-m", "pip", "install", "-q",
         "scipy", "matplotlib", "PyYAML", "g2p-en", "inflect", "pypinyin", "Unidecode", "yiddish"])
    run([sys.executable, "-c",
         "import ssl, nltk; ssl._create_default_https_context = ssl._create_unverified_context; "
         "nltk.download('averaged_perceptron_tagger'); nltk.download('cmudict'); "
         "nltk.download('averaged_perceptron_tagger_eng')"])
    print("Setup complete.")


# ── text preprocessing ─────────────────────────────────────────────────────────

def preprocess_text(text: str, orth: str) -> str:
    try:
        import yiddish
    except ImportError:
        sys.exit("Package 'yiddish' not found. Run: python tts.py --setup")
    if orth == "hasidic":
        text = yiddish.strip_diacritics(text)
    else:
        text = yiddish.replace_with_precombined(text)
        if "respelled" in orth:
            text = yiddish.respell_loshn_koydesh(text)
    return re.sub(r"[\.!,;?\-\s]+$", "", text)


# ── in-process synthesis engine ────────────────────────────────────────────────

def _load_engine(orth: str):
    """Load FastSpeech2 + HiFi-GAN into memory. Called once per process."""
    import torch
    import yaml

    # Make FastSpeech2 modules importable and set CWD so relative paths work
    if str(REPO_DIR) not in sys.path:
        sys.path.insert(0, str(REPO_DIR))
    os.chdir(REPO_DIR)

    from utils.model import get_model, get_vocoder

    device = torch.device("cpu")
    config_base = REPO_DIR / "config" / orth
    preprocess_config = yaml.load(open(config_base / "preprocess.yaml"), Loader=yaml.FullLoader)
    model_config      = yaml.load(open(config_base / "model.yaml"),      Loader=yaml.FullLoader)
    train_config      = yaml.load(open(config_base / "train.yaml"),      Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    class _Args:
        restore_step = 100000

    model   = get_model(_Args(), configs, device, train=False)
    vocoder = get_vocoder(model_config, device)
    return {
        "model": model, "vocoder": vocoder, "configs": configs, "device": device,
        # Parsed once here rather than per request — it is read-only afterwards.
        "lexicon": _load_lexicon(preprocess_config["path"]["lexicon_path"]),
    }


def _load_lexicon(lexicon_path: str) -> dict:
    """Parse a FastSpeech2 lexicon file into {word: [phones]}."""
    lexicon = {}
    with open(lexicon_path) as f:
        for line in f:
            parts = re.split(r"\s+", line.strip())
            if parts and parts[0].lower() not in lexicon:
                lexicon[parts[0].lower()] = parts[1:]
    return lexicon


def _run_synthesis(engine: dict, text: str, orth: str, speaker_id: int) -> bytes:
    """Run inference in-process; return raw WAV bytes."""
    import numpy as np
    import torch
    from scipy.io import wavfile
    from utils.tools import to_device, synth_samples
    from text import text_to_sequence

    model, vocoder, configs, device = (
        engine["model"], engine["vocoder"], engine["configs"], engine["device"]
    )
    preprocess_config, model_config, train_config = configs

    processed = preprocess_text(text, orth)
    print(f"  text: {text!r} → preprocessed: {processed!r}")

    # Build phoneme sequence (Yiddish path from synthesize.py)
    lexicon_path = preprocess_config["path"]["lexicon_path"]
    lexicon = engine.get("lexicon") or _load_lexicon(lexicon_path)

    phones = []
    for w in re.split(r'([,;.\u201e\u201c"\-\?\!\(\)\s+])', processed):
        if w in lexicon:
            phones += lexicon[w]
        else:
            w = re.sub(r"[־']", "", w)
            for pair in [("ך","כ"),("ם","מ"),("ן","נ"),("ץ","צ")]:
                w = w.replace(*pair)
            if "yivo_respelled" in lexicon_path:
                w = w.replace("א", "")
            if "yivo" in lexicon_path:
                w = w.replace("ף", "פֿ")
            phones += list(w)

    phone_str = "{" + "}{".join(phones) + "}"
    phone_str = re.sub(r"\{[^\w\sאאַאָבבֿגדהווּװױזחטייִײײַכּכךלמםנןסעפפּפֿףצץקרששׂתּת]\}", "{sp}", phone_str)
    phone_str = phone_str.replace("}{", " ")

    cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
    sequence = np.array(text_to_sequence(phone_str, cleaners))

    ids       = raw_texts = [processed[:100]]
    speakers  = np.array([speaker_id])
    texts     = np.array([sequence])
    text_lens = np.array([len(sequence)])
    batch     = (ids, raw_texts, speakers, texts, text_lens, max(text_lens))
    batch     = to_device(batch, device)

    with torch.no_grad():
        output = model(*(batch[2:]), p_control=1.0, e_control=1.0, d_control=1.0)

    # Run vocoder and capture WAV in memory
    from utils.model import vocoder_infer
    mel_predictions = output[1].transpose(1, 2)
    lengths = output[9] * preprocess_config["preprocessing"]["stft"]["hop_length"]
    wavs = vocoder_infer(mel_predictions, vocoder, model_config, preprocess_config, lengths=lengths)

    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    buf = io.BytesIO()
    wavfile.write(buf, sampling_rate, wavs[0])
    return buf.getvalue()


def ensure_textgrids(orth):
    stats_file = REPO_DIR / "preprocessed_data" / f"yiddish_textgrids_{orth}" / "stats.json"
    if stats_file.exists():
        return
    url = FIGSHARE_TEXTGRIDS[orth]
    tg_zip = ROOT / f"textgrids_{orth}.zip"
    if not tg_zip.exists():
        print(f"Downloading textgrids for {orth} (~613 MB) ...")
        run(["curl", "-L", "--progress-bar", "-o", str(tg_zip), url])
    else:
        print(f"Unpacking textgrids for {orth} ...")
    run(["unzip", "-o", str(tg_zip), "-d", str(REPO_DIR)])


# ── CLI synthesis (subprocess, for one-shot use) ───────────────────────────────

def synthesize_cli(text: str, orth: str, speaker_id: int, output):
    if not REPO_DIR.exists():
        sys.exit("Repo not found. Run: python tts.py --setup")
    ensure_textgrids(orth)
    (REPO_DIR / "output" / "result" / orth).mkdir(parents=True, exist_ok=True)

    processed = preprocess_text(text, orth)
    print(f"Input text  : {text}")
    print(f"Preprocessed: {processed}")
    print(f"Orthography : {orth}  |  Speaker: {speaker_id} ({SPEAKERS[speaker_id]})")

    config_base = REPO_DIR / "config" / orth
    cmd = [
        sys.executable, str(REPO_DIR / "synthesize.py"),
        "--text", processed, "--speaker_id", str(speaker_id),
        "--restore_step", "100000", "--mode", "single",
        "-p", str(config_base / "preprocess.yaml"),
        "-m", str(config_base / "model.yaml"),
        "-t", str(config_base / "train.yaml"),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(cmd, cwd=str(REPO_DIR), env=env)
    if result.returncode != 0:
        sys.exit("Synthesis failed.")

    result_dir = REPO_DIR / "output" / "result" / orth
    wavs = sorted(result_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not wavs:
        print("Synthesis ran but no WAV file found in", result_dir)
        return
    latest = wavs[0]
    if output:
        import shutil
        shutil.copy2(latest, Path(output))
        print(f"\nSaved to: {Path(output).resolve()}")
    else:
        print(f"\nPlaying: {latest.resolve()}")
        subprocess.run(["afplay", str(latest)])


# ── web UI ─────────────────────────────────────────────────────────────────────

FAVICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
  <rect width="64" height="64" rx="14" fill="#f5edd6"/>
  <rect x="2" y="2" width="60" height="60" rx="12" fill="none" stroke="#8b6f47" stroke-width="3"/>
  <text x="32" y="47" font-family="Georgia, 'Times New Roman', serif" font-size="40" font-weight="700" text-anchor="middle" fill="#9b3a1a">ר</text>
</svg>"""
FAVICON_DATA_URI = "data:image/svg+xml," + quote(FAVICON_SVG)

AUTOPLAY_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>REYD TTS</title></head>
<body style="background:#1a1410;display:flex;align-items:center;justify-content:center;height:100vh;margin:0;">
<audio autoplay controls src="data:audio/wav;base64,{audio_b64}"></audio>
</body></html>"""

UI_HTML = """<!DOCTYPE html>
<html lang="yi" dir="rtl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>רייד — ייִדיש טעקסט-צו-רעדן</title>
<link rel="icon" type="image/svg+xml" href="__FAVICON_HREF__">
<style>
  @import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,600;1,400&family=Frank+Ruhl+Libre:wght@400;700&display=swap');

  :root {
    --parchment: #f5edd6;
    --ink: #1a1208;
    --sepia: #8b6f47;
    --rust: #9b3a1a;
    --cream: #faf4e8;
    --border: #c9a96e;
    --shadow: rgba(26,18,8,0.15);
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--parchment);
    background-image:
      url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='400' height='400'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/%3E%3CfeColorMatrix type='saturate' values='0'/%3E%3C/filter%3E%3Crect width='400' height='400' filter='url(%23n)' opacity='0.04'/%3E%3C/svg%3E");
    color: var(--ink);
    font-family: 'EB Garamond', Georgia, serif;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 2rem 1rem 4rem;
  }

  header {
    text-align: center;
    margin-bottom: 2.5rem;
    position: relative;
  }

  .masthead-rule {
    width: 100%;
    max-width: 560px;
    height: 3px;
    background: linear-gradient(90deg, transparent, var(--sepia), var(--rust), var(--sepia), transparent);
    margin: 0.5rem auto;
  }

  .masthead-rule.thin {
    height: 1px;
    margin-top: 0.25rem;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
  }

  h1 {
    font-family: 'Frank Ruhl Libre', 'Times New Roman', serif;
    font-size: clamp(2.2rem, 6vw, 3.4rem);
    font-weight: 700;
    letter-spacing: 0.02em;
    color: var(--ink);
    line-height: 1.1;
    direction: rtl;
  }

  .subtitle {
    font-family: 'EB Garamond', serif;
    font-style: italic;
    font-size: 1rem;
    color: var(--sepia);
    letter-spacing: 0.08em;
    direction: ltr;
    margin-top: 0.3rem;
  }

  main {
    width: 100%;
    max-width: 560px;
    background: var(--cream);
    border: 1px solid var(--border);
    border-radius: 2px;
    box-shadow: 0 4px 24px var(--shadow), inset 0 1px 0 rgba(255,255,255,0.6);
    padding: 2rem;
  }

  .field { margin-bottom: 1.4rem; }

  label {
    display: block;
    font-size: 0.78rem;
    font-family: 'EB Garamond', serif;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--sepia);
    margin-bottom: 0.4rem;
    direction: ltr;
  }

  textarea {
    width: 100%;
    min-height: 130px;
    padding: 0.8rem 1rem;
    font-family: 'Frank Ruhl Libre', serif;
    font-size: 1.35rem;
    line-height: 1.7;
    direction: rtl;
    unicode-bidi: embed;
    background: var(--parchment);
    border: 1px solid var(--border);
    border-radius: 1px;
    color: var(--ink);
    resize: vertical;
    outline: none;
    transition: border-color 0.2s, box-shadow 0.2s;
  }

  textarea:focus {
    border-color: var(--rust);
    box-shadow: 0 0 0 3px rgba(155,58,26,0.1);
  }

  textarea::placeholder { color: #b8a07a; }

  .row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    direction: ltr;
  }

  select {
    width: 100%;
    padding: 0.55rem 0.8rem;
    font-family: 'EB Garamond', serif;
    font-size: 0.95rem;
    background: var(--parchment);
    border: 1px solid var(--border);
    border-radius: 1px;
    color: var(--ink);
    cursor: pointer;
    outline: none;
    appearance: none;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M0 0l5 6 5-6z' fill='%238b6f47'/%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: right 0.7rem center;
    padding-right: 2rem;
    transition: border-color 0.2s;
  }

  select:focus { border-color: var(--rust); }

  button {
    width: 100%;
    padding: 0.8rem 1.5rem;
    margin-top: 0.5rem;
    font-family: 'Frank Ruhl Libre', serif;
    font-size: 1.15rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    background: var(--ink);
    color: var(--parchment);
    border: none;
    border-radius: 1px;
    cursor: pointer;
    transition: background 0.2s, transform 0.1s;
    position: relative;
    overflow: hidden;
    direction: rtl;
  }

  button:hover { background: #2d1f0a; }
  button:active { transform: scale(0.99); }
  button:disabled { background: var(--sepia); cursor: not-allowed; }

  .spinner {
    display: none;
    width: 18px; height: 18px;
    border: 2px solid rgba(245,237,214,0.3);
    border-top-color: var(--parchment);
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    margin: 0 auto;
  }

  @keyframes spin { to { transform: rotate(360deg); } }

  .btn-text { display: block; }

  #player-wrap {
    display: none;
    margin-top: 1.6rem;
    padding-top: 1.4rem;
    border-top: 1px solid var(--border);
    direction: ltr;
  }

  .player-label {
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--sepia);
    margin-bottom: 0.6rem;
  }

  audio {
    width: 100%;
    accent-color: var(--rust);
  }

  .error-msg {
    display: none;
    margin-top: 1rem;
    padding: 0.7rem 1rem;
    background: #fdf0ec;
    border: 1px solid #e8b4a0;
    border-radius: 1px;
    color: var(--rust);
    font-size: 0.9rem;
    direction: ltr;
  }

  footer {
    margin-top: 2rem;
    font-size: 0.78rem;
    color: var(--sepia);
    letter-spacing: 0.06em;
    text-align: center;
    direction: ltr;
    line-height: 1.8;
  }
</style>
</head>
<body>

<header>
  <div class="masthead-rule"></div>
  <h1>רייד</h1>
  <div class="subtitle">Yiddish Text-to-Speech &middot; REYD-TTS</div>
  <div class="masthead-rule thin"></div>
</header>

<main>
  <div class="field">
    <label>טעקסט &mdash; Text</label>
    <textarea id="text">אַ גוטן טאָג</textarea>
  </div>

  <div class="row">
    <div class="field">
      <label>Orthography</label>
      <select id="orth">
        <option value="yivo_respelled">YIVO Respelled</option>
        <option value="yivo_original">YIVO Original</option>
        <option value="hasidic">Hasidic</option>
      </select>
    </div>
    <div class="field">
      <label>Speaker</label>
      <select id="speaker">
        <option value="0">Male &mdash; Lithuanian</option>
        <option value="1">Female &mdash; Lithuanian</option>
        <option value="2">Male &mdash; Polish</option>
      </select>
    </div>
  </div>

  <button id="btn" onclick="synthesize()">
    <span class="btn-text">רייד &mdash; Speak</span>
    <span class="spinner" id="spinner"></span>
  </button>

  <div class="error-msg" id="error"></div>

  <div id="player-wrap">
    <div class="player-label">Generated Audio</div>
    <audio id="audio" controls></audio>
  </div>
</main>

<footer>
  REYD &mdash; Reading Electronic Yiddish Documents &middot; <a href="/swagger" style="color:inherit;">API docs</a><br>
  &copy; 2026 Oleksandr Turevskiy &middot; <a href="https://turevskiy.com" style="color:inherit;">turevskiy.com</a>
</footer>

<script>
function updateShareURL() {
  const text    = document.getElementById('text').value;
  const orth    = document.getElementById('orth').value;
  const speaker = document.getElementById('speaker').value;
  const params  = new URLSearchParams();
  if (text) params.set('text', text);
  params.set('orth', orth);
  params.set('speaker', speaker);
  history.replaceState(null, '', `${location.pathname}?${params.toString()}`);
}

window.addEventListener('DOMContentLoaded', () => {
  const params = new URLSearchParams(location.search);
  const text = params.get('text');
  if (text) document.getElementById('text').value = text;
  const orth = params.get('orth');
  if (orth) document.getElementById('orth').value = orth;
  const speaker = params.get('speaker');
  if (speaker) document.getElementById('speaker').value = speaker;

  document.getElementById('text').addEventListener('input', updateShareURL);
  document.getElementById('orth').addEventListener('change', updateShareURL);
  document.getElementById('speaker').addEventListener('change', updateShareURL);

  if (params.get('autoplay') === '1') synthesize();
});

async function synthesize() {
  const text = document.getElementById('text').value.trim();
  if (!text) { showError('Please enter some Yiddish text.'); return; }

  const orth    = document.getElementById('orth').value;
  const speaker = document.getElementById('speaker').value;
  const btn     = document.getElementById('btn');
  const spinner = document.getElementById('spinner');
  const btnText = btn.querySelector('.btn-text');

  btn.disabled = true;
  btnText.style.display = 'none';
  spinner.style.display = 'block';
  hideError();
  document.getElementById('player-wrap').style.display = 'none';

  try {
    const res = await fetch(`/synthesize?orth=${orth}&speaker=${speaker}`, {
      method: 'POST',
      headers: { 'Content-Type': 'text/plain; charset=utf-8' },
      body: text
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: 'Synthesis failed' }));
      showError(err.error || 'Synthesis failed');
      return;
    }

    const blob = await res.blob();
    const url  = URL.createObjectURL(blob);
    const audio = document.getElementById('audio');
    audio.src = url;
    document.getElementById('player-wrap').style.display = 'block';
    audio.play();
  } catch (e) {
    showError('Network error: ' + e.message);
  } finally {
    btn.disabled = false;
    btnText.style.display = 'block';
    spinner.style.display = 'none';
  }
}

function showError(msg) {
  const el = document.getElementById('error');
  el.textContent = msg;
  el.style.display = 'block';
}
function hideError() {
  document.getElementById('error').style.display = 'none';
}

document.getElementById('text').addEventListener('keydown', e => {
  if (e.ctrlKey && e.key === 'Enter') synthesize();
});
</script>
</body>
</html>"""

UI_HTML = UI_HTML.replace("__FAVICON_HREF__", FAVICON_DATA_URI)


# ── API docs (Swagger/OpenAPI) ───────────────────────────────────────────────────

OPENAPI_SPEC = {
    "openapi": "3.0.3",
    "info": {
        "title": "REYD Yiddish TTS API",
        "description": "Synthesizes Yiddish speech from text using FastSpeech2 + HiFi-GAN.",
        "version": "1.0.0",
    },
    "paths": {
        "/status": {
            "get": {
                "summary": "Health check",
                "responses": {
                    "200": {
                        "description": "Service status and available options",
                        "content": {"application/json": {"schema": {
                            "type": "object",
                            "properties": {
                                "status": {"type": "string", "example": "ok"},
                                "orthographies": {"type": "array", "items": {"type": "string"}},
                                "speakers": {"type": "object"},
                            },
                        }}},
                    }
                },
            }
        },
        "/synthesize": {
            "parameters": [
                {"name": "text", "in": "query", "required": False, "schema": {"type": "string"},
                 "description": "Yiddish text to synthesize. Required unless sent as the POST request body."},
                {"name": "orth", "in": "query", "required": False,
                 "schema": {"type": "string", "enum": list(ORTHOGRAPHIES), "default": "yivo_respelled"},
                 "description": "Orthography/transliteration scheme."},
                {"name": "speaker", "in": "query", "required": False,
                 "schema": {"type": "integer", "enum": list(SPEAKERS.keys()), "default": 0},
                 "description": "0 = male Lithuanian, 1 = female Lithuanian, 2 = male Polish."},
                {"name": "autoplay", "in": "query", "required": False,
                 "schema": {"type": "string", "enum": ["0", "1"], "default": "0"},
                 "description": "If 1, returns an HTML page with an autoplaying <audio> element instead of raw WAV bytes."},
            ],
            "get": {
                "summary": "Synthesize speech (text via query parameter)",
                "parameters": [],
                "responses": {
                    "200": {
                        "description": "Synthesized audio (or an autoplaying HTML page if autoplay=1)",
                        "content": {
                            "audio/wav": {"schema": {"type": "string", "format": "binary"}},
                            "text/html": {"schema": {"type": "string"}},
                        },
                    },
                    "400": {"description": "Invalid request (missing/invalid text, orth, or speaker)"},
                },
            },
            "post": {
                "summary": "Synthesize speech (text via query parameter or plain-text request body)",
                "parameters": [],
                "requestBody": {
                    "required": False,
                    "content": {"text/plain": {"schema": {"type": "string"}, "example": "אַ גוטן טאָג"}},
                },
                "responses": {
                    "200": {
                        "description": "Synthesized audio (or an autoplaying HTML page if autoplay=1)",
                        "content": {
                            "audio/wav": {"schema": {"type": "string", "format": "binary"}},
                            "text/html": {"schema": {"type": "string"}},
                        },
                    },
                    "400": {"description": "Invalid request (missing/invalid text, orth, or speaker)"},
                },
            },
        },
    },
}

SWAGGER_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>REYD TTS API Docs</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/5.29.1/swagger-ui.css">
</head>
<body>
<div id="swagger-ui"></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/5.29.1/swagger-ui-bundle.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/5.29.1/swagger-ui-standalone-preset.js"></script>
<script>
  window.onload = () => {
    SwaggerUIBundle({
      url: '/openapi.json',
      dom_id: '#swagger-ui',
      presets: [SwaggerUIBundle.presets.apis, SwaggerUIStandalonePreset],
    });
  };
</script>
</body>
</html>"""


# ── web server ─────────────────────────────────────────────────────────────────

def serve(default_orth: str, default_speaker: int, port: int):
    import torch

    # Requests are served concurrently, so bound both how many syntheses run at
    # once and how many CPU threads each one may use. Letting torch fan out over
    # every core per request just oversubscribes the box once more than one user
    # shows up.
    max_workers = max(1, min(4, (os.cpu_count() or 4) // 4))
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // max_workers))
    synth_sem = threading.Semaphore(max_workers)

    ensure_textgrids(default_orth)

    # Load every orthography up front. _load_engine() calls os.chdir(), which is
    # process-global and would race between threads if it ran per request.
    print(f"Loading models {ORTHOGRAPHIES} ...")
    engines = {}
    for _orth in ORTHOGRAPHIES:
        print(f"  {_orth} ...")
        engines[_orth] = _load_engine(_orth)

    print(f"Models loaded ({max_workers} workers, "
          f"{torch.get_num_threads()} torch threads each). "
          f"Listening on http://0.0.0.0:{port}")
    print("POST /synthesize  {\"text\": \"...\", \"orth\": \"yivo_respelled\", \"speaker\": 0}")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            print(f"  {self.address_string()} {fmt % args}")

        def do_GET(self):
            path, _, query = self.path.partition("?")
            path = path.rstrip("/")
            if path == "":
                body = UI_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", len(body))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/status":
                body = json.dumps({"status": "ok", "orthographies": ORTHOGRAPHIES, "speakers": SPEAKERS}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", len(body))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/favicon.ico":
                body = FAVICON_SVG.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "image/svg+xml")
                self.send_header("Content-Length", len(body))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/swagger":
                body = SWAGGER_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", len(body))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/openapi.json":
                body = json.dumps(OPENAPI_SPEC).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", len(body))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/synthesize":
                qs = {k: v[0] for k, v in parse_qs(query).items()}
                self._synthesize(qs)
            else:
                self.send_error(404)

        def do_POST(self):
            path, _, query = self.path.partition("?")
            if path.rstrip("/") != "/synthesize":
                self.send_error(404)
                return
            qs = {k: v[0] for k, v in parse_qs(query).items()}
            length = int(self.headers.get("Content-Length", 0))
            if length and "text" not in qs:
                qs["text"] = self.rfile.read(length).decode("utf-8")
            elif length:
                self.rfile.read(length)  # drain body even if text came from the query string
            self._synthesize(qs)

        def _synthesize(self, qs):
            try:
                text     = qs.get("text", "").strip()
                orth     = qs.get("orth", default_orth)
                speaker  = int(qs.get("speaker", default_speaker))
                autoplay = qs.get("autoplay", "").lower() in ("1", "true", "yes")
                if not text:
                    raise ValueError("'text' is required (as a query parameter or request body)")
                if orth not in ORTHOGRAPHIES:
                    raise ValueError(f"'orth' must be one of {ORTHOGRAPHIES}")
                if speaker not in SPEAKERS:
                    raise ValueError(f"'speaker' must be 0, 1, or 2")

                eng = engines[orth]
                with synth_sem:
                    wav_bytes = _run_synthesis(eng, text, orth, speaker)

                if autoplay:
                    b64 = base64.b64encode(wav_bytes).decode("ascii")
                    body = AUTOPLAY_HTML.format(audio_b64=b64).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", len(body))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_response(200)
                    self.send_header("Content-Type", "audio/wav")
                    self.send_header("Content-Length", len(wav_bytes))
                    self.send_header("Content-Disposition", 'attachment; filename="speech.wav"')
                    self.end_headers()
                    self.wfile.write(wav_bytes)

            except (ValueError, KeyError, json.JSONDecodeError) as e:
                msg = json.dumps({"error": str(e)}).encode()
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", len(msg))
                self.end_headers()
                self.wfile.write(msg)
            except Exception as e:
                msg = json.dumps({"error": str(e)}).encode()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", len(msg))
                self.end_headers()
                self.wfile.write(msg)

    class ReusableHTTPServer(ThreadingHTTPServer):
        allow_reuse_address = True
        daemon_threads = True

    ReusableHTTPServer(("0.0.0.0", port), Handler).serve_forever()


# ── study mode ─────────────────────────────────────────────────────────────────

def study(lines: list, orth: str, speaker_id: int, show_for: float, interval: float):
    """Speak each line, display it, then pause before the next."""
    import time

    ensure_textgrids(orth)
    print(f"Loading model ({orth}) ...")
    engine = _load_engine(orth)
    total = len(lines)

    for i, line in enumerate(lines, 1):
        os.system("clear")
        print(f"\n{'─' * 50}")
        print(f"  [{i}/{total}]")
        print(f"\n  {line}\n")
        print(f"{'─' * 50}")

        wav_bytes = _run_synthesis(engine, line, orth, speaker_id)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            tmp = f.name

        subprocess.Popen(["afplay", tmp])   # non-blocking so display stays up
        time.sleep(show_for)
        subprocess.run(["killall", "afplay"], capture_output=True)
        os.unlink(tmp)

        if i < total:
            time.sleep(interval)

    os.system("clear")
    print("Done.")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="REYD Yiddish TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("text", nargs="?", help="Yiddish text to synthesise")
    parser.add_argument("--file", "-f", help="Text file to read input from")
    parser.add_argument("--orth", "-o", choices=ORTHOGRAPHIES, default="yivo_respelled")
    parser.add_argument("--speaker", "-s", type=int, choices=list(SPEAKERS), default=0,
                        metavar="{0,1,2}",
                        help="0=male Lithuanian, 1=female Lithuanian, 2=male Polish")
    parser.add_argument("--output", "-O", help="Save WAV to this path (default: play audio)")
    parser.add_argument("--web", action="store_true", help="Start web server")
    parser.add_argument("--port", type=int, default=8080, help="Web server port (default: 8080)")
    parser.add_argument("--study", action="store_true",
                        help="Study mode: display and speak each line with pauses")
    parser.add_argument("--setup", action="store_true", help="Download models and dependencies")

    args = parser.parse_args()

    if args.setup:
        setup()
        return

    if args.web:
        serve(args.orth, args.speaker, args.port)
        return

    if args.file:
        p = Path(args.file)
        if not p.exists():
            sys.exit(f"File not found: {args.file}")
        text = p.read_text(encoding="utf-8").strip()
    elif args.text:
        text = args.text.strip()
    else:
        parser.error("Provide text as argument, --file, or use --web.")

    if not text:
        sys.exit("Input text is empty.")

    if args.study:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        study(lines, args.orth, args.speaker, show_for=8.0, interval=4.0)
    else:
        synthesize_cli(text, args.orth, args.speaker, args.output)


if __name__ == "__main__":
    main()
