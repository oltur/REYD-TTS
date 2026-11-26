# REYD TTS — Yiddish Text-to-Speech

A console and web application for synthesising Yiddish speech, wrapping the [REYD-TTS/FastSpeech2](https://github.com/REYD-TTS/FastSpeech2) pretrained models.

## Setup

```bash
python3 tts.py --setup
```

This is the **only** setup step — no need to separately clone or download anything. It clones [REYD-TTS/FastSpeech2](https://github.com/REYD-TTS/FastSpeech2) into `./FastSpeech2/` (not part of this repo — see `.gitignore`), downloads the pretrained models (~1 GB) and preprocessed data (~1.8 GB) from figshare, and installs all Python dependencies (`torch`, `scipy`, and the rest). Run once, needs a few GB of disk and a stable connection. Safe to re-run — it skips anything already present.

## Usage

### Console

```bash
# Speak text directly (plays audio on macOS)
python3 tts.py "אַ גוטן טאָג"

# From a file
python3 tts.py --file text.txt

# Save to WAV instead of playing
python3 tts.py "אַ גוטן טאָג" --output hello.wav

# Change speaker or orthography
python3 tts.py "אַ גוטן טאָג" --speaker 1 --orth hasidic

# Study mode — displays and speaks each line with pauses
python3 tts.py --file text.txt --study
```

### Web server

```bash
python3 tts.py --web              # default port 8080
python3 tts.py --web --port 9000  # custom port
```

Open `http://localhost:8080` in a browser for the UI, or call the API directly:

```bash
# Text as a query parameter (GET or POST)
curl "http://localhost:8080/synthesize?text=%D7%90%D7%B7%20%D7%92%D7%95%D7%98%D7%9F%20%D7%98%D7%90%D6%B8%D7%92" --output speech.wav

# Open in a browser and it plays immediately instead of downloading
http://localhost:8080/synthesize?text=...&autoplay=1

# Plain text body, returns WAV
curl -X POST http://localhost:8080/synthesize \
  --data-binary "אַ גוטן טאָג" --output speech.wav

# With options as query params
curl -X POST "http://localhost:8080/synthesize?orth=hasidic&speaker=1" \
  --data-binary "אַ גוטן טאָג" --output speech.wav

# From a file
curl -X POST http://localhost:8080/synthesize \
  --data-binary @text.txt --output speech.wav
```

## Options

| Flag | Values | Default |
|------|--------|---------|
| `--orth` | `yivo_respelled`, `yivo_original`, `hasidic` | `yivo_respelled` |
| `--speaker` | `0` male Lithuanian, `1` female Lithuanian, `2` male Polish | `0` |
| `--output` | path to `.wav` | play audio (console) / return bytes (web) |
| `--port` | any port number | `8080` |

## Server deployment (Ubuntu / Debian)

Copy files to the server and run the setup script:

```bash
rsync -avz /path/to/REYD-TTS/ user@server:/opt/reyd-tts/
ssh user@server "sudo bash /opt/reyd-tts/server_setup.sh"
```

The script installs Python (upgrading to 3.10 if needed), creates a systemd service, and starts the app on port 8080. Configure a Cloudflare Tunnel to expose it publicly.

## Credits

All of the speech here comes from the [REYD](https://github.com/REYD-TTS) project — Reading
Electronic Yiddish Documents — which built the first Yiddish text-to-speech dataset and system
and released the whole thing publicly. This repository is only an interface to their work.

**System and paper**

> Jacob J. Webber, Samuel K. Lo and Isaac L. Bleaman. *REYD — The First Yiddish Text-to-Speech
> Dataset and System.* Proceedings of Interspeech 2022, pp. 2363–2367.
> [doi:10.21437/Interspeech.2022-789](https://doi.org/10.21437/Interspeech.2022-789)

**Corpus**

> Webber, Jacob; Bleaman, Isaac; Lo, Samuel; King, Simon. (2022). *REYD Yiddish TTS Corpus*
> [dataset]. Centre for Speech Technology Research (CSTR), University of Edinburgh.
> [doi:10.7488/ds/3424](https://doi.org/10.7488/ds/3424)

The corpus was recorded at the Montreal Jewish Public Library and the University of Haifa. Its
source texts and audio are in the public domain, and it was made public with the permission of
the surviving relatives of the three readers.

**Also**

- Text preprocessing: the [`yiddish`](https://pypi.org/project/yiddish/) library by Isaac L. Bleaman
- Vocoder: [HiFi-GAN](https://github.com/jik876/hifi-gan)
- Base implementation: [FastSpeech2](https://github.com/ming024/FastSpeech2) by Chung-Ming Chien (MIT)

## License

© 2026 Oleksandr Turevskiy — [turevskiy.com](https://turevskiy.com)
