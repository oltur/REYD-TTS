# REYD TTS — Yiddish Text-to-Speech

A console and web application for synthesising Yiddish speech, wrapping the [REYD-TTS/FastSpeech2](https://github.com/REYD-TTS/FastSpeech2) pretrained models.

## Setup

```bash
python3 tts.py --setup
```

Downloads the pretrained models (~1 GB) and preprocessed data (~1.8 GB), installs Python dependencies. Run once.

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

- Models: [REYD-TTS](https://github.com/REYD-TTS) by Jacob J. Webber, Samuel Lo, Isaac L. Bleaman
- Vocoder: [HiFi-GAN](https://github.com/jik876/hifi-gan)
- Dataset: University of Edinburgh DataShare ([19350539](https://figshare.com/articles/dataset/19350539))
