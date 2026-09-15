#!/usr/bin/env bash
# REYD TTS — Ubuntu server setup
# Cloudflare Tunnel is configured separately.
# Usage: sudo bash server_setup.sh

set -euo pipefail

APP_DIR="${1:-/opt/reyd-tts}"
TTS_PORT=8080
TTS_USER="reyd"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  App dir : $APP_DIR"
echo "  Port    : $TTS_PORT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── 1. system packages ─────────────────────────────────────────────────────────
echo "▶ Installing system packages ..."
apt-get update -qq
apt-get install -y -qq curl unzip software-properties-common

# Ensure Python 3.8+ (server may have only 3.6)
PY_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)" 2>/dev/null || echo 0)
if [[ "$PY_MINOR" -lt 8 ]]; then
    echo "▶ Python 3.6 detected — installing Python 3.10 via pyenv ..."
    apt-get install -y -qq make build-essential libssl-dev zlib1g-dev \
        libbz2-dev libreadline-dev libsqlite3-dev libffi-dev \
        libncurses5-dev liblzma-dev
    export PYENV_ROOT="/opt/pyenv"
    curl -fsSL https://pyenv.run | PYENV_ROOT="$PYENV_ROOT" bash
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$($PYENV_ROOT/bin/pyenv init -)"
    pyenv install 3.10.14
    pyenv global 3.10.14
    chmod -R a+rX "$PYENV_ROOT"
    PYTHON="$PYENV_ROOT/versions/3.10.14/bin/python3"
else
    apt-get install -y -qq python3-venv
    PYTHON=python3
fi
echo "  Using: $($PYTHON --version)"

# ── 2. dedicated user ──────────────────────────────────────────────────────────
echo "▶ Creating system user '$TTS_USER' …"
id "$TTS_USER" &>/dev/null || useradd -r -m -d "$APP_DIR" -s /bin/bash "$TTS_USER"

# ── 3. copy app ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ "$SCRIPT_DIR" != "$APP_DIR" ]]; then
    echo "▶ Copying app to $APP_DIR …"
    mkdir -p "$APP_DIR"
    cp -r "$SCRIPT_DIR"/. "$APP_DIR/"
else
    echo "▶ App already at $APP_DIR."
fi
chown -R "$TTS_USER:$TTS_USER" "$APP_DIR"

# ── 4. python venv + deps ──────────────────────────────────────────────────────
echo "▶ Setting up Python virtualenv ..."
sudo -u "$TTS_USER" $PYTHON -m venv "$APP_DIR/venv"
sudo -u "$TTS_USER" "$APP_DIR/venv/bin/pip" install -q --upgrade pip
ARCH=$(uname -m)
if [[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]]; then
    echo "  ARM detected — installing torch for aarch64 ..."
    sudo -u "$TTS_USER" "$APP_DIR/venv/bin/pip" install -q torch
else
    echo "  x86_64 detected — installing torch CPU wheel ..."
    sudo -u "$TTS_USER" "$APP_DIR/venv/bin/pip" install -q \
        torch --index-url https://download.pytorch.org/whl/cpu
fi
sudo -u "$TTS_USER" "$APP_DIR/venv/bin/pip" install -q \
    matplotlib PyYAML g2p-en pypinyin Unidecode yiddish scipy inflect

# ── 5. models ──────────────────────────────────────────────────────────────────
CKPT="$APP_DIR/FastSpeech2/output/ckpt/yivo_respelled/100000.pth.tar"
if [[ -f "$CKPT" ]]; then
    echo "▶ Models already present, skipping download."
else
    echo "▶ Downloading models (~3 GB) …"
    sudo -u "$TTS_USER" "$APP_DIR/venv/bin/python" "$APP_DIR/tts.py" --setup
fi

# ── 6. systemd service ─────────────────────────────────────────────────────────
echo "▶ Creating systemd service …"
cat > /etc/systemd/system/reyd-tts.service << EOF
[Unit]
Description=REYD Yiddish TTS
After=network.target

[Service]
User=$TTS_USER
WorkingDirectory=$APP_DIR
ExecStart=$APP_DIR/venv/bin/python $APP_DIR/tts.py --web --port $TTS_PORT
Environment=PYTHONIOENCODING=utf-8
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable reyd-tts
systemctl restart reyd-tts

echo
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✓ TTS service: $(systemctl is-active reyd-tts)"
echo "  Logs:    journalctl -u reyd-tts -f"
echo "  Restart: systemctl restart reyd-tts"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
