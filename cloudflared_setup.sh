#!/usr/bin/env bash
# Install cloudflared and join the existing yiddish-tts.turevskiy.kharkiv.ua tunnel
# from this server. Run with: sudo bash cloudflared_setup.sh <TUNNEL_TOKEN>
#   or: CLOUDFLARE_TUNNEL_TOKEN=... sudo -E bash cloudflared_setup.sh
set -euo pipefail

TOKEN="${1:-${CLOUDFLARE_TUNNEL_TOKEN:-}}"
if [[ -z "$TOKEN" ]]; then
    echo "Usage: sudo bash cloudflared_setup.sh <TUNNEL_TOKEN>" >&2
    echo "   or: CLOUDFLARE_TUNNEL_TOKEN=... sudo -E bash cloudflared_setup.sh" >&2
    exit 1
fi

echo "▶ Downloading cloudflared (amd64) ..."
curl -fsSL -o /tmp/cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
dpkg -i /tmp/cloudflared.deb
rm -f /tmp/cloudflared.deb

echo "▶ Installing as a system service using the tunnel token ..."
cloudflared service install "$TOKEN"

sleep 2
echo
echo "✓ cloudflared: $(systemctl is-active cloudflared)"
echo "  Logs:    journalctl -u cloudflared -f"
echo "  Version: $(cloudflared --version)"
