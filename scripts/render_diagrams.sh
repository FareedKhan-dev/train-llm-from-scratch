#!/usr/bin/env bash
# Render the hand-drawn, colour-coded diagram sources (docs/diagrams/src/*.mmd) to SVGs
# (docs/diagrams/*.svg) that are embedded as images in the docs. We pre-render because
# GitHub's live Mermaid does not reliably support `look: handDrawn`; embedding the SVG makes
# the hand-drawn look show everywhere.
#
# One-time setup:
#   sudo apt-get install -y nodejs                      # Node >= 18
#   sudo npm install -g @mermaid-js/mermaid-cli         # provides `mmdc`
#   # a Chrome/Chromium for headless rendering (e.g. google-chrome-stable)
#
# Usage (from repo root):
#   bash scripts/render_diagrams.sh
set -euo pipefail
cd "$(dirname "$0")/.."

SRC=docs/diagrams/src
OUT=docs/diagrams
CHROME="${CHROME:-/usr/bin/google-chrome-stable}"
PP=$(mktemp)
echo "{\"executablePath\":\"$CHROME\",\"args\":[\"--no-sandbox\",\"--disable-gpu\",\"--disable-dev-shm-usage\"]}" > "$PP"

for m in "$SRC"/*.mmd; do
  base=$(basename "$m" .mmd)
  mmdc -p "$PP" -i "$m" -o "$OUT/$base.png" -b white -s 2   # PNG @2x: renders in every viewer (GitHub, VS Code preview)
  echo "rendered $OUT/$base.png"
done
rm -f "$PP"
echo "Done. Edit a .mmd in $SRC, re-run this script, and the embedded image updates."
