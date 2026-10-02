#!/usr/bin/env bash
# Render the hand-drawn, colour-coded diagram sources (docs/diagrams/src/*.mmd) to PNGs
# (docs/diagrams/*.png) that are embedded as images in the docs. We pre-render because
# GitHub's live Mermaid does not reliably support `look: handDrawn`; an embedded PNG shows
# the hand-drawn look everywhere.
#
# One-time setup (Node >= 18 and a Chrome/Chromium for headless rendering):
#   npm install -g @mermaid-js/mermaid-cli                         # provides `mmdc`
#   # or keep it inside the repo (.tools/ is git-ignored):
#   npm install --prefix .tools/mermaid @mermaid-js/mermaid-cli
#
# Usage (from repo root):
#   bash scripts/render_diagrams.sh
#   MMDC=.tools/mermaid/node_modules/.bin/mmdc CHROME="/path/to/chrome" bash scripts/render_diagrams.sh
set -euo pipefail
cd "$(dirname "$0")/.."

SRC=docs/diagrams/src
OUT=docs/diagrams
CHROME="${CHROME:-/usr/bin/google-chrome-stable}"
MMDC="${MMDC:-mmdc}"
PP=$(mktemp)
echo "{\"executablePath\":\"$CHROME\",\"args\":[\"--no-sandbox\",\"--disable-gpu\",\"--disable-dev-shm-usage\"]}" > "$PP"

for m in "$SRC"/*.mmd; do
  base=$(basename "$m" .mmd)
  "$MMDC" -p "$PP" -i "$m" -o "$OUT/$base.png" -b white -s 2   # PNG @2x: renders in every viewer (GitHub, VS Code preview)
  echo "rendered $OUT/$base.png"
done
rm -f "$PP"
echo "Done. Edit a .mmd in $SRC, re-run this script, and the embedded image updates."
