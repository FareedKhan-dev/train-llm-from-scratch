# Diagrams

The documentation diagrams are **hand-drawn, colour-coded Mermaid sketches**, pre-rendered to PNG and
embedded as images in the docs. We pre-render (rather than rely on live ```` ```mermaid ```` blocks)
because GitHub's live Mermaid renderer does not reliably support the `look: handDrawn` style, and some
markdown viewers (e.g. the VS Code preview) block SVGs — an embedded **PNG** shows the hand-drawn look
everywhere. Each doc also keeps the editable Mermaid source in a collapsible *"Mermaid source"* block
under its image.

## Files

- `src/*.mmd` — the canonical hand-drawn Mermaid sources (with `look: handDrawn` + the colour palette).
- `*.png` — the rendered images embedded by the docs (and `README.png` for the top-level README).

## Regenerate after editing

Edit the relevant `src/<name>.mmd`, then from the repo root:

```bash
bash scripts/render_diagrams.sh
```

That re-renders every `src/*.mmd` to `docs/diagrams/<name>.png`. Requires the Mermaid CLI
(`npm i -g @mermaid-js/mermaid-cli`) and a Chrome/Chromium for headless rendering
(set `CHROME=/path/to/chrome` if it isn't at `/usr/bin/google-chrome-stable`).

## Colour legend

🟩 data / corpus · 🟦 preprocessing · teal storage (HDF5 / JSONL) · 🟨 model / training loop ·
🟧 RL / reward · 🟥 loss / objective · 🟪 evaluation · ⬜ checkpoint
