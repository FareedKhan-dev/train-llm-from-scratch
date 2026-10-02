# Diagrams

The documentation diagrams are **hand-drawn, colour-coded Mermaid sketches**, pre-rendered to PNG and
embedded as images in the docs. We pre-render (rather than rely on live ```` ```mermaid ```` blocks)
because GitHub's live Mermaid renderer does not reliably support the `look: handDrawn` style, and some
markdown viewers (e.g. the VS Code preview) block SVGs, so an embedded **PNG** shows the hand-drawn look
everywhere. Each doc also keeps the editable Mermaid source in a collapsible *"Mermaid source"* block
under its image.

## Files

- `src/*.mmd`: the canonical hand-drawn Mermaid sources (with `look: handDrawn` + the colour palette).
- `*.png`: the rendered images embedded by the docs (and `README.png` for the top-level README).
- `make_plots.py`: the data plots (learning-rate schedules, RoPE, KV cache memory, attention
  masks, Muon's effect on singular values, training curves). Each one is computed from the
  repo's own code or training logs, so the figures follow the code when it changes.

| Files | Topic |
|---|---|
| `00` to `09` | the post-training pipeline, one diagram per stage |
| `16` to `19` | the modern decoder: the block, GQA, the KV cache, Mixture of Experts |
| `20` to `25` | GRPO variants, the laptop track, LoRA, speculative decoding, BPE, typed configs |

The diagrams in the top-level README live in `images/`, with their own sources.

## Regenerate after editing

Edit the relevant `src/<name>.mmd`, then from the repo root:

```bash
bash scripts/render_diagrams.sh
```

That re-renders every `src/*.mmd` to `docs/diagrams/<name>.png`. Requires the Mermaid CLI
(`npm i -g @mermaid-js/mermaid-cli`, or `npm install --prefix .tools/mermaid @mermaid-js/mermaid-cli`
to keep it inside the repo and then `MMDC=.tools/mermaid/node_modules/.bin/mmdc`) and a
Chrome/Chromium for headless rendering (set `CHROME=/path/to/chrome` if it isn't at
`/usr/bin/google-chrome-stable`).

The plots need matplotlib (part of the dev dependencies):

```bash
python docs/diagrams/make_plots.py            # every plot
python docs/diagrams/make_plots.py rope       # one plot
```

## Colour legend

🟩 data / corpus · 🟦 preprocessing · teal storage (HDF5 / JSONL) · 🟨 model / training loop ·
🟧 RL / reward · 🟥 loss / objective · 🟪 evaluation · ⬜ checkpoint
