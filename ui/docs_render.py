"""
Render a ``docs/*.md`` theory page inside Streamlit.

Streamlit's ``st.markdown`` can't load local-disk images and won't render the ``<details>``
mermaid blocks, so we: (1) show the stage diagram with ``st.image`` (absolute path), and
(2) strip the markdown image line + the ``<details>…</details>`` mermaid block before
rendering the prose. The ``docs/*.md`` files stay the single source of truth (the MkDocs
site publishes the same files).
"""

from __future__ import annotations

import os
import re

from ui.stages import ABS_DOC


def _clean_markdown(text: str) -> str:
    # drop the embedded PNG line (we render it via st.image instead)
    text = re.sub(r"^!\[[^\]]*\]\([^)]*\)\s*$", "", text, flags=re.MULTILINE)
    # drop the <details>…</details> live-mermaid block
    text = re.sub(r"<details>.*?</details>", "", text, flags=re.DOTALL)
    # drop the leading H1 (the page already has a header)
    text = re.sub(r"^#\s+.*$", "", text, count=1, flags=re.MULTILINE)
    return text.strip()


def render_doc(st, doc_md: str, diagram_png: str | None = None) -> None:
    """Render the diagram (if any) + the prose of a docs markdown file."""
    if diagram_png:
        abs_png = ABS_DOC(diagram_png)
        if os.path.exists(abs_png):
            st.image(abs_png, use_container_width=True)
    abs_md = ABS_DOC(doc_md)
    if os.path.exists(abs_md):
        with open(abs_md) as f:
            st.markdown(_clean_markdown(f.read()))
    else:
        st.info(f"Theory doc not found: {doc_md}")
