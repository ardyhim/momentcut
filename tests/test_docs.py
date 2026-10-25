from __future__ import annotations

import json
from pathlib import Path


def test_colab_notebook_is_valid_and_its_code_compiles():
    path = Path(__file__).resolve().parents[1] / "notebooks" / "momentcut_colab.ipynb"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["nbformat"] == 4
    cells = payload["cells"]
    assert len(cells) >= 15
    code_cells = [cell for cell in cells if cell["cell_type"] == "code"]
    assert code_cells
    for index, cell in enumerate(code_cells):
        source = "".join(cell["source"])
        compile(source, f"notebook-code-cell-{index}", "exec")

    text = "\n".join("".join(cell["source"]) for cell in cells)
    assert "MomentCut" in text
    assert "manual" in text.lower()
    assert "MOMENTCUT_API_BASE_URL" in text
    assert "faster-whisper" in text
    assert "requirements.txt" in text
    assert "render_manifest.json" in text


def test_readme_links_existing_docs():
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    for relative in (
        "docs/FINDINGS.md",
        "docs/ARCHITECTURE.md",
        "docs/JSON_CONTRACT.md",
        "docs/COLAB.md",
        "docs/TESTING.md",
        "notebooks/momentcut_colab.ipynb",
    ):
        assert relative in readme
        assert (root / relative).exists(), relative
