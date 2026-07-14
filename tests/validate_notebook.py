from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "youtube_ai_slop_detector_colab.ipynb"


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["kernelspec"]["language"] == "python"

    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(code_cells) >= 8
    for index, cell in enumerate(code_cells, start=1):
        source = "".join(cell["source"])
        try:
            ast.parse(source)
        except SyntaxError as exc:
            raise AssertionError(f"Code cell {index} does not compile: {exc}") from exc

    all_source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    required_fragments = [
        "openai/whisper-small",
        "GeorgeDrayson/modernbert-ai-detection-raid-mage",
        "Speech-Arena-2025/DF_Arena_1B_V_1",
        "uniform_control",
        "clip_scores.csv",
    ]
    for fragment in required_fragments:
        assert fragment in all_source, f"Missing required notebook content: {fragment}"

    print(f"Validated {NOTEBOOK.name}: {len(code_cells)} code cells compile.")


if __name__ == "__main__":
    main()
