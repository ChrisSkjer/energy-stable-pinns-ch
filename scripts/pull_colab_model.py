"""Pull a model trained via notebooks/train_pinn.ipynb out of the saved notebook.

When train_pinn.ipynb runs against a Colab kernel from VS Code (rather than the
colab.research.google.com web UI), google.colab.files.download() has no browser
to hand the file to, so nothing ever reaches disk. The notebook instead prints
the trained model as base64 text between marker lines; that output gets saved
into the .ipynb file locally when you press Ctrl+S in VS Code. This script pulls
it back out and writes it under results/pinn_models/.

Usage:
    python scripts/pull_colab_model.py [notebook_path]
"""

import base64
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MARKER_RE = re.compile(
    r"===BEGIN_MODEL_B64:(?P<name>[^=\n]+)===\n(?P<data>.*?)\n===END_MODEL_B64===",
    re.S,
)


def main(notebook_path: str) -> None:
    nb = json.loads(Path(notebook_path).read_text(encoding="utf-8"))

    found = 0
    for cell in nb.get("cells", []):
        for output in cell.get("outputs", []):
            text = "".join(output.get("text", []))
            for match in MARKER_RE.finditer(text):
                out_path = REPO_ROOT / "results" / "pinn_models" / match.group("name")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(base64.b64decode(match.group("data")))
                print(f"wrote {out_path.relative_to(REPO_ROOT)}")
                found += 1

    if not found:
        print(
            "No embedded model data found. Run the notebook's export cell, "
            "then save the notebook (Ctrl+S) before running this script."
        )


if __name__ == "__main__":
    default_path = REPO_ROOT / "notebooks" / "train_pinn.ipynb"
    main(sys.argv[1] if len(sys.argv) > 1 else str(default_path))
