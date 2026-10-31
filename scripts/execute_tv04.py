from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT = Path.cwd()
nb_path = ROOT / "notebooks/03_xgboost_importance.ipynb"
nb = nbformat.read(nb_path, as_version=4)

client = NotebookClient(
    nb,
    timeout=600,
    resources={"metadata": {"path": str(ROOT / "notebooks")}}
)
print("Executing notebook...")
client.execute()
print("Execution complete, writing inplace...")
with open(nb_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)
print("Successfully updated 03_xgboost_importance.ipynb with outputs!")
