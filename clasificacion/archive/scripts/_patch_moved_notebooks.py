"""Parchea los notebooks movidos: raíz del proyecto robusta + referencias entre notebooks."""
import glob
import json
import re

OLD_LINE = 'PROJECT_ROOT = Path("..").resolve()'
NEW_BLOCK = '''# Raíz del proyecto: se sube desde el directorio actual hasta encontrar src/ y requirements.txt
# (funciona desde notebooks/, notebooks/best_model/ o notebooks/other_experiments/)
PROJECT_ROOT = next(
    directory
    for directory in (Path.cwd().resolve(), *Path.cwd().resolve().parents)
    if (directory / "src").is_dir() and (directory / "requirements.txt").is_file()
)'''

OTHER_EXPERIMENT_NOTEBOOKS = ("02_train_resnet152", "02.1_train_resnet152", "03_optuna_resnet152", "04_compare_models")
BEST_MODEL_NOTEBOOKS = ("02_train_resnet152_nocrop", "02.1_train_resnet152_nocrop", "05_improve_recall")


def patch_source(source_text: str, cell_type: str, folder: str) -> str:
    if cell_type == "code":
        return source_text.replace(OLD_LINE, NEW_BLOCK)
    # Markdown: referencias a notebooks que ahora viven en la otra carpeta
    other_folder = "other_experiments" if folder == "best_model" else "best_model"
    other_names = OTHER_EXPERIMENT_NOTEBOOKS if folder == "best_model" else BEST_MODEL_NOTEBOOKS
    for name in other_names:
        source_text = re.sub(
            rf"(?<![\w/]){re.escape(name)}\.ipynb", f"{other_folder}/{name}.ipynb", source_text
        )
    return source_text


for folder in ("best_model", "other_experiments"):
    for path in sorted(glob.glob(f"notebooks/{folder}/*.ipynb")):
        notebook = json.load(open(path, encoding="utf-8"))
        changed_cells = 0
        for cell in notebook["cells"]:
            original = "".join(cell["source"])
            patched = patch_source(original, cell["cell_type"], folder)
            if patched != original:
                cell["source"] = patched.splitlines(keepends=True)
                changed_cells += 1
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(notebook, handle, indent=1, ensure_ascii=False)
            handle.write("\n")
        print(path, "celdas modificadas:", changed_cells)
