import glob
import json
import sys

pattern = sys.argv[1] if len(sys.argv) > 1 else "notebooks/**/*.ipynb"
with open("artifacts/logs/_nb_dump.txt", "w", encoding="utf-8") as out:
    for path in sorted(glob.glob(pattern, recursive=True)):
        notebook = json.load(open(path, encoding="utf-8"))
        out.write("\n########## " + path + "\n")
        for index, cell in enumerate(notebook["cells"]):
            out.write("--- [%d] %s\n" % (index, cell["cell_type"]))
            out.write("".join(cell["source"]) + "\n")
