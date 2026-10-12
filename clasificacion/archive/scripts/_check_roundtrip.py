import glob
import json

for path in sorted(glob.glob("notebooks/*.ipynb")):
    raw = open(path, encoding="utf-8", newline="").read()
    notebook = json.loads(raw)
    for indent in (1, 2):
        for ensure_ascii in (False, True):
            for newline in ("\n", ""):
                dumped = json.dumps(notebook, indent=indent, ensure_ascii=ensure_ascii) + newline
                if dumped == raw:
                    print(path, "roundtrip OK", indent, ensure_ascii, repr(newline))
    print(path, "crlf" if "\r\n" in raw else "lf", notebook["nbformat"], notebook["nbformat_minor"])
