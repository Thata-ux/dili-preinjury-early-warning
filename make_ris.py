"""Build a Mendeley/Zotero-importable .ris from references.json.

references.json is the manuscript's reference list in CSL-JSON, in citation order
([1]..[36]), with complete author lists and journal abbreviations verified against
Crossref. It is the same data embedded in the manuscript's Mendeley citation fields
(citation style: Computational Toxicology = Elsevier numeric, with titles).

    python3 make_ris.py references.json references.ris
"""
import json, sys
from pathlib import Path

SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2])
TY = {"article-journal": "JOUR", "paper-conference": "CONF", "chapter": "CHAP"}
lines = []
for it in json.loads(SRC.read_text(encoding="utf-8")):
    lines.append(f"TY  - {TY.get(it['type'], 'GEN')}")
    for a in it["author"]:
        lines.append(f"AU  - {a['family']}, {a['given']}")
    lines.append(f"TI  - {it['title']}")
    tag = "T2" if it["type"] == "paper-conference" else "JO"
    lines.append(f"{tag}  - {it['container-title']}")
    if it.get("container-title-short") and it["type"] != "paper-conference":
        lines.append(f"J2  - {it['container-title-short']}")
    lines.append(f"PY  - {it['issued']['date-parts'][0][0]}")
    for k, t in (("volume", "VL"), ("issue", "IS"), ("publisher", "PB"), ("publisher-place", "CY")):
        if it.get(k):
            lines.append(f"{t}  - {it[k]}")
    if it.get("page"):
        p = it["page"].split("-")
        lines.append(f"SP  - {p[0]}")
        if len(p) > 1:
            lines.append(f"EP  - {p[1]}")
    if it.get("DOI"):
        lines.append(f"DO  - {it['DOI']}")
        lines.append(f"UR  - https://doi.org/{it['DOI']}")
    elif it.get("URL"):
        lines.append(f"UR  - {it['URL']}")
    lines.append(f"N1  - Manuscript reference [{it['id'][3:]}]")
    lines.append("ER  - ")
    lines.append("")
OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"{OUT}: {sum(1 for l in lines if l.startswith('TY'))} references")
