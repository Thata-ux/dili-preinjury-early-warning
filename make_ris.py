"""Build a Mendeley-importable .ris from the manuscript's reference list.

Author lists in the manuscript use 'et al.' for multi-author works, so those
co-authors are not recoverable from the text. This script never invents them:
it writes the authors it can see and marks the entry so Mendeley's
"Update details from DOI" can complete it.
"""
import re, sys
from pathlib import Path

SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2])
refs = SRC.read_text().split("# References\n", 1)[1]
entries = [b.strip().replace("\n", " ") for b in refs.strip().split("\n\n") if b.strip()]

# Full author lists verified against the publisher record for the 11 added here.
VERIFIED = {
 "10.1111/j.2517-6161.1995.tb02031.x": ["Benjamini, Y.", "Hochberg, Y."],
 "10.1023/A:1010933404324": ["Breiman, L."],
 "10.1145/2939672.2939785": ["Chen, T.", "Guestrin, C."],
 "10.1007/978-3-642-00663-0_9": ["Copple, I.M.","Goldring, C.E.","Kitteringham, N.R.","Park, B.K."],
 "10.1093/nar/gni179": ["Dai, M.","Wang, P.","Boyd, A.D.","Kostov, G.","Athey, B.","Jones, E.G.",
    "Bunney, W.E.","Myers, R.M.","Speed, T.P.","Akil, H.","Watson, S.J.","Meng, F."],
 "10.1177/0192623312448935": ["Hall, A.P.","Elcombe, C.R.","Foster, J.R.","Harada, T.",
    "Kaufmann, W.","Knippel, A.","Küttler, K.","Malarkey, D.E.","Maronpot, R.R.","Nishikawa, A.",
    "Nolte, T.","Schulte, A.","Strauss, V.","York, M.J."],
 "10.1093/nar/gku955": ["Igarashi, Y.","Nakatsu, N.","Yamashita, T.","Ono, A.","Ohno, Y.",
    "Urushidani, T.","Yamada, H."],
 "10.1111/ecog.02881": ["Roberts, D.R.","Bahn, V.","Ciuti, S.","Boyce, M.S.","Elith, J.",
    "Guillera-Arroita, G.","Hauenstein, S.","Lahoz-Monfort, J.J.","Schröder, B.","Thuiller, W.",
    "Warton, D.I.","Wintle, B.A.","Hartig, F.","Dormann, C.F."],
 "10.1007/s12272-019-01192-3": ["Taguchi, K.", "Kensler, T.W."],
 "10.1177/0192623310386499": ["Thoolen, B.","Maronpot, R.R.","Harada, T.","Nyska, A.",
    "Rousseaux, C.","Nolte, T.","Malarkey, D.E.","Kaufmann, W.","Küttler, K.","Deschl, U.",
    "Nakae, D.","Gregson, R.","Vinlove, M.P.","Brix, A.E.","Singh, B.","Belpoggi, F.","Ward, J.M."],
 "10.1080/01926230252824699": ["Williams, G.M.", "Iatropoulos, M.J."],
}
LUNDBERG = ["Lundberg, S.M.", "Lee, S.-I."]

def strip_md(t):
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)      # links -> text
    return re.sub(r"[*_]", "", t).strip()

out, problems = [], []
for e in entries:
    doi_m = re.search(r"doi\.org/(10\.[^\)\s\]]+)", e)
    url_m = re.search(r"\((https?://[^\)]+)\)", e)
    doi = doi_m.group(1).rstrip(".") if doi_m else ""
    head, _, rest = e.partition(") ")
    year = re.search(r"\((\d{4})\)", e)
    title = re.search(r"'([^']+)'|‘([^’]+)’", e)
    if not year or not title:
        problems.append(e[:70]); continue
    year = year.group(1)
    title = strip_md(title.group(1) or title.group(2)).rstrip(".,")
    tail = e.split("'", 2)[-1] if "'" in e else rest
    jour = re.search(r"\*([^*]+)\*", e.split("'")[-1] if "'" in e else e)
    journal = strip_md(jour.group(1)).strip(" ,.") if jour else ""
    vol = re.search(r",\s*(\d+)\s*\(([^)]+)\)", tail)
    volume, issue = (vol.group(1), vol.group(2)) if vol else ("", "")
    if not volume:
        v2 = re.search(r",\s*(\d+),", tail)
        volume = v2.group(1) if v2 else ""
    pg = re.search(r"pp?\.\s*([0-9A-Za-z]+)[–-]([0-9A-Za-z]+)", tail)
    sp, ep = (pg.group(1), pg.group(2)) if pg else ("", "")
    if not sp:
        one = re.search(r"p\.\s*([0-9A-Za-z]+)", tail)
        sp = ep = one.group(1) if one else ""

    if doi in VERIFIED:
        authors, complete = VERIFIED[doi], True
    elif "neurips" in e:
        authors, complete = LUNDBERG, True
    else:
        # surnames may be several words ("Hossam Abdelmonem", "Van den Broeck")
        first = re.match(
            r"^((?:[A-ZÁ-Ü][a-zA-Zá-ü\-']*\s+(?:den\s+)?)*[A-ZÁ-Ü][a-zA-Zá-ü\-']+,"
            r"\s*(?:[A-Z]\.\s*)+)", e)
        authors = [first.group(1).strip().rstrip(",")] if first else []
        complete = "et al." not in e and " and " not in e
        if not complete and " and " in e.split("(")[0]:
            seg = e.split("(")[0]
            authors = [a.strip().rstrip(",") for a in re.split(r",\s*(?=[A-ZÁ-Ü])|\s+and\s+", seg)
                       if re.match(r"^[A-ZÁ-Ü]", a.strip())]
            authors = [a for a in authors if a]
            complete = True

    ty = "CONF" if ("Proceedings" in e or "Advances in Neural" in e) else \
         "CHAP" if "Handbook of Experimental" in e else "JOUR"
    r = [f"TY  - {ty}"]
    r += [f"AU  - {a}" for a in authors]
    r += [f"PY  - {year}", f"TI  - {title}"]
    if journal: r.append(f"{'T2' if ty != 'JOUR' else 'JO'}  - {journal}")
    if volume:  r.append(f"VL  - {volume}")
    if issue:   r.append(f"IS  - {issue}")
    if sp:      r.append(f"SP  - {sp}")
    if ep and ep != sp: r.append(f"EP  - {ep}")
    if doi:     r.append(f"DO  - {doi}")
    if url_m:   r.append(f"UR  - {url_m.group(1)}")
    if not complete:
        r.append("N1  - Author list incomplete in source; use Mendeley 'Update details from DOI'.")
    r.append("ER  - ")
    out.append("\n".join(r))

OUT.write_text("\n\n".join(out) + "\n", encoding="utf-8")
print(f"wrote {len(out)} records to {OUT}")
if problems:
    print("UNPARSED:", problems)
