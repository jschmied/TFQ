# Compare two conversion output dirs byte by byte. python3 -I compare.py <dirA> <dirB>
import hashlib, os, struct, sys
a, b = sys.argv[1], sys.argv[2]

def sections(path):
    with open(path, "rb") as f:
        h = f.read(64)
    magic, ver, nt, _, off_meta, off_scales, off_weights = struct.unpack_from("<IIIIQQQ", h)
    names = []
    with open(path, "rb") as f:
        f.seek(off_meta)
        for _ in range(nt):
            rec = f.read(128)
            names.append(rec[:80].split(b"\0", 1)[0].decode())
    return [(0, "header"), (off_meta, "meta"), (off_scales, "scales"), (off_weights, "weights")], names

for fn in sorted(set(os.listdir(a)) | set(os.listdir(b))):
    pa, pb = os.path.join(a, fn), os.path.join(b, fn)
    if not (os.path.exists(pa) and os.path.exists(pb)):
        print(f"{fn}: only in {'A' if os.path.exists(pa) else 'B'}"); continue
    da, db = open(pa, "rb").read(), open(pb, "rb").read()
    ha, hb = hashlib.sha256(da).hexdigest(), hashlib.sha256(db).hexdigest()
    if da == db:
        print(f"{fn}: identical {ha}"); continue
    n = min(len(da), len(db))
    diff = [i for i in range(n) if da[i] != db[i]]
    first = diff[0] if diff else n
    where = ""
    if fn.endswith((".tffpw", ".tffp4")):
        secs, names = sections(pa)
        sec = [s for o, s in secs if o <= first][-1]
        per = {}
        for i in diff:
            s = [s for o, s in secs if o <= i][-1]; per[s] = per.get(s, 0) + 1
        where = f" first in section '{sec}', differing bytes per section {per}"
    print(f"{fn}: DIFFER A={ha} B={hb} sizes {len(da)}/{len(db)} differing bytes {len(diff)} of {n}, first at offset {first}{where}")
