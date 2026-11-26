#!/usr/bin/env python3
"""Decode selected 128x128 blocks of a TFPN tensor with reference/tfq3_reference.py and compare them
with the dequantized MLX-q8 source. Prints one line per block and exits 1 if any block is below --min-db.

Usage: python3 -I check_glm_rows.py --repo TFQ_CHECKOUT --model SYNTH_DIR --out CONVERTED_DIR
       [--tensor lm_head] [--rows 0,63,64,65,127,128,1151,1152,1209] [--cols 0,31]
"""
import argparse
import json
import math
import os
import struct
import sys

import numpy as np


def load_ref(repo):
    sys.path.insert(0, os.path.join(repo, "reference"))
    import tfq3_reference as ref
    return ref


def read_tfpn(path, name):
    with open(path, "rb") as f:
        magic, ver, nt, _, off_meta, off_scales, off_weights, _, _, _ = struct.unpack("<IIIIQQQQQQ", f.read(64))
        f.seek(off_meta)
        for _ in range(nt):
            rec = f.read(128)
            if rec[:80].split(b"\x00", 1)[0].decode() == name:
                R, C, bits, sc_off, w_off, w_cnt, snr, _, _, _ = struct.unpack("<IIIIQQfIII", rec[80:])
                break
        else:
            raise KeyError(name)
        f.seek(off_scales + sc_off * 2)
        scales = np.frombuffer(f.read((R // 128) * (C // 128) * 2), dtype=np.float16).reshape(R // 128, C // 128)
        f.seek(off_weights + w_off * 4)
        words = np.frombuffer(f.read(w_cnt * 4), dtype=np.uint32)
    return R, C, bits, snr, scales, words


def decode_block(ref, tbl, words, scales, C, bits, rb, cb):
    """Same walk as ref.dequantize_tfpn_tensor, for one (rb, cb) block."""
    n_bc, wpb = C // 128, bits * 2
    blk = np.zeros((128, 128), dtype=np.float32)
    sc = float(scales[rb, cb])
    for s in range(8):
        base = (rb * 8 + s) * n_bc * wpb * 32
        for lane in range(32):
            st = (ref.INIT_ST_BASE ^ ((lane * ref.HASH_MULT) & 0xFFFFFFFF)) & 0xFFFFFFFF
            if cb > 0:
                st = int(words[base + (cb * 2 - 1) * 128 + lane * 4 + 3]) if bits == 4 else int(words[base + (cb * wpb - 1) * 32 + lane])
            hc, hr = ref._frag_home(lane)
            for h in range(2):
                if bits == 4:
                    idx = base + (cb * 2 + h) * 128 + lane * 4
                    groups = [("4", int(words[idx + k])) for k in range(4)]
                elif bits == 3:
                    idx = base + (cb * 6 + h * 3) * 32 + lane
                    a0, a1, a2 = int(words[idx]), int(words[idx + 32]), int(words[idx + 64])
                    groups = [("3", m) for m in (a0 >> 8, ((a0 << 16) | (a1 >> 16)) & 0xFFFFFF, ((a1 << 8) | (a2 >> 24)) & 0xFFFFFF, a2 & 0xFFFFFF)]
                else:
                    idx = base + (cb * 10 + h * 5) * 32 + lane
                    w0, w1, w2, w3, w4 = [int(words[idx + j * 32]) for j in range(5)]
                    groups = [("5", p) for p in ((w0 >> 24, ((w0 << 8) | (w1 >> 24)) & 0xFFFFFFFF), ((w1 >> 16) & 255, ((w1 << 16) | (w2 >> 16)) & 0xFFFFFFFF),
                                                 ((w2 >> 8) & 255, ((w2 << 24) | (w3 >> 8)) & 0xFFFFFFFF), (w3 & 255, w4 & 0xFFFFFFFF))]
                for k, (kind, g) in enumerate(groups):
                    if kind == "4":
                        st, q = ref._decode_8_4b(st, g, tbl)
                    elif kind == "3":
                        st, q = ref._decode_8_3b(st, g, tbl)
                    else:
                        st, q = ref._decode_8_5b(st, g[0], g[1], tbl)
                    c0 = h * 64 + k * 16 + hc
                    blk[s * 16 + hr, c0:c0 + 4] = q[:4] * sc
                    blk[s * 16 + hr + 8, c0:c0 + 4] = q[4:] * sc
    return ref.unrotate_block_2d(blk, r_off=rb * 128, c_off=cb * 128)


def source_block(model, name, rb, cb):
    idx = json.load(open(os.path.join(model, "model.safetensors.index.json")))["weight_map"]
    fn = os.path.join(model, idx[name + ".weight"])
    with open(fn, "rb") as f:
        hl = int.from_bytes(f.read(8), "little")
        hdr = json.loads(f.read(hl))
    mm = np.memmap(fn, dtype=np.uint8, mode="r")
    base = 8 + hl
    def arr(suf, dt):
        m = hdr[name + suf]; a, b = m["data_offsets"]
        return np.frombuffer(mm[base + a:base + b], dtype=dt).reshape(m["shape"])
    q = arr(".weight", np.uint8)[rb * 128:(rb + 1) * 128, cb * 128:(cb + 1) * 128].astype(np.float32)
    bf = lambda x: (x.astype(np.uint32) << 16).view(np.float32)
    s = bf(arr(".scales", np.uint16)[rb * 128:(rb + 1) * 128, cb * 2:cb * 2 + 2])
    b = bf(arr(".biases", np.uint16)[rb * 128:(rb + 1) * 128, cb * 2:cb * 2 + 2])
    return q * np.repeat(s, 64, axis=1) + np.repeat(b, 64, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tensor", default="lm_head")
    ap.add_argument("--rows", default="0,63,64,65,127,128,1151,1152,1209")
    ap.add_argument("--cols", default="0,31")
    ap.add_argument("--min-db", type=float, default=15.0)
    a = ap.parse_args()
    ref = load_ref(a.repo)
    tbl = ref.load_tables(os.path.join(a.out, "pw_v3_tables.bin"))
    R, C, bits, snr, scales, words = read_tfpn(os.path.join(a.out, "nonmoe.tffpw"), a.tensor)
    print(f"{a.tensor} R={R} C={C} bits={bits} converter_snr={snr:.2f} dB")
    bad = 0
    for rb in [int(x) for x in a.rows.split(",")]:
        for cb in [int(x) for x in a.cols.split(",")]:
            d = decode_block(ref, tbl, words, scales, C, bits, rb, cb)
            w = source_block(a.model, a.tensor, rb, cb)
            db = 10 * math.log10(float((w.astype(np.float64) ** 2).sum()) / max(float(((w - d).astype(np.float64) ** 2).sum()), 1e-30))
            ok = db >= a.min_db
            bad += not ok
            print(f"rows {rb * 128:6d}-{rb * 128 + 127:6d} cols {cb * 128:5d}-{cb * 128 + 127:5d}  SNR {db:7.2f} dB  {'PASS' if ok else 'FAIL'}")
    print("RESULT", "PASS" if bad == 0 else f"FAIL ({bad} blocks below {a.min_db} dB)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
