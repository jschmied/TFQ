#!/usr/bin/env python3
"""Write a synthetic MLX-8bit (q8, group 64) checkpoint with exactly the tensors the GLM
converter's `--mode nonmoe` reads (no attention: the MLA branch skips absent q_b_proj/o_proj).

lm_head is [154880, 4096] (> 16384 rows), so the converter encodes it in 8192-row chunks.
Values are random; the dequantized source is reproducible from the files themselves.
Usage: python3 -I make_glm_synth.py OUT_DIR
"""
import json
import os
import sys

import numpy as np

P = "model.language_model.layers"


def tensors():
    for l in range(3):
        yield f"{P}.{l}.mlp.gate_proj", 12288, 4096
        yield f"{P}.{l}.mlp.up_proj", 12288, 4096
        yield f"{P}.{l}.mlp.down_proj", 4096, 12288
    for l in range(7, 45):
        yield f"{P}.{l}.mlp.shared_experts.gate_proj", 2048, 4096
        yield f"{P}.{l}.mlp.shared_experts.up_proj", 2048, 4096
        yield f"{P}.{l}.mlp.shared_experts.down_proj", 4096, 2048
    yield "lm_head", 154880, 4096


def gen(rng, R, C):
    q = np.clip(np.rint(128.0 + 32.0 * rng.standard_normal((R, C), dtype=np.float32)), 0, 255).astype(np.uint8)
    s = (0.001 + 0.002 * rng.random((R, C // 64), dtype=np.float32)).astype(np.float32)
    b = -128.0 * s + 0.0005 * rng.standard_normal((R, C // 64), dtype=np.float32)
    to_bf16 = lambda x: (x.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)  # truncation, exact on read back
    return q, to_bf16(s), to_bf16(b)


def write_shard(path, items):
    hdr, off = {}, 0
    for name, R, C, q, s, b in items:
        for suf, arr, dt, shape in ((".weight", q, "U8", [R, C]), (".scales", s, "BF16", [R, C // 64]), (".biases", b, "BF16", [R, C // 64])):
            hdr[name + suf] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + arr.nbytes]}
            off += arr.nbytes
    h = json.dumps(hdr, separators=(",", ":")).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    with open(path, "wb") as f:
        f.write(len(h).to_bytes(8, "little"))
        f.write(h)
        for _, _, _, q, s, b in items:
            f.write(q.tobytes()); f.write(s.tobytes()); f.write(b.tobytes())


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(20261126)
    wmap, small = {}, []
    for name, R, C in tensors():
        q, s, b = gen(rng, R, C)
        fn = "model-00002-of-00002.safetensors" if name == "lm_head" else "model-00001-of-00002.safetensors"
        for suf in (".weight", ".scales", ".biases"):
            wmap[name + suf] = fn
        if name == "lm_head":
            write_shard(os.path.join(out, fn), [(name, R, C, q, s, b)])
        else:
            small.append((name, R, C, q, s, b))
    write_shard(os.path.join(out, "model-00001-of-00002.safetensors"), small)
    with open(os.path.join(out, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": wmap}, f, indent=1)
    print(f"wrote {len(wmap) // 3} tensors to {out}")


if __name__ == "__main__":
    main()
