#!/usr/bin/env python3
"""merge_prep.py -- turn the raw CUDA driver CSVs into a merge-ready table.

The drivers emit
    op,prec,dim,nnz,cpu_time,gpu_time,cpu_kind,relerr
which carries no information about *which machine* produced it.  This script
prefixes the platform / toolchain / library metadata so that runs from several
hosts (e.g. this Arm+GB10 box and an x86+H100 box) can simply be concatenated.

Usage
-----
  # annotate a raw run of this machine
  python3 bench/cuda/merge_prep.py annotate \\
      --raw bench/cuda/out/cuda_results.csv --lib gdtq \\
      --raw bench/cuda/out/mpf_results.csv  --lib mpc_cuda \\
      --platform arm-gb10 --out bench/cuda/out/gpu_bench_merged.csv

  # merge this machine's table with one produced elsewhere
  python3 bench/cuda/merge_prep.py merge \\
      bench/cuda/out/gpu_bench_merged.csv /path/from/x86/gpu_bench_merged.csv \\
      --out gpu_bench_all.csv

`annotate` auto-detects host/CPU/GPU/CUDA/compiler; every field can be
overridden from the command line so the x86 side can be annotated here too.
"""
import argparse, csv, os, platform, re, subprocess, sys, datetime

SCHEMA = [
    "platform", "host", "cpu", "cpu_threads", "gpu", "gpu_arch", "cuda", "compiler",
    "lib", "op", "kernel", "prec", "base", "ncomp", "prec_bits",
    "dim", "nnz", "cpu_time", "gpu_time", "speedup", "cpu_kind", "relerr",
    "fma", "tree", "date",
]

# prec tag -> (base, ncomp, working bits, complex?)
PREC_INFO = {
    "f":  ("binary32", 1, 24),   "d":  ("binary64", 1, 53),
    "ds": ("binary32", 2, 48),   "ts": ("binary32", 3, 72),  "qs": ("binary32", 4, 96),
    "dd": ("binary64", 2, 106),  "td": ("binary64", 3, 159), "qd": ("binary64", 4, 212),
}

# op -> the expression the driver actually measures, per library.
# NOTE: the gdtq "axpy" driver measures cmul_g<P>vector_dev, i.e. c = alpha*a
# (a SCAL), because the GPU library has no fused alpha*x+y kernel.  The label is
# kept so the two machines join, but the real expression is recorded here.
KERNEL = {
    ("gdtq", "axpy"):     "c = alpha*a  (SCAL; no fused AXPY kernel in the GPU lib)",
    ("gdtq", "matvec"):   "y = A*x",
    ("gdtq", "matmul"):   "C = A*B",
    ("gdtq", "spmv"):     "y = A*x (sparse)",
    ("mpc_cuda", "axpy"):   "y = alpha*x + y  (fused)",
    ("mpc_cuda", "matvec"): "y = A*x",
    ("mpc_cuda", "matmul"): "C = A*B",
}


def sh(cmd, default=""):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception:
        return default


def detect():
    # LC_ALL=C: lscpu localises its field names (e.g. Japanese "モデル名")
    cpu = sh("LC_ALL=C lscpu | sed -n 's/^Model name: *//p' | head -1")
    if not cpu:
        cpu = sh("LC_ALL=C lscpu | sed -n 's/^BIOS Model name: *//p' | head -1")
    if not cpu:
        cpu = sh("sed -n 's/^model name[ \t]*: *//p' /proc/cpuinfo | head -1")
    if not cpu:
        impl = sh("LC_ALL=C lscpu | sed -n 's/^Model: *//p' | head -1")
        cpu = ("%s %s" % (platform.machine(), impl)).strip()
    if not cpu:
        cpu = platform.processor() or platform.machine()
    gpu = sh("nvidia-smi --query-gpu=name --format=csv,noheader | head -1")
    cc = sh("nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1")
    cuda = sh("nvcc --version | sed -n 's/.*release \\([0-9.]*\\).*/\\1/p' | head -1")
    gcc = sh("LC_ALL=C gcc --version | head -1")
    return {
        "host": platform.node(),
        "cpu": cpu,
        "cpu_threads": sh("nproc", "?"),
        "gpu": gpu,
        "gpu_arch": ("sm_" + cc.replace(".", "")) if cc else "",
        "cuda": cuda,
        "compiler": gcc,
    }


def prec_fields(prec):
    """-> (base, ncomp, bits) for both the gdtq tags and mpfNNN."""
    m = re.fullmatch(r"mpf(\d+)", prec)
    if m:
        return ("mpf", "", int(m.group(1)))
    cplx = prec.startswith("c") and prec[1:] in PREC_INFO
    tag = prec[1:] if cplx else prec
    if tag in PREC_INFO:
        base, n, bits = PREC_INFO[tag]
        return (("complex " if cplx else "") + base, n, bits)
    return ("", "", "")


def annotate(args):
    meta = detect()
    for k in ("host", "cpu", "cpu_threads", "gpu", "gpu_arch", "cuda", "compiler"):
        v = getattr(args, k, None)
        if v:
            meta[k] = v
    date = args.date or datetime.date.today().isoformat()

    rows = []
    for raw, lib in zip(args.raw, args.lib):
        if not os.path.exists(raw):
            print("warning: missing %s (skipped)" % raw, file=sys.stderr)
            continue
        with open(raw) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("op,") or line.startswith("#"):
                    continue
                if line.startswith("RESULT,"):
                    line = line[len("RESULT,"):]
                f = line.split(",")
                if len(f) != 8:
                    continue
                op, prec, dim, nnz, tcpu, tgpu, kind, relerr = f
                base, ncomp, bits = prec_fields(prec)
                try:
                    tc, tg = float(tcpu), float(tgpu)
                    spd = (tc / tg) if (tc > 0 and tg > 0) else ""
                except ValueError:
                    tc = tg = spd = ""
                rows.append({
                    "platform": args.platform, **meta,
                    "lib": lib, "op": op,
                    "kernel": KERNEL.get((lib, op), ""),
                    "prec": prec, "base": base, "ncomp": ncomp, "prec_bits": bits,
                    "dim": dim, "nnz": nnz,
                    "cpu_time": tcpu, "gpu_time": tgpu,
                    "speedup": ("%.4f" % spd) if spd != "" else "",
                    "cpu_kind": kind, "relerr": relerr,
                    "fma": args.fma, "tree": args.tree, "date": date,
                })

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SCHEMA)
        w.writeheader()
        w.writerows(rows)
    print("wrote %s (%d rows, %d columns)" % (args.out, len(rows), len(SCHEMA)))
    libs = sorted({r["lib"] for r in rows})
    for lib in libs:
        sub = [r for r in rows if r["lib"] == lib]
        ops = sorted({r["op"] for r in sub})
        precs = sorted({r["prec"] for r in sub})
        print("  %-9s %4d rows  ops=%s  prec=%s" % (lib, len(sub), ",".join(ops),
                                                    ",".join(precs)))
    return 0


def merge(args):
    out_rows, seen_cols = [], None
    for path in args.inputs:
        with open(path) as fh:
            rd = csv.DictReader(fh)
            cols = rd.fieldnames
            if seen_cols is None:
                seen_cols = cols
            elif cols != seen_cols:
                print("ERROR: %s has a different schema\n  expected %s\n  found    %s"
                      % (path, seen_cols, cols), file=sys.stderr)
                return 1
            n = 0
            for row in rd:
                out_rows.append(row); n += 1
            print("  %-50s %5d rows  platform=%s" %
                  (path, n, ",".join(sorted({r['platform'] for r in out_rows[-n:]}))))
    # duplicate key check
    key = lambda r: (r["platform"], r["lib"], r["op"], r["prec"], r["dim"])
    seen, dups = set(), 0
    for r in out_rows:
        k = key(r)
        if k in seen:
            dups += 1
        seen.add(k)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=seen_cols)
        w.writeheader(); w.writerows(out_rows)
    print("wrote %s (%d rows)" % (args.out, len(out_rows)))
    if dups:
        print("note: %d duplicate (platform,lib,op,prec,dim) keys" % dups)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("annotate", help="add platform metadata to raw driver CSVs")
    a.add_argument("--raw", action="append", required=True)
    a.add_argument("--lib", action="append", required=True,
                   help="library tag for the preceding --raw (gdtq | mpc_cuda)")
    a.add_argument("--platform", required=True, help="e.g. arm-gb10, x86-h100")
    a.add_argument("--out", required=True)
    a.add_argument("--fma", default="pre-FMA",
                   help="state of the branch-free FMA integration (default: pre-FMA)")
    a.add_argument("--tree", default="bncmatmul-0.24")
    a.add_argument("--date", default=None)
    for k in ("host", "cpu", "cpu_threads", "gpu", "gpu_arch", "cuda", "compiler"):
        a.add_argument("--" + k.replace("_", "-"), dest=k, default=None)
    a.set_defaults(func=annotate)

    m = sub.add_parser("merge", help="concatenate annotated CSVs from several hosts")
    m.add_argument("inputs", nargs="+")
    m.add_argument("--out", required=True)
    m.set_defaults(func=merge)

    args = ap.parse_args()
    if args.cmd == "annotate" and len(args.raw) != len(args.lib):
        ap.error("--raw and --lib must be given in matching pairs")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
