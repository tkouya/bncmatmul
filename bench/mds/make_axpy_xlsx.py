#!/usr/bin/env python3
# Build axpy_compare.xlsx from the AXPY benchmark CSVs:
#   bench/mds/axpy_cpu.csv   (serial=1 thread, cpu_max=nproc threads; BNC + MPFR)
#   bench/cuda/axpy_gpu.csv  (gpu_32 = 32 CUDA threads, gpu_max = full occupancy)
#
# CSV schema (both): backend,nthreads,family,type,bits,N,seconds,iters,mflops
#
# Sheets:
#   Overview      - what was measured / how to read it
#   Time (s)      - AXPY wall-clock seconds, per config, type x N
#   MFLOPS        - throughput, per config, type x N
#   EFT vs MPFR   - headline: BNC EFT vs MPFR of the SAME mantissa length
#   Speedups      - serial -> cpu_max / gpu_max scaling
#   Raw           - every measured row
import csv, os, sys
sys.path.insert(0, "/tmp/pylibs")
import xlsxwriter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CPU_CSV = os.path.join(ROOT, "bench/mds/axpy_cpu.csv")
GPU_CSV = os.path.join(ROOT, "bench/cuda/axpy_gpu.csv")
OUT     = os.path.join(ROOT, "axpy_compare.xlsx")

# BNC EFT types in ascending mantissa order, paired with the matching MPFR prec.
EFT = [   # (type, bits, family, human description)
    ("float", 24,  "float-based",  "binary32"),
    ("ds",    48,  "float-based",  "double-single (float x2)"),
    ("double",53,  "double-based", "binary64"),
    ("ts",    72,  "float-based",  "triple-single (float x3)"),
    ("qs",    96,  "float-based",  "quad-single (float x4)"),
    ("dd",    106, "double-based", "double-double (double x2)"),
    ("td",    159, "double-based", "triple-double (double x3)"),
    ("qd",    212, "double-based", "quad-double (double x4)"),
]
MPFR_OF = {b: "mpfr%d" % b for (_, b, _, _) in EFT}
CONFIGS = ["serial", "cpu_max", "gpu_32", "gpu_max"]
CONFIG_LABEL = {
    "serial":  "CPU serial (1 thread)",
    "cpu_max": "CPU max threads (32)",
    "gpu_32":  "GPU 32 threads (1 warp)",
    "gpu_max": "GPU max threads (full occupancy)",
}

def config_of(backend, nt):
    if backend == "cuda":
        return "gpu_32" if nt == 32 else "gpu_max"
    return "serial" if nt == 1 else "cpu_max"

# data[config][type][N] = {"sec":..,"mflops":..,"nt":..}
data = {c: {} for c in CONFIGS}
sizes = set()
gpu_max_nt = {}   # N -> grid thread count

def load(path):
    if not os.path.exists(path):
        print("WARN missing", path); return
    with open(path) as f:
        for row in csv.reader(f):
            if not row or row[0] == "backend":
                continue
            backend, nt, fam, typ, bits, N, sec, it, mf = row
            nt = int(nt); N = int(N)
            cfg = config_of(backend, nt)
            data[cfg].setdefault(typ, {})[N] = {
                "sec": float(sec), "mflops": float(mf), "nt": nt, "bits": int(bits)}
            sizes.add(N)
            if cfg == "gpu_max":
                gpu_max_nt[N] = nt

load(CPU_CSV); load(GPU_CSV)
SIZES = sorted(sizes)

def get(cfg, typ, N, key):
    return data.get(cfg, {}).get(typ, {}).get(N, {}).get(key)

wb = xlsxwriter.Workbook(OUT)
F = {
 "title":  wb.add_format({"bold":True,"font_size":15}),
 "h":      wb.add_format({"bold":True,"font_size":12,"font_color":"#1F4E79"}),
 "hdr":    wb.add_format({"bold":True,"bg_color":"#1F4E79","font_color":"white",
                          "align":"center","valign":"vcenter","border":1,"text_wrap":True}),
 "sub":    wb.add_format({"bold":True,"bg_color":"#D6E4F0","border":1,"align":"center"}),
 "typ":    wb.add_format({"bold":True,"bg_color":"#F2F2F2","border":1}),
 "mpfr":   wb.add_format({"italic":True,"bg_color":"#FBE5D6","border":1}),
 "sci":    wb.add_format({"num_format":"0.000E+00","border":1}),
 "mf":     wb.add_format({"num_format":"#,##0","border":1}),
 "ratio":  wb.add_format({"num_format":"0.0\"x\"","border":1}),
 "ratiob": wb.add_format({"num_format":"0.0\"x\"","border":1,"bold":True,"bg_color":"#E2EFDA"}),
 "txt":    wb.add_format({"text_wrap":True,"valign":"top"}),
 "cell":   wb.add_format({"border":1}),
 "na":     wb.add_format({"border":1,"align":"center","font_color":"#999999"}),
}

# ----------------------------------------------------------------- Overview
ws = wb.add_worksheet("Overview")
ws.set_column(0, 0, 2); ws.set_column(1, 1, 22); ws.set_column(2, 8, 16)
ws.write(1, 1, "AXPY benchmark : BNCmatmul EFT types vs MPFR (matching mantissa)", F["title"])
ov = [
 "",
 "Operation : AXPY  c = a + alpha * b   (add_cmul over a length-N vector).",
 "FLOP count per call = 2N (one multiply + one add per element).",
 "",
 "Compared precisions (BNC extended-floating-point, EFT) and the MPFR",
 "arbitrary-precision type set to the SAME mantissa bit length:",
 "    float (24 bits)  <-> mpfr24       ts (72)  <-> mpfr72",
 "    ds    (48 bits)  <-> mpfr48       qs (96)  <-> mpfr96",
 "    double(53 bits)  <-> mpfr53       dd (106) <-> mpfr106",
 "                                      td (159) <-> mpfr159",
 "                                      qd (212) <-> mpfr212",
 "",
 "Four execution configurations:",
 "    serial   - CPU, 1 thread          (AVX-512 kernel)",
 "    cpu_max  - CPU, 32 threads         (AVX-512 kernel, OpenMP over vector slices)",
 "    gpu_32   - GPU, 32 CUDA threads    (1 warp; the 'same thread count as CPU' case)",
 "    gpu_max  - GPU, full occupancy     (one thread per element on an NVIDIA H100 NVL)",
 "",
 "Notes:",
 " - CPU serial/cpu_max use the same AVX-512 kernel; only the thread count differs.",
 " - MPFR has no GPU implementation, so the GPU columns cover the BNC EFT types only.",
 " - GPU times are kernel-only (device-resident data; host<->device copies excluded).",
 " - gpu_32 is intentionally pathological (a single warp): it shows that 'matching the",
 "   CPU thread count' starves the GPU. It is capped at N<=65536 (heavy EFT types take",
 "   tens of seconds per call as a single warp). Read it as a lower bound, not a tuned GPU.",
 " - EFT GPU (gpu_max) is measured to N=1048576 (throughput already saturates); native",
 "   double/float reach N=4194304. MPFR rows (CPU) cover the full size range.",
 " - Each measurement is the best (min) per-call time over repeated runs.",
]
r = 3
for line in ov:
    ws.write(r, 1, line, F["txt"]); r += 1
ws.write(r+1, 1, "GPU max-config grid thread count per N:", F["h"])
r += 2
ws.write(r, 1, "N", F["sub"])
for j, N in enumerate(SIZES):
    ws.write(r, 2+j, N, F["sub"])
ws.write(r+1, 1, "threads", F["typ"])
for j, N in enumerate(SIZES):
    ws.write(r+1, 2+j, gpu_max_nt.get(N, "-"), F["cell"])

# ------------------------------------------------- helper: per-config table
def write_table(ws, top, key, numfmt, title):
    """One block per config: rows = EFT types (+ matching MPFR), cols = N."""
    ws.write(top, 0, title, F["h"]); top += 1
    for cfg in CONFIGS:
        ws.write(top, 0, CONFIG_LABEL[cfg], F["sub"])
        top += 1
        ws.write(top, 0, "type", F["hdr"]); ws.write(top, 1, "bits", F["hdr"])
        for j, N in enumerate(SIZES):
            ws.write(top, 2+j, "N=%d" % N, F["hdr"])
        top += 1
        for (typ, bits, fam, desc) in EFT:
            ws.write(top, 0, typ, F["typ"]); ws.write(top, 1, bits, F["cell"])
            for j, N in enumerate(SIZES):
                v = get(cfg, typ, N, key)
                ws.write(top, 2+j, v, F[numfmt]) if v is not None \
                    else ws.write(top, 2+j, "n/a", F["na"])
            top += 1
            # matching MPFR row (CPU configs only)
            mp = MPFR_OF[bits]
            ws.write(top, 0, mp, F["mpfr"]); ws.write(top, 1, bits, F["mpfr"])
            for j, N in enumerate(SIZES):
                v = get(cfg, mp, N, key)
                ws.write(top, 2+j, v, F[numfmt]) if v is not None \
                    else ws.write(top, 2+j, "n/a", F["na"])
            top += 1
        top += 1
    return top

ws = wb.add_worksheet("Time (s)")
ws.set_column(0, 0, 10); ws.set_column(1, 1, 6); ws.set_column(2, 2+len(SIZES), 13)
write_table(ws, 0, "sec", "sci", "AXPY wall-clock time per call [seconds]  (lower = faster)")

ws = wb.add_worksheet("MFLOPS")
ws.set_column(0, 0, 10); ws.set_column(1, 1, 6); ws.set_column(2, 2+len(SIZES), 13)
write_table(ws, 0, "mflops", "mf", "AXPY throughput [MFLOPS = 2N / time]  (higher = faster)")

# ------------------------------------------------------------- EFT vs MPFR
ws = wb.add_worksheet("EFT vs MPFR")
ws.set_column(0, 0, 10); ws.set_column(1, 1, 7); ws.set_column(2, 40, 13)
ws.write(0, 0, "How many times faster the BNC EFT type is than MPFR at the SAME mantissa length",
         F["h"])
ws.write(1, 0, "ratio = MPFR_time / EFT_time  (e.g. 50x means EFT computes the AXPY 50x faster than MPFR)",
         F["txt"])
top = 3
for cfg in ("serial", "cpu_max"):     # MPFR exists on CPU only
    ws.write(top, 0, CONFIG_LABEL[cfg], F["sub"]); top += 1
    ws.write(top, 0, "type", F["hdr"]); ws.write(top, 1, "bits", F["hdr"])
    for j, N in enumerate(SIZES):
        ws.write(top, 2+j, "N=%d" % N, F["hdr"])
    top += 1
    for (typ, bits, fam, desc) in EFT:
        ws.write(top, 0, typ, F["typ"]); ws.write(top, 1, bits, F["cell"])
        for j, N in enumerate(SIZES):
            te = get(cfg, typ, N, "sec"); tm = get(cfg, MPFR_OF[bits], N, "sec")
            if te and tm and te > 0:
                fmt = "ratiob" if N == SIZES[-1] else "ratio"
                ws.write(top, 2+j, tm/te, F[fmt])
            else:
                ws.write(top, 2+j, "n/a", F["na"])
        top += 1
    top += 1
# small explanatory note + averages at the largest N
ws.write(top, 0, "Highlighted column = largest N (most representative of steady-state throughput).",
         F["txt"])

# -------------------------------------------------------------- Speedups
ws = wb.add_worksheet("Speedups")
ws.set_column(0, 0, 10); ws.set_column(1, 1, 7); ws.set_column(2, 40, 15)
ws.write(0, 0, "AXPY speedup vs CPU serial (= serial_time / config_time), per type", F["h"])
top = 2

def speedup_block(top, Nrep, configs):
    ws.write(top, 0, "at N = %d" % Nrep, F["sub"]); top += 1
    ws.write(top, 0, "type", F["hdr"]); ws.write(top, 1, "bits", F["hdr"])
    for j, (_, lab) in enumerate(configs):
        ws.write(top, 2+j, lab, F["hdr"])
    top += 1
    for (typ, bits, fam, desc) in EFT:
        ws.write(top, 0, typ, F["typ"]); ws.write(top, 1, bits, F["cell"])
        ts_ = get("serial", typ, Nrep, "sec")
        for j, (cfg, lab) in enumerate(configs):
            tc = get(cfg, typ, Nrep, "sec")
            v = (ts_/tc) if (ts_ and tc and tc > 0) else None
            ws.write(top, 2+j, v, F["ratio"]) if v else ws.write(top, 2+j, "n/a", F["na"])
        top += 1
    return top + 1

# 65536 is the largest N common to ALL FOUR configs (gpu_32 is capped there)
top = speedup_block(top, 65536,
                    [("cpu_max","CPU 32 thr"), ("gpu_32","GPU 32 thr"),
                     ("gpu_max","GPU max")])
# 1048576: CPU + GPU-max (EFT GPU saturates by here)
top = speedup_block(top, 1048576,
                    [("cpu_max","CPU 32 thr"), ("gpu_max","GPU max")])
ws.write(top, 0, "Note: GPU 32-thread (1 warp) is intentionally starved and is typically SLOWER "
                 "than CPU serial for heavy EFT types - that is the point of the 'equal thread "
                 "count' comparison. Real GPU value shows in the 'GPU max' column.", F["txt"])

# ------------------------------------------------------------------ Charts
# Throughput chart per config at a size all CPU+GPU-max configs share.
NCHART = 1048576
wsc = wb.add_worksheet("Chart data")
wsc.write(0, 0, "type (bits)")
for j, cfg in enumerate(CONFIGS):
    wsc.write(0, 1+j, CONFIG_LABEL[cfg])
rr = 1
for (typ, bits, fam, desc) in EFT:
    for nm in (typ, MPFR_OF[bits]):
        wsc.write(rr, 0, "%s(%d)" % (nm, bits))
        for j, cfg in enumerate(CONFIGS):
            v = get(cfg, nm, NCHART, "mflops")
            wsc.write(rr, 1+j, v if v is not None else 0)
        rr += 1
nrows = rr - 1

chart = wb.add_chart({"type": "column"})
for j, cfg in enumerate(CONFIGS):
    chart.add_series({
        "name":       CONFIG_LABEL[cfg],
        "categories": ["Chart data", 1, 0, nrows, 0],
        "values":     ["Chart data", 1, 1+j, nrows, 1+j],
    })
chart.set_title({"name": "AXPY throughput at N=%d (MFLOPS, log scale)" % NCHART})
chart.set_x_axis({"name": "type (mantissa bits)"})
chart.set_y_axis({"name": "MFLOPS", "log_base": 10})
chart.set_size({"width": 1180, "height": 560})
chart.set_legend({"position": "top"})
wsc.insert_chart(1, 7, chart)

# Second chart: EFT vs MPFR speedup (CPU serial) across N, for the heavy types.
wsc.write(nrows+3, 0, "EFT-over-MPFR speedup (CPU serial), by N", F["h"])
hdr_r = nrows+4
wsc.write(hdr_r, 0, "N")
heavy = [t for t in EFT]   # all
for j, (typ, bits, _, _) in enumerate(heavy):
    wsc.write(hdr_r, 1+j, "%s/%d" % (typ, bits))
for i, N in enumerate(SIZES):
    wsc.write(hdr_r+1+i, 0, N)
    for j, (typ, bits, _, _) in enumerate(heavy):
        te = get("serial", typ, N, "sec"); tm = get("serial", MPFR_OF[bits], N, "sec")
        wsc.write(hdr_r+1+i, 1+j, (tm/te) if (te and tm and te>0) else 0)
ch2 = wb.add_chart({"type": "line"})
for j, (typ, bits, _, _) in enumerate(heavy):
    ch2.add_series({
        "name":       "%s (%d bits)" % (typ, bits),
        "categories": ["Chart data", hdr_r+1, 0, hdr_r+len(SIZES), 0],
        "values":     ["Chart data", hdr_r+1, 1+j, hdr_r+len(SIZES), 1+j],
    })
ch2.set_title({"name": "How many times faster EFT is than MPFR (CPU serial)"})
ch2.set_x_axis({"name": "vector length N"})
ch2.set_y_axis({"name": "speedup (x)"})
ch2.set_size({"width": 1180, "height": 520})
wsc.insert_chart(nrows+3, 7, ch2)

# ------------------------------------------------------------------- Raw
ws = wb.add_worksheet("Raw")
hdr = ["config", "backend_nthreads", "family", "type", "bits", "N", "seconds", "mflops"]
for j, h in enumerate(hdr):
    ws.write(0, j, h, F["hdr"])
ws.set_column(0, 0, 16); ws.set_column(2, 3, 13)
rr = 1
for cfg in CONFIGS:
    for typ in sorted(data[cfg].keys()):
        for N in sorted(data[cfg][typ].keys()):
            d = data[cfg][typ][N]
            ws.write_row(rr, 0, [cfg, d["nt"], "", typ, d["bits"], N, d["sec"], d["mflops"]])
            rr += 1

wb.close()
print("wrote", OUT, "(%d raw rows)" % (rr-1))
