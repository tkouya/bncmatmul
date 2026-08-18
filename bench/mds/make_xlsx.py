#!/usr/bin/env python3
# Build mds_gpu_01.xlsx from results.csv (+ results_cplx.csv) using xlsxwriter.
# Tables of computation time / MFLOPS and serial->avx2/avx512 speedup, plus
# per (type,op) combo charts: computation time (log axis) + speedup (2nd axis).
import csv, os, sys
sys.path.insert(0, "/tmp/pylibs")
import xlsxwriter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT  = os.path.join(ROOT, "mds_gpu_01.xlsx")

SIZES    = [128, 256, 512, 1024, 2048]
BACKENDS = ["serial", "avx2", "avx512"]
OPS      = ["axpy", "matvec", "matmul", "spmv"]
REAL_TYPES = ["double", "dd", "td", "qd", "float", "ds", "ts", "qs"]
CPLX_TYPES = ["cd", "cdd", "ctd", "cqd", "cf", "cds", "cts", "cqs"]

TYPE_DESC = {
 "double":"double (binary64, ~16 digits)", "dd":"double-double (~32 digits)",
 "td":"triple-double (~48 digits)", "qd":"quad-double (~64 digits)",
 "float":"float (binary32, ~7 digits)", "ds":"double-single (~14 digits)",
 "ts":"triple-single (~21 digits)", "qs":"quad-single (~28 digits)",
 "cd":"complex double", "cdd":"complex double-double", "ctd":"complex triple-double",
 "cqd":"complex quad-double",
 "cf":"complex float", "cds":"complex double-single", "cts":"complex triple-single",
 "cqs":"complex quad-single",
}

def load(path, store):
    if not os.path.exists(path):
        return
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                key = (row["type"], row["op"], int(row["N"]), row["backend"])
                store[key] = (float(row["seconds"]), float(row["mflops"]))
            except (ValueError, KeyError):
                pass

DATA = {}
load(os.path.join(HERE, "results.csv"), DATA)
load(os.path.join(HERE, "results_cplx.csv"), DATA)

# memplus: real SuiteSparse matrix (Hamm/memplus) SpMV, keyed (type,backend) -> (sec,mflops)
MEMPLUS = {}
def load_memplus(path):
    if not os.path.exists(path):
        return
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                MEMPLUS[(row["type"], row["backend"])] = (float(row["seconds"]), float(row["mflops"]))
            except (ValueError, KeyError):
                pass
load_memplus(os.path.join(HERE, "results_memplus.csv"))

def get(t, op, n, b):
    return DATA.get((t, op, n, b))   # -> (sec, mflops) or None

wb = xlsxwriter.Workbook(OUT, {"nan_inf_to_errors": True})

# ---- formats ----
f_title  = wb.add_format({"bold": True, "font_size": 16})
f_h2     = wb.add_format({"bold": True, "font_size": 13, "font_color": "#1F4E78"})
f_hdr    = wb.add_format({"bold": True, "bg_color": "#DDEBF7", "border": 1, "align": "center", "valign":"vcenter", "text_wrap":True})
f_typ    = wb.add_format({"bold": True, "font_size": 12, "font_color": "#C00000"})
f_int    = wb.add_format({"border": 1, "align": "center"})
f_sec    = wb.add_format({"border": 1, "num_format": "0.000E+00"})
f_mf     = wb.add_format({"border": 1, "num_format": "#,##0.0"})
f_sp     = wb.add_format({"border": 1, "num_format": "0.00", "bg_color": "#E2EFDA"})
f_txt    = wb.add_format({"text_wrap": True, "valign": "top"})
f_na     = wb.add_format({"border": 1, "align": "center", "font_color": "#999999"})

# ============================== README sheet ==============================
ws = wb.add_worksheet("概要")
ws.set_column(0, 0, 22); ws.set_column(1, 1, 95)
ws.write(0, 0, "BNCmatmul ベンチマーク結果", f_title)
rows = [
 ("ライブラリ", "BNCmatmul 0.24 (多倍長・複素 行列演算ライブラリ)"),
 ("CPU", "Intel(R) Xeon(R) Gold 6526Y, 32 threads (single-thread benchmark)"),
 ("コンパイラ", "g++ 13.3.0, -O3 -ffp-contract=off"),
 ("バックエンド", "serial (スカラ) / avx2 (-mavx2 -mfma) / avx512 (-mavx512f -mfma)"),
 ("", "バックエンドはリンクするライブラリ (libbncmatmul-0.24[_avx2|_avx512].a) で選択"),
 ("演算", "axpy = add_cmul (c = a + α·b) / matvec (A·x) / matmul (A·B) / spmv (疎行列×ベクトル, 帯行列 half-bw=3)"),
 ("実数型", "double, dd(double-double), td(triple-double), qd(quad-double), float, ds(double-single), ts(triple-single), qs(quad-single)"),
 ("複素型(副疎=複素演算版)", "cd(complex double), cdd, ctd, cqd (ネイティブ複素型) ＋ cf(complex float), cds(complex double-single), cts(complex triple-single), cqs(complex quad-single)"),
 ("複素float型 (cf/cds/cts/cqs) の計測", "これらはライブラリにネイティブ複素型が無いため、実部・虚部に分解し既存の実数float/float-EFT (f/ds/ts/qs) ルーチンで計測 (ネイティブ cdd等が内部で用いる re/im 分解と同方式: (a+bi)(c+di)=(ac-bd)+(ad+bc)i)。FLOP基準は他の複素型と同一。"),
 ("行列サイズ N", "128, 256, 512, 1024, 2048"),
 ("計測", "各演算を tmin=0.2s に達するまで反復、1回あたりの平均秒を記録 (大規模は1回)"),
 ("MFLOPS", "axpy:2N, matvec:2N^2, matmul:2N^3, spmv:2·nnz を演算回数として算出"),
 ("性能向上比 (Speedup)", "serial時間 / avx2(または avx512)時間。各演算シートの表と、計算時間グラフ上に第2軸で描画。"),
 ("AVX-512 対応強化 (本改修)", "ds/ts/qs の AVX-512 カーネル (dslinear/tslinear/qslinear.c) を修正、および多倍長疎行列ベクトル積 (sparse_dd/td/qd.c) の AVX-512 境界バグ (segfault) を修正。全演算 serial と完全一致を確認済。"),
 ("精度線形計算の新規実装 (本改修)", "float系の実数疎行列ベクトル積 (SpMV) を新規追加: float (sparse_float.c), ds (sparse_ds.c), ts (sparse_ts.c), qs (sparse_qs.c)。各々スカラ+AVX2+AVX-512 カーネルを実装し、serial と完全一致を確認済。"),
 ("疎行列 spmv の型", "double/dd/td/qd, float/ds/ts/qs (本改修で追加), 複素 cd/cdd/ctd/cqd。"),
 ("実疎行列ベンチ (memplus)", "合成帯行列に加え、実際の疎行列 SuiteSparse Hamm/memplus (17758x17758, nnz=126150, ~7.1 nnz/行) の SpMV を memplus シートに収録。Matrix Market を読み込み全精度型で計測 (serial と一致確認済)。この検証で double 実疎行列 SpMV の AVX2/AVX-512 の padding 取りこぼし + jres=1 添字バグを発見・修正。"),
]
r = 2
for k, v in rows:
    ws.write(r, 0, k, wb.add_format({"bold": True, "valign":"top"}))
    ws.write(r, 1, v, f_txt); r += 1
ws.write(r+1, 0, "シート構成", f_h2)
ws.write(r+2, 1, "raw: 全測定値 / axpy・matvec・matmul・spmv: 演算別の時間・MFLOPS・速度向上比の表とグラフ", f_txt)

# ============================== raw sheet =================================
wsr = wb.add_worksheet("raw")
hdr = ["backend", "type", "op", "N", "seconds", "iters_per_meas", "MFLOPS"]
for c, h in enumerate(hdr): wsr.write(0, c, h, f_hdr)
wsr.set_column(0, 2, 9); wsr.set_column(3, 6, 12)
rr = 1
allrows = []
_seen = {}
def collect(path):
    if not os.path.exists(path): return
    with open(path) as f:
        for row in csv.reader(f):
            if row and row[0] != "backend" and len(row) >= 7:
                _seen[(row[0], row[1], row[2], row[3])] = row  # dedup, last wins
collect(os.path.join(HERE, "results.csv"))
collect(os.path.join(HERE, "results_cplx.csv"))
def _ord(r):
    bo = {"serial":0,"avx2":1,"avx512":2}.get(r[0],9)
    return (r[2], r[1], bo, int(r[3]))
allrows = sorted(_seen.values(), key=_ord)
for row in allrows:
    try:
        wsr.write(rr,0,row[0]); wsr.write(rr,1,row[1]); wsr.write(rr,2,row[2])
        wsr.write_number(rr,3,int(row[3])); wsr.write_number(rr,4,float(row[4]))
        wsr.write_number(rr,5,int(row[5])); wsr.write_number(rr,6,float(row[6]))
        rr += 1
    except (ValueError, IndexError):
        pass
wsr.freeze_panes(1, 0)

# ===================== per-operation sheets with charts ===================
def build_op_sheet(op, types):
    ws = wb.add_worksheet(op)
    ws.set_column(0, 0, 10)
    ws.set_column(1, 9, 12)
    ws.write(0, 0, f"演算: {op}", f_title)
    ws.write(1, 0, "時間[s] / MFLOPS / 速度向上比 (serial基準) / avx2[s]÷avx512[s]。グラフ左: 計算時間(対数)＋速度向上比(右軸)、右: avx2時間÷avx512時間", f_txt)
    row = 3
    COLN, COLser, COLa2, COLa5 = 0, 1, 2, 3
    COLmser, COLma2, COLma5 = 4, 5, 6
    COLsp2, COLsp5 = 7, 8
    COLr = 9   # avx2[s] / avx512[s]  (avx512 が avx2 の何倍速いか)
    headers = ["N", "serial[s]", "avx2[s]", "avx512[s]",
               "serial MFLOPS", "avx2 MFLOPS", "avx512 MFLOPS",
               "speedup avx2", "speedup avx512", "avx2[s]/avx512[s]"]
    for t in types:
        ws.write(row, 0, f"{t}  —  {TYPE_DESC.get(t,'')}", f_typ)
        hrow = row + 1
        for c, h in enumerate(headers): ws.write(hrow, c, h, f_hdr)
        first = hrow + 1
        for i, n in enumerate(SIZES):
            rrow = first + i
            ws.write_number(rrow, COLN, n, f_int)
            secs = {}
            for b, col, mcol in (("serial",COLser,COLmser),("avx2",COLa2,COLma2),("avx512",COLa5,COLma5)):
                v = get(t, op, n, b)
                if v:
                    ws.write_number(rrow, col, v[0], f_sec)
                    ws.write_number(rrow, mcol, v[1], f_mf)
                    secs[b] = v[0]
                else:
                    ws.write(rrow, col, "N/A", f_na)
                    ws.write(rrow, mcol, "N/A", f_na)
            # speedups
            if "serial" in secs and "avx2" in secs and secs["avx2"] > 0:
                ws.write_number(rrow, COLsp2, secs["serial"]/secs["avx2"], f_sp)
            else:
                ws.write(rrow, COLsp2, "N/A", f_na)
            if "serial" in secs and "avx512" in secs and secs["avx512"] > 0:
                ws.write_number(rrow, COLsp5, secs["serial"]/secs["avx512"], f_sp)
            else:
                ws.write(rrow, COLsp5, "N/A", f_na)
            # avx2[s] / avx512[s]  ( >1 なら avx512 の方が高速 )
            if "avx2" in secs and "avx512" in secs and secs["avx512"] > 0:
                ws.write_number(rrow, COLr, secs["avx2"]/secs["avx512"], f_sp)
            else:
                ws.write(rrow, COLr, "N/A", f_na)
        last = first + len(SIZES) - 1
        # ---- combo chart: time (log, primary) + speedup (secondary) ----
        ch = wb.add_chart({"type": "line"})
        cats = [op, first, COLN, last, COLN]
        for name, col, color in (("serial", COLser, "#4472C4"),
                                 ("avx2",   COLa2,  "#ED7D31"),
                                 ("avx512", COLa5,  "#70AD47")):
            ch.add_series({
                "name": f"{name} [s]",
                "categories": cats,
                "values": [op, first, col, last, col],
                "line": {"color": color, "width": 2.0},
                "marker": {"type": "circle", "size": 4},
            })
        ch2 = wb.add_chart({"type": "line"})
        for name, col, color in (("speedup avx2", COLsp2, "#ED7D31"),
                                 ("speedup avx512", COLsp5, "#70AD47")):
            ch2.add_series({
                "name": name,
                "categories": cats,
                "values": [op, first, col, last, col],
                "line": {"color": color, "width": 1.25, "dash_type": "dash"},
                "marker": {"type": "diamond", "size": 4},
                "y2_axis": True,
            })
        ch.combine(ch2)
        ch.set_title({"name": f"{t}  {op}"})
        ch.set_x_axis({"name": "N", "num_format": "0"})
        ch.set_y_axis({"name": "time [s]", "log_base": 10, "major_gridlines": {"visible": True}})
        ch2.set_y2_axis({"name": "speedup (×)", "min": 0})
        ch.set_legend({"position": "bottom"})
        ch.set_size({"width": 520, "height": 300})
        ws.insert_chart(hrow, 10, ch)
        # ---- dedicated chart: avx2[s] / avx512[s]  ( >1 → avx512 が高速 ) ----
        chs = wb.add_chart({"type": "line"})
        chs.add_series({
            "name": "avx2[s] / avx512[s]",
            "categories": cats,
            "values": [op, first, COLr, last, COLr],
            "line": {"color": "#7030A0", "width": 2.25},
            "marker": {"type": "diamond", "size": 6},
            "data_labels": {"value": True, "num_format": "0.00", "position": "above"},
        })
        chs.set_title({"name": f"{t}  {op}  — avx2時間 ÷ avx512時間"})
        chs.set_x_axis({"name": "N", "num_format": "0"})
        chs.set_y_axis({"name": "avx2[s] / avx512[s]  (>1で avx512 高速)", "min": 0,
                        "major_gridlines": {"visible": True}})
        chs.set_legend({"position": "bottom"})
        chs.set_size({"width": 520, "height": 300})
        ws.insert_chart(hrow, 19, chs)
        row = last + 2 + 14   # leave room for the chart (~14 rows)
    ws.freeze_panes(3, 0)

for op in OPS:
    types = (REAL_TYPES + CPLX_TYPES) if op != "spmv" else (["double","dd","td","qd","float","ds","ts","qs"] + CPLX_TYPES)
    build_op_sheet(op, types)

# ===================== memplus sheet (real-matrix SpMV) ===================
def build_memplus_sheet():
    if not MEMPLUS:
        return
    ws = wb.add_worksheet("memplus")
    ws.set_column(0, 0, 28); ws.set_column(1, 9, 13)
    ws.write(0, 0, "実疎行列 SpMV ベンチマーク: Hamm/memplus", f_title)
    notes = [
        "行列: SuiteSparse (旧 UF) Collection の Hamm/memplus — 回路シミュレーション由来の実非対称疎行列",
        "サイズ: 17758 x 17758, 非零要素 126150 (平均 ~7.1 nnz/行)",
        "演算: y = A·x (SpMV)。MFLOPS = 2·nnz / 時間。Matrix Market 形式を読み込み、各精度の疎行列を",
        "      double 構造から limb0=値・他=0 でコピー構築 (密行列を経由しない)。serial と完全一致を確認済。",
        "注: 平均 ~7 nnz/行と非常に疎なため、float系 (AVX-512 は 16 幅padding) は1行あたりの padding 無駄が",
        "    大きく SIMD が serial を下回ることがある。double系 (8幅) は padding 無駄が小さく SIMD が有効。",
    ]
    r = 1
    for n in notes:
        ws.write(r, 0, n, f_txt); r += 1
    r += 1
    headers = ["type", "serial[s]", "avx2[s]", "avx512[s]",
               "serial MFLOPS", "avx2 MFLOPS", "avx512 MFLOPS",
               "speedup avx2", "speedup avx512", "avx2[s]/avx512[s]"]
    for c, h in enumerate(headers): ws.write(r, c, h, f_hdr)
    first = r + 1
    rtypes = ["double", "dd", "td", "qd", "float", "ds", "ts", "qs"]
    for i, t in enumerate(rtypes):
        rr = first + i
        ws.write(rr, 0, f"{t}  —  {TYPE_DESC.get(t,'')}", f_typ)
        secs = {}
        for b, sc, mc in (("serial",1,4),("avx2",2,5),("avx512",3,6)):
            v = MEMPLUS.get((t, b))
            if v:
                ws.write_number(rr, sc, v[0], f_sec); ws.write_number(rr, mc, v[1], f_mf); secs[b] = v[0]
            else:
                ws.write(rr, sc, "N/A", f_na); ws.write(rr, mc, "N/A", f_na)
        if "serial" in secs and "avx2"  in secs and secs["avx2"]  > 0: ws.write_number(rr, 7, secs["serial"]/secs["avx2"],  f_sp)
        else: ws.write(rr, 7, "N/A", f_na)
        if "serial" in secs and "avx512" in secs and secs["avx512"] > 0: ws.write_number(rr, 8, secs["serial"]/secs["avx512"], f_sp)
        else: ws.write(rr, 8, "N/A", f_na)
        if "avx2" in secs and "avx512" in secs and secs["avx512"] > 0: ws.write_number(rr, 9, secs["avx2"]/secs["avx512"], f_sp)
        else: ws.write(rr, 9, "N/A", f_na)
    last = first + len(rtypes) - 1
    # MFLOPS bar chart per type x backend
    ch = wb.add_chart({"type": "column"})
    cats = ["memplus", first, 0, last, 0]
    for name, col, color in (("serial", 4, "#4472C4"), ("avx2", 5, "#ED7D31"), ("avx512", 6, "#70AD47")):
        ch.add_series({"name": name, "categories": cats,
                       "values": ["memplus", first, col, last, col],
                       "fill": {"color": color}})
    ch.set_title({"name": "memplus SpMV — MFLOPS"})
    ch.set_x_axis({"name": "type"}); ch.set_y_axis({"name": "MFLOPS"})
    ch.set_legend({"position": "bottom"}); ch.set_size({"width": 640, "height": 340})
    ws.insert_chart(first, 11, ch)
    ws.freeze_panes(first, 0)
build_memplus_sheet()

wb.close()
print("wrote", OUT, "rows(raw)=", len(allrows))
