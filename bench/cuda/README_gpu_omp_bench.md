# GPU (gdtq / mpc_cuda) vs CPU-OMP 同精度ベンチマーク 一式

BNCmatmul 0.24 の**基本線形計算(AXPY・行列×ベクトル・行列×行列)と疎行列ベクトル積(SpMV)**を
GPU (実数EFT = gdtq, 複素多倍長MPC = mpc_cuda) で実行し、**同一精度の CPU OpenMP 並列ルーチン**
(`_bncomp_*`, AVX-512, 32スレッド) と実行時間・正当性を比較する一式です。別環境でそのまま
再現できるよう、パス類はすべて `env.sh` の環境変数で切り替えられます。

対象精度: 実数 dd(106bit) td(159) qd(212) / ds(48) ts(72) qs(96)、複素 cmpf 106/212/424 bit。

---

## 1. ファイル一覧 (すべて `bench/cuda/`)

### 設定 / 実行スクリプト
| ファイル | 役割 |
|---|---|
| `env.sh`            | 共通設定 (PREFIX, CUDA_HOME, ARCH, ライブラリ名)。他スクリプトが source |
| `run_all.sh`        | **一括**: ビルド→実行→レポート生成 (推奨エントリポイント) |
| `build_gpu_omp.sh`  | 実数EFTベンチ (dense + SpMV) をビルド |
| `build_cmpf.sh`     | 複素 mpc_cuda ベンチを精度ごとにビルド (`build_cmpf.sh 106 212 424`) |
| `run_gpu_omp.sh`    | 実数スイープ実行 → `gpu_omp_real.csv` |
| `run_cmpf.sh`       | 複素スイープ実行 → `gpu_omp_cmpf.csv` |
| `make_report.py`    | CSV を集約し `gpu_report.xlsx` + `gpu_report.pdf` を生成 |

### ベンチマーク本体 (ソース)
| ファイル | 内容 |
|---|---|
| `gpu_omp_bench.cu`  | 密行列 matvec / matmul。GPU `mul_g<P>matrix_g<P>vec` / `mul_g<P>matrix_dev` vs CPU `_bncomp_mul_<P>matrix_<P>vec` / `_bncomp_mul_<P>matrix` |
| `gpu_spmv_bench.cu` | 疎行列 SpMV。型ごとに `-DBENCH_{DD,TD,QD,DS,TS,QS}` でコンパイル。GPU `mul_g<P>spmatrix` vs CPU `_bncomp_mul_<P>rsmatrix_<P>vec` (dd/td/qd) / serial (ds/ts/qs) |
| `cmpf_bench.cu`     | 複素多倍長MPC。`cu_mpc_t` (mpc_cuda) の GPU カーネル vs 同一MPC演算のOpenMP並列。`-DPREC=<bits>` |

### 結果データ / レポート (再生成される成果物)
| ファイル | 内容 |
|---|---|
| `gpu_omp_real.csv`  | 実数EFT結果 (matvec/matmul/spmv 各24行) |
| `gpu_omp_cmpf.csv`  | 複素MPC結果 (axpy/matvec/matmul × 3精度 = 27行) |
| `axpy_gpu.csv`      | 実数AXPY GPU (別セッション実測、レポートに再掲。`cuda_axpy_bench.cu` 由来) |
| `../mds/axpy_cpu.csv` | 実数AXPY CPU (serial+OMP)。`../mds/axpy_bench.cc` 由来 |
| `gpu_report.xlsx`   | 6シート: 概要/matvec/matmul/spmv/axpy/complex_mpc |
| `gpu_report.pdf`    | 7ページ: 手法・4演算の速度向上図・まとめ表・考察 |

CSV スキーマ: `op,family,type,bits,N,nnz,gpu_sec,cpu_sec,cpu_backend,speedup,relerr,gpu_mflops,cpu_mflops`
(`speedup = cpu_sec/gpu_sec`、`relerr` = GPU と CPU の相対誤差最大値=正当性。0 なら下位ビットまで一致)。

---

## 2. 前提 (別環境で用意するもの)

### 2-1. 外部ライブラリ (BNCmatmul 本体と同じ依存)
`$PREFIX` (既定 `/home/tkouya/local`) の下に以下がインストールされていること:
- **gdtq** (GPU実数多倍長): `$PREFIX/include/gdtq/` (gqd_type.h, gqd.cu ほか)
- **mpc_cuda** (GPU複素多倍長): `$PREFIX/include/mpc_cuda*`, `$PREFIX/lib/libmpc_cuda.a`
- **QD**: `$PREFIX/include/qd/`, `$PREFIX/lib/libqd`
- **GMP / MPFR / MPC**: `$PREFIX/lib/lib{gmp,mpfr,mpc}`
- `rds.h` (本リポジトリ `include/` にも同梱)
- **CUDA Toolkit** ($CUDA_HOME、既定 `/usr/local/cuda`)、OpenMP (`libgomp`)

### 2-2. BNCmatmul の静的ライブラリ (本リポジトリでビルド)
ベンチは以下 2 つの静的ライブラリにリンクします (リポジトリ直下):
- `libbncmatmul-0.24-omp_avx512.a`  … OpenMP EFT ルーチン (`_bncomp_*`)
- `libbncmatmul-0.24_avx512.a`       … 直列 EFT + ホスト補助関数 (init_/set_/subst_)

作り方 (本リポジトリで):
```bash
./configure --with-gmp=$PREFIX --with-qd=$PREFIX --with-gdtq=$PREFIX   # 必要に応じて
make avx512            # 直列/OMP の avx512 版 .a を生成 (Makefile.legacy 経由)
```
> AVX-512 非対応 CPU の場合は `env.sh` の `OMPLIB`/`SERLIB` を `*_avx2.a` に変更してください。

### 2-3. レポート生成用 (任意)
- Python3 + `xlsxwriter numpy matplotlib` (`pip install --user xlsxwriter numpy matplotlib`)
- 日本語PDF用に **CJKフォント** (Noto Sans CJK 等)。無ければ英数字のみで生成。
  別パスなら `export CJK_FONT=/path/to/font.ttc`。
- (任意) `pdftoppm` / `libreoffice` … PDF確認用

---

## 3. 実行手順

### 3-1. 一括 (推奨)
```bash
cd bncmatmul-0.24
# 自環境に合わせて上書き (例): A100 なら ARCH=sm_80
export PREFIX=/home/tkouya/local CUDA_HOME=/usr/local/cuda ARCH=sm_90
bench/cuda/run_all.sh
```
→ `bench/cuda/gpu_report.{xlsx,pdf}` が出力されます。

### 3-2. 個別
```bash
. bench/cuda/env.sh                 # 環境変数を設定
bash bench/cuda/build_gpu_omp.sh    # 実数ベンチをビルド
bash bench/cuda/build_cmpf.sh       # 複素ベンチをビルド (106 212 424)
bash bench/cuda/run_gpu_omp.sh      # 実数スイープ → gpu_omp_real.csv
bash bench/cuda/run_cmpf.sh         # 複素スイープ → gpu_omp_cmpf.csv
python3 bench/cuda/make_report.py "device: ... / OpenMP 32t"   # レポート生成
```

### 3-3. よく変えるパラメータ
- `ARCH` : **GPU の compute capability**。H100=sm_90, A100=sm_80, RTX40=sm_89 等。**要変更**
- `OMP_NUM_THREADS` : CPU 並列スレッド数 (既定 nproc)
- `run_gpu_omp.sh`: `--mv-sizes / --mm-sizes / --sizes / --band / --reps`
- SpMV サイズは既定で最大 262144 (下記の注意参照)
- `build_cmpf.sh 106 212` のように精度を絞れる

---

## 4. 手法・計測の注意 (レポートにも記載)

- **GPU時間**: カーネル実行のみ (データはデバイス常駐, `cudaDeviceSynchronize`, reps回の最小値)。
- **CPU時間**: `_bncomp_*` (OpenMP, AVX-512) の reps回最小値。SpMV の ds/ts/qs はライブラリに
  OMP版が無いため直列(serial)を基準。複素の CPU 基準は同一 MPC 演算の OpenMP 並列化。
- **正当性**: 全演算で GPU==CPU (relerr=0、単精度系のみ丸め由来 ~1e-9〜1e-12)。
- **matvec / SpMV の小〜中N** は CPU-OMP のスレッド生成オーバヘッド(~7ms)律速で、
  speedup はレイテンシ限界を反映。
- **SpMV 大N**: ライブラリの OMP版 EFT SpMV は行オフセット前置和を毎行再計算する **O(N²)** 実装
  のため N=1e6 で実用外。既定で **N≤262144** に制限。大Nの高速比にはこの CPU 側非効率が含まれる。
- **実数EFT matmul** はナイーブGPUカーネル(1スレッド1要素)で N≤2048 では CPU-OMP に及ばず
  (0.2〜1.0倍, N増で改善)。一方 **複素MPC matmul は 27〜59倍** (高演算密度で GPU 優位)。

### ハーネス側で回避しているライブラリの遅さ (ソース修正はしていない)
本ベンチは以下を回避するため、デバイス確保/転送を手動バルク化しています (`gpu_omp_bench.cu` の
`FAST_MAT/FAST_VEC` と SoA→AoS pack、`gpu_spmv_bench.cu` の nzero_index パディング):
1. `init_g*matrix_dev` / `subst_g*matrix_dev_*mat` が要素ごと cudaMemcpy (O(N²), ~10s@N=1024)。
2. `set_*rsmatrix_ij` は nzero_index が事前設定済みの列のみ書込む → 先に index を設定して呼ぶ。
   OMP AVX512 SpMV は SIMD パディング長まで gather するため nzero_index の後ろを実長へ realloc。

---

## 5. トラブルシュート
- `nvlink: Multiple definition ...` → 疎カーネルは各 `g<P>sparse.o` が算術を自己内包するため
  **型ごとに別バイナリ**にしています (`build_gpu_omp.sh` の `build_spmv`)。密は gdd/gds のみが
  算術を持つので 6 オブジェクトを一括リンク可。
- `undefined reference to init_ddrsmatrix ...` → `bncsparse.h` は extern"C" 非対応。
  ドライバでは `extern "C" { #include "bncsparse.h" }` 済み。C側ライブラリのリンク順は
  `--start-group ... --end-group` で解決。
- 複素ビルドで `gmp.h` 再定義エラー → mpc_cuda は system の gmp/mpc と衝突するため、`cmpf_bench.cu`
  は `cu_`接頭辞API + `mpc_cuda.cuh` のみ使用 (system mpc.h を include しない)。
- PDF の日本語が□になる → CJK フォント未検出。`CJK_FONT` を設定するか Noto Sans CJK を導入。
