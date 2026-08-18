# attic/ — retired sources

2026-08 のソース整理で現役ツリーから退避した「ビルド対象外・参照ゼロ」のファイル置き場。
削除ではなく退避なので、必要になったら `git mv` で戻せる。
`make dist` には含まれない(配布物は現役ソースのみ)。

## 分類

- `src/*_old.c, *_org.c, *_20250811.c` — 旧版スナップショット
- `src/*_avx.c, *_avx2.c, *_neon*.c, bncavx.c, ddlu_nosimd.c` — SIMD 手動移植の旧版。
  現行の SIMD は include/{avx2,neon,sve2}/ のヘッダに集約され、同一 .c を
  ISA 別に再コンパイルする方式に置き換わった
- `src/*_omp.c` (krylov/cg/cocg/dka) — OpenMP 旧版(現行は bncomp_* 系 + OMP_STRASSEN_SOURCES)
- `src/c_dd.c, c_qd.c, rdd.c` — c_/r 層の旧・外部シンボル実装。
  現行は c_dd_qd.h / rdd.h の static inline と rdtq_func.c が担う
- `src/short_mmbench_*, serial_lu_bench.c, test_*.c, cond_mat.c, get_linear_system.c`
  — src/ に紛れ込んでいたベンチ・テスト類(現役ベンチは bench/ 配下)
- `src/clinear.c, clu.c, dlu.c, mpclu*.c, power.c, iterative.c, ...` — Makefile.legacy
  時代にはビルドされていたが automake 移行時に外れた機能持ち孤児
- `include/` — 上記に対応する旧版ヘッダ(rsd.h は rds.h とインクルードガードが
  衝突する旧版、mpfr_dd_td_qd.h は mpfr_dtq_sd.h に完全包含)
- `test/` — 旧版テストドライバ
