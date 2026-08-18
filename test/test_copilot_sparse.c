#include <stdio.h>
#include "mkl.h"

int main() {
    // CSR形式で疎行列 A を定義（3x3 の行列）
    MKL_INT rows = 3;
    MKL_INT cols = 3;

    // CSR形式データ（例: A = [1 0 0; 0 2 0; 0 0 3]）
    MKL_INT rowIndex[] = {0, 1, 2, 3};
    MKL_INT columns[] = {0, 1, 2};
    double values[] = {1.0, 2.0, 3.0};

    // 密ベクトル x, y の定義
    double x[] = {1.0, 1.0, 1.0};  // 入力ベクトル
    double y[] = {0.0, 0.0, 0.0};  // 出力ベクトル（初期化）

    sparse_matrix_t A;
    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    // 疎行列ハンドルを作成
    mkl_sparse_d_create_csr(&A, SPARSE_INDEX_BASE_ZERO, rows, cols, rowIndex, rowIndex + 1, columns, values);

    // 疎行列と密ベクトルの積を計算： y = 1.0 * A * x + 0.0 * y
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A, descr, x, 0.0, y);

    // 結果の出力
    printf("結果 y = [");
    for (int i = 0; i < rows; i++) {
        printf(" %.2f", y[i]);
    }
    printf(" ]\n");

    // メモリ解放
    mkl_sparse_destroy(A);

    return 0;
}
