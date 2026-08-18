// Small matrix multiplicaiton

// 4x4
void mul_small_ddmatrix_4x4(DDMatrix ret, DDMatrix mat_a, DDMatrix mat_b)
{
    long row_dim, mid_dim, col_dim;
    long i, j, k, ij, ik, kj;
    double in_ret[16][DDSIZE]; // 4x4
    double in_mat_a[16][DDSIZE]; // 4x4
    double in_mat_b[16][DDSIZE];
    double tmp[4][DDSIZE];

    // in_ret, in_mat_a, in_mat_b := ret, mat_a, mat_b
    row_dim = mat_a->row_dim;
    mid_dim = mat_a->col_dim;
    col_dim = mat_b->col_dim;

    // in_mat_a := 0
    in_mat_a[0][0]  = 0.0; in_mat_a[1][0]  = 0.0; in_mat_a[2][0]  = 0.0; in_mat_a[3][0]  = 0.0;
    in_mat_a[0][1]  = 0.0; in_mat_a[1][1]  = 0.0; in_mat_a[2][1]  = 0.0; in_mat_a[3][1]  = 0.0;
    in_mat_a[4][0]  = 0.0; in_mat_a[5][0]  = 0.0; in_mat_a[6][0]  = 0.0; in_mat_a[7][0]  = 0.0;
    in_mat_a[4][1]  = 0.0; in_mat_a[5][1]  = 0.0; in_mat_a[6][1]  = 0.0; in_mat_a[7][1]  = 0.0;
    in_mat_a[8][0]  = 0.0; in_mat_a[9][0]  = 0.0; in_mat_a[10][0] = 0.0; in_mat_a[11][0] = 0.0;
    in_mat_a[8][1]  = 0.0; in_mat_a[9][1]  = 0.0; in_mat_a[10][1] = 0.0; in_mat_a[11][1] = 0.0;
    in_mat_a[12][0] = 0.0; in_mat_a[13][0] = 0.0; in_mat_a[14][0] = 0.0; in_mat_a[15][0] = 0.0;
    in_mat_a[12][1] = 0.0; in_mat_a[13][1] = 0.0; in_mat_a[14][1] = 0.0; in_mat_a[15][1] = 0.0;

    // in_mat_b := 0
    in_mat_b[0][0]  = 0.0; in_mat_b[1][0]  = 0.0; in_mat_b[2][0]  = 0.0; in_mat_b[3][0]  = 0.0;
    in_mat_b[0][1]  = 0.0; in_mat_b[1][1]  = 0.0; in_mat_b[2][1]  = 0.0; in_mat_b[3][1]  = 0.0;
    in_mat_b[4][0]  = 0.0; in_mat_b[5][0]  = 0.0; in_mat_b[6][0]  = 0.0; in_mat_b[7][0]  = 0.0;
    in_mat_b[4][1]  = 0.0; in_mat_b[5][1]  = 0.0; in_mat_b[6][1]  = 0.0; in_mat_b[7][1]  = 0.0;
    in_mat_b[8][0]  = 0.0; in_mat_b[9][0]  = 0.0; in_mat_b[10][0] = 0.0; in_mat_b[11][0] = 0.0;
    in_mat_b[8][1]  = 0.0; in_mat_b[9][1]  = 0.0; in_mat_b[10][1] = 0.0; in_mat_b[11][1] = 0.0;
    in_mat_b[12][0] = 0.0; in_mat_b[13][0] = 0.0; in_mat_b[14][0] = 0.0; in_mat_b[15][0] = 0.0;
    in_mat_b[12][1] = 0.0; in_mat_b[13][1] = 0.0; in_mat_b[14][1] = 0.0; in_mat_b[15][1] = 0.0;

    // in_mat_a := mat_a
    for(i = 0; i < row_dim; i++)
    {
        for(j = 0; j < mid_dim; j++)
        {
            ij = i * mid_dim + j;
            in_mat_a[ij][0] = mat_a->element[0][ij]; 
            in_mat_a[ij][1] = mat_a->element[1][ij];
        }
    }

    // in_mat_b := mat_b
    for(i = 0; i < mid_dim; i++)
    {
        for(j = 0; j < col_dim; j++)
        {
            ij = i * col_dim + j;
            in_mat_b[ij][0] = mat_b->element[0][ij]; 
            in_mat_b[ij][1] = mat_b->element[1][ij];
        }
    }

    // in_ret := 0
    in_ret[0][0]  = 0.0; in_ret[1][0]  = 0.0; in_ret[2][0]  = 0.0; in_ret[3][0]  = 0.0;
    in_ret[0][1]  = 0.0; in_ret[1][1]  = 0.0; in_ret[2][1]  = 0.0; in_ret[3][1]  = 0.0;
    in_ret[4][0]  = 0.0; in_ret[5][0]  = 0.0; in_ret[6][0]  = 0.0; in_ret[7][0]  = 0.0;
    in_ret[4][1]  = 0.0; in_ret[5][1]  = 0.0; in_ret[6][1]  = 0.0; in_ret[7][1]  = 0.0;
    in_ret[8][0]  = 0.0; in_ret[9][0]  = 0.0; in_ret[10][0] = 0.0; in_ret[11][0] = 0.0;
    in_ret[8][1]  = 0.0; in_ret[9][1]  = 0.0; in_ret[10][1] = 0.0; in_ret[11][1] = 0.0;
    in_ret[12][0] = 0.0; in_ret[13][0] = 0.0; in_ret[14][0] = 0.0; in_ret[15][0] = 0.0;
    in_ret[12][1] = 0.0; in_ret[13][1] = 0.0; in_ret[14][1] = 0.0; in_ret[15][1] = 0.0;

    // in_ret := in_mat_a * in_mat_b
    // ret[0, 0]
    rdd_mul(tmp[0], in_mat_a[0], in_mat_b[0]);
    rdd_mul(tmp[1], in_mat_a[1], in_mat_b[4]);
    rdd_mul(tmp[2], in_mat_a[2], in_mat_b[8]);
    rdd_mul(tmp[3], in_mat_a[3], in_mat_b[12]);
    rdd_add(in_ret[0], in_ret[0], tmp[0]);
    rdd_add(in_ret[0], in_ret[0], tmp[1]);
    rdd_add(in_ret[0], in_ret[0], tmp[2]);
    rdd_add(in_ret[0], in_ret[0], tmp[3]);

    // ret[0, 1]
    rdd_mul(tmp[0], in_mat_a[0], in_mat_b[1]);
    rdd_mul(tmp[1], in_mat_a[1], in_mat_b[5]);
    rdd_mul(tmp[2], in_mat_a[2], in_mat_b[9]);
    rdd_mul(tmp[3], in_mat_a[3], in_mat_b[13]);
    rdd_add(in_ret[0], in_ret[0], tmp[0]);
    rdd_add(in_ret[0], in_ret[0], tmp[1]);
    rdd_add(in_ret[0], in_ret[0], tmp[2]);
    rdd_add(in_ret[0], in_ret[0], tmp[3]);

    // ret[0, 2]
    rdd_mul(tmp[0], in_mat_a[0], in_mat_b[2]);
    rdd_mul(tmp[1], in_mat_a[1], in_mat_b[6]);
    rdd_mul(tmp[2], in_mat_a[2], in_mat_b[10]);
    rdd_mul(tmp[3], in_mat_a[3], in_mat_b[14]);
    rdd_add(in_ret[0], in_ret[0], tmp[0]);
    rdd_add(in_ret[0], in_ret[0], tmp[1]);
    rdd_add(in_ret[0], in_ret[0], tmp[2]);
    rdd_add(in_ret[0], in_ret[0], tmp[3]);

    // ret[0, 3]
    rdd_mul(tmp[0], in_mat_a[0], in_mat_b[3]);
    rdd_mul(tmp[1], in_mat_a[1], in_mat_b[7]);
    rdd_mul(tmp[2], in_mat_a[2], in_mat_b[11]);
    rdd_mul(tmp[3], in_mat_a[3], in_mat_b[15]);
    rdd_add(in_ret[0], in_ret[0], tmp[0]);
    rdd_add(in_ret[0], in_ret[0], tmp[1]);
    rdd_add(in_ret[0], in_ret[0], tmp[2]);
    rdd_add(in_ret[0], in_ret[0], tmp[3]);

    // ret[1, 0]
    rdd_mul(tmp[0], in_mat_a[4], in_mat_b[0]);
    rdd_mul(tmp[1], in_mat_a[5], in_mat_b[4]);
    rdd_mul(tmp[2], in_mat_a[6], in_mat_b[8]);
    rdd_mul(tmp[3], in_mat_a[7], in_mat_b[12]);
    rdd_add(in_ret[0], in_ret[0], tmp[0]);
    rdd_add(in_ret[0], in_ret[0], tmp[1]);
    rdd_add(in_ret[0], in_ret[0], tmp[2]);
    rdd_add(in_ret[0], in_ret[0], tmp[3]);

    // ret[1, 1]
    rdd_mul(tmp[0], in_mat_a[4], in_mat_b[1]);
    rdd_mul(tmp[1], in_mat_a[5], in_mat_b[5]);
    rdd_mul(tmp[2], in_mat_a[6], in_mat_b[9]);
    rdd_mul(tmp[3], in_mat_a[7], in_mat_b[13]);
    rdd_add(in_ret[0], in_ret[0], tmp[0]);
    rdd_add(in_ret[0], in_ret[0], tmp[1]);
    rdd_add(in_ret[0], in_ret[0], tmp[2]);
    rdd_add(in_ret[0], in_ret[0], tmp[3]);

    // ret[1, 2]
    rdd_mul(tmp[0], in_mat_a[4], in_mat_b[2]);
    rdd_mul(tmp[1], in_mat_a[5], in_mat_b[6]);
    rdd_mul(tmp[2], in_mat_a[6], in_mat_b[10]);
    rdd_mul(tmp[3], in_mat_a[7], in_mat_b[14]);
    rdd_add(in_ret[0], in_ret[0], tmp[0]);
    rdd_add(in_ret[0], in_ret[0], tmp[1]);
    rdd_add(in_ret[0], in_ret[0], tmp[2]);
    rdd_add(in_ret[0], in_ret[0], tmp[3]);

    // ret[1, 3]
    rdd_mul(tmp[0], in_mat_a[4], in_mat_b[3]);
    rdd_mul(tmp[1], in_mat_a[5], in_mat_b[7]);
    rdd_mul(tmp[2], in_mat_a[6], in_mat_b[11]);
    rdd_mul(tmp[3], in_mat_a[7], in_mat_b[15]);
    rdd_add(in_ret[0], in_ret[0], tmp[0]);
    rdd_add(in_ret[0], in_ret[0], tmp[1]);
    rdd_add(in_ret[0], in_ret[0], tmp[2]);
    rdd_add(in_ret[0], in_ret[0], tmp[3]);

    // ret[2, 0]
    rdd_mul(tmp[0], in_mat_a[8], in_mat_b[0]);
    rdd_mul(tmp[1], in_mat_a[9], in_mat_b[4]);
    rdd_mul(tmp[2], in_mat_a[10], in_mat_b[8]);
    rdd_mul(tmp[3], in_mat_a[11], in_mat_b[12]);
    rdd_add(in_ret[0], in_ret[0], tmp[0]);
    rdd_add(in_ret[0], in_ret[0], tmp[1]);
    rdd_add(in_ret[0], in_ret[0], tmp[2]);
    rdd_add(in_ret[0], in_ret[0], tmp[3]);

    // ret[2, 1]
    rdd_mul(tmp[0], in_mat_a[8], in_mat_b[1]);
    rdd_mul(tmp[1], in_mat_a[9], in_mat_b[5]);
    rdd_mul(tmp[2], in_mat_a[10], in_mat_b[9]);
    rdd_mul(tmp[3], in_mat_a[11], in_mat_b[13]);
    rdd_add(in_ret[0], in_ret[0], tmp[0]);
    rdd_add(in_ret[0], in_ret[0], tmp[1]);
    rdd_add(in_ret[0], in_ret[0], tmp[2]);
    rdd_add(in_ret[0], in_ret[0], tmp[3]);

    // ret[2, 2]
    rdd_mul(tmp[0], in_mat_a[8], in_mat_b[2]);
    rdd_mul(tmp[1], in_mat_a[9], in_mat_b[6]);
    rdd_mul(tmp[2], in_mat_a[10], in_mat_b[10]);
    rdd_mul(tmp[3], in_mat_a[11], in_mat_b[14]);
    rdd_add(in_ret[0], in_ret[0], tmp[0]);
    rdd_add(in_ret[0], in_ret[0], tmp[1]);
    rdd_add(in_ret[0], in_ret[0], tmp[2]);
    rdd_add(in_ret[0], in_ret[0], tmp[3]);

    // ret[2, 3]
    rdd_mul(tmp[0], in_mat_a[8], in_mat_b[3]);
    rdd_mul(tmp[1], in_mat_a[9], in_mat_b[7]);
    rdd_mul(tmp[2], in_mat_a[10], in_mat_b[11]);
    rdd_mul(tmp[3], in_mat_a[11], in_mat_b[15]);
    rdd_add(in_ret[0], in_ret[0], tmp[0]);
    rdd_add(in_ret[0], in_ret[0], tmp[1]);
    rdd_add(in_ret[0], in_ret[0], tmp[2]);
    rdd_add(in_ret[0], in_ret[0], tmp[3]);

    // ret[3, 0]
    rdd_mul(tmp[0], in_mat_a[12], in_mat_b[0]);
    rdd_mul(tmp[1], in_mat_a[13], in_mat_b[4]);
    rdd_mul(tmp[2], in_mat_a[14], in_mat_b[8]);
    rdd_mul(tmp[3], in_mat_a[15], in_mat_b[12]);
    rdd_add(in_ret[0], in_ret[0], tmp[0]);
    rdd_add(in_ret[0], in_ret[0], tmp[1]);
    rdd_add(in_ret[0], in_ret[0], tmp[2]);
    rdd_add(in_ret[0], in_ret[0], tmp[3]);

    // ret[3, 1]
    rdd_mul(tmp[0], in_mat_a[12], in_mat_b[1]);
    rdd_mul(tmp[1], in_mat_a[13], in_mat_b[5]);
    rdd_mul(tmp[2], in_mat_a[14], in_mat_b[9]);
    rdd_mul(tmp[3], in_mat_a[15], in_mat_b[13]);
    rdd_add(in_ret[0], in_ret[0], tmp[0]);
    rdd_add(in_ret[0], in_ret[0], tmp[1]);
    rdd_add(in_ret[0], in_ret[0], tmp[2]);
    rdd_add(in_ret[0], in_ret[0], tmp[3]);

    // ret[3, 2]
    rdd_mul(tmp[0], in_mat_a[12], in_mat_b[2]);
    rdd_mul(tmp[1], in_mat_a[13], in_mat_b[6]);
    rdd_mul(tmp[2], in_mat_a[14], in_mat_b[10]);
    rdd_mul(tmp[3], in_mat_a[15], in_mat_b[14]);
    rdd_add(in_ret[0], in_ret[0], tmp[0]);
    rdd_add(in_ret[0], in_ret[0], tmp[1]);
    rdd_add(in_ret[0], in_ret[0], tmp[2]);
    rdd_add(in_ret[0], in_ret[0], tmp[3]);

    // ret[3, 3]
    rdd_mul(tmp[0], in_mat_a[12], in_mat_b[3]);
    rdd_mul(tmp[1], in_mat_a[13], in_mat_b[7]);
    rdd_mul(tmp[2], in_mat_a[14], in_mat_b[11]);
    rdd_mul(tmp[3], in_mat_a[15], in_mat_b[15]);
    rdd_add(in_ret[0], in_ret[0], tmp[0]);
    rdd_add(in_ret[0], in_ret[0], tmp[1]);
    rdd_add(in_ret[0], in_ret[0], tmp[2]);
    rdd_add(in_ret[0], in_ret[0], tmp[3]);

    // ret := in_ret
    for(i = 0; i < row_dim; i++)
    {
        for(j = 0; j < col_dim; j++)
        {
            ij = i * mid_dim + j;
            ret->element[0][ij] = in_ret[ij][0]; 
            ret->element[1][ij] = in_ret[ij][1];
        }
    }
}

// 8x8

// 16x16

// 32x32
void mul_small_ddmatrix_32x32(DDMatrix ret, DDMatrix mat_a, DDMatrix mat_b)
{
    long row_dim, mid_dim, col_dim;
    long i, j, k, ij, ik, kj;
    double in_ret[1024][DDSIZE]; // 4x4
    double in_mat_a[1024][DDSIZE]; // 4x4
    double in_mat_b[1024][DDSIZE];
    double tmp[32][DDSIZE];

    // in_ret, in_mat_a, in_mat_b := ret, mat_a, mat_b
    row_dim = mat_a->row_dim;
    mid_dim = mat_a->col_dim;
    col_dim = mat_b->col_dim;

    // in_mat_a := 0
    for(i = 0; i < 1024; i++)
        rdd_set0(in_mat_a[i]);
    
    // in_mat_b := 0
    for(i = 0; i < 1024; i++)
        rdd_set0(in_mat_b[i]);

    // in_mat_a := mat_a
    for(i = 0; i < row_dim; i++)
    {
        for(j = 0; j < mid_dim; j++)
        {
            ij = i * mid_dim + j;
            in_mat_a[ij][0] = mat_a->element[0][ij]; 
            in_mat_a[ij][1] = mat_a->element[1][ij];
        }
    }

    // in_mat_b := mat_b
    for(i = 0; i < mid_dim; i++)
    {
        for(j = 0; j < col_dim; j++)
        {
            ij = i * col_dim + j;
            in_mat_b[ij][0] = mat_b->element[0][ij]; 
            in_mat_b[ij][1] = mat_b->element[1][ij];
        }
    }

    // in_ret := 0
    for(i = 0; i < 1024; i++)
        rdd_set0(in_ret[i]);

    // in_ret := in_mat_a * in_mat_b
    for(i = 0; i < 32; i++)
    {
        ik = i * mid_dim; 
        for(j = 0; j < 32; j++)
        {
            ij = i * col_dim + j;
            rdd_mul(tmp[ 0], in_mat_a[ik +  0], in_mat_b[mid_dim *  0 + j]);
            rdd_mul(tmp[ 1], in_mat_a[ik +  1], in_mat_b[mid_dim *  1 + j]);
            rdd_mul(tmp[ 2], in_mat_a[ik +  2], in_mat_b[mid_dim *  2 + j]);
            rdd_mul(tmp[ 3], in_mat_a[ik +  3], in_mat_b[mid_dim *  3 + j]);
            rdd_mul(tmp[ 4], in_mat_a[ik +  4], in_mat_b[mid_dim *  4 + j]);
            rdd_mul(tmp[ 5], in_mat_a[ik +  5], in_mat_b[mid_dim *  5 + j]);
            rdd_mul(tmp[ 6], in_mat_a[ik +  6], in_mat_b[mid_dim *  6 + j]);
            rdd_mul(tmp[ 7], in_mat_a[ik +  7], in_mat_b[mid_dim *  7 + j]);
            rdd_mul(tmp[ 8], in_mat_a[ik +  8], in_mat_b[mid_dim *  8 + j]);
            rdd_mul(tmp[ 9], in_mat_a[ik +  9], in_mat_b[mid_dim *  9 + j]);
            rdd_mul(tmp[10], in_mat_a[ik + 10], in_mat_b[mid_dim * 10 + j]);
            rdd_mul(tmp[11], in_mat_a[ik + 11], in_mat_b[mid_dim * 11 + j]);
            rdd_mul(tmp[12], in_mat_a[ik + 12], in_mat_b[mid_dim * 12 + j]);
            rdd_mul(tmp[13], in_mat_a[ik + 13], in_mat_b[mid_dim * 13 + j]);
            rdd_mul(tmp[14], in_mat_a[ik + 14], in_mat_b[mid_dim * 14 + j]);
            rdd_mul(tmp[15], in_mat_a[ik + 15], in_mat_b[mid_dim * 15 + j]);
            rdd_mul(tmp[16], in_mat_a[ik + 16], in_mat_b[mid_dim * 16 + j]);
            rdd_mul(tmp[17], in_mat_a[ik + 17], in_mat_b[mid_dim * 17 + j]);
            rdd_mul(tmp[18], in_mat_a[ik + 18], in_mat_b[mid_dim * 18 + j]);
            rdd_mul(tmp[19], in_mat_a[ik + 19], in_mat_b[mid_dim * 19 + j]);
            rdd_mul(tmp[20], in_mat_a[ik + 20], in_mat_b[mid_dim * 20 + j]);
            rdd_mul(tmp[21], in_mat_a[ik + 21], in_mat_b[mid_dim * 21 + j]);
            rdd_mul(tmp[22], in_mat_a[ik + 22], in_mat_b[mid_dim * 22 + j]);
            rdd_mul(tmp[23], in_mat_a[ik + 23], in_mat_b[mid_dim * 23 + j]);
            rdd_mul(tmp[24], in_mat_a[ik + 24], in_mat_b[mid_dim * 24 + j]);
            rdd_mul(tmp[25], in_mat_a[ik + 25], in_mat_b[mid_dim * 25 + j]);
            rdd_mul(tmp[26], in_mat_a[ik + 26], in_mat_b[mid_dim * 26 + j]);
            rdd_mul(tmp[27], in_mat_a[ik + 27], in_mat_b[mid_dim * 27 + j]);
            rdd_mul(tmp[28], in_mat_a[ik + 28], in_mat_b[mid_dim * 28 + j]);
            rdd_mul(tmp[29], in_mat_a[ik + 29], in_mat_b[mid_dim * 29 + j]);
            rdd_mul(tmp[30], in_mat_a[ik + 30], in_mat_b[mid_dim * 30 + j]);
            rdd_mul(tmp[31], in_mat_a[ik + 31], in_mat_b[mid_dim * 31 + j]);

            rdd_add(in_ret[ij], in_ret[ij], tmp[ 0]);
            rdd_add(in_ret[ij], in_ret[ij], tmp[ 1]);
            rdd_add(in_ret[ij], in_ret[ij], tmp[ 2]);
            rdd_add(in_ret[ij], in_ret[ij], tmp[ 3]);
            rdd_add(in_ret[ij], in_ret[ij], tmp[ 4]);
            rdd_add(in_ret[ij], in_ret[ij], tmp[ 5]);
            rdd_add(in_ret[ij], in_ret[ij], tmp[ 6]);
            rdd_add(in_ret[ij], in_ret[ij], tmp[ 7]);
            rdd_add(in_ret[ij], in_ret[ij], tmp[ 8]);
            rdd_add(in_ret[ij], in_ret[ij], tmp[ 9]);
            rdd_add(in_ret[ij], in_ret[ij], tmp[10]);
            rdd_add(in_ret[ij], in_ret[ij], tmp[11]);
            rdd_add(in_ret[ij], in_ret[ij], tmp[12]);
            rdd_add(in_ret[ij], in_ret[ij], tmp[13]);
            rdd_add(in_ret[ij], in_ret[ij], tmp[14]);
            rdd_add(in_ret[ij], in_ret[ij], tmp[15]);
            rdd_add(in_ret[ij], in_ret[ij], tmp[16]);
            rdd_add(in_ret[ij], in_ret[ij], tmp[17]);
            rdd_add(in_ret[ij], in_ret[ij], tmp[18]);
            rdd_add(in_ret[ij], in_ret[ij], tmp[19]);
            rdd_add(in_ret[ij], in_ret[ij], tmp[20]);
            rdd_add(in_ret[ij], in_ret[ij], tmp[21]);
            rdd_add(in_ret[ij], in_ret[ij], tmp[22]);
            rdd_add(in_ret[ij], in_ret[ij], tmp[23]);
            rdd_add(in_ret[ij], in_ret[ij], tmp[24]);
            rdd_add(in_ret[ij], in_ret[ij], tmp[25]);
            rdd_add(in_ret[ij], in_ret[ij], tmp[26]);
            rdd_add(in_ret[ij], in_ret[ij], tmp[27]);
            rdd_add(in_ret[ij], in_ret[ij], tmp[28]);
            rdd_add(in_ret[ij], in_ret[ij], tmp[29]);
            rdd_add(in_ret[ij], in_ret[ij], tmp[30]);
            rdd_add(in_ret[ij], in_ret[ij], tmp[31]);

        }
    }

    // ret := in_ret
    for(i = 0; i < row_dim; i++)
    {
        for(j = 0; j < col_dim; j++)
        {
            ij = i * mid_dim + j;
            ret->element[0][ij] = in_ret[ij][0]; 
            ret->element[1][ij] = in_ret[ij][1];
        }
    }
}
