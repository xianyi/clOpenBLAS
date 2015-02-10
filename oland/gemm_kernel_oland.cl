#pragma OPENCL EXTENSION cl_khr_fp64 : enable

typedef union GPtr {
    __global float *f;
    __global double *d;
    __global float2 *f2v;
    __global double2 *d2v;
    __global float4 *f4v;
    __global double4 *d4v;
    __global float8 *f8v;
    __global double8 *d8v;
    __global float16 *f16v;
    __global double16 *d16v;
} GPtr;

typedef union LPtr {
    __local float *f;
    __local double *d;
    __local float2 *f2v;
    __local double2 *d2v;
    __local float4 *f4v;
    __local double4 *d4v;
    __local float8 *f8v;
    __local double8 *d8v;
    __local float16 *f16v;
    __local double16 *d16v;
} LPtr;

typedef union PPtr {
    float *f;
    double *d;
    float2 *f2v;
    double2 *d2v;
    float4 *f4v;
    double4 *d4v;
    float8 *f8v;
    double8 *d8v;
    float16 *f16v;
    double16 *d16v;
} PPtr;


__attribute__((reqd_work_group_size(8, 8, 1)))
void __kernel
sgemm_kernel( const uint M, const uint N, const uint K, const float alpha, const __global float8 *restrict A, const __global float8 *restrict B, __global float8 *C )
{
    float8 a0, a1, a2, a3;
    float8 b0, b1, b2, b3;
    float8 c0, c1, c2, c3, c4, c5, c6, c7;

    uint4 coord = 0u; /* contains coordB, coordA, k */

    uint lda = M / 8;
    uint ldb = N / 8;
    uint ldc = M;

    A += (uint) get_global_id(0);

    uint get_group_id_1 = (get_group_id(0) + get_group_id(1))% get_num_groups(1);
    uint get_global_id_1 = get_group_id_1 * get_local_size(1) + get_local_id(1);

    uint kif = ( (N % 512) != 0);

    get_global_id_1 = (kif*(uint)get_global_id(1)) + ((1-kif)*get_global_id_1);

    B += get_global_id_1;

    coord.y = 8u * (uint)get_global_id(0);
    coord.x = 8u * (uint)get_global_id_1;

    c0 = 0;
    c1 = 0;
    c2 = 0;
    c3 = 0;
    c4 = 0;
    c5 = 0;
    c6 = 0;
    c7 = 0;

    for (uint k1 = 0; k1 < K; k1 += 4) {
        /* -- Tiles multiplier -- */
        b0 = B[0];
        b1 = B[ldb];
        b2 = B[(ldb << 1)];
        b3 = B[mad24(3u, ldb, 0u)];

        a0 = A[0];
        a1 = A[lda];
        a2 = A[(lda << 1)];
        a3 = A[mad24(3u, lda, 0u)];

        c0 = mad(a0, b0.s0, c0);
        c1 = mad(a0, b0.s1, c1);
        c2 = mad(a0, b0.s2, c2);
        c3 = mad(a0, b0.s3, c3);
        c4 = mad(a0, b0.s4, c4);
        c5 = mad(a0, b0.s5, c5);
        c6 = mad(a0, b0.s6, c6);
        c7 = mad(a0, b0.s7, c7);

        c0 = mad(a1, b1.s0, c0);
        c1 = mad(a1, b1.s1, c1);
        c2 = mad(a1, b1.s2, c2);
        c3 = mad(a1, b1.s3, c3);
        c4 = mad(a1, b1.s4, c4);
        c5 = mad(a1, b1.s5, c5);
        c6 = mad(a1, b1.s6, c6);
        c7 = mad(a1, b1.s7, c7);

        c0 = mad(a2, b2.s0, c0);
        c1 = mad(a2, b2.s1, c1);
        c2 = mad(a2, b2.s2, c2);
        c3 = mad(a2, b2.s3, c3);
        c4 = mad(a2, b2.s4, c4);
        c5 = mad(a2, b2.s5, c5);
        c6 = mad(a2, b2.s6, c6);
        c7 = mad(a2, b2.s7, c7);

        c0 = mad(a3, b3.s0, c0);
        c1 = mad(a3, b3.s1, c1);
        c2 = mad(a3, b3.s2, c2);
        c3 = mad(a3, b3.s3, c3);
        c4 = mad(a3, b3.s4, c4);
        c5 = mad(a3, b3.s5, c5);
        c6 = mad(a3, b3.s6, c6);
        c7 = mad(a3, b3.s7, c7);

        A += (lda << 2);
        B += (ldb << 2);
        /* ---------------------- */
    }


    GPtr uC;

    uC.f8v = C + (coord.x * ldc + coord.y)/8;

    uint ldc3 = ldc >> 3;

    __global float8 *pC = uC.f8v;

    pC[0] 				= mad(c0, alpha, 0);
    pC[ldc3] 				= mad(c1, alpha, 0);
    pC[(ldc >> 2)] 			= mad(c2, alpha, 0);
    pC[mad24(3u, ldc3, 0u)] 		= mad(c3, alpha, 0);
    pC[(ldc >> 1)] 			= mad(c4, alpha, 0);
    pC[mad24(5u, ldc3, 0u)] 		= mad(c5, alpha, 0);
    pC[mad24(6u, ldc3, 0u)] 		= mad(c6, alpha, 0);
    pC[mad24(7u, ldc3, 0u)] 		= mad(c7, alpha, 0);
}



__attribute__((reqd_work_group_size(8, 8, 1)))
void __kernel
 cgemm_kernel( uint M, uint N, uint K, const float2 alpha, const __global float8 *restrict A, const __global float8 *restrict B, __global float8 *C)
{
    float8 a0, a1, a2, a3;
    float8 b0, b1, b2, b3;
    float8 c0, c1, c2, c3;
    uint4 coord = 0u; /* contains coordB, coordA, k */

    uint lda = M/4;
    uint ldb = N/4;
    uint ldc = M;

    uint kif;
    uint get_group_id_1;
    uint get_global_id_1;

    A += (uint)get_global_id(0);
    get_group_id_1 = (get_group_id(0) + get_group_id(1))% get_num_groups(1);
    get_global_id_1 = get_group_id_1 * get_local_size(1) + get_local_id(1);

    kif = (N % 256 != 0);
    get_global_id_1 = (kif*(uint)get_global_id(1)) + ((1-kif)*get_global_id_1);

    B += get_global_id_1;
    coord.y = 4u * (uint)get_global_id(0);
    coord.x = 4u * (uint)get_global_id_1;

    c0 = 0;
    c1 = 0;
    c2 = 0;
    c3 = 0;

    for (uint k1 = 0; k1 < K; k1 += 4) {
        /* -- Tiles multiplier -- */
        b0 = B[0];
        b1 = B[ldb];
        b2 = B[(ldb << 1)];
        b3 = B[mad24(3u, ldb, 0u)];

        a0 = A[0];
        a1 = A[lda];
        a2 = A[(lda << 1)];
        a3 = A[mad24(3u, lda, 0u)];

        c0.s01 = mad(b0.s01, a0.s0, c0.s01);
        c0.s01 = mad(b0.s10, (float2)(-a0.s1, a0.s1), c0.s01);
        c1.s01 = mad(b0.s23, a0.s0, c1.s01);
        c1.s01 = mad(b0.s32, (float2)(-a0.s1, a0.s1), c1.s01);
        c2.s01 = mad(b0.s45, a0.s0, c2.s01);
        c2.s01 = mad(b0.s54, (float2)(-a0.s1, a0.s1), c2.s01);
        c3.s01 = mad(b0.s67, a0.s0, c3.s01);
        c3.s01 = mad(b0.s76, (float2)(-a0.s1, a0.s1), c3.s01);
        c0.s23 = mad(b0.s01, a0.s2, c0.s23);
        c0.s23 = mad(b0.s10, (float2)(-a0.s3, a0.s3), c0.s23);
        c1.s23 = mad(b0.s23, a0.s2, c1.s23);
        c1.s23 = mad(b0.s32, (float2)(-a0.s3, a0.s3), c1.s23);
        c2.s23 = mad(b0.s45, a0.s2, c2.s23);
        c2.s23 = mad(b0.s54, (float2)(-a0.s3, a0.s3), c2.s23);
        c3.s23 = mad(b0.s67, a0.s2, c3.s23);
        c3.s23 = mad(b0.s76, (float2)(-a0.s3, a0.s3), c3.s23);
        c0.s45 = mad(b0.s01, a0.s4, c0.s45);
        c0.s45 = mad(b0.s10, (float2)(-a0.s5, a0.s5), c0.s45);
        c1.s45 = mad(b0.s23, a0.s4, c1.s45);
        c1.s45 = mad(b0.s32, (float2)(-a0.s5, a0.s5), c1.s45);
        c2.s45 = mad(b0.s45, a0.s4, c2.s45);
        c2.s45 = mad(b0.s54, (float2)(-a0.s5, a0.s5), c2.s45);
        c3.s45 = mad(b0.s67, a0.s4, c3.s45);
        c3.s45 = mad(b0.s76, (float2)(-a0.s5, a0.s5), c3.s45);
        c0.s67 = mad(b0.s01, a0.s6, c0.s67);
        c0.s67 = mad(b0.s10, (float2)(-a0.s7, a0.s7), c0.s67);
        c1.s67 = mad(b0.s23, a0.s6, c1.s67);
        c1.s67 = mad(b0.s32, (float2)(-a0.s7, a0.s7), c1.s67);
        c2.s67 = mad(b0.s45, a0.s6, c2.s67);
        c2.s67 = mad(b0.s54, (float2)(-a0.s7, a0.s7), c2.s67);
        c3.s67 = mad(b0.s67, a0.s6, c3.s67);
        c3.s67 = mad(b0.s76, (float2)(-a0.s7, a0.s7), c3.s67);

        c0.s01 = mad(b1.s01, a1.s0, c0.s01);
        c0.s01 = mad(b1.s10, (float2)(-a1.s1, a1.s1), c0.s01);
        c1.s01 = mad(b1.s23, a1.s0, c1.s01);
        c1.s01 = mad(b1.s32, (float2)(-a1.s1, a1.s1), c1.s01);
        c2.s01 = mad(b1.s45, a1.s0, c2.s01);
        c2.s01 = mad(b1.s54, (float2)(-a1.s1, a1.s1), c2.s01);
        c3.s01 = mad(b1.s67, a1.s0, c3.s01);
        c3.s01 = mad(b1.s76, (float2)(-a1.s1, a1.s1), c3.s01);
        c0.s23 = mad(b1.s01, a1.s2, c0.s23);
        c0.s23 = mad(b1.s10, (float2)(-a1.s3, a1.s3), c0.s23);
        c1.s23 = mad(b1.s23, a1.s2, c1.s23);
        c1.s23 = mad(b1.s32, (float2)(-a1.s3, a1.s3), c1.s23);
        c2.s23 = mad(b1.s45, a1.s2, c2.s23);
        c2.s23 = mad(b1.s54, (float2)(-a1.s3, a1.s3), c2.s23);
        c3.s23 = mad(b1.s67, a1.s2, c3.s23);
        c3.s23 = mad(b1.s76, (float2)(-a1.s3, a1.s3), c3.s23);
        c0.s45 = mad(b1.s01, a1.s4, c0.s45);
        c0.s45 = mad(b1.s10, (float2)(-a1.s5, a1.s5), c0.s45);
        c1.s45 = mad(b1.s23, a1.s4, c1.s45);
        c1.s45 = mad(b1.s32, (float2)(-a1.s5, a1.s5), c1.s45);
        c2.s45 = mad(b1.s45, a1.s4, c2.s45);
        c2.s45 = mad(b1.s54, (float2)(-a1.s5, a1.s5), c2.s45);
        c3.s45 = mad(b1.s67, a1.s4, c3.s45);
        c3.s45 = mad(b1.s76, (float2)(-a1.s5, a1.s5), c3.s45);
        c0.s67 = mad(b1.s01, a1.s6, c0.s67);
        c0.s67 = mad(b1.s10, (float2)(-a1.s7, a1.s7), c0.s67);
        c1.s67 = mad(b1.s23, a1.s6, c1.s67);
        c1.s67 = mad(b1.s32, (float2)(-a1.s7, a1.s7), c1.s67);
        c2.s67 = mad(b1.s45, a1.s6, c2.s67);
        c2.s67 = mad(b1.s54, (float2)(-a1.s7, a1.s7), c2.s67);
        c3.s67 = mad(b1.s67, a1.s6, c3.s67);
        c3.s67 = mad(b1.s76, (float2)(-a1.s7, a1.s7), c3.s67);

        c0.s01 = mad(b2.s01, a2.s0, c0.s01);
        c0.s01 = mad(b2.s10, (float2)(-a2.s1, a2.s1), c0.s01);
        c1.s01 = mad(b2.s23, a2.s0, c1.s01);
        c1.s01 = mad(b2.s32, (float2)(-a2.s1, a2.s1), c1.s01);
        c2.s01 = mad(b2.s45, a2.s0, c2.s01);
        c2.s01 = mad(b2.s54, (float2)(-a2.s1, a2.s1), c2.s01);
        c3.s01 = mad(b2.s67, a2.s0, c3.s01);
        c3.s01 = mad(b2.s76, (float2)(-a2.s1, a2.s1), c3.s01);
        c0.s23 = mad(b2.s01, a2.s2, c0.s23);
        c0.s23 = mad(b2.s10, (float2)(-a2.s3, a2.s3), c0.s23);
        c1.s23 = mad(b2.s23, a2.s2, c1.s23);
        c1.s23 = mad(b2.s32, (float2)(-a2.s3, a2.s3), c1.s23);
        c2.s23 = mad(b2.s45, a2.s2, c2.s23);
        c2.s23 = mad(b2.s54, (float2)(-a2.s3, a2.s3), c2.s23);
        c3.s23 = mad(b2.s67, a2.s2, c3.s23);
        c3.s23 = mad(b2.s76, (float2)(-a2.s3, a2.s3), c3.s23);
        c0.s45 = mad(b2.s01, a2.s4, c0.s45);
        c0.s45 = mad(b2.s10, (float2)(-a2.s5, a2.s5), c0.s45);
        c1.s45 = mad(b2.s23, a2.s4, c1.s45);
        c1.s45 = mad(b2.s32, (float2)(-a2.s5, a2.s5), c1.s45);
        c2.s45 = mad(b2.s45, a2.s4, c2.s45);
        c2.s45 = mad(b2.s54, (float2)(-a2.s5, a2.s5), c2.s45);
        c3.s45 = mad(b2.s67, a2.s4, c3.s45);
        c3.s45 = mad(b2.s76, (float2)(-a2.s5, a2.s5), c3.s45);
        c0.s67 = mad(b2.s01, a2.s6, c0.s67);
        c0.s67 = mad(b2.s10, (float2)(-a2.s7, a2.s7), c0.s67);
        c1.s67 = mad(b2.s23, a2.s6, c1.s67);
        c1.s67 = mad(b2.s32, (float2)(-a2.s7, a2.s7), c1.s67);
        c2.s67 = mad(b2.s45, a2.s6, c2.s67);
        c2.s67 = mad(b2.s54, (float2)(-a2.s7, a2.s7), c2.s67);
        c3.s67 = mad(b2.s67, a2.s6, c3.s67);
        c3.s67 = mad(b2.s76, (float2)(-a2.s7, a2.s7), c3.s67);

        c0.s01 = mad(b3.s01, a3.s0, c0.s01);
        c0.s01 = mad(b3.s10, (float2)(-a3.s1, a3.s1), c0.s01);
        c1.s01 = mad(b3.s23, a3.s0, c1.s01);
        c1.s01 = mad(b3.s32, (float2)(-a3.s1, a3.s1), c1.s01);
        c2.s01 = mad(b3.s45, a3.s0, c2.s01);
        c2.s01 = mad(b3.s54, (float2)(-a3.s1, a3.s1), c2.s01);
        c3.s01 = mad(b3.s67, a3.s0, c3.s01);
        c3.s01 = mad(b3.s76, (float2)(-a3.s1, a3.s1), c3.s01);
        c0.s23 = mad(b3.s01, a3.s2, c0.s23);
        c0.s23 = mad(b3.s10, (float2)(-a3.s3, a3.s3), c0.s23);
        c1.s23 = mad(b3.s23, a3.s2, c1.s23);
        c1.s23 = mad(b3.s32, (float2)(-a3.s3, a3.s3), c1.s23);
        c2.s23 = mad(b3.s45, a3.s2, c2.s23);
        c2.s23 = mad(b3.s54, (float2)(-a3.s3, a3.s3), c2.s23);
        c3.s23 = mad(b3.s67, a3.s2, c3.s23);
        c3.s23 = mad(b3.s76, (float2)(-a3.s3, a3.s3), c3.s23);
        c0.s45 = mad(b3.s01, a3.s4, c0.s45);
        c0.s45 = mad(b3.s10, (float2)(-a3.s5, a3.s5), c0.s45);
        c1.s45 = mad(b3.s23, a3.s4, c1.s45);
        c1.s45 = mad(b3.s32, (float2)(-a3.s5, a3.s5), c1.s45);
        c2.s45 = mad(b3.s45, a3.s4, c2.s45);
        c2.s45 = mad(b3.s54, (float2)(-a3.s5, a3.s5), c2.s45);
        c3.s45 = mad(b3.s67, a3.s4, c3.s45);
        c3.s45 = mad(b3.s76, (float2)(-a3.s5, a3.s5), c3.s45);
        c0.s67 = mad(b3.s01, a3.s6, c0.s67);
        c0.s67 = mad(b3.s10, (float2)(-a3.s7, a3.s7), c0.s67);
        c1.s67 = mad(b3.s23, a3.s6, c1.s67);
        c1.s67 = mad(b3.s32, (float2)(-a3.s7, a3.s7), c1.s67);
        c2.s67 = mad(b3.s45, a3.s6, c2.s67);
        c2.s67 = mad(b3.s54, (float2)(-a3.s7, a3.s7), c2.s67);
        c3.s67 = mad(b3.s67, a3.s6, c3.s67);
        c3.s67 = mad(b3.s76, (float2)(-a3.s7, a3.s7), c3.s67);

        A += (lda << 2);
        B += (ldb << 2);
        /* ---------------------- */
    }


    GPtr uC;

    uC.f2v = C + (coord.x * ldc + coord.y)/4;

    __global float8 *pC = uC.f8v;

    float8 tempC0, tempC1, tempC2, tempC3;

    tempC0.s01 = alpha * c0.s0 + alpha.s10 * (float2)(-c0.s1, c0.s1);
    tempC0.s23 = alpha * c0.s2 + alpha.s10 * (float2)(-c0.s3, c0.s3);
    tempC0.s45 = alpha * c0.s4 + alpha.s10 * (float2)(-c0.s5, c0.s5);
    tempC0.s67 = alpha * c0.s6 + alpha.s10 * (float2)(-c0.s7, c0.s7);
    tempC1.s01 = alpha * c1.s0 + alpha.s10 * (float2)(-c1.s1, c1.s1);
    tempC1.s23 = alpha * c1.s2 + alpha.s10 * (float2)(-c1.s3, c1.s3);
    tempC1.s45 = alpha * c1.s4 + alpha.s10 * (float2)(-c1.s5, c1.s5);
    tempC1.s67 = alpha * c1.s6 + alpha.s10 * (float2)(-c1.s7, c1.s7);
    tempC2.s01 = alpha * c2.s0 + alpha.s10 * (float2)(-c2.s1, c2.s1);
    tempC2.s23 = alpha * c2.s2 + alpha.s10 * (float2)(-c2.s3, c2.s3);
    tempC2.s45 = alpha * c2.s4 + alpha.s10 * (float2)(-c2.s5, c2.s5);
    tempC2.s67 = alpha * c2.s6 + alpha.s10 * (float2)(-c2.s7, c2.s7);
    tempC3.s01 = alpha * c3.s0 + alpha.s10 * (float2)(-c3.s1, c3.s1);
    tempC3.s23 = alpha * c3.s2 + alpha.s10 * (float2)(-c3.s3, c3.s3);
    tempC3.s45 = alpha * c3.s4 + alpha.s10 * (float2)(-c3.s5, c3.s5);
    tempC3.s67 = alpha * c3.s6 + alpha.s10 * (float2)(-c3.s7, c3.s7);

    pC[0] = tempC0;
    pC[(ldc >> 2)] = tempC1;
    pC[(ldc >> 1)] = tempC2;
    pC[mad24(3u, (ldc >> 2), 0u)] = tempC3;
}







__attribute__((reqd_work_group_size(8, 8, 1)))
void __kernel
dgemm_kernel( const uint M, const uint N, const uint K, const double alpha, const __global double4 *restrict A, const __global double4 *restrict B, __global double4 *C )
{
    double4 a0, a1, a2, a3;
    double4 b0, b1, b2, b3;
    double4 c0, c1, c2, c3;

    uint4 coord = 0u; /* contains coordB, coordA, k */

    uint lda = M / 4;
    uint ldb = N / 4;
    uint ldc = M;

    A += (uint) get_global_id(0);

    uint get_group_id_1 = (get_group_id(0) + get_group_id(1))% get_num_groups(1);
    uint get_global_id_1 = get_group_id_1 * get_local_size(1) + get_local_id(1);

    uint kif = ( (N % 256) != 0);

    get_global_id_1 = (kif*(uint)get_global_id(1)) + ((1-kif)*get_global_id_1);

    B += get_global_id_1;

    coord.y = 4u * (uint)get_global_id(0);
    coord.x = 4u * (uint)get_global_id_1;

    c0 = 0;
    c1 = 0;
    c2 = 0;
    c3 = 0;

    for (uint k1 = 0; k1 < K; k1 += 4) {
        /* -- Tiles multiplier -- */
        b0 = B[0];
        b1 = B[ldb];
        b2 = B[(ldb << 1)];
        b3 = B[mad24(3u, ldb, 0u)];

        a0 = A[0];
        a1 = A[lda];
        a2 = A[(lda << 1)];
        a3 = A[mad24(3u, lda, 0u)];

        c0 = mad(a0, b0.s0, c0);
        c1 = mad(a0, b0.s1, c1);
        c2 = mad(a0, b0.s2, c2);
        c3 = mad(a0, b0.s3, c3);

        c0 = mad(a1, b1.s0, c0);
        c1 = mad(a1, b1.s1, c1);
        c2 = mad(a1, b1.s2, c2);
        c3 = mad(a1, b1.s3, c3);

        c0 = mad(a2, b2.s0, c0);
        c1 = mad(a2, b2.s1, c1);
        c2 = mad(a2, b2.s2, c2);
        c3 = mad(a2, b2.s3, c3);

        c0 = mad(a3, b3.s0, c0);
        c1 = mad(a3, b3.s1, c1);
        c2 = mad(a3, b3.s2, c2);
        c3 = mad(a3, b3.s3, c3);

        A += (lda << 2);
        B += (ldb << 2);
        /* ---------------------- */
    }


    GPtr uC;

    uC.d4v = C + (coord.x * ldc + coord.y)/4;

    __global double4 *pC = uC.d4v;

    pC[0] 				= mad(c0, alpha, 0);
    pC[(ldc >> 2)] 			= mad(c1, alpha, 0);
    pC[(ldc >> 1)] 			= mad(c2, alpha, 0);
    pC[mad24(3u,(ldc >> 2), 0u)] 	= mad(c3, alpha, 0);

}


