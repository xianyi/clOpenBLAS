
 __attribute__((reqd_work_group_size(16,16,1)))
__kernel void sgemm_kernel(unsigned int M, unsigned int N, unsigned int K, float ALPHA, __global float* A, __global float* B, __global float* C)
{
    float rC[8][4] = {{(float)0}};
    float rA[8];
    float rB[4];

    __local float lA[2064];
    __local float lB[1040];

    uint gidx = get_group_id(0);
    uint gidy = get_group_id(1);
    uint idx = get_local_id(0);
    uint idy = get_local_id(1);
    
    uint idt = 16*idy + idx;
    uint idxT = idt % 32;
    uint idyT = idt / 32;
    
    A += (gidx*128 + idxT) + idyT*M;
    B += (gidy*64 + idxT) + idyT*N;
    
    for(unsigned int block_k=0; block_k < K; block_k+=16)
    {
        __local float* plA = lA + idyT*129 + idxT;
        __local float* plB = lB + idyT*65  + idxT;

        barrier(CLK_LOCAL_MEM_FENCE);

        (plA + 0)[0]    = A[0*M + 0];
        (plA + 32)[0]   = A[0*M + 32];
        (plA + 64)[0]   = A[0*M + 64];
        (plA + 96)[0]   = A[0*M + 96];
        (plA + 1032)[0] = A[8*M + 0];
        (plA + 1064)[0] = A[8*M + 32];
        (plA + 1096)[0] = A[8*M + 64];
        (plA + 1128)[0] = A[8*M + 96];
        (plB + 0)[0]    = B[0*N + 0];
        (plB + 32)[0]   = B[0*N + 32];
        (plB + 520)[0]  = B[8*N + 0];
        (plB + 552)[0]  = B[8*N + 32];

        barrier(CLK_LOCAL_MEM_FENCE);

        uint offA = 1*idx;
        uint offB = 1*idy;
        for(unsigned int k = 0; k < 16; k+=1)
	{

            #pragma unroll 8
            for(unsigned int mm = 0; mm < 8; mm++)
            {
                rA[mm*1+0] = lA[offA + mm*16+0+ 0*129];
            }

            #pragma unroll 4
            for(unsigned int nn = 0; nn < 4; nn++)
            {
                rB[nn*1+0] = lB[offB + nn*16+0+ 0*65];
            }
            offA += 129;
            offB += 65;

            rC[0][0]=fma(rA[0],rB[0],rC[0][0]);
            rC[1][0]=fma(rA[1],rB[0],rC[1][0]);
            rC[2][0]=fma(rA[2],rB[0],rC[2][0]);
            rC[3][0]=fma(rA[3],rB[0],rC[3][0]);
            rC[4][0]=fma(rA[4],rB[0],rC[4][0]);
            rC[5][0]=fma(rA[5],rB[0],rC[5][0]);
            rC[6][0]=fma(rA[6],rB[0],rC[6][0]);
            rC[7][0]=fma(rA[7],rB[0],rC[7][0]);
            rC[0][1]=fma(rA[0],rB[1],rC[0][1]);
            rC[1][1]=fma(rA[1],rB[1],rC[1][1]);
            rC[2][1]=fma(rA[2],rB[1],rC[2][1]);
            rC[3][1]=fma(rA[3],rB[1],rC[3][1]);
            rC[4][1]=fma(rA[4],rB[1],rC[4][1]);
            rC[5][1]=fma(rA[5],rB[1],rC[5][1]);
            rC[6][1]=fma(rA[6],rB[1],rC[6][1]);
            rC[7][1]=fma(rA[7],rB[1],rC[7][1]);
            rC[0][2]=fma(rA[0],rB[2],rC[0][2]);
            rC[1][2]=fma(rA[1],rB[2],rC[1][2]);
            rC[2][2]=fma(rA[2],rB[2],rC[2][2]);
            rC[3][2]=fma(rA[3],rB[2],rC[3][2]);
            rC[4][2]=fma(rA[4],rB[2],rC[4][2]);
            rC[5][2]=fma(rA[5],rB[2],rC[5][2]);
            rC[6][2]=fma(rA[6],rB[2],rC[6][2]);
            rC[7][2]=fma(rA[7],rB[2],rC[7][2]);
            rC[0][3]=fma(rA[0],rB[3],rC[0][3]);
            rC[1][3]=fma(rA[1],rB[3],rC[1][3]);
            rC[2][3]=fma(rA[2],rB[3],rC[2][3]);
            rC[3][3]=fma(rA[3],rB[3],rC[3][3]);
            rC[4][3]=fma(rA[4],rB[3],rC[4][3]);
            rC[5][3]=fma(rA[5],rB[3],rC[5][3]);
            rC[6][3]=fma(rA[6],rB[3],rC[6][3]);
            rC[7][3]=fma(rA[7],rB[3],rC[7][3]);

        }
        A += 16*M;
        B += 16*N;
    }
    C += gidx*128;
    C += idx;
    C += gidy*64*M;
    C += idy*M;

    C[0*M]  = rC[0][0]*ALPHA ;
    C[16*M] = rC[0][1]*ALPHA ;
    C[32*M] = rC[0][2]*ALPHA ;
    C[48*M] = rC[0][3]*ALPHA ;
    C += 16;
    C[0*M]  = rC[1][0]*ALPHA ;
    C[16*M] = rC[1][1]*ALPHA ;
    C[32*M] = rC[1][2]*ALPHA ;
    C[48*M] = rC[1][3]*ALPHA ;
    C += 16;
    C[0*M]  = rC[2][0]*ALPHA ;
    C[16*M] = rC[2][1]*ALPHA ;
    C[32*M] = rC[2][2]*ALPHA ;
    C[48*M] = rC[2][3]*ALPHA ;
    C += 16;
    C[0*M]  = rC[3][0]*ALPHA ;
    C[16*M] = rC[3][1]*ALPHA ;
    C[32*M] = rC[3][2]*ALPHA ;
    C[48*M] = rC[3][3]*ALPHA ;
    C += 16;
    C[0*M]  = rC[4][0]*ALPHA ;
    C[16*M] = rC[4][1]*ALPHA ;
    C[32*M] = rC[4][2]*ALPHA ;
    C[48*M] = rC[4][3]*ALPHA ;
    C += 16;
    C[0*M]  = rC[5][0]*ALPHA ;
    C[16*M] = rC[5][1]*ALPHA ;
    C[32*M] = rC[5][2]*ALPHA ;
    C[48*M] = rC[5][3]*ALPHA ;
    C += 16;
    C[0*M]  = rC[6][0]*ALPHA ;
    C[16*M] = rC[6][1]*ALPHA ;
    C[32*M] = rC[6][2]*ALPHA ;
    C[48*M] = rC[6][3]*ALPHA ;
    C += 16;
    C[0*M]  = rC[7][0]*ALPHA ;
    C[16*M] = rC[7][1]*ALPHA ;
    C[32*M] = rC[7][2]*ALPHA ;
    C[48*M] = rC[7][3]*ALPHA ;
    C += 16;
}

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

 __attribute__((reqd_work_group_size(16,16,1)))
__kernel void dgemm_kernel(unsigned int M, unsigned int N, unsigned int K, double ALPHA, __global double* A, __global double* B, __global double* C)
{
    double rC[8][4] = {{(double)0}};
    double rA[8];
    double rB[4];

    __local double lA[2064];
    __local double lB[1040];

    uint gidx = get_group_id(0);
    uint gidy = get_group_id(1);
    uint idx = get_local_id(0);
    uint idy = get_local_id(1);
    
    uint idt = 16*idy + idx;
    uint idxT = idt % 32;
    uint idyT = idt / 32;
    
    A += (gidx*128 + idxT) + idyT*M;
    B += (gidy*64 + idxT) + idyT*N;
    
    for(unsigned int block_k=0; block_k < K; block_k+=16)
    {
        __local double* plA = lA + idyT*129 + idxT;
        __local double* plB = lB + idyT*65  + idxT;

        barrier(CLK_LOCAL_MEM_FENCE);

        (plA + 0)[0]    = A[0*M + 0];
        (plA + 32)[0]   = A[0*M + 32];
        (plA + 64)[0]   = A[0*M + 64];
        (plA + 96)[0]   = A[0*M + 96];
        (plA + 1032)[0] = A[8*M + 0];
        (plA + 1064)[0] = A[8*M + 32];
        (plA + 1096)[0] = A[8*M + 64];
        (plA + 1128)[0] = A[8*M + 96];
        (plB + 0)[0]    = B[0*N + 0];
        (plB + 32)[0]   = B[0*N + 32];
        (plB + 520)[0]  = B[8*N + 0];
        (plB + 552)[0]  = B[8*N + 32];

        barrier(CLK_LOCAL_MEM_FENCE);

        uint offA = 1*idx;
        uint offB = 1*idy;
        for(unsigned int k = 0; k < 16; k+=1)
	{

            #pragma unroll 8
            for(unsigned int mm = 0; mm < 8; mm++)
            {
                rA[mm*1+0] = lA[offA + mm*16+0+ 0*129];
            }

            #pragma unroll 4
            for(unsigned int nn = 0; nn < 4; nn++)
            {
                rB[nn*1+0] = lB[offB + nn*16+0+ 0*65];
            }
            offA += 129;
            offB += 65;

            rC[0][0]=fma(rA[0],rB[0],rC[0][0]);
            rC[1][0]=fma(rA[1],rB[0],rC[1][0]);
            rC[2][0]=fma(rA[2],rB[0],rC[2][0]);
            rC[3][0]=fma(rA[3],rB[0],rC[3][0]);
            rC[4][0]=fma(rA[4],rB[0],rC[4][0]);
            rC[5][0]=fma(rA[5],rB[0],rC[5][0]);
            rC[6][0]=fma(rA[6],rB[0],rC[6][0]);
            rC[7][0]=fma(rA[7],rB[0],rC[7][0]);
            rC[0][1]=fma(rA[0],rB[1],rC[0][1]);
            rC[1][1]=fma(rA[1],rB[1],rC[1][1]);
            rC[2][1]=fma(rA[2],rB[1],rC[2][1]);
            rC[3][1]=fma(rA[3],rB[1],rC[3][1]);
            rC[4][1]=fma(rA[4],rB[1],rC[4][1]);
            rC[5][1]=fma(rA[5],rB[1],rC[5][1]);
            rC[6][1]=fma(rA[6],rB[1],rC[6][1]);
            rC[7][1]=fma(rA[7],rB[1],rC[7][1]);
            rC[0][2]=fma(rA[0],rB[2],rC[0][2]);
            rC[1][2]=fma(rA[1],rB[2],rC[1][2]);
            rC[2][2]=fma(rA[2],rB[2],rC[2][2]);
            rC[3][2]=fma(rA[3],rB[2],rC[3][2]);
            rC[4][2]=fma(rA[4],rB[2],rC[4][2]);
            rC[5][2]=fma(rA[5],rB[2],rC[5][2]);
            rC[6][2]=fma(rA[6],rB[2],rC[6][2]);
            rC[7][2]=fma(rA[7],rB[2],rC[7][2]);
            rC[0][3]=fma(rA[0],rB[3],rC[0][3]);
            rC[1][3]=fma(rA[1],rB[3],rC[1][3]);
            rC[2][3]=fma(rA[2],rB[3],rC[2][3]);
            rC[3][3]=fma(rA[3],rB[3],rC[3][3]);
            rC[4][3]=fma(rA[4],rB[3],rC[4][3]);
            rC[5][3]=fma(rA[5],rB[3],rC[5][3]);
            rC[6][3]=fma(rA[6],rB[3],rC[6][3]);
            rC[7][3]=fma(rA[7],rB[3],rC[7][3]);

        }
        A += 16*M;
        B += 16*N;
    }
    C += gidx*128;
    C += idx;
    C += gidy*64*M;
    C += idy*M;

    C[0*M]  = rC[0][0]*ALPHA ;
    C[16*M] = rC[0][1]*ALPHA ;
    C[32*M] = rC[0][2]*ALPHA ;
    C[48*M] = rC[0][3]*ALPHA ;
    C += 16;
    C[0*M]  = rC[1][0]*ALPHA ;
    C[16*M] = rC[1][1]*ALPHA ;
    C[32*M] = rC[1][2]*ALPHA ;
    C[48*M] = rC[1][3]*ALPHA ;
    C += 16;
    C[0*M]  = rC[2][0]*ALPHA ;
    C[16*M] = rC[2][1]*ALPHA ;
    C[32*M] = rC[2][2]*ALPHA ;
    C[48*M] = rC[2][3]*ALPHA ;
    C += 16;
    C[0*M]  = rC[3][0]*ALPHA ;
    C[16*M] = rC[3][1]*ALPHA ;
    C[32*M] = rC[3][2]*ALPHA ;
    C[48*M] = rC[3][3]*ALPHA ;
    C += 16;
    C[0*M]  = rC[4][0]*ALPHA ;
    C[16*M] = rC[4][1]*ALPHA ;
    C[32*M] = rC[4][2]*ALPHA ;
    C[48*M] = rC[4][3]*ALPHA ;
    C += 16;
    C[0*M]  = rC[5][0]*ALPHA ;
    C[16*M] = rC[5][1]*ALPHA ;
    C[32*M] = rC[5][2]*ALPHA ;
    C[48*M] = rC[5][3]*ALPHA ;
    C += 16;
    C[0*M]  = rC[6][0]*ALPHA ;
    C[16*M] = rC[6][1]*ALPHA ;
    C[32*M] = rC[6][2]*ALPHA ;
    C[48*M] = rC[6][3]*ALPHA ;
    C += 16;
    C[0*M]  = rC[7][0]*ALPHA ;
    C[16*M] = rC[7][1]*ALPHA ;
    C[32*M] = rC[7][2]*ALPHA ;
    C[48*M] = rC[7][3]*ALPHA ;
    C += 16;
}


 __attribute__((reqd_work_group_size(16,16,1)))
__kernel void cgemm_kernel(unsigned int M, unsigned int N, unsigned int K, float2 ALPHA, __global float2* A, __global float2* B, __global float2* C)
{
    float2 rC[8][4] = {{0}};
    float2 rA[8];
    float2 rB[4];

    __local float2 lA[2064];
    __local float2 lB[1040];

    uint gidx = get_group_id(0);
    uint gidy = get_group_id(1);
    uint idx = get_local_id(0);
    uint idy = get_local_id(1);
    
    uint idt = 16*idy + idx;
    uint idxT = idt % 32;
    uint idyT = idt / 32;
    
    A += (gidx*128 + idxT) + idyT*M;
    B += (gidy*64 + idxT) + idyT*N;
    
    for(unsigned int block_k=0; block_k < K; block_k+=16)
    {
        __local float2* plA = lA + idyT*129 + idxT;
        __local float2* plB = lB + idyT*65  + idxT;

        barrier(CLK_LOCAL_MEM_FENCE);

        (plA + 0)[0]    = A[0*M + 0];
        (plA + 32)[0]   = A[0*M + 32];
        (plA + 64)[0]   = A[0*M + 64];
        (plA + 96)[0]   = A[0*M + 96];
        (plA + 1032)[0] = A[8*M + 0];
        (plA + 1064)[0] = A[8*M + 32];
        (plA + 1096)[0] = A[8*M + 64];
        (plA + 1128)[0] = A[8*M + 96];
        (plB + 0)[0]    = B[0*N + 0];
        (plB + 32)[0]   = B[0*N + 32];
        (plB + 520)[0]  = B[8*N + 0];
        (plB + 552)[0]  = B[8*N + 32];

        barrier(CLK_LOCAL_MEM_FENCE);

        uint offA = 1*idx;
        uint offB = 1*idy;
        for(unsigned int k = 0; k < 16; k+=1)
	{

            #pragma unroll 8
            for(unsigned int mm = 0; mm < 8; mm++)
            {
                rA[mm*1+0] = lA[offA + mm*16+0+ 0*129];
            }

            #pragma unroll 4
            for(unsigned int nn = 0; nn < 4; nn++)
            {
                rB[nn*1+0] = lB[offB + nn*16+0+ 0*65];
            }
            offA += 129;
            offB += 65;

            rC[0][0].s0 = fma(rA[0].s0,rB[0].s0,rC[0][0].s0);
            rC[1][0].s0 = fma(rA[1].s0,rB[0].s0,rC[1][0].s0);
            rC[2][0].s0 = fma(rA[2].s0,rB[0].s0,rC[2][0].s0);
            rC[3][0].s0 = fma(rA[3].s0,rB[0].s0,rC[3][0].s0);
            rC[4][0].s0 = fma(rA[4].s0,rB[0].s0,rC[4][0].s0);
            rC[5][0].s0 = fma(rA[5].s0,rB[0].s0,rC[5][0].s0);
            rC[6][0].s0 = fma(rA[6].s0,rB[0].s0,rC[6][0].s0);
            rC[7][0].s0 = fma(rA[7].s0,rB[0].s0,rC[7][0].s0);
            rC[0][1].s0 = fma(rA[0].s0,rB[1].s0,rC[0][1].s0);
            rC[1][1].s0 = fma(rA[1].s0,rB[1].s0,rC[1][1].s0);
            rC[2][1].s0 = fma(rA[2].s0,rB[1].s0,rC[2][1].s0);
            rC[3][1].s0 = fma(rA[3].s0,rB[1].s0,rC[3][1].s0);
            rC[4][1].s0 = fma(rA[4].s0,rB[1].s0,rC[4][1].s0);
            rC[5][1].s0 = fma(rA[5].s0,rB[1].s0,rC[5][1].s0);
            rC[6][1].s0 = fma(rA[6].s0,rB[1].s0,rC[6][1].s0);
            rC[7][1].s0 = fma(rA[7].s0,rB[1].s0,rC[7][1].s0);
            rC[0][2].s0 = fma(rA[0].s0,rB[2].s0,rC[0][2].s0);
            rC[1][2].s0 = fma(rA[1].s0,rB[2].s0,rC[1][2].s0);
            rC[2][2].s0 = fma(rA[2].s0,rB[2].s0,rC[2][2].s0);
            rC[3][2].s0 = fma(rA[3].s0,rB[2].s0,rC[3][2].s0);
            rC[4][2].s0 = fma(rA[4].s0,rB[2].s0,rC[4][2].s0);
            rC[5][2].s0 = fma(rA[5].s0,rB[2].s0,rC[5][2].s0);
            rC[6][2].s0 = fma(rA[6].s0,rB[2].s0,rC[6][2].s0);
            rC[7][2].s0 = fma(rA[7].s0,rB[2].s0,rC[7][2].s0);
            rC[0][3].s0 = fma(rA[0].s0,rB[3].s0,rC[0][3].s0);
            rC[1][3].s0 = fma(rA[1].s0,rB[3].s0,rC[1][3].s0);
            rC[2][3].s0 = fma(rA[2].s0,rB[3].s0,rC[2][3].s0);
            rC[3][3].s0 = fma(rA[3].s0,rB[3].s0,rC[3][3].s0);
            rC[4][3].s0 = fma(rA[4].s0,rB[3].s0,rC[4][3].s0);
            rC[5][3].s0 = fma(rA[5].s0,rB[3].s0,rC[5][3].s0);
            rC[6][3].s0 = fma(rA[6].s0,rB[3].s0,rC[6][3].s0);
            rC[7][3].s0 = fma(rA[7].s0,rB[3].s0,rC[7][3].s0);

            rC[0][0].s0 = fma(-rA[0].s1,rB[0].s1,rC[0][0].s0);
            rC[1][0].s0 = fma(-rA[1].s1,rB[0].s1,rC[1][0].s0);
            rC[2][0].s0 = fma(-rA[2].s1,rB[0].s1,rC[2][0].s0);
            rC[3][0].s0 = fma(-rA[3].s1,rB[0].s1,rC[3][0].s0);
            rC[4][0].s0 = fma(-rA[4].s1,rB[0].s1,rC[4][0].s0);
            rC[5][0].s0 = fma(-rA[5].s1,rB[0].s1,rC[5][0].s0);
            rC[6][0].s0 = fma(-rA[6].s1,rB[0].s1,rC[6][0].s0);
            rC[7][0].s0 = fma(-rA[7].s1,rB[0].s1,rC[7][0].s0);
            rC[0][1].s0 = fma(-rA[0].s1,rB[1].s1,rC[0][1].s0);
            rC[1][1].s0 = fma(-rA[1].s1,rB[1].s1,rC[1][1].s0);
            rC[2][1].s0 = fma(-rA[2].s1,rB[1].s1,rC[2][1].s0);
            rC[3][1].s0 = fma(-rA[3].s1,rB[1].s1,rC[3][1].s0);
            rC[4][1].s0 = fma(-rA[4].s1,rB[1].s1,rC[4][1].s0);
            rC[5][1].s0 = fma(-rA[5].s1,rB[1].s1,rC[5][1].s0);
            rC[6][1].s0 = fma(-rA[6].s1,rB[1].s1,rC[6][1].s0);
            rC[7][1].s0 = fma(-rA[7].s1,rB[1].s1,rC[7][1].s0);
            rC[0][2].s0 = fma(-rA[0].s1,rB[2].s1,rC[0][2].s0);
            rC[1][2].s0 = fma(-rA[1].s1,rB[2].s1,rC[1][2].s0);
            rC[2][2].s0 = fma(-rA[2].s1,rB[2].s1,rC[2][2].s0);
            rC[3][2].s0 = fma(-rA[3].s1,rB[2].s1,rC[3][2].s0);
            rC[4][2].s0 = fma(-rA[4].s1,rB[2].s1,rC[4][2].s0);
            rC[5][2].s0 = fma(-rA[5].s1,rB[2].s1,rC[5][2].s0);
            rC[6][2].s0 = fma(-rA[6].s1,rB[2].s1,rC[6][2].s0);
            rC[7][2].s0 = fma(-rA[7].s1,rB[2].s1,rC[7][2].s0);
            rC[0][3].s0 = fma(-rA[0].s1,rB[3].s1,rC[0][3].s0);
            rC[1][3].s0 = fma(-rA[1].s1,rB[3].s1,rC[1][3].s0);
            rC[2][3].s0 = fma(-rA[2].s1,rB[3].s1,rC[2][3].s0);
            rC[3][3].s0 = fma(-rA[3].s1,rB[3].s1,rC[3][3].s0);
            rC[4][3].s0 = fma(-rA[4].s1,rB[3].s1,rC[4][3].s0);
            rC[5][3].s0 = fma(-rA[5].s1,rB[3].s1,rC[5][3].s0);
            rC[6][3].s0 = fma(-rA[6].s1,rB[3].s1,rC[6][3].s0);
            rC[7][3].s0 = fma(-rA[7].s1,rB[3].s1,rC[7][3].s0);




            rC[0][0].s1 = fma(rA[0].s1,rB[0].s0,rC[0][0].s1);
            rC[1][0].s1 = fma(rA[1].s1,rB[0].s0,rC[1][0].s1);
            rC[2][0].s1 = fma(rA[2].s1,rB[0].s0,rC[2][0].s1);
            rC[3][0].s1 = fma(rA[3].s1,rB[0].s0,rC[3][0].s1);
            rC[4][0].s1 = fma(rA[4].s1,rB[0].s0,rC[4][0].s1);
            rC[5][0].s1 = fma(rA[5].s1,rB[0].s0,rC[5][0].s1);
            rC[6][0].s1 = fma(rA[6].s1,rB[0].s0,rC[6][0].s1);
            rC[7][0].s1 = fma(rA[7].s1,rB[0].s0,rC[7][0].s1);
            rC[0][1].s1 = fma(rA[0].s1,rB[1].s0,rC[0][1].s1);
            rC[1][1].s1 = fma(rA[1].s1,rB[1].s0,rC[1][1].s1);
            rC[2][1].s1 = fma(rA[2].s1,rB[1].s0,rC[2][1].s1);
            rC[3][1].s1 = fma(rA[3].s1,rB[1].s0,rC[3][1].s1);
            rC[4][1].s1 = fma(rA[4].s1,rB[1].s0,rC[4][1].s1);
            rC[5][1].s1 = fma(rA[5].s1,rB[1].s0,rC[5][1].s1);
            rC[6][1].s1 = fma(rA[6].s1,rB[1].s0,rC[6][1].s1);
            rC[7][1].s1 = fma(rA[7].s1,rB[1].s0,rC[7][1].s1);
            rC[0][2].s1 = fma(rA[0].s1,rB[2].s0,rC[0][2].s1);
            rC[1][2].s1 = fma(rA[1].s1,rB[2].s0,rC[1][2].s1);
            rC[2][2].s1 = fma(rA[2].s1,rB[2].s0,rC[2][2].s1);
            rC[3][2].s1 = fma(rA[3].s1,rB[2].s0,rC[3][2].s1);
            rC[4][2].s1 = fma(rA[4].s1,rB[2].s0,rC[4][2].s1);
            rC[5][2].s1 = fma(rA[5].s1,rB[2].s0,rC[5][2].s1);
            rC[6][2].s1 = fma(rA[6].s1,rB[2].s0,rC[6][2].s1);
            rC[7][2].s1 = fma(rA[7].s1,rB[2].s0,rC[7][2].s1);
            rC[0][3].s1 = fma(rA[0].s1,rB[3].s0,rC[0][3].s1);
            rC[1][3].s1 = fma(rA[1].s1,rB[3].s0,rC[1][3].s1);
            rC[2][3].s1 = fma(rA[2].s1,rB[3].s0,rC[2][3].s1);
            rC[3][3].s1 = fma(rA[3].s1,rB[3].s0,rC[3][3].s1);
            rC[4][3].s1 = fma(rA[4].s1,rB[3].s0,rC[4][3].s1);
            rC[5][3].s1 = fma(rA[5].s1,rB[3].s0,rC[5][3].s1);
            rC[6][3].s1 = fma(rA[6].s1,rB[3].s0,rC[6][3].s1);
            rC[7][3].s1 = fma(rA[7].s1,rB[3].s0,rC[7][3].s1);


            rC[0][0].s1 = fma(rA[0].s0,rB[0].s1,rC[0][0].s1);
            rC[1][0].s1 = fma(rA[1].s0,rB[0].s1,rC[1][0].s1);
            rC[2][0].s1 = fma(rA[2].s0,rB[0].s1,rC[2][0].s1);
            rC[3][0].s1 = fma(rA[3].s0,rB[0].s1,rC[3][0].s1);
            rC[4][0].s1 = fma(rA[4].s0,rB[0].s1,rC[4][0].s1);
            rC[5][0].s1 = fma(rA[5].s0,rB[0].s1,rC[5][0].s1);
            rC[6][0].s1 = fma(rA[6].s0,rB[0].s1,rC[6][0].s1);
            rC[7][0].s1 = fma(rA[7].s0,rB[0].s1,rC[7][0].s1);
            rC[0][1].s1 = fma(rA[0].s0,rB[1].s1,rC[0][1].s1);
            rC[1][1].s1 = fma(rA[1].s0,rB[1].s1,rC[1][1].s1);
            rC[2][1].s1 = fma(rA[2].s0,rB[1].s1,rC[2][1].s1);
            rC[3][1].s1 = fma(rA[3].s0,rB[1].s1,rC[3][1].s1);
            rC[4][1].s1 = fma(rA[4].s0,rB[1].s1,rC[4][1].s1);
            rC[5][1].s1 = fma(rA[5].s0,rB[1].s1,rC[5][1].s1);
            rC[6][1].s1 = fma(rA[6].s0,rB[1].s1,rC[6][1].s1);
            rC[7][1].s1 = fma(rA[7].s0,rB[1].s1,rC[7][1].s1);
            rC[0][2].s1 = fma(rA[0].s0,rB[2].s1,rC[0][2].s1);
            rC[1][2].s1 = fma(rA[1].s0,rB[2].s1,rC[1][2].s1);
            rC[2][2].s1 = fma(rA[2].s0,rB[2].s1,rC[2][2].s1);
            rC[3][2].s1 = fma(rA[3].s0,rB[2].s1,rC[3][2].s1);
            rC[4][2].s1 = fma(rA[4].s0,rB[2].s1,rC[4][2].s1);
            rC[5][2].s1 = fma(rA[5].s0,rB[2].s1,rC[5][2].s1);
            rC[6][2].s1 = fma(rA[6].s0,rB[2].s1,rC[6][2].s1);
            rC[7][2].s1 = fma(rA[7].s0,rB[2].s1,rC[7][2].s1);
            rC[0][3].s1 = fma(rA[0].s0,rB[3].s1,rC[0][3].s1);
            rC[1][3].s1 = fma(rA[1].s0,rB[3].s1,rC[1][3].s1);
            rC[2][3].s1 = fma(rA[2].s0,rB[3].s1,rC[2][3].s1);
            rC[3][3].s1 = fma(rA[3].s0,rB[3].s1,rC[3][3].s1);
            rC[4][3].s1 = fma(rA[4].s0,rB[3].s1,rC[4][3].s1);
            rC[5][3].s1 = fma(rA[5].s0,rB[3].s1,rC[5][3].s1);
            rC[6][3].s1 = fma(rA[6].s0,rB[3].s1,rC[6][3].s1);
            rC[7][3].s1 = fma(rA[7].s0,rB[3].s1,rC[7][3].s1);



        }
        A += 16*M;
        B += 16*N;
    }
    C += gidx*128;
    C += idx;
    C += gidy*64*M;
    C += idy*M;

    C[ 0*M].s0      = rC[0][0].s0 * ALPHA.s0 - rC[0][0].s1 * ALPHA.s1 ;
    C[ 0*M].s1      = rC[0][0].s0 * ALPHA.s1 + rC[0][0].s1 * ALPHA.s0 ;
    C[16*M].s0      = rC[0][1].s0 * ALPHA.s0 - rC[0][1].s1 * ALPHA.s1 ;
    C[16*M].s1      = rC[0][1].s0 * ALPHA.s1 + rC[0][1].s1 * ALPHA.s0 ;
    C[32*M].s0      = rC[0][2].s0 * ALPHA.s0 - rC[0][2].s1 * ALPHA.s1 ;
    C[32*M].s1      = rC[0][2].s0 * ALPHA.s1 + rC[0][2].s1 * ALPHA.s0 ;
    C[48*M].s0      = rC[0][3].s0 * ALPHA.s0 - rC[0][3].s1 * ALPHA.s1 ;
    C[48*M].s1      = rC[0][3].s0 * ALPHA.s1 + rC[0][3].s1 * ALPHA.s0 ;
    C += 16;

    C[ 0*M].s0      = rC[1][0].s0 * ALPHA.s0 - rC[1][0].s1 * ALPHA.s1 ;
    C[ 0*M].s1      = rC[1][0].s0 * ALPHA.s1 + rC[1][0].s1 * ALPHA.s0 ;
    C[16*M].s0      = rC[1][1].s0 * ALPHA.s0 - rC[1][1].s1 * ALPHA.s1 ;
    C[16*M].s1      = rC[1][1].s0 * ALPHA.s1 + rC[1][1].s1 * ALPHA.s0 ;
    C[32*M].s0      = rC[1][2].s0 * ALPHA.s0 - rC[1][2].s1 * ALPHA.s1 ;
    C[32*M].s1      = rC[1][2].s0 * ALPHA.s1 + rC[1][2].s1 * ALPHA.s0 ;
    C[48*M].s0      = rC[1][3].s0 * ALPHA.s0 - rC[1][3].s1 * ALPHA.s1 ;
    C[48*M].s1      = rC[1][3].s0 * ALPHA.s1 + rC[1][3].s1 * ALPHA.s0 ;
    C += 16;

    C[ 0*M].s0      = rC[2][0].s0 * ALPHA.s0 - rC[2][0].s1 * ALPHA.s1 ;
    C[ 0*M].s1      = rC[2][0].s0 * ALPHA.s1 + rC[2][0].s1 * ALPHA.s0 ;
    C[16*M].s0      = rC[2][1].s0 * ALPHA.s0 - rC[2][1].s1 * ALPHA.s1 ;
    C[16*M].s1      = rC[2][1].s0 * ALPHA.s1 + rC[2][1].s1 * ALPHA.s0 ;
    C[32*M].s0      = rC[2][2].s0 * ALPHA.s0 - rC[2][2].s1 * ALPHA.s1 ;
    C[32*M].s1      = rC[2][2].s0 * ALPHA.s1 + rC[2][2].s1 * ALPHA.s0 ;
    C[48*M].s0      = rC[2][3].s0 * ALPHA.s0 - rC[2][3].s1 * ALPHA.s1 ;
    C[48*M].s1      = rC[2][3].s0 * ALPHA.s1 + rC[2][3].s1 * ALPHA.s0 ;
    C += 16;

    C[ 0*M].s0      = rC[3][0].s0 * ALPHA.s0 - rC[3][0].s1 * ALPHA.s1 ;
    C[ 0*M].s1      = rC[3][0].s0 * ALPHA.s1 + rC[3][0].s1 * ALPHA.s0 ;
    C[16*M].s0      = rC[3][1].s0 * ALPHA.s0 - rC[3][1].s1 * ALPHA.s1 ;
    C[16*M].s1      = rC[3][1].s0 * ALPHA.s1 + rC[3][1].s1 * ALPHA.s0 ;
    C[32*M].s0      = rC[3][2].s0 * ALPHA.s0 - rC[3][2].s1 * ALPHA.s1 ;
    C[32*M].s1      = rC[3][2].s0 * ALPHA.s1 + rC[3][2].s1 * ALPHA.s0 ;
    C[48*M].s0      = rC[3][3].s0 * ALPHA.s0 - rC[3][3].s1 * ALPHA.s1 ;
    C[48*M].s1      = rC[3][3].s0 * ALPHA.s1 + rC[3][3].s1 * ALPHA.s0 ;
    C += 16;

    C[ 0*M].s0      = rC[4][0].s0 * ALPHA.s0 - rC[4][0].s1 * ALPHA.s1 ;
    C[ 0*M].s1      = rC[4][0].s0 * ALPHA.s1 + rC[4][0].s1 * ALPHA.s0 ;
    C[16*M].s0      = rC[4][1].s0 * ALPHA.s0 - rC[4][1].s1 * ALPHA.s1 ;
    C[16*M].s1      = rC[4][1].s0 * ALPHA.s1 + rC[4][1].s1 * ALPHA.s0 ;
    C[32*M].s0      = rC[4][2].s0 * ALPHA.s0 - rC[4][2].s1 * ALPHA.s1 ;
    C[32*M].s1      = rC[4][2].s0 * ALPHA.s1 + rC[4][2].s1 * ALPHA.s0 ;
    C[48*M].s0      = rC[4][3].s0 * ALPHA.s0 - rC[4][3].s1 * ALPHA.s1 ;
    C[48*M].s1      = rC[4][3].s0 * ALPHA.s1 + rC[4][3].s1 * ALPHA.s0 ;
    C += 16;

    C[ 0*M].s0      = rC[5][0].s0 * ALPHA.s0 - rC[5][0].s1 * ALPHA.s1 ;
    C[ 0*M].s1      = rC[5][0].s0 * ALPHA.s1 + rC[5][0].s1 * ALPHA.s0 ;
    C[16*M].s0      = rC[5][1].s0 * ALPHA.s0 - rC[5][1].s1 * ALPHA.s1 ;
    C[16*M].s1      = rC[5][1].s0 * ALPHA.s1 + rC[5][1].s1 * ALPHA.s0 ;
    C[32*M].s0      = rC[5][2].s0 * ALPHA.s0 - rC[5][2].s1 * ALPHA.s1 ;
    C[32*M].s1      = rC[5][2].s0 * ALPHA.s1 + rC[5][2].s1 * ALPHA.s0 ;
    C[48*M].s0      = rC[5][3].s0 * ALPHA.s0 - rC[5][3].s1 * ALPHA.s1 ;
    C[48*M].s1      = rC[5][3].s0 * ALPHA.s1 + rC[5][3].s1 * ALPHA.s0 ;
    C += 16;

    C[ 0*M].s0      = rC[6][0].s0 * ALPHA.s0 - rC[6][0].s1 * ALPHA.s1 ;
    C[ 0*M].s1      = rC[6][0].s0 * ALPHA.s1 + rC[6][0].s1 * ALPHA.s0 ;
    C[16*M].s0      = rC[6][1].s0 * ALPHA.s0 - rC[6][1].s1 * ALPHA.s1 ;
    C[16*M].s1      = rC[6][1].s0 * ALPHA.s1 + rC[6][1].s1 * ALPHA.s0 ;
    C[32*M].s0      = rC[6][2].s0 * ALPHA.s0 - rC[6][2].s1 * ALPHA.s1 ;
    C[32*M].s1      = rC[6][2].s0 * ALPHA.s1 + rC[6][2].s1 * ALPHA.s0 ;
    C[48*M].s0      = rC[6][3].s0 * ALPHA.s0 - rC[6][3].s1 * ALPHA.s1 ;
    C[48*M].s1      = rC[6][3].s0 * ALPHA.s1 + rC[6][3].s1 * ALPHA.s0 ;
    C += 16;

    C[ 0*M].s0      = rC[7][0].s0 * ALPHA.s0 - rC[7][0].s1 * ALPHA.s1 ;
    C[ 0*M].s1      = rC[7][0].s0 * ALPHA.s1 + rC[7][0].s1 * ALPHA.s0 ;
    C[16*M].s0      = rC[7][1].s0 * ALPHA.s0 - rC[7][1].s1 * ALPHA.s1 ;
    C[16*M].s1      = rC[7][1].s0 * ALPHA.s1 + rC[7][1].s1 * ALPHA.s0 ;
    C[32*M].s0      = rC[7][2].s0 * ALPHA.s0 - rC[7][2].s1 * ALPHA.s1 ;
    C[32*M].s1      = rC[7][2].s0 * ALPHA.s1 + rC[7][2].s1 * ALPHA.s0 ;
    C[48*M].s0      = rC[7][3].s0 * ALPHA.s0 - rC[7][3].s1 * ALPHA.s1 ;
    C[48*M].s1      = rC[7][3].s0 * ALPHA.s1 + rC[7][3].s1 * ALPHA.s0 ;
    C += 16;

}


