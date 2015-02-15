

struct gpu_context
{
        int                     platform;
        int                     device;
        int                     unified_mem;
        cl_platform_id          platform_id;
        cl_device_id            device_id;
        cl_context              context;
        cl_program              program;
        cl_kernel               kernel;
        cl_command_queue        command_queue;
        cl_mem                  A;
        cl_mem                  B;
        cl_mem                  C;
        void*                   hA;
        void*                   hB;
        void*                   hC;
        char                    device_name[64];
};

static int  create_gpu_context(struct gpu_context *gpu);
static int  create_gpu_program_nonunified(struct gpu_context *gpu, char *func ,size_t bufsize);
static void destroy_gpu_context(struct gpu_context *gpu);
static void release_gpu_program(struct gpu_context *gpu);
static void open_gpu()  __attribute__((constructor));
static void close_gpu() __attribute__((destructor));

static void sgemm_gpu_ccopy(int M, int N, float *A, int LDA, float *B, blasint LDB, float beta) __attribute__ ((noinline));
static void sgemm_gpu_acopy(int M, int N, float *A , blasint LDA, float *B, int PAD_M, int PAD_N) __attribute__ ((noinline));
static void sgemm_gpu_btcopy(int M, int N, float *A , blasint LDA, float *B, int PAD_M, int PAD_N) __attribute__ ((noinline));
static void sgemm_gpu_bcopy(int M, int N, float *A , blasint LDA, float *B, int PAD_M, int PAD_N) __attribute__ ((noinline));
static int  sgemm_gpu_kernel(struct gpu_context *gpu_ptr, int M, int N, int K, float ALPHA, int acopy, int bcopy, double *ktime) __attribute__ ((noinline));

static void dgemm_gpu_ccopy(int M, int N, double *A, int LDA, double *B, blasint LDB, double beta) __attribute__ ((noinline));
static void dgemm_gpu_acopy(int M, int N, double *A , blasint LDA, double *B, int PAD_M, int PAD_N) __attribute__ ((noinline));
static void dgemm_gpu_btcopy(int M, int N, double *A , blasint LDA, double *B, int PAD_M, int PAD_N) __attribute__ ((noinline));
static void dgemm_gpu_bcopy(int M, int N, double *A , blasint LDA, double *B, int PAD_M, int PAD_N) __attribute__ ((noinline));
static int  dgemm_gpu_kernel(struct gpu_context *gpu_ptr, int M, int N, int K, double ALPHA, int acopy, int bcopy, double *ktime) __attribute__ ((noinline));

#if defined(CGEMM_GLOBAL0_DIV)

static void cgemm_gpu_ccopy(int M, int N, float *A, int LDA, float *B, blasint LDB, float *beta) __attribute__ ((noinline));
static void cgemm_gpu_acopy(int M, int N, float *A , blasint LDA, float *B, int PAD_M, int PAD_N) __attribute__ ((noinline));
static void cgemm_gpu_btcopy(int M, int N, float *A , blasint LDA, float *B, int PAD_M, int PAD_N) __attribute__ ((noinline));
static void cgemm_gpu_btcopy_conj(int M, int N, float *A , blasint LDA, float *B, int PAD_M, int PAD_N) __attribute__ ((noinline));
static void cgemm_gpu_bcopy(int M, int N, float *A , blasint LDA, float *B, int PAD_M, int PAD_N) __attribute__ ((noinline));
static void cgemm_gpu_bcopy_conj(int M, int N, float *A , blasint LDA, float *B, int PAD_M, int PAD_N) __attribute__ ((noinline));
static int  cgemm_gpu_kernel(struct gpu_context *gpu_ptr, int M, int N, int K, float *ALPHA, int acopy, int bcopy, double *ktime) __attribute__ ((noinline));

#endif

#if defined(ZGEMM_GLOBAL0_DIV)

static void zgemm_gpu_ccopy(int M, int N, double *A, int LDA, double *B, blasint LDB, double *beta) __attribute__ ((noinline));
static void zgemm_gpu_acopy(int M, int N, double *A , blasint LDA, double *B, int PAD_M, int PAD_N) __attribute__ ((noinline));
static void zgemm_gpu_btcopy(int M, int N, double *A , blasint LDA, double *B, int PAD_M, int PAD_N) __attribute__ ((noinline));
static void zgemm_gpu_btcopy_conj(int M, int N, double *A , blasint LDA, double *B, int PAD_M, int PAD_N) __attribute__ ((noinline));
static void zgemm_gpu_bcopy(int M, int N, double *A , blasint LDA, double *B, int PAD_M, int PAD_N) __attribute__ ((noinline));
static void zgemm_gpu_bcopy_conj(int M, int N, double *A , blasint LDA, double *B, int PAD_M, int PAD_N) __attribute__ ((noinline));
static int  zgemm_gpu_kernel(struct gpu_context *gpu_ptr, int M, int N, int K, double *ALPHA, int acopy, int bcopy, double *ktime) __attribute__ ((noinline));


#endif


#if defined(HAVE_DSGEMM)

static void dsgemm_gpu_ccopy(int M, int N, float *A, int LDA, double *B, blasint LDB, double alpha, double beta) __attribute__ ((noinline));
static void dsgemm_gpu_acopy(int M, int N, double *A , blasint LDA, float *B, int PAD_M, int PAD_N) __attribute__ ((noinline));
static void dsgemm_gpu_btcopy(int M, int N, double *A , blasint LDA, float *B, int PAD_M, int PAD_N) __attribute__ ((noinline));
static void dsgemm_gpu_bcopy(int M, int N, double *A , blasint LDA, float *B, int PAD_M, int PAD_N) __attribute__ ((noinline));

#endif
