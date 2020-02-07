const char *dgemm_desc = "Simple blocked dgemm.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 4
#endif

#ifndef MatrixPartitionSize
#define MatrixPartitionSize 4
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))

#include <immintrin.h>

/*
 * This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N.
 */
static void do_block(int lda, int M, int N, int K, double *A, double *B, double *C) {
    // For each row i of A
    for (int i = 0; i < M; ++i) {
        //For each column j of B
        for (int j = 0; j < N; ++j) {
            // Compute C(i,j)
            for (int k = 0; k < K; ++k) {
                C[i + j * lda] += A[i + k * lda] * B[k + j * lda];
            }
        }
    }
}


static void do_block(int lda, int M, int N, int K, double *A, double *B, double *C) {
    double block_A[4 * K], block_B[4 * K];
    double *pt_a = A, *pt_b = B, *pt_c = C;
    int floor_M = M / 4, floor_N = N/4;
    int res_M = M % 4, res_N = N % 4;

    for(int j = 0; j < floor_N; j++){
      pt_b += 4 * lda * j;
      mem_B(lda, K, pt_b, j);
      for(int i = 0; i < floor_M; i++){
        pt_a += 4 * K * i;
        matmul_4x4(lda, K, pt_a, pt_b, pt_c);
        pt_c += 4;
      }
      pt_c += (4 * lda - floor_M * 4);
    }

    if(res_M != 0){
      for(int i = 0; i < )
    }    
}

static inline void mem_A(int lda, int K, double *a_src, double *a_dst){
  for(int i = 0; i < K; i++){
    *a_dst++ = *a_src++;
    *a_dst++ = *a_src++;
    *a_dst++ = *a_src++;
    *a_dst++ = *a_src++;
    a_src += (lda - 4);
  }
}

static inline void mem_B(int lda, int K, double *b_src, double *b_dst){
  for(int i = 0; i < 4; i++){
    for(int j = 0; j < K; j++){
      *b_dst++ = *b_src++;
    }
    b_src += (lda - K);
  }
}

static inline void matmul_4x4(int lda, int K, double* a, double* b, double* c){

  __m256d col_a;
  __m256d b0, b1, b2, b3;

  double* c0 = c;
  double* c1 = c + lda;
  double* c2 = c + 2 * lda;
  double* c3 = c + 3 * lda;

    // load old value of c
  __m256d col_c0 = _mm256_loadu_pd(c);
  __m256d col_c1 = _mm256_loadu_pd(c1);
  __m256d col_c2 = _mm256_loadu_pd(c2);
  __m256d col_c3 = _mm256_loadu_pd(c3);

    // for every column of a (or every row of b)
  for (int i = 0; i < K; ++i){
    col_a = _mm256_load_pd(a);
    a += 4;

    b0 = _mm256_broadcast_sd(b++);
    b1 = _mm256_broadcast_sd(b++);
    b2 = _mm256_broadcast_sd(b++);
    b3 = _mm256_broadcast_sd(b++);

    col_c0 = _mm256_add_pd(col_c0, _mm256_mul_pd(col_a, b0));
    col_c1 = _mm256_add_pd(col_c1, _mm256_mul_pd(col_a, b1));
    col_c2 = _mm256_add_pd(col_c2, _mm256_mul_pd(col_a, b2));
    col_c3 = _mm256_add_pd(col_c3, _mm256_mul_pd(col_a, b3));
  }

  _mm256_storeu_pd(c, col_c0);
  _mm256_storeu_pd(c1, col_c1);
  _mm256_storeu_pd(c2, col_c2);
  _mm256_storeu_pd(c3, col_c3);
}


static void do_block_simd(int lda, double* A, double *B, double *C) {
   __m256d row1 = _mm256_loadu_pd(&A[0]);
   __m256d row2 = _mm256_loadu_pd(&A[lda]);
   __m256d row3 = _mm256_loadu_pd(&A[2 * lda]);
   __m256d row4 = _mm256_loadu_pd(&A[3 * lda]);
   for (int i = 0; i < 4; i++) {
       __m256d brod1 = _mm256_set1_pd(B[lda * i + 0]);
       __m256d brod2 = _mm256_set1_pd(B[lda * i + 1]);
       __m256d brod3 = _mm256_set1_pd(B[lda * i + 2]);
       __m256d brod4 = _mm256_set1_pd(B[lda * i + 3]);
       __m256d row = _mm256_add_pd(
           _mm256_add_pd(
             _mm256_mul_pd(brod1, row1),
             _mm256_mul_pd(brod2, row2)),
           _mm256_add_pd(
             _mm256_mul_pd(brod3, row3),
             _mm256_mul_pd(brod4, row4)));
       _mm256_storeu_pd(&C[lda*i], _mm256_add_pd(_mm256_loadu_pd(&C[lda*i]), row));
     
   }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */
void square_dgemm(int lda, double *A, double *B, double *C) {
    // For each block-row of A
    for (int i = 0; i < lda; i += BLOCK_SIZE) {
        // For each block-column of B
        for (int j = 0; j < lda; j += BLOCK_SIZE) {
            // Accumulate block dgemms into block of C
            for (int k = 0; k < lda; k += BLOCK_SIZE) {
                // Correct block dimensions if block "goes off edge of" the matrix
                int M = min(BLOCK_SIZE, lda - i);
                int N = min(BLOCK_SIZE, lda - j);
                int K = min(BLOCK_SIZE, lda - k);
                // Perform individual block dgemm
    if (M != BLOCK_SIZE || N != BLOCK_SIZE || K != BLOCK_SIZE) {
                    do_block(lda, M, N, K, A + i + k * lda, B + k + j * lda, C + i + j * lda);
    } else {
              do_block_simd(lda, A + i + k * lda, B + k + j * lda, C + i + j * lda);
          }
            }
        }
    }
}