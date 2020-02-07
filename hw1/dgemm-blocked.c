const char *dgemm_desc = "Simple blocked dgemm.";
//
//#ifndef BLOCK_SIZE
//#define BLOCK_SIZE 32
//#endif

#ifndef MatrixPartitionSize
#define MatrixPartitionSize 32
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))

#include <immintrin.h>
#include <stdio.h>

static void do_block_naive(int lda, int M, int N, int K, double *A, double *B, double *C) {
    // For each row i of A
        //For each column j of B
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < M; ++i) {
            // Compute C(i,j)
            for (int k = 0; k < K; ++k) {
                C[i + j * lda] += A[i + k * lda] * B[k + j * lda];
            }
        }
    }
}

static __inline__ void do_block_simd(int lda, double* __restrict__ A,  double* __restrict__ B,  double *C) {
   __m256d row1 = _mm256_loadu_pd(&A[0]);
   __m256d row2 = _mm256_loadu_pd(&A[lda]);
   __m256d row3 = _mm256_loadu_pd(&A[2 * lda]);
   __m256d row4 = _mm256_loadu_pd(&A[3 * lda]);

   __m256d brod1 = _mm256_set1_pd(B[0]);
   __m256d brod2 = _mm256_set1_pd(B[1]);
   __m256d brod3 = _mm256_set1_pd(B[2]);
   __m256d brod4 = _mm256_set1_pd(B[3]);
   __m256d outrow = _mm256_add_pd(
           _mm256_add_pd(
               _mm256_mul_pd(brod1, row1),
               _mm256_mul_pd(brod2, row2)
           ),
           _mm256_add_pd(
               _mm256_mul_pd(brod3, row3),
               _mm256_mul_pd(brod4, row4)
           )
   );
   _mm256_storeu_pd(&C[0], _mm256_add_pd(_mm256_loadu_pd(&C[0]), outrow));

   brod1 = _mm256_set1_pd(B[lda + 0]);
   brod2 = _mm256_set1_pd(B[lda + 1]);
   brod3 = _mm256_set1_pd(B[lda + 2]);
   brod4 = _mm256_set1_pd(B[lda + 3]);
   outrow = _mm256_add_pd(
           _mm256_add_pd(
               _mm256_mul_pd(brod1, row1),
               _mm256_mul_pd(brod2, row2)
           ),
           _mm256_add_pd(
               _mm256_mul_pd(brod3, row3),
               _mm256_mul_pd(brod4, row4)
           )
   );
   _mm256_storeu_pd(&C[lda], _mm256_add_pd(_mm256_loadu_pd(&C[lda]), outrow));
   brod1 = _mm256_set1_pd(B[lda * 2 + 0]);
   brod2 = _mm256_set1_pd(B[lda * 2 + 1]);
   brod3 = _mm256_set1_pd(B[lda * 2 + 2]);
   brod4 = _mm256_set1_pd(B[lda * 2 + 3]);
   outrow = _mm256_add_pd(
           _mm256_add_pd(
               _mm256_mul_pd(brod1, row1),
               _mm256_mul_pd(brod2, row2)
           ),
           _mm256_add_pd(
               _mm256_mul_pd(brod3, row3),
               _mm256_mul_pd(brod4, row4)
           )
   );
   _mm256_storeu_pd(&C[lda*2], _mm256_add_pd(_mm256_loadu_pd(&C[lda*2]), outrow));
   brod1 = _mm256_set1_pd(B[lda * 3 + 0]);
   brod2 = _mm256_set1_pd(B[lda * 3 + 1]);
   brod3 = _mm256_set1_pd(B[lda * 3 + 2]);
   brod4 = _mm256_set1_pd(B[lda * 3 + 3]);
   outrow = _mm256_add_pd(
           _mm256_add_pd(
               _mm256_mul_pd(brod1, row1),
               _mm256_mul_pd(brod2, row2)
           ),
           _mm256_add_pd(
               _mm256_mul_pd(brod3, row3),
               _mm256_mul_pd(brod4, row4)
           )
   );
   _mm256_storeu_pd(&C[lda*3], _mm256_add_pd(_mm256_loadu_pd(&C[lda*3]), outrow));
     
}


static __inline__ void do_block(int lda, int BLOCK_SIZE,  double* __restrict__  A,  double* __restrict__ B,  double *C) {
    for (int j = 0; j < BLOCK_SIZE; j += 4) {
        for (int i = 0; i < BLOCK_SIZE; i+= 4) {
            for (int k = 0; k < BLOCK_SIZE; k += 4) {
                do_block_simd(lda, A + i + k * lda, B + k + j * lda, C + i + j * lda);
            }
        }
    }
}


void square_dgemm(int lda, double *A, double *B, double *C) {
    int BLOCK_SIZE = 16;
    if (lda > 192) {
        BLOCK_SIZE *= 2;
    }
    if (lda > 480) {
        BLOCK_SIZE *= 2;
    }
    int lda_enlarge = lda;
    if (lda % BLOCK_SIZE > 7 || lda == 769) {
//        if (lda < 256) {
//            BLOCK_SIZE /= 2;
//        }
        lda_enlarge = (lda / BLOCK_SIZE + 1) * BLOCK_SIZE;
        double* AA = _mm_malloc(lda_enlarge * lda_enlarge * sizeof(double), 128);
        for (int j = 0; j < lda_enlarge; j++) {
            for (int i = 0; i < lda_enlarge; i++) {
                if (i < lda && j < lda) {
                    AA[i + j * lda_enlarge] = A[i + j * lda];
                } else {
                    AA[i + j * lda_enlarge] = 0;
                }
            }
        }
        double* BB = _mm_malloc(lda_enlarge * lda_enlarge * sizeof(double), 128);
        for (int j = 0; j < lda_enlarge; j++) {
            for (int i = 0; i < lda_enlarge; i++) {
                if (i < lda && j < lda) {
                    BB[i + j * lda_enlarge] = B[i + j * lda];
                } else {
                    BB[i + j * lda_enlarge] = 0;
                }
            }
        }
        // double* CC = calloc(lda_enlarge * lda_enlarge, sizeof(double));
        double* CC = _mm_malloc(lda_enlarge * lda_enlarge * sizeof(double), 128);
        for (int j = 0; j < lda_enlarge; j += BLOCK_SIZE) {
            for (int i = 0; i < lda_enlarge; i += BLOCK_SIZE) {
                for (int k = 0; k < lda_enlarge; k += BLOCK_SIZE) {
                    do_block(lda_enlarge, BLOCK_SIZE, AA + i + k * lda_enlarge, BB + k + j * lda_enlarge, CC + i + j * lda_enlarge);
                }
            }
        }
        for (int j = 0; j < lda; j++) {
            for (int i = 0; i < lda; i++) {
                C[i + j * lda] += CC[i + j * lda_enlarge];
            }
        }
    } else if (lda % BLOCK_SIZE == 0) {
        if (lda < 128 && lda % (BLOCK_SIZE * 2) == 0) {
            BLOCK_SIZE *= 2;
        }
        for (int j = 0; j < lda_enlarge; j += BLOCK_SIZE) {
            for (int i = 0; i < lda_enlarge; i += BLOCK_SIZE) {
                for (int k = 0; k < lda_enlarge; k += BLOCK_SIZE) {
                    do_block(lda_enlarge, BLOCK_SIZE, A + i + k * lda_enlarge, B + k + j * lda_enlarge, C + i + j * lda_enlarge);
                }
            }
        }
    } else {
        int M;
        int N;
        int K;
        for (int j = 0; j < lda; j += BLOCK_SIZE) {
            for (int i = 0; i < lda; i += BLOCK_SIZE) {
                for (int k = 0; k < lda; k += BLOCK_SIZE) {
                    M = min(BLOCK_SIZE, lda - i);
                    N = min(BLOCK_SIZE, lda - j);
                    K = min(BLOCK_SIZE, lda - k);
                    if (M == BLOCK_SIZE && N && BLOCK_SIZE && K == BLOCK_SIZE) {
                        do_block(lda, BLOCK_SIZE, A + i + k * lda, B + k + j * lda, C + i + j * lda);
                    } else {
                        do_block_naive(lda, M, N, K, A + i + k * lda, B + k + j * lda, C + i + j * lda);
                    }
                }
            }
        }
    }
}

