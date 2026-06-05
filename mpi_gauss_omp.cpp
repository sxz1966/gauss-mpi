#include <iostream>
#include <random>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <arm_neon.h>
#include <omp.h>
#include "mpi.h"

using namespace std;

void init_matrix(float* A, float* b, int n) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(-1.0f, 1.0f);
    vector<float> x_true(n);
    for (int i = 0; i < n; i++) x_true[i] = dis(gen);
    for (int i = 0; i < n; i++) {
        float row_sum = 0.0f;
        for (int j = 0; j < n; j++) {
            if (i != j) {
                float val = dis(gen);
                A[i*n + j] = val;
                row_sum += fabs(val);
            }
        }
        A[i*n + i] = row_sum + 1.0f;
    }
    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++) sum += A[i*n + j] * x_true[j];
        b[i] = sum;
    }
}

void back_substitution(float* A, float* b, int n) {
    for (int i = n-1; i >= 0; --i) {
        float sum = b[i];
        for (int j = i+1; j < n; ++j) sum -= A[i*n + j] * b[j];
        b[i] = sum / A[i*n + i];
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int omp_threads = 1;
    if (const char* env_omp = getenv("OMP_NUM_THREADS")) omp_threads = atoi(env_omp);
    omp_set_num_threads(omp_threads);
    if (rank == 0) {
        cout << "Hybrid MPI+OpenMP: " << size << " processes, " 
             << omp_threads << " threads per process" << endl;
    }

    int n = 2048;
    if (argc > 1) n = atoi(argv[1]);

    float* A = (float*)aligned_alloc(64, n * n * sizeof(float));
    float* b = (float*)aligned_alloc(64, n * sizeof(float));

    if (rank == 0) init_matrix(A, b, n);
    MPI_Bcast(A, n*n, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(b, n, MPI_FLOAT, 0, MPI_COMM_WORLD);

    int rows_per_proc = n / size;
    int remainder = n % size;
    int start_row = (rank < remainder) ? rank * (rows_per_proc+1) : rank * rows_per_proc + remainder;
    int end_row = start_row + ((rank < remainder) ? rows_per_proc+1 : rows_per_proc);
    if (end_row > n) end_row = n;

    double start_time = MPI_Wtime();

    for (int k = 0; k < n; ++k) {
        int pivot_owner = -1;
        int p_start = 0;
        for (int p = 0; p < size; ++p) {
            int p_rows = n / size;
            int p_rem = n % size;
            int p_end = p_start + ((p < p_rem) ? p_rows+1 : p_rows);
            if (k >= p_start && k < p_end) {
                pivot_owner = p;
                break;
            }
            p_start = p_end;
        }

        // 仅由一个线程执行通信和归一化，其他线程等待
        #pragma omp single
        {
            if (rank == pivot_owner) {
                float pivot = A[k * n + k];
                for (int j = k+1; j < n; ++j) A[k * n + j] /= pivot;
                A[k * n + k] = 1.0f;
            }
            MPI_Bcast(A + k * n + k, n - k, MPI_FLOAT, pivot_owner, MPI_COMM_WORLD);
        }

        // 所有线程并行消去本进程负责的行（行号 > k）
        #pragma omp parallel for schedule(static)
        for (int i = max(start_row, k+1); i < end_row; ++i) {
            float factor = A[i * n + k];
            float32x4_t v_factor = vdupq_n_f32(factor);
            int j = k+1;
            for (; j <= n-4; j += 4) {
                float32x4_t v_Ai = vld1q_f32(&A[i*n + j]);
                float32x4_t v_Ak = vld1q_f32(&A[k*n + j]);
                float32x4_t v_res = vmlsq_f32(v_Ai, v_factor, v_Ak);
                vst1q_f32(&A[i*n + j], v_res);
            }
            for (; j < n; ++j) {
                A[i*n + j] -= factor * A[k*n + j];
            }
            A[i*n + k] = 0.0f;
            b[i] -= factor * b[k];
        }
    }

    double end_time = MPI_Wtime();

    if (rank == 0) {
        back_substitution(A, b, n);
        cout << fixed << setprecision(3);
        cout << n << "," << size << "," << omp_threads << "," 
             << (end_time - start_time) * 1000 << endl;
    }

    free(A);
    free(b);
    MPI_Finalize();
    return 0;
}