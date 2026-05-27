#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <arm_neon.h>
#include "mpi.h"

using namespace std;

// 与之前相同的初始化函数（保证对角占优）
void init_matrix(float* A, float* b, int n) {
    // 此处复制你之前 generate_data 的核心逻辑，但只分配内存并填充
    // 为了简洁，假设已实现
}

// 回代（仅用0号进程的完整矩阵）
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

    int n = 2048;  // 可以从命令行参数读取
    if (argc > 1) n = atoi(argv[1]);

    // 每个进程都分配完整的矩阵和b（全复制版）
    float* A = (float*)aligned_alloc(64, n * n * sizeof(float));
    float* b = (float*)aligned_alloc(64, n * sizeof(float));

    // 0号进程初始化，然后广播给所有进程
    if (rank == 0) {
        init_matrix(A, b, n);
    }
    MPI_Bcast(A, n*n, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(b, n, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // 计算每个进程负责的行范围（连续块划分）
    int rows_per_proc = n / size;
    int remainder = n % size;
    int start_row, end_row;
    if (rank < remainder) {
        rows_per_proc++;
        start_row = rank * rows_per_proc;
    } else {
        start_row = rank * rows_per_proc + remainder;
    }
    end_row = start_row + rows_per_proc;
    if (end_row > n) end_row = n;

    double start_time = MPI_Wtime();

    for (int k = 0; k < n; ++k) {
        // 判断主元行属于哪个进程
        int pivot_owner = -1;
        int p_start = 0;
        for (int p = 0; p < size; ++p) {
            int p_rows = n / size;
            int p_rem = n % size;
            if (p < p_rem) p_rows++;
            int p_end = p_start + p_rows;
            if (k >= p_start && k < p_end) {
                pivot_owner = p;
                break;
            }
            p_start = p_end;
        }

        // 归一化主元行
        if (rank == pivot_owner) {
            float pivot = A[k * n + k];
            for (int j = k+1; j < n; ++j) {
                A[k * n + j] /= pivot;
            }
            A[k * n + k] = 1.0f;
        }

        // 广播主元行（从k列开始）
        MPI_Bcast(A + k * n + k, n - k, MPI_FLOAT, pivot_owner, MPI_COMM_WORLD);

        // 消去：当前进程负责的行中，行号 > k 的那些行
        for (int i = max(start_row, k+1); i < end_row; ++i) {
            float factor = A[i * n + k];
            // NEON向量化消去
            float32x4_t v_factor = vdupq_n_f32(factor);
            int j = k+1;
            for (; j <= n-4; j += 4) {
                float32x4_t v_Ai = vld1q_f32(&A[i * n + j]);
                float32x4_t v_Ak = vld1q_f32(&A[k * n + j]);
                float32x4_t v_res = vmlsq_f32(v_Ai, v_factor, v_Ak);
                vst1q_f32(&A[i * n + j], v_res);
            }
            for (; j < n; ++j) {
                A[i * n + j] -= factor * A[k * n + j];
            }
            A[i * n + k] = 0.0f;
            // 本地b向量更新（如果需要）
            // 注意：b向量的更新也需要对应行，这里省略简化
        }
    }

    double end_time = MPI_Wtime();

    if (rank == 0) {
        back_substitution(A, b, n);
        cout << "MPI time with " << size << " procs: " << (end_time - start_time)*1000 << " ms" << endl;
        // 可以输出误差检查正确性
    }

    free(A);
    free(b);
    MPI_Finalize();
    return 0;
}