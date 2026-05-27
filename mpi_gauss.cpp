#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <arm_neon.h>
#include "mpi.h"

using namespace std;

// 生成随机对角占优矩阵（与之前实验一致）
void init_matrix(float* A, float* b, int n) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    // 生成随机解 x_true
    vector<float> x_true(n);
    for (int i = 0; i < n; i++) x_true[i] = dis(gen);
    
    // 生成严格对角占优矩阵
    for (int i = 0; i < n; i++) {
        float row_sum = 0.0f;
        for (int j = 0; j < n; j++) {
            if (i != j) {
                float val = dis(gen);
                A[i * n + j] = val;
                row_sum += fabs(val);
            }
        }
        A[i * n + i] = row_sum + 1.0f;
    }
    
    // 计算 b = A * x_true
    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++) sum += A[i * n + j] * x_true[j];
        b[i] = sum;
    }
}

// 回代（使用完整矩阵）
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

    int n = 1024;  // 默认规模，可从命令行参数读取
    if (argc > 1) n = atoi(argv[1]);

    // 每个进程都分配完整的矩阵和b（全复制版，便于通信）
    float* A = (float*)aligned_alloc(64, n * n * sizeof(float));
    float* b = (float*)aligned_alloc(64, n * sizeof(float));

    // 0号进程初始化，然后广播给所有进程
    if (rank == 0) {
        init_matrix(A, b, n);
    }
    MPI_Bcast(A, n*n, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(b, n, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // 每个进程负责的行范围（连续块划分）
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
            // 更新b向量（对应行）
            b[i] -= factor * b[k];
        }
    }

    double end_time = MPI_Wtime();

    if (rank == 0) {
        back_substitution(A, b, n);
        cout << "MPI time with " << size << " procs: " << (end_time - start_time)*1000 << " ms" << endl;
        
        // 可选：打印误差（需事先保存x_true，此处省略）
    }

    free(A);
    free(b);
    MPI_Finalize();
    return 0;
}