#include <iostream>
#include <random>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <arm_neon.h>
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

    int n = 2048;
    if (argc > 1) n = atoi(argv[1]);

    float* A = (float*)aligned_alloc(64, n * n * sizeof(float));
    float* b = (float*)aligned_alloc(64, n * sizeof(float));

    if (rank == 0) init_matrix(A, b, n);
    MPI_Bcast(A, n*n, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(b, n, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // 行块划分（负载均衡）
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

    // 接收缓冲区（非阻塞接收时需要独立缓冲区）
    float* recv_buf = (float*)malloc(n * sizeof(float));

    for (int k = 0; k < n; ++k) {
        // 计算主元行所属进程
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

        MPI_Request req_recv = MPI_REQUEST_NULL;
        MPI_Request req_send = MPI_REQUEST_NULL;
        MPI_Status status;

        // 非阻塞接收（如果不是主元进程）
        if (rank != pivot_owner) {
            MPI_Irecv(recv_buf + k, n - k, MPI_FLOAT, pivot_owner, 
                      k, MPI_COMM_WORLD, &req_recv);
        }

        // 主元进程：归一化并发送
        if (rank == pivot_owner) {
            float pivot = A[k * n + k];
            for (int j = k+1; j < n; ++j) A[k * n + j] /= pivot;
            A[k * n + k] = 1.0f;
            // 非阻塞发送到所有其他进程
            for (int p = 0; p < size; ++p) {
                if (p != rank) {
                    MPI_Isend(A + k * n + k, n - k, MPI_FLOAT, p, 
                              k, MPI_COMM_WORLD, &req_send);
                }
            }
        }

        // 通信与计算重叠：主元进程立即开始消去自己负责的行
        if (rank == pivot_owner) {
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
                for (; j < n; ++j) A[i*n + j] -= factor * A[k*n + j];
                A[i*n + k] = 0.0f;
                b[i] -= factor * b[k];
            }
        } else {
            // 非主元进程：等待接收完成
            MPI_Wait(&req_recv, &status);
            // 将接收到的数据复制到矩阵中（替换主元行）
            memcpy(A + k*n + k, recv_buf + k, (n - k) * sizeof(float));
            // 消去自己负责的行
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
                for (; j < n; ++j) A[i*n + j] -= factor * A[k*n + j];
                A[i*n + k] = 0.0f;
                b[i] -= factor * b[k];
            }
        }

        // 同步所有进程，确保进入下一轮时通信已全部完成
        MPI_Barrier(MPI_COMM_WORLD);
    }

    double end_time = MPI_Wtime();

    if (rank == 0) {
        back_substitution(A, b, n);
        cout << fixed << setprecision(3);
        cout << "Non-blocking, n=" << n << ", procs=" << size 
             << ", time=" << (end_time - start_time)*1000 << " ms" << endl;
    }

    free(recv_buf);
    free(A);
    free(b);
    MPI_Finalize();
    return 0;
}