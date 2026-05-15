#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <arm_neon.h>   // NEON 头文件
#include <cstring>   // for std::memcpy
#include <cstdlib>   // for aligned_alloc, free
#include <pthread.h>    // Pthread 库
#include <semaphore.h>  // 信号量
// 全局变量，供 Pthread 线程函数访问
float* g_A = nullptr;
float* g_b = nullptr;
int g_n = 0;

// Pthread 同步 barrier
pthread_barrier_t barrier_div;   // 除法同步点
pthread_barrier_t barrier_elim;  // 消去同步点

// Pthread 线程参数结构体
typedef struct {
    int t_id;           // 线程 ID (0 ~ num_threads-1)
    int num_threads;    // 总线程数
} threadParam_t;


// Pthread 线程函数：所有线程（包括主线程）都执行此函数
void* pthread_simd_thread_func(void* arg) {
    threadParam_t* p = (threadParam_t*)arg;
    int t_id = p->t_id;
    int nth = p->num_threads;

    for (int k = 0; k < g_n; ++k) {
        // ---------- 除法：只有 t_id == 0 的线程执行 ----------
        if (t_id == 0) {
            float pivot = g_A[k * g_n + k];
            // 标量除法（也可以用向量化，但没必要）
            for (int j = k + 1; j < g_n; ++j) {
                g_A[k * g_n + j] /= pivot;
            }
            g_A[k * g_n + k] = 1.0f;
        }
        // 第一个 barrier：确保除法完成
        pthread_barrier_wait(&barrier_div);

        // ---------- 消去：水平划分连续行块 ----------
        int rows = g_n - k - 1;  // 需要处理的总行数
        if (rows > 0) {
            int rows_per_thread = (rows + nth - 1) / nth;
            int start = k + 1 + t_id * rows_per_thread;
            int end = start + rows_per_thread;
            if (end > g_n) end = g_n;

            for (int i = start; i < end; ++i) {
                float factor = g_A[i * g_n + k];   // A[i][k]
                // 内层循环：使用 NEON 向量化消去一行
                float32x4_t v_factor = vdupq_n_f32(factor);
                int j = k + 1;
                for (; j <= g_n - 4; j += 4) {
                    float32x4_t v_Ai = vld1q_f32(&g_A[i * g_n + j]);
                    float32x4_t v_Ak = vld1q_f32(&g_A[k * g_n + j]);
                    float32x4_t v_res = vmlsq_f32(v_Ai, v_factor, v_Ak);
                    vst1q_f32(&g_A[i * g_n + j], v_res);
                }
                for (; j < g_n; ++j) {
                    g_A[i * g_n + j] -= factor * g_A[k * g_n + j];
                }
                // 更新右端项
                g_b[i] -= factor * g_b[k];
                // 置零 A[i][k]
                g_A[i * g_n + k] = 0.0f;
            }
        }

        // 第二个 barrier：等待所有线程完成消去
        pthread_barrier_wait(&barrier_elim);
    }
    return nullptr;
}
//纯标量pthread线程函数
void* pthread_scalar_thread_func(void* arg) {
    threadParam_t* p = (threadParam_t*)arg;
    int t_id = p->t_id;
    int nth = p->num_threads;

    for (int k = 0; k < g_n; ++k) {
        // 除法：只有 t_id == 0 的线程执行（标量，与 SIMD 版本相同）
        if (t_id == 0) {
            float pivot = g_A[k * g_n + k];
            for (int j = k + 1; j < g_n; ++j) {
                g_A[k * g_n + j] /= pivot;
            }
            g_A[k * g_n + k] = 1.0f;
        }
        pthread_barrier_wait(&barrier_div);

        // 消去：水平划分（与 SIMD 版本相同，但内层循环是标量）
        int rows = g_n - k - 1;
        if (rows > 0) {
            int rows_per_thread = (rows + nth - 1) / nth;
            int start = k + 1 + t_id * rows_per_thread;
            int end = start + rows_per_thread;
            if (end > g_n) end = g_n;

            for (int i = start; i < end; ++i) {
                float factor = g_A[i * g_n + k];
                // 标量消去：最简单的三重循环
                for (int j = k + 1; j < g_n; ++j) {
                    g_A[i * g_n + j] -= factor * g_A[k * g_n + j];
                }
                g_b[i] -= factor * g_b[k];
                g_A[i * g_n + k] = 0.0f;
            }
        }
        pthread_barrier_wait(&barrier_elim);
    }
    return nullptr;
}
// ---------- 测试数据生成 ----------
struct TestData {
    std::vector<float> A;      // 矩阵 (n*n)
    std::vector<float> b;      // 右端项 (n)
    std::vector<float> x_true; // 真实解 (n)
    int n;
};

// 生成严格对角占优矩阵，确保高斯消去稳定
TestData generate_data(int n) {
    TestData data;
    data.n = n;
    data.A.resize(n * n, 0.0f);
    data.b.resize(n, 0.0f);
    data.x_true.resize(n, 0.0f);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    // 1. 随机生成真实解
    for (int i = 0; i < n; ++i) {
        data.x_true[i] = dis(gen);
    }

    // 2. 构造严格对角占优矩阵
    for (int i = 0; i < n; ++i) {
        float row_sum = 0.0f;
        for (int j = 0; j < n; ++j) {
            if (i != j) {
                float val = dis(gen);
                data.A[i * n + j] = val;
                row_sum += std::abs(val);
            }
        }
        data.A[i * n + i] = row_sum + 1.0f;  // 对角线 = 非对角绝对值之和 + 1
    }

    // 3. 计算 b = A * x_true
    for (int i = 0; i < n; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < n; ++j) {
            sum += data.A[i * n + j] * data.x_true[j];
        }
        data.b[i] = sum;
    }
    return data;
}
// ---------- 回代函数 ----------
void back_substitution(float* A, float* b, int n) {
    std::vector<float> x(n);
    for (int i = n - 1; i >= 0; --i) {
        float sum = b[i];
        for (int j = i + 1; j < n; ++j) {
            sum -= A[i * n + j] * x[j];
        }
        x[i] = sum / A[i * n + i];
    }
    for (int i = 0; i < n; ++i) {
        b[i] = x[i];
    }
}

//添加 对齐内存分配辅助函数
float* aligned_alloc_float(size_t n) {
    float* ptr = static_cast<float*>(aligned_alloc(16, n * sizeof(float)));
    if (!ptr) throw std::bad_alloc();
    return ptr;
}

//添加对齐部分的代码
// 对齐版本的高斯消去（与 gauss_simd_neon 完全相同，但数据来自对齐内存）
void gauss_simd_neon_aligned(float* A, float* b, int n) {
    // 前向消去
    for (int k = 0; k < n; ++k) {
        float pivot = A[k * n + k];
        for (int i = k + 1; i < n; ++i) {
            float factor = A[i * n + k] / pivot;
            float32x4_t v_factor = vdupq_n_f32(factor);
            int j = k + 1;
            for (; j <= n - 4; j += 4) {
                float32x4_t v_Ai = vld1q_f32(&A[i * n + j]);
                float32x4_t v_Ak = vld1q_f32(&A[k * n + j]);
                float32x4_t v_res = vmlsq_f32(v_Ai, v_factor, v_Ak);
                vst1q_f32(&A[i * n + j], v_res);
            }
            for (; j < n; ++j) {
                A[i * n + j] -= factor * A[k * n + j];
            }
            b[i] -= factor * b[k];
            A[i * n + k] = 0.0f;
        }
    }
        // 回代求解
    back_substitution(A, b, n);
}


// ---------- 串行高斯消去（单精度） ----------
void gauss_serial(float* A, float* b, int n) {
    // 前向消去
    for (int k = 0; k < n; ++k) {
        float pivot = A[k * n + k];
        for (int i = k + 1; i < n; ++i) {
            float factor = A[i * n + k] / pivot;
            for (int j = k + 1; j < n; ++j) {
                A[i * n + j] -= factor * A[k * n + j];
            }
            b[i] -= factor * b[k];
            A[i * n + k] = 0.0f;
        }
    }
        // 回代求解
    back_substitution(A, b, n);
}


// ---------- NEON SIMD 高斯消去 版本A 向量化消去  不对齐----------
void gauss_simd_neon(float* A, float* b, int n) {
    // 前向消去
    for (int k = 0; k < n; ++k) {
        float pivot = A[k * n + k];
        // 将主元广播到向量寄存器（供后续可能的向量化除法使用，这里仅用于消去）
        float32x4_t v_pivot = vdupq_n_f32(pivot);

        for (int i = k + 1; i < n; ++i) {
            float factor = A[i * n + k] / pivot;
            float32x4_t v_factor = vdupq_n_f32(factor);  // 将 factor 广播到 4 个通道

            int j = k + 1;
            // 向量化处理：每次处理 4 个元素
            for (; j <= n - 4; j += 4) {
                float32x4_t v_Ai = vld1q_f32(&A[i * n + j]);   // 加载 A[i][j..j+3]
                float32x4_t v_Ak = vld1q_f32(&A[k * n + j]);   // 加载 A[k][j..j+3]
                // 核心计算：A[i] = A[i] - factor * A[k]
                float32x4_t v_res = vmlsq_f32(v_Ai, v_factor, v_Ak);
                vst1q_f32(&A[i * n + j], v_res);              // 存储结果
            }
            // 处理剩余不足 4 个的元素（尾部串行）
            for (; j < n; ++j) {
                A[i * n + j] -= factor * A[k * n + j];
            }
            b[i] -= factor * b[k];
            A[i * n + k] = 0.0f;
        }
    }

        // 回代求解
    back_substitution(A, b, n);
}


// ---------- NEON SIMD 高斯消去 版本B：同时向量化除法和消去 ----------
void gauss_simd_neon_v2(float* A, float* b, int n) {
    // 临时数组存储 factor
    std::vector<float> factor_array(n);

    for (int k = 0; k < n; ++k) {
        float pivot = A[k * n + k];
        float32x4_t v_pivot = vdupq_n_f32(pivot);

        // ----- 阶段1：向量化计算 factor = A[i][k] / pivot -----
        int i = k + 1;
        for (; i <= n - 4; i += 4) {
            // 加载4个 A[i][k]（注意：这是列访问，地址不连续）
            // 需要逐个加载并组合，因为列访问在行主序下不是连续存储
            float32x4_t v_Aik;
            v_Aik = vsetq_lane_f32(A[(i+0) * n + k], v_Aik, 0);
            v_Aik = vsetq_lane_f32(A[(i+1) * n + k], v_Aik, 1);
            v_Aik = vsetq_lane_f32(A[(i+2) * n + k], v_Aik, 2);
            v_Aik = vsetq_lane_f32(A[(i+3) * n + k], v_Aik, 3);
            
            // 向量除法：4个factor同时计算
            float32x4_t v_factor = vdivq_f32(v_Aik, v_pivot);
            
            // 存储到 factor_array
            vst1q_f32(&factor_array[i], v_factor);
        }
        // 处理尾部剩余行
        for (; i < n; ++i) {
            factor_array[i] = A[i * n + k] / pivot;
        }

        // ----- 阶段2：向量化消去（与版本A相同，但从数组读取factor）-----
        for (int i = k + 1; i < n; ++i) {
            float factor = factor_array[i];
            float32x4_t v_factor = vdupq_n_f32(factor);

            int j = k + 1;
            for (; j <= n - 4; j += 4) {
                float32x4_t v_Ai = vld1q_f32(&A[i * n + j]);
                float32x4_t v_Ak = vld1q_f32(&A[k * n + j]);
                float32x4_t v_res = vmlsq_f32(v_Ai, v_factor, v_Ak);
                vst1q_f32(&A[i * n + j], v_res);
            }
            for (; j < n; ++j) {
                A[i * n + j] -= factor * A[k * n + j];
            }
            b[i] -= factor * b[k];
            A[i * n + k] = 0.0f;
        }
    }

    // 回代求解
    back_substitution(A, b, n);
}
// ========== Pthread +simd版本 ==========
void gauss_pthread_simd(float* A, float* b, int n, int num_threads) {
    // 设置全局指针（供线程函数使用）
    g_A = A;
    g_b = b;
    g_n = n;

    // 初始化 barrier
    pthread_barrier_init(&barrier_div, nullptr, num_threads);
    pthread_barrier_init(&barrier_elim, nullptr, num_threads);

    
    // 创建 num_threads 个线程，主线程也执行线程函数，但需要让主线程也调用线程函数。
    // 为避免主线程阻塞在 pthread_join，通常只创建 num_threads-1 个子线程，主线程直接执行线程函数。
    // 下面采用推荐方式：创建 num_threads-1 个子线程，主线程自己执行线程函数（作为最后一个线程）。
    std::vector<pthread_t> threads(num_threads - 1);
    std::vector<threadParam_t> params(num_threads);
    
    // 为子线程分配参数（t_id 从 0 到 num_threads-2）
    for (int i = 0; i < num_threads - 1; ++i) {
        params[i].t_id = i;
        params[i].num_threads = num_threads;
        pthread_create(&threads[i], nullptr, pthread_simd_thread_func, &params[i]);
    }
    
    // 主线程自己也作为最后一个线程（t_id = num_threads-1）
    threadParam_t main_param;
    main_param.t_id = num_threads - 1;
    main_param.num_threads = num_threads;
    pthread_simd_thread_func(&main_param);
    
    // 等待所有子线程结束
    for (int i = 0; i < num_threads - 1; ++i) {
        pthread_join(threads[i], nullptr);
    }
    
    // 销毁 barrier
    pthread_barrier_destroy(&barrier_div);
    pthread_barrier_destroy(&barrier_elim);
    
    // 回代
    back_substitution(A, b, n);
}
// ========== Pthread 纯标量版本 ==========
void gauss_pthread(float* A, float* b, int n, int num_threads) {
    g_A = A;
    g_b = b;
    g_n = n;

    pthread_barrier_init(&barrier_div, nullptr, num_threads);
    pthread_barrier_init(&barrier_elim, nullptr, num_threads);

    std::vector<pthread_t> threads(num_threads - 1);
    std::vector<threadParam_t> params(num_threads);

    for (int i = 0; i < num_threads - 1; ++i) {
        params[i].t_id = i;
        params[i].num_threads = num_threads;
        pthread_create(&threads[i], nullptr, pthread_scalar_thread_func, &params[i]);
    }

    threadParam_t main_param;
    main_param.t_id = num_threads - 1;
    main_param.num_threads = num_threads;
    pthread_scalar_thread_func(&main_param);

    for (int i = 0; i < num_threads - 1; ++i) {
        pthread_join(threads[i], nullptr);
    }

    pthread_barrier_destroy(&barrier_div);
    pthread_barrier_destroy(&barrier_elim);

    back_substitution(A, b, n);
}


// ---------- 分块优化版本 --------//
void gauss_blocked_simd(float* A, float* b, int n, int block_size) {
    // 前向消去（分块）
    for (int kk = 0; kk < n; kk += block_size) {
        int k_end = std::min(kk + block_size, n);
        
        // 1. 对当前对角块进行标准消去
        for (int k = kk; k < k_end; ++k) {
            float pivot = A[k * n + k];
            for (int i = k + 1; i < k_end; ++i) {
                float factor = A[i * n + k] / pivot;
                float32x4_t v_factor = vdupq_n_f32(factor);
                int j = k + 1;
                for (; j <= n - 4; j += 4) {
                    float32x4_t v_Ai = vld1q_f32(&A[i * n + j]);
                    float32x4_t v_Ak = vld1q_f32(&A[k * n + j]);
                    float32x4_t v_res = vmlsq_f32(v_Ai, v_factor, v_Ak);
                    vst1q_f32(&A[i * n + j], v_res);
                }
                for (; j < n; ++j) {
                    A[i * n + j] -= factor * A[k * n + j];
                }
                b[i] -= factor * b[k];
                A[i * n + k] = 0.0f;
            }
        }
        
        // 2. 更新右侧剩余列块,
        for (int k = kk; k < k_end; ++k) {
            float pivot = A[k * n + k];

            for (int i = k_end; i < n; ++i) {
                float factor = A[i * n + k] / pivot;
                float32x4_t v_factor = vdupq_n_f32(factor);
                int j = k_end;
                for (; j <= n - 4; j += 4) {
                    float32x4_t v_Ai = vld1q_f32(&A[i * n + j]);
                    float32x4_t v_Ak = vld1q_f32(&A[k * n + j]);
                    float32x4_t v_res = vmlsq_f32(v_Ai, v_factor, v_Ak);
                    vst1q_f32(&A[i * n + j], v_res);
                }
                for (; j < n; ++j) {
                    A[i * n + j] -= factor * A[k * n + j];
                }
                b[i] -= factor * b[k];
                A[i * n + k] = 0.0f;
            }
        }
    }

        // 回代求解
    back_substitution(A, b, n);
}


//所有模式的性能和误差计算都在 main() 中通过注释切换，默认是模式1（串行 vs SIMD-A vs SIMD-B 多规模表格）。其他模式可以通过取消相应代码块的注释来测试。

// ---------- 误差计算 ----------
float compute_max_error(const std::vector<float>& x_comp, const std::vector<float>& x_true) {
    float max_err = 0.0f;
    for (size_t i = 0; i < x_comp.size(); ++i) {
        max_err = std::max(max_err, std::abs(x_comp[i] - x_true[i]));
    }
    return max_err;
}

// ---------- 主函数（通过注释切换测试内容） ----------
int main(int argc, char* argv[]) {
    // 解析命令行参数，默认规模为 1024（perf 模式通常固定一个规模）
    int n = 1024;
    if (argc >= 2) {
        n = std::atoi(argv[1]);
    }

    std::cout << std::fixed << std::setprecision(3);
    

    TestData data = generate_data(n);

    

    // -------------------- 模式1：串行 vs SIMD-A vs SIMD-B（多规模自动表格）--------------------
    //测试不同规模下的性能和误差，规模 512, 1024, 2048
    
    std::vector<int> sizes = {512, 1024, 2048};
    std::cout << "\n===============================================================================\n";
    std::cout << "  Gaussian Elimination Final Benchmark (Serial vs SIMD-A vs SIMD-B)\n";
    std::cout << "===============================================================================\n";
    std::cout << "Size\tSerial(ms)\tSIMD-A(ms)\tSpeedup-A\tSIMD-B(ms)\tSpeedup-B\tMaxError\n";
    for (int sz : sizes) {
        TestData d = generate_data(sz);
        // 串行
        std::vector<float> A_ser = d.A;
        std::vector<float> b_ser = d.b;
        auto t1 = std::chrono::high_resolution_clock::now();
        gauss_serial(A_ser.data(), b_ser.data(), sz);
        auto t2 = std::chrono::high_resolution_clock::now();
        double time_ser = std::chrono::duration<double, std::milli>(t2 - t1).count();
        // SIMD-A
        std::vector<float> A_simdA = d.A;
        std::vector<float> b_simdA = d.b;
        t1 = std::chrono::high_resolution_clock::now();
        gauss_simd_neon(A_simdA.data(), b_simdA.data(), sz);
        t2 = std::chrono::high_resolution_clock::now();
        double time_simdA = std::chrono::duration<double, std::milli>(t2 - t1).count();
        // SIMD-B
        std::vector<float> A_simdB = d.A;
        std::vector<float> b_simdB = d.b;
        t1 = std::chrono::high_resolution_clock::now();
        gauss_simd_neon_v2(A_simdB.data(), b_simdB.data(), sz);
        t2 = std::chrono::high_resolution_clock::now();
        double time_simdB = std::chrono::duration<double, std::milli>(t2 - t1).count();

        float err = compute_max_error(b_simdB, d.x_true);
        float speedupA = time_ser / time_simdA;
        float speedupB = time_ser / time_simdB;
        std::cout << sz << "\t" << time_ser << "\t\t" << time_simdA << "\t\t"
                  << speedupA << "x\t\t" << time_simdB << "\t\t" << speedupB << "x\t\t"
                  << std::scientific << err << "\n";
    }
    std::cout << "===============================================================================\n";
    return 0;
    

    // -------------------- 模式2：对齐 vs 不对齐（单独测试，固定规模 1024）--------------------
    
   /*
    std::cout << "\n--- Aligned vs Unaligned Comparison (n = " << n << ") ---\n";
    // 不对齐版本（使用 std::vector）
    std::vector<float> A_un = data.A;
    std::vector<float> b_un = data.b;
    auto t1 = std::chrono::high_resolution_clock::now();
    gauss_simd_neon(A_un.data(), b_un.data(), n);
    auto t2 = std::chrono::high_resolution_clock::now();
    double time_un = std::chrono::duration<double, std::milli>(t2 - t1).count();
    float err_un = compute_max_error(b_un, data.x_true);
    std::cout << "SIMD-Unaligned time: " << time_un << " ms, error: " << std::scientific << err_un << "\n";

    // 对齐版本（使用 aligned_alloc）
    float* A_al = aligned_alloc_float(n * n);
    float* b_al = aligned_alloc_float(n);
    std::memcpy(A_al, data.A.data(), n * n * sizeof(float));
    std::memcpy(b_al, data.b.data(), n * sizeof(float));
    t1 = std::chrono::high_resolution_clock::now();
    gauss_simd_neon_aligned(A_al, b_al, n);
    t2 = std::chrono::high_resolution_clock::now();
    double time_al = std::chrono::duration<double, std::milli>(t2 - t1).count();
    std::vector<float> x_al(n);
    std::memcpy(x_al.data(), b_al, n * sizeof(float));
    float err_al = compute_max_error(x_al, data.x_true);
    std::cout << "SIMD-Aligned time:   " << time_al << " ms, error: " << std::scientific << err_al << "\n";
    free(A_al);
    free(b_al);
    return 0;
    //
    */
    // -------------------- 模式3：分块优化测试（不同块大小，规模 2048）--------------------
    
    /*
    std::vector<int> block_sizes = {32, 64, 128, 256};
    std::cout << "\n--- Blocked SIMD Results (n = " << n << ") ---\n";
    std::cout << "BlockSize\tTime(ms)\t\tMaxError\n";
    for (int bs : block_sizes) {
        std::vector<float> A_blk = data.A;
        std::vector<float> b_blk = data.b;
        auto t1 = std::chrono::high_resolution_clock::now();
        gauss_blocked_simd(A_blk.data(), b_blk.data(), n, bs);
        auto t2 = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        float err = compute_max_error(b_blk, data.x_true);
        std::cout << bs << "\t\t" << time_ms << "\t\t" << std::scientific << err << "\n";
    }
    return 0;
    */

    // -------------------- 模式4： perf 串行--------------------
    
    /*
    std::vector<float> A_ser = data.A;
    std::vector<float> b_ser = data.b;
    gauss_serial(A_ser.data(), b_ser.data(), n);
    std::cout << "Serial done. Error: " << std::scientific << compute_max_error(b_ser, data.x_true) << "\n";
    return 0;
    //
    */
    // -------------------- 模式5：perf SIMD-A（仅消去向量化，不对齐）--------------------
    /*
    std::vector<float> A_simd = data.A;
    std::vector<float> b_simd = data.b;
    gauss_simd_neon(A_simd.data(), b_simd.data(), n);
    std::cout << "SIMD-A done. Error: " << std::scientific << compute_max_error(b_simd, data.x_true) << "\n";
    return 0;
    //
     */
    // -------------------- 模式6：perf SIMD-B（除法+消去均向量化）--------------------
    /*
    std::vector<float> A_v2 = data.A;
    std::vector<float> b_v2 = data.b;
    gauss_simd_neon_v2(A_v2.data(), b_v2.data(), n);
    std::cout << "SIMD-B done. Error: " << std::scientific << compute_max_error(b_v2, data.x_true) << "\n";
    return 0;
    //
    */
    // -------------------- 模式7： perf SIMD-对齐（对齐内存分配）--------------------
    /*
    float* A_al = aligned_alloc_float(n * n);
    float* b_al = aligned_alloc_float(n);
    std::memcpy(A_al, data.A.data(), n * n * sizeof(float));
    std::memcpy(b_al, data.b.data(), n * sizeof(float));
    gauss_simd_neon_aligned(A_al, b_al, n);
    std::vector<float> x_al(n);
    std::memcpy(x_al.data(), b_al, n * sizeof(float));
    std::cout << "SIMD-Aligned done. Error: " << std::scientific << compute_max_error(x_al, data.x_true) << "\n";
    free(A_al);
    free(b_al);
    return 0;
    //
    */
        // -------------------- 模式8： perf 专用（分块优化，固定块大小）--------------------
    
    /*
    int block_size = 64;   // 可手动修改为 32, 64, 128, 256 等
    std::vector<float> A_blk = data.A;
    std::vector<float> b_blk = data.b;
    gauss_blocked_simd(A_blk.data(), b_blk.data(), n, block_size);
    std::cout << "Blocked SIMD (bs=" << block_size << ") done. Error: "
              << std::scientific << compute_max_error(b_blk, data.x_true) << "\n";
    return 0;
    //*/
    // -------------------- 默认模式（防止所有模式都被注释时报错）--------------------
    std::cout << "Please uncomment one of the test modes in main()!\n";
    return 0;
}