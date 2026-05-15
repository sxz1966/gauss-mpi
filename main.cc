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
#include <omp.h>   // OpenMP 头文件
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
// ========== OpenMP + SIMD 版本 ==========
void gauss_omp_simd(float* A, float* b, int n, int num_threads) {
    // 设置线程数（也可以在 parallel 指令中用 num_threads 子句）
    omp_set_num_threads(num_threads);

    // 整个外层循环放在 parallel 区域内，只创建一次线程池
    #pragma omp parallel
    {
        // 私有变量：每个线程独立
        int i, j;
        float factor;

        // 外层 k 循环（所有线程都执行，但通过 single/for 分工）
        for (int k = 0; k < n; ++k) {
            // ---------- 除法：只有一个线程执行 ----------
            #pragma omp single
            {
                float pivot = A[k * n + k];
                for (int j = k + 1; j < n; ++j) {
                    A[k * n + j] /= pivot;
                }
                A[k * n + k] = 1.0f;
            }
            // 隐式 barrier 保证除法完成

            // ---------- 消去：并行化 i 循环（行划分）----------
            #pragma omp for schedule(static) private(i, j, factor)
            for (i = k + 1; i < n; ++i) {
                factor = A[i * n + k];
                // 使用 NEON 向量化消去一行
                float32x4_t v_factor = vdupq_n_f32(factor);
                j = k + 1;
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
            // 循环结束有隐式 barrier，保证所有线程完成本轮消去
        }
    }
    // 并行区域结束，回代
    back_substitution(A, b, n);
}

// ========== OpenMP 纯标量版本 ==========
void gauss_omp(float* A, float* b, int n, int num_threads) {
    omp_set_num_threads(num_threads);

    #pragma omp parallel
    {
        int i, j;
        float factor;

        for (int k = 0; k < n; ++k) {
            #pragma omp single
            {
                float pivot = A[k * n + k];
                for (int j = k + 1; j < n; ++j) {
                    A[k * n + j] /= pivot;
                }
                A[k * n + k] = 1.0f;
            }

            #pragma omp for schedule(static) private(i, j, factor)
            for (i = k + 1; i < n; ++i) {
                factor = A[i * n + k];
                // 标量消去
                for (j = k + 1; j < n; ++j) {
                    A[i * n + j] -= factor * A[k * n + j];
                }
                b[i] -= factor * b[k];
                A[i * n + k] = 0.0f;
            }
        }
    }
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

int main(int argc, char* argv[]) {
    // 参数解析
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <n> <version> [num_threads]\n";
        std::cerr << "  version: serial, simd, simd_v2, pthread, pthread_simd, omp, omp_simd, blocked\n";
        return 1;
    }
    int n = std::atoi(argv[1]);
    std::string version = argv[2];
    int num_threads = (argc >= 4) ? std::atoi(argv[3]) : 1;

    // 生成测试数据
    TestData data = generate_data(n);
    std::vector<float> A = data.A;
    std::vector<float> b = data.b;

    auto t1 = std::chrono::high_resolution_clock::now();

    if (version == "serial") {
        gauss_serial(A.data(), b.data(), n);
    } else if (version == "simd") {
        gauss_simd_neon(A.data(), b.data(), n);
    } else if (version == "simd_v2") {
        gauss_simd_neon_v2(A.data(), b.data(), n);
    } else if (version == "pthread") {
        gauss_pthread(A.data(), b.data(), n, num_threads);
    } else if (version == "pthread_simd") {
        gauss_pthread_simd(A.data(), b.data(), n, num_threads);
    } else if (version == "omp") {
        gauss_omp(A.data(), b.data(), n, num_threads);
    } else if (version == "omp_simd") {
        gauss_omp_simd(A.data(), b.data(), n, num_threads);
    } else if (version == "blocked") {
        int block_size = (argc >= 5) ? std::atoi(argv[4]) : 64;
        gauss_blocked_simd(A.data(), b.data(), n, block_size);
    } else {
        std::cerr << "Unknown version: " << version << "\n";
        return 1;
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

    float error = compute_max_error(b, data.x_true);

    // 输出格式：时间(ms) 误差（科学计数法），便于 test.sh 收集
    std::cout << std::fixed << std::setprecision(3) << time_ms << " "
              << std::scientific << std::setprecision(6) << error << std::endl;

    return 0;
}