#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <cstring>
#include <cstdlib>
#include <pthread.h>
#include <arm_neon.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

// ==================== 手动实现 Barrier（C++11 兼容）====================
class Barrier {
public:
    explicit Barrier(int count) : m_count(count), m_generation(0), m_total(count) {}
    
    void wait() {
        std::unique_lock<std::mutex> lock(m_mutex);
        int my_gen = m_generation;
        if (--m_count == 0) {
            m_generation++;
            m_count = m_total;
            lock.unlock();
            m_cv.notify_all();
        } else {
            m_cv.wait(lock, [this, my_gen] { return my_gen != m_generation; });
        }
    }
    
private:
    std::mutex m_mutex;
    std::condition_variable m_cv;
    int m_count;
    int m_generation;
    int m_total;
};

// ==================== 公共函数 ====================
void back_substitution(float* A, float* b, int n) {
    for (int i = n - 1; i >= 0; --i) {
        float sum = b[i];
        for (int j = i + 1; j < n; ++j) sum -= A[i * n + j] * b[j];
        b[i] = sum / A[i * n + i];
    }
}

struct TestData {
    float* A;
    float* b;
    std::vector<float> x_true;
    int n;
};

TestData generate_data(int n) {
    TestData data;
    data.n = n;
    data.A = static_cast<float*>(aligned_alloc(64, n * n * sizeof(float)));
    data.b = static_cast<float*>(aligned_alloc(64, n * sizeof(float)));
    data.x_true.resize(n);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (int i = 0; i < n; ++i) data.x_true[i] = dis(gen);
    for (int i = 0; i < n; ++i) {
        float row_sum = 0.0f;
        for (int j = 0; j < n; ++j) {
            if (i != j) {
                float val = dis(gen);
                data.A[i * n + j] = val;
                row_sum += std::abs(val);
            }
        }
        data.A[i * n + i] = row_sum + 1.0f;
    }
    for (int i = 0; i < n; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < n; ++j) sum += data.A[i * n + j] * data.x_true[j];
        data.b[i] = sum;
    }
    return data;
}

void free_testdata(TestData& data) { free(data.A); free(data.b); }

float compute_max_error(const float* x_comp, const std::vector<float>& x_true, int n) {
    float max_err = 0.0f;
    for (int i = 0; i < n; ++i) {
        float diff = std::abs(x_comp[i] - x_true[i]);
        if (diff > max_err) max_err = diff;
    }
    return max_err;
}

// ==================== 串行标量（基准） ====================
void gauss_serial(float* A, float* b, int n) {
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
    back_substitution(A, b, n);
}

// ==================== Pthread + SIMD 连续行块划分 (block) ====================
static float* g_A = nullptr;
static float* g_b = nullptr;
static int g_n = 0;
static int g_num_threads = 0;
static pthread_barrier_t barrier_elim_block;

void* pthread_worker_block(void* arg) {
    int tid = *(int*)arg;
    for (int k = 0; k < g_n; ++k) {
        float pivot = g_A[k * g_n + k];
        int rows = g_n - k - 1;
        if (rows > 0) {
            int rows_per_thread = (rows + g_num_threads - 1) / g_num_threads;
            int start = k + 1 + tid * rows_per_thread;
            int end = start + rows_per_thread;
            if (end > g_n) end = g_n;
            for (int i = start; i < end; ++i) {
                float factor = g_A[i * g_n + k] / pivot;
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
                g_b[i] -= factor * g_b[k];
                g_A[i * g_n + k] = 0.0f;
            }
        }
        pthread_barrier_wait(&barrier_elim_block);
    }
    return nullptr;
}

void gauss_pthread_block(float* A, float* b, int n, int num_threads) {
    if (num_threads <= 1) {
        gauss_serial(A, b, n);
        return;
    }
    g_A = A;
    g_b = b;
    g_n = n;
    g_num_threads = num_threads;
    pthread_barrier_init(&barrier_elim_block, nullptr, num_threads);
    std::vector<pthread_t> threads(num_threads - 1);
    std::vector<int> tids(num_threads);
    for (int i = 0; i < num_threads - 1; ++i) {
        tids[i] = i;
        pthread_create(&threads[i], nullptr, pthread_worker_block, &tids[i]);
    }
    tids[num_threads - 1] = num_threads - 1;
    pthread_worker_block(&tids[num_threads - 1]);
    for (int i = 0; i < num_threads - 1; ++i) pthread_join(threads[i], nullptr);
    pthread_barrier_destroy(&barrier_elim_block);
    back_substitution(g_A, g_b, g_n);
}

// ==================== C++ std::thread + 手动 Barrier 连续块划分 ====================
void gauss_stdthread_block(float* A, float* b, int n, int num_threads) {
    if (num_threads <= 1) {
        gauss_serial(A, b, n);
        return;
    }
    Barrier sync_point(num_threads);
    
    auto worker = [&](int tid) {
        for (int k = 0; k < n; ++k) {
            float pivot = A[k * n + k];
            int rows = n - k - 1;
            if (rows > 0) {
                int rows_per_thread = (rows + num_threads - 1) / num_threads;
                int start = k + 1 + tid * rows_per_thread;
                int end = start + rows_per_thread;
                if (end > n) end = n;
                for (int i = start; i < end; ++i) {
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
            sync_point.wait();
        }
    };
    
    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back(worker, t);
    }
    for (auto& th : threads) {
        th.join();
    }
    back_substitution(A, b, n);
}

// ==================== 主函数：性能测试 ====================
int main(int argc, char* argv[]) {
    std::vector<int> sizes = {512, 1024, 2048};
    std::vector<int> thread_counts = {1, 2, 4, 8};
    
    if (argc >= 2) {
        sizes.clear();
        sizes.push_back(std::atoi(argv[1]));
        if (argc >= 3) {
            thread_counts.clear();
            thread_counts.push_back(std::atoi(argv[2]));
        }
    }
    
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "============================================================\n";
    std::cout << " Comparison: Pthread vs std::thread (with manual Barrier)\n";
    std::cout << "============================================================\n\n";
    
    for (int n : sizes) {
        std::cout << "----------- Matrix Size: " << n << " x " << n << " -----------\n";
        TestData base_data = generate_data(n);
        
        // 串行基准
        float* A_ser = static_cast<float*>(aligned_alloc(64, n * n * sizeof(float)));
        float* b_ser = static_cast<float*>(aligned_alloc(64, n * sizeof(float)));
        std::memcpy(A_ser, base_data.A, n * n * sizeof(float));
        std::memcpy(b_ser, base_data.b, n * sizeof(float));
        auto t1 = std::chrono::high_resolution_clock::now();
        gauss_serial(A_ser, b_ser, n);
        auto t2 = std::chrono::high_resolution_clock::now();
        double time_ser = std::chrono::duration<double, std::milli>(t2 - t1).count();
        float err_ser = compute_max_error(b_ser, base_data.x_true, n);
        std::cout << "Serial (baseline) time = " << time_ser << " ms, error = " << std::scientific << err_ser << "\n";
        free(A_ser); free(b_ser);
        std::cout << "\n";
        
        for (int thr : thread_counts) {
            std::cout << "--- Threads = " << thr << " ---\n";
            
            // Pthread block
            float* A_pth = static_cast<float*>(aligned_alloc(64, n * n * sizeof(float)));
            float* b_pth = static_cast<float*>(aligned_alloc(64, n * sizeof(float)));
            std::memcpy(A_pth, base_data.A, n * n * sizeof(float));
            std::memcpy(b_pth, base_data.b, n * sizeof(float));
            t1 = std::chrono::high_resolution_clock::now();
            gauss_pthread_block(A_pth, b_pth, n, thr);
            t2 = std::chrono::high_resolution_clock::now();
            double time_pth = std::chrono::duration<double, std::milli>(t2 - t1).count();
            float err_pth = compute_max_error(b_pth, base_data.x_true, n);
            double speedup_pth = time_ser / time_pth;
            std::cout << "  Pthread block    : " << time_pth << " ms, error = " << std::scientific << err_pth
                      << ", speedup = " << std::fixed << std::setprecision(2) << speedup_pth << "x\n";
            free(A_pth); free(b_pth);
            
            // std::thread block
            float* A_std = static_cast<float*>(aligned_alloc(64, n * n * sizeof(float)));
            float* b_std = static_cast<float*>(aligned_alloc(64, n * sizeof(float)));
            std::memcpy(A_std, base_data.A, n * n * sizeof(float));
            std::memcpy(b_std, base_data.b, n * sizeof(float));
            t1 = std::chrono::high_resolution_clock::now();
            gauss_stdthread_block(A_std, b_std, n, thr);
            t2 = std::chrono::high_resolution_clock::now();
            double time_std = std::chrono::duration<double, std::milli>(t2 - t1).count();
            float err_std = compute_max_error(b_std, base_data.x_true, n);
            double speedup_std = time_ser / time_std;
            std::cout << "  std::thread block: " << time_std << " ms, error = " << std::scientific << err_std
                      << ", speedup = " << std::fixed << std::setprecision(2) << speedup_std << "x\n";
            free(A_std); free(b_std);
            
            std::cout << "\n";
        }
        free_testdata(base_data);
        std::cout << "============================================================\n\n";
    }
    return 0;
}