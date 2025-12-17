#define SQRT_IMPLEMENTATION
#include "sqrt.h"

#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <functional>
#include <string>
#include <print>

constexpr int SINGLE_ITERATIONS{ 10'000'000 };
constexpr int BATCH_SIZE {10'000};
constexpr int BATCH_ITERATIONS{ 1'000 };

class Timer {
public:
    void start() {
        m_start = std::chrono::high_resolution_clock::now();
    }

    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - m_start).count();
    }

private:
    std::chrono::high_resolution_clock::time_point m_start{};
};

std::vector<float> generate_floats(int count) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.1f, 1000.0f);
    std::vector<float> data(count);
    for (auto& v : data) v = dist(rng);
    return data;
}

std::vector<double> generate_doubles(int count) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.1, 1000.0);
    std::vector<double> data(count);
    for (auto& v : data) v = dist(rng);
    return data;
}

std::vector<unsigned int> generate_uints(int count) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<unsigned int> dist(1, 1'000'000);
    std::vector<unsigned int> data(count);
    for (auto& v : data) v = dist(rng);
    return data;
}

void print_header(const std::string& title) {
    std::println("\n{}", std::string(60, '='));
    std::println(" {}", title);
    std::println("{}", std::string(60, '='));
}

void print_result(const std::string& name, double time_ms, double ops_per_sec) {
    std::println("{:<30}{:>12.2f} ms{:>15.2e} ops/s", name, time_ms, ops_per_sec);
}

template<typename Func>
double benchmark_single(Func func, const std::vector<float>& data, int iterations) {
    Timer timer;
    volatile float result = 0;
    timer.start();
    for (int i = 0; i < iterations; ++i) {
        result = func(data[i % data.size()]);
    }
    return timer.elapsed_ms();
}

template<typename Func>
double benchmark_single_double(Func func, const std::vector<double>& data, int iterations) {
    Timer timer;
    volatile double result = 0;
    timer.start();
    for (int i = 0; i < iterations; ++i) {
        result = func(data[i % data.size()]);
    }
    return timer.elapsed_ms();
}

template<typename Func>
double benchmark_single_uint(Func func, const std::vector<unsigned int>& data, int iterations) {
    Timer timer;
    volatile unsigned int result = 0;
    timer.start();
    for (int i = 0; i < iterations; ++i) {
        result = func(data[i % data.size()]);
    }
    return timer.elapsed_ms();
}

template<typename Func>
double benchmark_batch_float(Func func, const std::vector<float>& input, std::vector<float>& output, int iterations) {
    Timer timer;
    timer.start();
    for (int i = 0; i < iterations; ++i) {
        func(input.data(), output.data(), static_cast<int>(input.size()));
    }
    return timer.elapsed_ms();
}

template<typename Func>
double benchmark_batch_double(Func func, const std::vector<double>& input, std::vector<double>& output, int iterations) {
    Timer timer;
    timer.start();
    for (int i = 0; i < iterations; ++i) {
        func(input.data(), output.data(), static_cast<int>(input.size()));
    }
    return timer.elapsed_ms();
}

void verify_accuracy() {
    print_header("ACCURACY VERIFICATION (value = 25.0)");
    
    float f_val = 25.0f;
    double d_val = 25.0;
    unsigned int u_val = 25;
    
    std::println("std::sqrt(float):      {:.10f}", std::sqrt(f_val));
    std::println("std::sqrt(double):     {:.10f}\n", std::sqrt(d_val));
    
    std::println("hw_sqrt_ss:            {:.10f}", hw_sqrt_ss(f_val));
    std::println("hw_sqrt_sd:            {:.10f}", hw_sqrt_sd(d_val));
    std::println("hw_sqrt_approx_ss:     {:.10f}\n", hw_sqrt_approx_ss(f_val));
    
    std::println("sf_sqrt:               {:.10f}", sf_sqrt(f_val));
    std::println("sf_sqrt_integer:       {}", sf_sqrt_integer(u_val));
    std::println("sf_sqrt_approx:        {:.10f}", sf_sqrt_approx(f_val));
}

void benchmark_single_float() {
    print_header(std::format("SINGLE FLOAT BENCHMARKS ({} iterations)", SINGLE_ITERATIONS));
    
    auto data = generate_floats(1000);
    double time_ms, ops;
    
    time_ms = benchmark_single([](float x) { return std::sqrt(x); }, data, SINGLE_ITERATIONS);
    ops = SINGLE_ITERATIONS / (time_ms / 1000.0);
    print_result("std::sqrt (float)", time_ms, ops);
    
    time_ms = benchmark_single(hw_sqrt_ss, data, SINGLE_ITERATIONS);
    ops = SINGLE_ITERATIONS / (time_ms / 1000.0);
    print_result("hw_sqrt_ss (AVX)", time_ms, ops);
    
    time_ms = benchmark_single(hw_sqrt_approx_ss, data, SINGLE_ITERATIONS);
    ops = SINGLE_ITERATIONS / (time_ms / 1000.0);
    print_result("hw_sqrt_approx_ss (AVX)", time_ms, ops);
    
    time_ms = benchmark_single(sf_sqrt, data, SINGLE_ITERATIONS);
    ops = SINGLE_ITERATIONS / (time_ms / 1000.0);
    print_result("sf_sqrt (Newton-Raphson)", time_ms, ops);
    
    time_ms = benchmark_single(sf_sqrt_approx, data, SINGLE_ITERATIONS);
    ops = SINGLE_ITERATIONS / (time_ms / 1000.0);
    print_result("sf_sqrt_approx (Quake III)", time_ms, ops);
}

void benchmark_single_double() {
    print_header(std::format("SINGLE DOUBLE BENCHMARKS ({} iterations)", SINGLE_ITERATIONS));
    
    auto data = generate_doubles(1000);
    double time_ms, ops;
    
    time_ms = benchmark_single_double([](double x) { return std::sqrt(x); }, data, SINGLE_ITERATIONS);
    ops = SINGLE_ITERATIONS / (time_ms / 1000.0);
    print_result("std::sqrt (double)", time_ms, ops);
    
    time_ms = benchmark_single_double(hw_sqrt_sd, data, SINGLE_ITERATIONS);
    ops = SINGLE_ITERATIONS / (time_ms / 1000.0);
    print_result("hw_sqrt_sd (AVX)", time_ms, ops);
}

void benchmark_single_integer() {
    print_header(std::format("INTEGER SQRT BENCHMARK ({} iterations)", SINGLE_ITERATIONS));
    
    auto data = generate_uints(1000);
    double time_ms, ops;
    
    time_ms = benchmark_single_uint(sf_sqrt_integer, data, SINGLE_ITERATIONS);
    ops = SINGLE_ITERATIONS / (time_ms / 1000.0);
    print_result("sf_sqrt_integer (bit magic)", time_ms, ops);
}

void benchmark_batch_float() {
    print_header(std::format("BATCH FLOAT BENCHMARKS ({} elements x {} iterations)", BATCH_SIZE, BATCH_ITERATIONS));
    
    auto input = generate_floats(BATCH_SIZE);
    std::vector<float> output(BATCH_SIZE);
    double time_ms, ops;
    long long total_ops = static_cast<long long>(BATCH_SIZE) * BATCH_ITERATIONS;
    
    Timer timer;
    timer.start();
    for (int iter = 0; iter < BATCH_ITERATIONS; ++iter) {
        for (int i = 0; i < BATCH_SIZE; ++i) {
            output[i] = std::sqrt(input[i]);
        }
    }
    time_ms = timer.elapsed_ms();
    ops = total_ops / (time_ms / 1000.0);
    print_result("std::sqrt loop", time_ms, ops);
    
    time_ms = benchmark_batch_float(hw_sqrt_ps, input, output, BATCH_ITERATIONS);
    ops = total_ops / (time_ms / 1000.0);
    print_result("hw_sqrt_ps (AVX)", time_ms, ops);
    
    time_ms = benchmark_batch_float(hw_sqrt_approx_ps, input, output, BATCH_ITERATIONS);
    ops = total_ops / (time_ms / 1000.0);
    print_result("hw_sqrt_approx_ps (AVX rsqrt)", time_ms, ops);
}

void benchmark_batch_double() {
    print_header(std::format("BATCH DOUBLE BENCHMARKS ({} elements x {} iterations)", BATCH_SIZE, BATCH_ITERATIONS));
    
    auto input = generate_doubles(BATCH_SIZE);
    std::vector<double> output(BATCH_SIZE);
    double time_ms, ops;
    long long total_ops = static_cast<long long>(BATCH_SIZE) * BATCH_ITERATIONS;
    
    Timer timer;
    timer.start();
    for (int iter = 0; iter < BATCH_ITERATIONS; ++iter) {
        for (int i = 0; i < BATCH_SIZE; ++i) {
            output[i] = std::sqrt(input[i]);
        }
    }
    time_ms = timer.elapsed_ms();
    ops = total_ops / (time_ms / 1000.0);
    print_result("std::sqrt loop", time_ms, ops);
    
    time_ms = benchmark_batch_double(hw_sqrt_pd, input, output, BATCH_ITERATIONS);
    ops = total_ops / (time_ms / 1000.0);
    print_result("hw_sqrt_pd (AVX)", time_ms, ops);
}

void benchmark_cpp_api() {
    print_header("C++ API BENCHMARKS");
    
    auto input_f = generate_floats(BATCH_SIZE);
    auto input_d = generate_doubles(BATCH_SIZE);
    std::vector<float> output_f(BATCH_SIZE);
    std::vector<double> output_d(BATCH_SIZE);
    double time_ms, ops;
    long long total_ops = static_cast<long long>(BATCH_SIZE) * BATCH_ITERATIONS;
    
    Timer timer;
    timer.start();
    for (int i = 0; i < BATCH_ITERATIONS; ++i) {
        gg_sqrt::hw::sqrt(std::span<const float>(input_f), std::span<float>(output_f));
    }
    time_ms = timer.elapsed_ms();
    ops = total_ops / (time_ms / 1000.0);
    print_result("gg_sqrt::hw::sqrt (float span)", time_ms, ops);
    
    timer.start();
    for (int i = 0; i < BATCH_ITERATIONS; ++i) {
        gg_sqrt::hw::sqrt(std::span<const double>(input_d), std::span<double>(output_d));
    }
    time_ms = timer.elapsed_ms();
    ops = total_ops / (time_ms / 1000.0);
    print_result("gg_sqrt::hw::sqrt (double span)", time_ms, ops);
    
    timer.start();
    for (int i = 0; i < BATCH_ITERATIONS; ++i) {
        gg_sqrt::hw::sqrt_inplace(std::span<float>(output_f));
    }
    time_ms = timer.elapsed_ms();
    ops = total_ops / (time_ms / 1000.0);
    print_result("gg_sqrt::hw::sqrt_inplace", time_ms, ops);
}

int main() {
    std::println("SQRT Library Performance Benchmark");
    std::println("===================================");
    
    verify_accuracy();
    
    benchmark_single_float();
    benchmark_single_double();
    benchmark_single_integer();
    
    benchmark_batch_float();
    benchmark_batch_double();
    
    benchmark_cpp_api();
    
    std::println("\n{}", std::string(60, '='));
    std::println(" Benchmark complete!");
    std::println("{}", std::string(60, '='));
    
    return 0;
}
