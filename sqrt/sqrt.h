#ifndef SQRT_INCLUDE_H
#define SQRT_INCLUDE_H

#define SQRT_VERSION_MAJOR 1
#define SQRT_VERSION_MINOR 0
#define SQRT_VERSION_PATCH 0

#ifdef SQRT_STATIC
    #define SQRT_DEF static
#else
    #define SQRT_DEF extern
#endif

#if defined(_MSC_VER)
#define SQRT_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define SQRT_INLINE inline __attribute__((always_inline))
#else
#define SQRT_INLINE inline
#endif


/* ========================================================================= */
/* C API DECLARATIONS                                                        */
/* ========================================================================= */

#ifdef __cplusplus
extern "C" {
#endif

    SQRT_INLINE SQRT_DEF float hw_sqrt_ss(float x);
    SQRT_INLINE SQRT_DEF double hw_sqrt_sd(double x);

    SQRT_INLINE SQRT_DEF void hw_sqrt_ps(const float* input, float* output, int count);
    SQRT_INLINE SQRT_DEF void hw_sqrt_pd(const double* input, double* output, int count);

    SQRT_INLINE SQRT_DEF float hw_sqrt_approx_ss(float x);
    SQRT_INLINE SQRT_DEF void hw_sqrt_approx_ps(const float* input, float* output, int count);

    SQRT_INLINE SQRT_DEF float sf_sqrt(float number);
    SQRT_INLINE SQRT_DEF unsigned int sf_sqrt_integer(unsigned int n);
    SQRT_INLINE SQRT_DEF float sf_sqrt_approx(float number);

#ifdef __cplusplus
}
#endif

/* ========================================================================= */
/* C++ API DECLARATIONS                                                      */
/* ========================================================================= */

#ifdef __cplusplus

#include <span>
#include <cstddef>

namespace gg_sqrt {
    namespace hw {
        inline float sqrt(float x) { return hw_sqrt_ss(x); }
        inline double sqrt(double x) { return hw_sqrt_sd(x); }

        inline float sqrt_approx(float x) { return hw_sqrt_approx_ss(x); }

        inline void sqrt(std::span<const float> input, std::span<float> output) {
            hw_sqrt_ps(input.data(), output.data(), static_cast<int>(input.size()));
        }

        inline void sqrt(std::span<const double> input, std::span<double> output) {
            hw_sqrt_pd(input.data(), output.data(), static_cast<int>(input.size()));
        }

        inline void sqrt_approx(std::span<const float> input, std::span<float> output) {
            hw_sqrt_approx_ps(input.data(), output.data(), static_cast<int>(input.size()));
        }

        inline void sqrt_inplace(std::span<float> data) {
            hw_sqrt_ps(data.data(), data.data(), static_cast<int>(data.size()));
        }

        inline void sqrt_inplace(std::span<double> data) {
            hw_sqrt_pd(data.data(), data.data(), static_cast<int>(data.size()));
        }

        inline void sqrt_approx_inplace(std::span<float> data) {
            hw_sqrt_approx_ps(data.data(), data.data(), static_cast<int>(data.size()));
        }
    }

    namespace sf {
        inline float sqrt(float number) { return sf_sqrt(number); }
        inline float sqrt_integer(unsigned int n) { return sf_sqrt_integer(n); }
		inline float sqrt_approx(float number) { return sf_sqrt_approx(number); }
    }

} /* namespace gg_sqrt */

#endif /* __cplusplus */

#endif /* SQRT_INCLUDE_H */

/* ========================================================================= */
/* IMPLEMENTATION                                                            */
/* ========================================================================= */

#ifdef SQRT_IMPLEMENTATION

#include <immintrin.h>

#ifdef __cplusplus
extern "C" {
#endif

    SQRT_INLINE SQRT_DEF float hw_sqrt_ss(float x) {
        __m128 in = _mm_set_ss(x);
        __m128 out = _mm_sqrt_ss(in);
        return _mm_cvtss_f32(out);
    }

    SQRT_INLINE SQRT_DEF double hw_sqrt_sd(double x) {
        __m128d in = _mm_set_sd(x);
        __m128d out = _mm_sqrt_sd(in, in);
        return _mm_cvtsd_f64(out);
    }

    SQRT_INLINE SQRT_DEF void hw_sqrt_ps(const float* input, float* output, int count) {
        int i = 0;
        for (; i + 8 <= count; i += 8) {
            __m256 in_vec = _mm256_loadu_ps(&input[i]);
            __m256 out_vec = _mm256_sqrt_ps(in_vec);
            _mm256_storeu_ps(&output[i], out_vec);
        }

        for (; i < count; ++i) {
            output[i] = hw_sqrt_ss(input[i]);
        }
    }

    SQRT_INLINE SQRT_DEF void hw_sqrt_pd(const double* input, double* output, int count) {
        int i = 0;
        for (; i + 4 <= count; i += 4) {
            __m256d in_vec = _mm256_loadu_pd(&input[i]);
            __m256d out_vec = _mm256_sqrt_pd(in_vec);
            _mm256_storeu_pd(&output[i], out_vec);
        }

        for (; i < count; ++i) {
            output[i] = hw_sqrt_sd(input[i]);
        }
    }

    SQRT_INLINE SQRT_DEF float hw_sqrt_approx_ss(float x) {
        __m128 in = _mm_set_ss(x);
        __m128 rsqrt = _mm_rsqrt_ss(in);
        __m128 out = _mm_mul_ss(in, rsqrt);
        return _mm_cvtss_f32(out);
    }

    SQRT_INLINE SQRT_DEF void hw_sqrt_approx_ps(const float* input, float* output, int count) {
        int i = 0;
        for (; i + 8 <= count; i += 8) {
            __m256 in_vec = _mm256_loadu_ps(&input[i]);
            __m256 rsqrt = _mm256_rsqrt_ps(in_vec);
            __m256 out_vec = _mm256_mul_ps(in_vec, rsqrt);
            _mm256_storeu_ps(&output[i], out_vec);
        }

        for (; i < count; ++i) {
            output[i] = hw_sqrt_approx_ss(input[i]);
        }
    }


    SQRT_INLINE SQRT_DEF float sf_sqrt(float number) {
        if (number < 0) return -1.0f;
        if (number == 0) return 0.0f;

        float x = number * 0.5f;

        x = 0.5f * (x + number / x);
        x = 0.5f * (x + number / x);
        x = 0.5f * (x + number / x);
        x = 0.5f * (x + number / x);
        x = 0.5f * (x + number / x);

        return x;
    }

    SQRT_INLINE SQRT_DEF unsigned int sf_sqrt_integer(unsigned int n) {
        if (n == 0) return 0;

        unsigned int res = 0;
        unsigned int bit = 1 << 30;

        while (bit > n)
            bit >>= 2;

        while (bit != 0) {
            if (n >= res + bit) {
                n -= res + bit;
                res = (res >> 1) + bit;
            }
            else {
                res >>= 1;
            }
            bit >>= 2;
        }
        return res;
    }

    SQRT_INLINE SQRT_DEF float sf_sqrt_approx(float number) {
        const float xhalf = number * 0.5f;

        union {
            float f;
            int32_t i;
        } conv;

        conv.f = number;
        conv.i = 0x5f3759df - (conv.i >> 1);
        float y = conv.f;

        y = y * (1.5f - (xhalf * y * y));

        return number * y;
    }

#ifdef __cplusplus
}
#endif

#endif /* SQRT_IMPLEMENTATION */