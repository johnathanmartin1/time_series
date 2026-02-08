#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <iostream>
#include <array>
#include <bitset>
#include <immintrin.h> // For AVX intrinsics (optional)
#include "avx_check.h"

namespace py = pybind11;


std::vector<double> vector_slice(const std::vector<double>& vec, int begin, int ending)
{
    auto start = vec.begin() + begin;
    auto end = vec.begin() + ending;
    std::vector<double> result(end - start);
    copy(start, end, result.begin());
    return result;
}


std::vector<double> corr_128(const std::vector<double>& vec, const double ave, const int lags)
{

    /*Calculating the denominator for all lags*/
    std::vector<double> denominator(vec.size() + 100);
    for (int i = 0; i < vec.size(); i+=2)
    {
        __m128d vec1 = _mm_loadu_pd(vec.data() + i);
        __m128d av = _mm_set1_pd(ave);

        __m128d subtraction = _mm_sub_pd(vec1, av);
        __m128d res = _mm_mul_pd(subtraction, subtraction);

        _mm_storeu_pd(denominator.data() + i, res);
    }

    denominator = vector_slice(denominator, 0, vec.size());
    double ave_sum{};
    ave_sum = std::accumulate(denominator.begin(), denominator.end(), 0.0);

    /*calculating the numerators for a given number of lags then dividing by
    denominator and pushing to the correlation vector*/
    std::vector<double> correlation(lags);
    correlation[0] = 1;
    
    std::vector<double> results(vec.size() + 100);
    for (int i = 0; i < vec.size(); i += 2)
    {

        __m128d vec1 = _mm_loadu_pd(vec.data() + i);
        __m128d av = _mm_set1_pd(ave);
        __m128d res = _mm_sub_pd(vec1, av);

        _mm_storeu_pd(results.data() + i, res);
    }
    results = vector_slice(results, 0, vec.size());

    for (int j = 1; j < lags; j++)
    {
        std::vector<double> a = vector_slice(results, j, results.size());

        std::vector<double> b = vector_slice(results, 0, results.size() - j);
        for (int i = 0; i < 100; i++) {
            a.push_back(0);
            b.push_back(0);
        }
        std::vector<double> numerator(a.size() + 100);

        for (int i = 0; i < a.size(); i += 2)
        {
            __m128d vec1 = _mm_loadu_pd(a.data() + i);
            __m128d vec2 = _mm_loadu_pd(b.data() + i);
            __m128d mul = _mm_mul_pd(vec1, vec2);

            _mm_storeu_pd(numerator.data() + i, mul);
        }
        
        numerator = vector_slice(numerator, 0, results.size() - j);
        correlation[j] = std::accumulate(numerator.begin(), numerator.end(), 0.0) / ave_sum;

    }
    return  correlation;
}


std::vector<double> corr_256(const std::vector<double>& vec, const double ave, const int lags)
{

    /*Calculating the denominator for all lags*/
    std::vector<double> denominator(vec.size() + 100);
    for (int i = 0; i < vec.size(); i+=4)
    {
        __m256d vec1 = _mm256_loadu_pd(vec.data() + i);
        __m256d av = _mm256_set1_pd(ave);

        __m256d subtraction = _mm256_sub_pd(vec1, av);
        __m256d res = _mm256_mul_pd(subtraction, subtraction);

        _mm256_storeu_pd(denominator.data() + i, res);
    }

    denominator = vector_slice(denominator, 0, vec.size());
    double ave_sum{};
    ave_sum = std::accumulate(denominator.begin(), denominator.end(), 0.0);

    /*calculating the numerators for a given number of lags then dividing by
    denominator and pushing to the correlation vector*/
    std::vector<double> correlation(lags);
    correlation[0] = 1;
    
    std::vector<double> results(vec.size() + 100);
    for (int i = 0; i < vec.size(); i += 4)
    {

        __m256d vec1 = _mm256_loadu_pd(vec.data() + i);
        __m256d av = _mm256_set1_pd(ave);
        __m256d res = _mm256_sub_pd(vec1, av);

        _mm256_storeu_pd(results.data() + i, res);
    }
    results = vector_slice(results, 0, vec.size());

    for (int j = 1; j < lags; j++)
    {
        std::vector<double> a = vector_slice(results, j, results.size());

        std::vector<double> b = vector_slice(results, 0, results.size() - j);
        for (int i = 0; i < 100; i++) {
            a.push_back(0);
            b.push_back(0);
        }
        std::vector<double> numerator(a.size() + 100);

        for (int i = 0; i < a.size(); i += 4)
        {
            __m256d vec1 = _mm256_loadu_pd(a.data() + i);
            __m256d vec2 = _mm256_loadu_pd(b.data() + i);
            __m256d mul = _mm256_mul_pd(vec1, vec2);

            _mm256_storeu_pd(numerator.data() + i, mul);
        }
        
        numerator = vector_slice(numerator, 0, results.size() - j);
        
        correlation[j] = std::accumulate(numerator.begin(), numerator.end(), 0.0) / ave_sum;

    }
    return  correlation;
}


std::vector<double> acf_cpp_backend_calc(const std::vector<double>& lst, int lags) 
{
    double average{};
    average = std::accumulate(lst.begin(), lst.end(), 0.0) / lst.size();
    std::vector<double> correlation(lags);
    int avx_type = avx_support();
    if (avx_type == 2 || avx_type == 1)
    {
        correlation = corr_256(lst, average, lags);
    }
    else
    {
        correlation = corr_128(lst, average, lags);
    }
    return correlation;
}


PYBIND11_MODULE(acf_cpp_backend, m, py::mod_gil_not_used()) {
    m.doc() = "cpp backend for acf function. can be passed python lists or numpy arrays both ne dimensional only. The function will check the processor for the avx instructions sets avx2 or avx and choose avx2 ove avx.";
    m.def("acf_cpp_backend_calc", &acf_cpp_backend_calc, "calculates the acf of a series can be passed lists or numpy arrays");
}





