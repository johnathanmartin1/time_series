#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <numeric>
#include <iostream>
#include <array>
#include <bitset>
#include <algorithm>
#include <immintrin.h> // For AVX intrinsics (optional)
#include "avx_check.h" // For checking if AVX instruction sets are avaliable

namespace py = pybind11;

std::vector<double> vector_slice(const std::vector<double>& vec, int begin, int ending)
{
    auto start = vec.begin() + begin;
    auto end = vec.begin() + ending;
    std::vector<double> result(end - start);
    copy(start, end, result.begin());
    return result;
}


std::vector<double> pcorr_128(const std::vector<double>& acf, const int lags)
{
    std::vector<double> pcorr(acf.size());

    pcorr[0] = 1;
    pcorr[1] = acf[1];
    for (int lag = 2; lag < lags; lag++)
    {
        std::vector<double> acf_resize(acf.begin(), acf.begin()+lag+1);
        std::vector<double> pcorr_resize(pcorr.begin(), pcorr.begin()+lag+1);
        std::vector<double> acf_rev(acf.begin(), acf.begin()+lag+1);
        std::reverse(acf_rev.begin(), acf_rev.end());
        
        
        std::vector<double> mul_result(acf.size()+100);

        for (int i = 0; i < lags; i += 2)
        {
            __m128d pacf = _mm_loadu_pd(pcorr_resize.data() + i);
            __m128d acf1 = _mm_loadu_pd(acf_resize.data() + i);
            __m128d mul = _mm_mul_pd(pacf, acf1);
            _mm_storeu_pd(mul_result.data() + i, mul);
        }
        mul_result = vector_slice(mul_result, 1, lag);
        double pacfn_denominator =  1 - std::accumulate(mul_result.begin(), mul_result.end(), 0.0);
        

        std::vector<double> rev_mul_result(acf.size()+100);
       
        for (int i = 0; i < lags; i += 2)
        {
            __m128d pacf = _mm_loadu_pd(pcorr_resize.data() + i);
            __m128d rev_acf = _mm_load_pd(acf_rev.data() + i);
            __m128d mul = _mm_mul_pd(pacf, rev_acf);
            _mm_storeu_pd(rev_mul_result.data() + i, mul);
        }
        rev_mul_result = vector_slice(rev_mul_result, 1, lag);
        double pacfn_numerator = acf[lag] - std::accumulate(rev_mul_result.begin(), rev_mul_result.end(), 0.0);
        pcorr[lag] = pacfn_numerator / pacfn_denominator;
       
        
    }
    pcorr = vector_slice(pcorr, 0, 51);
    
    return pcorr;
}



std::vector<double> pcorr_256(const std::vector<double>& acf, const int lags)
{
    std::vector<double> pcorr(acf.size());

    pcorr[0] = 1;
    pcorr[1] = acf[1];
    for (int lag = 2; lag < lags; lag++)
    {
        std::vector<double> acf_resize(acf.begin(), acf.begin() + lag + 1);
        std::vector<double> pcorr_resize(pcorr.begin(), pcorr.begin() + lag + 1);
        std::vector<double> acf_rev(acf.begin(), acf.begin() + lag + 1);
        std::reverse(acf_rev.begin(), acf_rev.end());


        std::vector<double> mul_result(acf.size() + 100);

        for (int i = 0; i < lags; i += 4)
        {
            __m256d pacf = _mm256_loadu_pd(pcorr_resize.data() + i);
            __m256d acf1 = _mm256_loadu_pd(acf_resize.data() + i);
            __m256d mul = _mm256_mul_pd(pacf, acf1);
            _mm256_storeu_pd(mul_result.data() + i, mul);
        }
        mul_result = vector_slice(mul_result, 1, lag);
        double pacfn_denominator = 1 - std::accumulate(mul_result.begin(), mul_result.end(), 0.0);


        std::vector<double> rev_mul_result(acf.size() + 100);

        for (int i = 0; i < lags; i += 4)
        {
            __m256d pacf = _mm256_loadu_pd(pcorr_resize.data() + i);
            __m256d rev_acf = _mm256_load_pd(acf_rev.data() + i);
            __m256d mul = _mm256_mul_pd(pacf, rev_acf);
            _mm256_storeu_pd(rev_mul_result.data() + i, mul);
        }
        rev_mul_result = vector_slice(rev_mul_result, 1, lag);
        double pacfn_numerator = acf[lag] - std::accumulate(rev_mul_result.begin(), rev_mul_result.end(), 0.0);
        pcorr[lag] = pacfn_numerator / pacfn_denominator;


    }
    pcorr = vector_slice(pcorr, 0, 51);

    return pcorr;
}



std::vector<double> pacf_cpp_backend_calc(const std::vector<double>& acf, int lags, std::string avx)
{
    

    int avx_type = avx_support();
    if (avx == "auto")
    {
        if (avx_type == 2 || avx_type == 1)// && avx == "auto") || avx == "avx2")
        {
            return pcorr_256(acf, lags);
        }
        else if (avx_type == 0)// && avx == "auto") || avx == "avx")
        {
            return pcorr_128(acf, lags);
        }
        else
        {
            py::print("Device unable to run AVX or AVX2 SIMD instruction sets. Please use the alternative pacf_py function to calculate the partial auto correlation.");
            std::vector<double> vec(lags, 0);
            return vec;
        }
    }
    else
    {
        if (avx == "avx2")
        {
            return pcorr_256(acf, lags);
        }
        if (avx == "avx")
        {
            return pcorr_128(acf, lags);
        }
        else
        {
            py::print("Please select avx, avx2 or auto for the pacf function or use the alternative pacf_py function to calculate the partial auto correlation.");
            std::vector<double> vec(lags, 0);
            return vec;
        }

    }
}



PYBIND11_MODULE(pacf_cpp_backend, m, py::mod_gil_not_used()) {
    m.doc() = "cpp backend for pacf function. can be passed python lists or numpy arrays both one dimensional only. The function will check the processor for the avx instructions sets avx2 or avx and choose one.";
    m.def("pacf_cpp_backend_calc", &pacf_cpp_backend_calc, "calculates the pacf of a series can be passed lists or numpy arrays");
}

