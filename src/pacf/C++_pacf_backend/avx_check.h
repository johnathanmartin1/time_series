#ifndef AVXCHECK
#define AVXCHECK


#ifdef _MSC_VER
#include <intrin.h> // MSVC CPUID
#else
#include <cpuid.h>  // GCC/Clang CPUID
#endif


void cpuid1(int regs, int func_id, int sub_func_id);

bool os_supports_avx();

int avx_support();




#endif
