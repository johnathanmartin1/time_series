#include "avx_check.h"



// Cross-platform CPUID wrapper
void cpuid1(int regs[4], int func_id, int sub_func_id = 0) {
#ifdef _MSC_VER
    __cpuidex(regs, func_id, sub_func_id);
#else
    __cpuid_count(func_id, sub_func_id, regs[0], regs[1], regs[2], regs[3]);
#endif
}

// Check if OS supports saving/restoring YMM registers (needed for AVX)
bool os_supports_avx() {
    int regs[4];
    cpuid1(regs, 1);
    bool osxsave = (regs[2] & (1 << 27)) != 0; // OSXSAVE bit
    if (!osxsave) return false;

    // Check XGETBV for XMM/YMM state support
    unsigned int eax, edx;
#ifdef _MSC_VER
    eax = (unsigned int)_xgetbv(0);
    edx = (unsigned int)(_xgetbv(0) >> 32);
#else
    __asm__ volatile (".byte 0x0f, 0x01, 0xd0" : "=a"(eax), "=d"(edx) : "c"(0));
#endif
    return (eax & 0x6) == 0x6; // XMM (bit 1) and YMM (bit 2) state enabled
};

int avx_support() {
    int regs[4];

    // Check AVX support
    cpuid1(regs, 1);
    bool avx_cpu = (regs[2] & (1 << 28)) != 0; // AVX bit in ECX
    bool avx_supported = avx_cpu && os_supports_avx();

    // Check AVX2 support
    cpuid1(regs, 7, 0);
    bool avx2_cpu = (regs[1] & (1 << 5)) != 0; // AVX2 bit in EBX
    bool avx2_supported = avx2_cpu && os_supports_avx();

    if (avx_supported && avx2_supported) { return 2; };
    if (avx2_supported) { return 1; };
    if (avx_supported) { return 0; };
};