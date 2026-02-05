#include <pybind11/pybind11.h>
#include "avx_check.h"

namespace py = pybind11;

void bind_avx_check(py::module_& m) {
	m.def("avx_check", &avx_support, "checks the processor for avx support");
}