// Minimal TORCH_CHECK shim for the ported exllamav3 kernels: throws with the
// condition text, matching the throw-on-false contract of the original.
#pragma once
#include <stdexcept>
#include <string>

#define TORCH_CHECK(cond, ...) \
    do { if (!(cond)) { std::string msg = "exl3 check failed: " #cond; \
          throw std::runtime_error(msg); } } while (0)
#define TORCH_CHECK_DTYPE(x, dt) do {} while (0)
#define TORCH_CHECK_DIM(x, d) do {} while (0)
#define TORCH_CHECK_SHAPES(x, xi, y, yi, d) do {} while (0)
#define TORCH_CHECK_SHAPES_FULL(x, y) do {} while (0)
