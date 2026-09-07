"""Qwen3.8 TP2 K0 target-prologue CUDA kernel.

The first executable target node consumes the fixed seven-field QSA metadata
ABI and emits one deterministic descriptor per graph row. It proves the graph
can execute model-specific CUDA work after metadata upload. It does not perform
Q/K/V projection, index selection, sparse attention, or token sampling.

The stable device binding reserves two unsigned 64-bit words for a slab weight
pointer and authenticated chunk generation. Public host configuration keeps
both zero. The CUDA runtime fills its private device copy only after validating
the rank-slab projection payload.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

K0_TARGET_SCHEMA = "qwen3.8-flash-next:tp2:k0-target-prologue:v1"
TARGET_BINDING_BYTES = 16
TARGET_ROW_WORDS = 4
TARGET_ROWS = 16
TARGET_OUTPUT_BYTES = TARGET_ROWS * TARGET_ROW_WORDS * 8
DEFAULT_NVRTC = Path("/usr/local/cuda/lib64/libnvrtc.so")
DEFAULT_CUDA_DRIVER = Path("/lib/aarch64-linux-gnu/libcuda.so.1")

_SOURCE = rb"""
__device__ __forceinline__ float qwen38_pow2(int exponent) {
  const unsigned bits = static_cast<unsigned>(exponent + 127) << 23;
  return __uint_as_float(bits);
}

__device__ __forceinline__ float qwen38_e4m3(unsigned char value) {
  const int sign = value >> 7;
  const int exponent = (value >> 3) & 15;
  const int mantissa = value & 7;
  float result = exponent == 0
      ? static_cast<float>(mantissa) * qwen38_pow2(-9)
      : (1.0f + static_cast<float>(mantissa) * 0.125f) *
            qwen38_pow2(exponent - 7);
  return sign ? -result : result;
}

__device__ __forceinline__ float qwen38_e2m1(unsigned char value) {
  const float magnitude[8] = {0.0f, 0.5f, 1.0f, 1.5f,
                              2.0f, 3.0f, 4.0f, 6.0f};
  const float result = magnitude[value & 7];
  return value & 8 ? -result : result;
}

__device__ __forceinline__ float qwen38_bf16(unsigned short value) {
  return __uint_as_float(static_cast<unsigned>(value) << 16);
}

extern "C" __global__ void qwen38_k0_target_prologue(
    const int* query_start_loc,
    const int* seq_lens,
    const int* stream_slots,
    const int* token_to_req,
    const long long* logical_positions,
    const int* raw_ring_offsets,
    const long long* compressed_positions,
    const unsigned long long* slab_binding,
    long long* output,
    int graph_batch) {
  const int row = int(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= 16) return;
  const int request = row < graph_batch ? token_to_req[row] : -1;
  const bool live = request >= 0 && request < graph_batch &&
                    query_start_loc[request] == row &&
                    query_start_loc[request + 1] == row + 1;
  const int base = row * 4;
  if (!live) {
    output[base] = -1;
    output[base + 1] = -1;
    output[base + 2] = -1;
    output[base + 3] = -1;
    return;
  }
  const long long logical = logical_positions[row];
  (void)slab_binding;
  output[base] = stream_slots[request];
  output[base + 1] = logical;
  output[base + 2] =
      (static_cast<long long>(seq_lens[request]) << 32) |
      static_cast<unsigned int>(raw_ring_offsets[row]);
  output[base + 3] = compressed_positions[row];
}

extern "C" __global__ void qwen38_k0_qkv_projection(
    const long long* target_rows,
    const unsigned char* packed_activations,
    const float* activation_scales,
    const unsigned char* packed_weights,
    const unsigned char* block_scales,
    const float* global_scales,
    float* output) {
  __shared__ float reduction[256];
  const int output_row = blockIdx.x;
  const int batch_row = blockIdx.y;
  float partial = 0.0f;
  if (target_rows[batch_row * 4] >= 0) {
    const unsigned char* packed = packed_weights + output_row * 1280;
    const unsigned char* scales = block_scales + output_row * 160;
    const unsigned char* input = packed_activations + batch_row * 1280;
    const float* input_scales = activation_scales + batch_row * 160;
    for (int column = threadIdx.x; column < 2560; column += 256) {
      const unsigned char input_pair = input[column >> 1];
      const unsigned char input_nibble =
          column & 1 ? static_cast<unsigned char>(input_pair >> 4)
                     : static_cast<unsigned char>(input_pair & 15);
      const unsigned char pair = packed[column >> 1];
      const unsigned char nibble =
          column & 1 ? static_cast<unsigned char>(pair >> 4)
                     : static_cast<unsigned char>(pair & 15);
      partial += qwen38_e2m1(input_nibble) * input_scales[column >> 4] *
                 qwen38_e2m1(nibble) * qwen38_e4m3(scales[column >> 4]);
    }
  }
  reduction[threadIdx.x] = partial;
  __syncthreads();
  for (int stride = 128; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride)
      reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    output[batch_row * 12 + output_row] =
        reduction[0] * global_scales[output_row >> 2];
  }
}

extern "C" __global__ void qwen38_k0_activation_requant(
    const unsigned short* activations,
    unsigned char* packed,
    float* block_scales) {
  __shared__ unsigned char codes[16];
  __shared__ float scale;
  const int group = blockIdx.x;
  const int batch = blockIdx.y;
  const int column = group * 16 + threadIdx.x;
  const float value = qwen38_bf16(activations[batch * 2560 + column]);
  codes[threadIdx.x] = 0;
  __shared__ float magnitudes[16];
  magnitudes[threadIdx.x] = value < 0.0f ? -value : value;
  __syncthreads();
  for (int stride = 8; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride && magnitudes[threadIdx.x + stride] > magnitudes[threadIdx.x])
      magnitudes[threadIdx.x] = magnitudes[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    scale = magnitudes[0] / 6.0f;
    block_scales[batch * 160 + group] = scale;
  }
  __syncthreads();
  const float normalized = scale > 0.0f ? value / scale : 0.0f;
  float best_error = 1.0e30f;
  unsigned char best = 0;
  for (unsigned char code = 0; code < 16; ++code) {
    const float error = normalized - qwen38_e2m1(code);
    const float absolute = error < 0.0f ? -error : error;
    if (absolute < best_error) {
      best_error = absolute;
      best = code;
    }
  }
  codes[threadIdx.x] = best;
  __syncthreads();
  if (threadIdx.x < 8) {
    packed[batch * 1280 + group * 8 + threadIdx.x] =
        codes[threadIdx.x * 2] | (codes[threadIdx.x * 2 + 1] << 4);
  }
}
"""


class K0TargetError(RuntimeError):
    """NVRTC, CUDA Driver, or K0 target ABI failure."""


@dataclass(frozen=True)
class K0TargetBinding:
    """Immutable contents of the stable device-side slab binding."""

    slab_weight_table: int = 0
    slab_weight_generation: int = 0

    def __post_init__(self) -> None:
        for name, value in (
            ("slab_weight_table", self.slab_weight_table),
            ("slab_weight_generation", self.slab_weight_generation),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or value > 0xFFFF_FFFF_FFFF_FFFF
            ):
                raise K0TargetError(f"{name} must be an unsigned 64-bit integer")
        if self.slab_weight_table != 0 or self.slab_weight_generation != 0:
            raise K0TargetError("slab weights are reserved but not bound in prologue v1")


@dataclass(frozen=True)
class K0TargetRow:
    """Owned deterministic output from one target-prologue row."""

    stream_slot: int
    logical_position: int
    seq_len_and_raw_offset: int
    compressed_position: int


class _Nvrtc:
    def __init__(self, library: Path):
        try:
            self.lib = ctypes.CDLL(str(library))
        except OSError as exc:
            raise K0TargetError(f"cannot load NVRTC library: {library}") from exc
        pointer = ctypes.c_void_p
        size = ctypes.c_size_t
        integer = ctypes.c_int
        self._bind(
            "nvrtcCreateProgram",
            [
                ctypes.POINTER(pointer),
                ctypes.c_char_p,
                ctypes.c_char_p,
                integer,
                ctypes.POINTER(ctypes.c_char_p),
                ctypes.POINTER(ctypes.c_char_p),
            ],
        )
        self._bind(
            "nvrtcCompileProgram",
            [pointer, integer, ctypes.POINTER(ctypes.c_char_p)],
        )
        self._bind("nvrtcGetProgramLogSize", [pointer, ctypes.POINTER(size)])
        self._bind("nvrtcGetProgramLog", [pointer, ctypes.c_char_p])
        self._bind("nvrtcGetPTXSize", [pointer, ctypes.POINTER(size)])
        self._bind("nvrtcGetPTX", [pointer, ctypes.c_char_p])
        self._bind("nvrtcDestroyProgram", [ctypes.POINTER(pointer)])
        self.lib.nvrtcGetErrorString.argtypes = [integer]
        self.lib.nvrtcGetErrorString.restype = ctypes.c_char_p

    def _bind(self, name: str, arguments: list[object]) -> None:
        try:
            function = getattr(self.lib, name)
        except AttributeError as exc:
            raise K0TargetError(f"NVRTC symbol is missing: {name}") from exc
        function.argtypes = arguments
        function.restype = ctypes.c_int

    def call(self, name: str, *arguments: object) -> None:
        code = getattr(self.lib, name)(*arguments)
        if code:
            raw = self.lib.nvrtcGetErrorString(code)
            detail = raw.decode("utf-8", "replace") if raw else "unknown"
            raise K0TargetError(f"{name} failed with NVRTC {code}: {detail}")

    def compile(self) -> bytes:
        program = ctypes.c_void_p()
        self.call(
            "nvrtcCreateProgram",
            ctypes.byref(program),
            _SOURCE,
            b"qwen38_k0_target.cu",
            0,
            None,
            None,
        )
        options = (ctypes.c_char_p * 2)(
            b"--gpu-architecture=compute_121", b"--std=c++17"
        )
        try:
            try:
                self.call("nvrtcCompileProgram", program, len(options), options)
            except K0TargetError as exc:
                size = ctypes.c_size_t()
                self.call("nvrtcGetProgramLogSize", program, ctypes.byref(size))
                log = ctypes.create_string_buffer(size.value)
                self.call("nvrtcGetProgramLog", program, log)
                detail = log.value.decode("utf-8", "replace").strip()
                raise K0TargetError(f"K0 target compilation failed: {detail}") from exc
            size = ctypes.c_size_t()
            self.call("nvrtcGetPTXSize", program, ctypes.byref(size))
            ptx = ctypes.create_string_buffer(size.value)
            self.call("nvrtcGetPTX", program, ptx)
            return ptx.raw
        finally:
            self.call("nvrtcDestroyProgram", ctypes.byref(program))


class _Driver:
    def __init__(self, library: Path):
        try:
            self.lib = ctypes.CDLL(str(library))
        except OSError as exc:
            raise K0TargetError(f"cannot load CUDA Driver library: {library}") from exc
        pointer = ctypes.c_void_p
        unsigned = ctypes.c_uint
        self._bind("cuInit", [unsigned])
        self._bind("cuModuleLoadData", [ctypes.POINTER(pointer), pointer])
        self._bind("cuModuleUnload", [pointer])
        self._bind(
            "cuModuleGetFunction",
            [ctypes.POINTER(pointer), pointer, ctypes.c_char_p],
        )
        self._bind(
            "cuLaunchKernel",
            [
                pointer,
                unsigned,
                unsigned,
                unsigned,
                unsigned,
                unsigned,
                unsigned,
                unsigned,
                pointer,
                ctypes.POINTER(pointer),
                ctypes.POINTER(pointer),
            ],
        )
        self.lib.cuGetErrorName.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        self.lib.cuGetErrorName.restype = ctypes.c_int

    def _bind(self, name: str, arguments: list[object]) -> None:
        try:
            function = getattr(self.lib, name)
        except AttributeError as exc:
            raise K0TargetError(f"CUDA Driver symbol is missing: {name}") from exc
        function.argtypes = arguments
        function.restype = ctypes.c_int

    def call(self, name: str, *arguments: object) -> None:
        code = getattr(self.lib, name)(*arguments)
        if code:
            raw = ctypes.c_char_p()
            self.lib.cuGetErrorName(code, ctypes.byref(raw))
            detail = raw.value.decode("utf-8", "replace") if raw.value else "unknown"
            raise K0TargetError(f"{name} failed with CUDA Driver {code}: {detail}")


class K0TargetKernel:
    """Own one JIT-loaded SM121 target prologue for captured graph launches."""

    _METADATA = (
        "query_start_loc",
        "seq_lens",
        "stream_slots",
        "token_to_req",
        "logical_positions",
        "raw_ring_offsets",
        "compressed_positions",
    )

    def __init__(
        self,
        nvrtc: Path = DEFAULT_NVRTC,
        driver: Path = DEFAULT_CUDA_DRIVER,
    ):
        if not isinstance(nvrtc, Path) or not isinstance(driver, Path):
            raise K0TargetError("CUDA libraries must be explicit Path values")
        self._driver = _Driver(driver)
        self._module = ctypes.c_void_p()
        self._function = ctypes.c_void_p()
        self._projection = ctypes.c_void_p()
        self._requant = ctypes.c_void_p()
        self._closed = False
        image = _Nvrtc(nvrtc).compile()
        retained = ctypes.create_string_buffer(image)
        self._driver.call("cuInit", 0)
        self._driver.call(
            "cuModuleLoadData", ctypes.byref(self._module), ctypes.addressof(retained)
        )
        try:
            self._driver.call(
                "cuModuleGetFunction",
                ctypes.byref(self._function),
                self._module,
                b"qwen38_k0_target_prologue",
            )
            self._driver.call(
                "cuModuleGetFunction",
                ctypes.byref(self._projection),
                self._module,
                b"qwen38_k0_qkv_projection",
            )
            self._driver.call(
                "cuModuleGetFunction",
                ctypes.byref(self._requant),
                self._module,
                b"qwen38_k0_activation_requant",
            )
        except BaseException:
            self.close()
            raise

    def capture_launch(
        self,
        metadata: Mapping[str, ctypes.c_void_p],
        slab_binding: ctypes.c_void_p,
        output: ctypes.c_void_p,
        graph_batch: int,
        stream: ctypes.c_void_p,
    ) -> None:
        if self._closed:
            raise K0TargetError("K0 target kernel is closed")
        if graph_batch not in (1, 2, 4, 8, 16):
            raise K0TargetError("K0 target graph batch is invalid")
        values = [ctypes.c_void_p(metadata[name].value) for name in self._METADATA]
        values.extend(
            [ctypes.c_void_p(slab_binding.value), ctypes.c_void_p(output.value)]
        )
        batch = ctypes.c_int(graph_batch)
        parameters = (ctypes.c_void_p * 10)(
            *(ctypes.cast(ctypes.byref(value), ctypes.c_void_p) for value in values),
            ctypes.cast(ctypes.byref(batch), ctypes.c_void_p),
        )
        self._driver.call(
            "cuLaunchKernel",
            self._function,
            1,
            1,
            1,
            32,
            1,
            1,
            0,
            stream,
            parameters,
            None,
        )

    def capture_projection(
        self,
        target_rows: ctypes.c_void_p,
        packed_activations: ctypes.c_void_p,
        activation_scales: ctypes.c_void_p,
        packed_weights: ctypes.c_void_p,
        block_scales: ctypes.c_void_p,
        global_scales: ctypes.c_void_p,
        output: ctypes.c_void_p,
        stream: ctypes.c_void_p,
    ) -> None:
        """Append the fixed layer-3 rank-0 W4A16 Q/K/V projection node."""

        if self._closed:
            raise K0TargetError("K0 target kernel is closed")
        values = [
            ctypes.c_void_p(pointer.value)
            for pointer in (
                target_rows,
                packed_activations,
                activation_scales,
                packed_weights,
                block_scales,
                global_scales,
                output,
            )
        ]
        parameters = (ctypes.c_void_p * len(values))(
            *(ctypes.cast(ctypes.byref(value), ctypes.c_void_p) for value in values)
        )
        self._driver.call(
            "cuLaunchKernel",
            self._projection,
            12,
            16,
            1,
            256,
            1,
            1,
            0,
            stream,
            parameters,
            None,
        )

    def launch_requant(
        self,
        activations: ctypes.c_void_p,
        packed: ctypes.c_void_p,
        block_scales: ctypes.c_void_p,
        stream: ctypes.c_void_p,
    ) -> None:
        """Launch fixed c16 x 2560 activation FP4 quantization and scale generation."""

        if self._closed:
            raise K0TargetError("K0 target kernel is closed")
        values = [ctypes.c_void_p(pointer.value) for pointer in (activations, packed, block_scales)]
        parameters = (ctypes.c_void_p * len(values))(
            *(ctypes.cast(ctypes.byref(value), ctypes.c_void_p) for value in values)
        )
        self._driver.call(
            "cuLaunchKernel",
            self._requant,
            160,
            16,
            1,
            16,
            1,
            1,
            0,
            stream,
            parameters,
            None,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._module.value:
            self._driver.call("cuModuleUnload", self._module)
