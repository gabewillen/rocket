// SPDX-License-Identifier: Apache-2.0
#include "linear_attention/gdn_flashinfer_wheel.h"

#include <dlfcn.h>
#include <fcntl.h>
#include <spawn.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>

#include "linear_attention/gdn_flashinfer_cutlass.h"

extern char** environ;

namespace rocket::qwen38::linear_attention {
namespace {

using WheelRunner = std::size_t (*)(
    void*, const void*, const void*, const void*, const void*, const float*,
    int, int, int, int, char*, std::size_t, cudaStream_t, const char*);

void cuda_check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

class ScopedFd final {
 public:
  explicit ScopedFd(int fd) : fd_(fd) {}
  ~ScopedFd() {
    if (fd_ >= 0) close(fd_);
  }
  int get() const { return fd_; }

 private:
  int fd_;
};

std::string sha256sum(int fd) {
  int output[2];
  if (pipe(output) != 0) throw std::runtime_error("create sha256sum pipe");
  posix_spawn_file_actions_t actions;
  if (posix_spawn_file_actions_init(&actions) != 0) {
    close(output[0]);
    close(output[1]);
    throw std::runtime_error("initialize sha256sum process");
  }
  posix_spawn_file_actions_adddup2(&actions, output[1], STDOUT_FILENO);
  posix_spawn_file_actions_adddup2(&actions, fd, STDIN_FILENO);
  posix_spawn_file_actions_addclose(&actions, output[0]);
  posix_spawn_file_actions_addclose(&actions, output[1]);
  std::array<char*, 3> arguments{const_cast<char*>("/usr/bin/sha256sum"),
                                 const_cast<char*>("-"), nullptr};
  pid_t child = -1;
  const int spawn_status = posix_spawn(&child, arguments[0], &actions, nullptr,
                                       arguments.data(), environ);
  posix_spawn_file_actions_destroy(&actions);
  close(output[1]);
  if (spawn_status != 0) {
    close(output[0]);
    throw std::runtime_error("launch sha256sum: " +
                             std::string(std::strerror(spawn_status)));
  }
  std::array<char, 256> buffer{};
  const ssize_t count = read(output[0], buffer.data(), buffer.size() - 1);
  close(output[0]);
  int status = 0;
  if (waitpid(child, &status, 0) != child || !WIFEXITED(status) ||
      WEXITSTATUS(status) != 0 || count < 64) {
    throw std::runtime_error("hash FlashInfer wheel shared object");
  }
  return std::string(buffer.data(), 64);
}

bool same_file(int expected_fd, const char* path) {
  struct stat expected {};
  struct stat observed {};
  return fstat(expected_fd, &expected) == 0 && stat(path, &observed) == 0 &&
         S_ISREG(expected.st_mode) && expected.st_dev == observed.st_dev &&
         expected.st_ino == observed.st_ino;
}

}  // namespace

std::string gdn_sha256_file(std::string_view selected_path) {
  const std::string path(selected_path);
  ScopedFd artifact(open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
  if (artifact.get() < 0) {
    throw std::runtime_error("open hash input: " +
                             std::string(std::strerror(errno)));
  }
  return sha256sum(artifact.get());
}

struct GdnFlashInferWheelGemm::Impl {
  int m = 0;
  int n = 0;
  int k = 0;
  const std::uint8_t* packed_a = nullptr;
  const std::uint8_t* sfa = nullptr;
  const std::uint8_t* packed_b = nullptr;
  const std::uint8_t* sfb = nullptr;
  const float* alpha = nullptr;
  __nv_bfloat16* output = nullptr;
  char* workspace = nullptr;
  std::size_t workspace_bytes = 0;
  void* library = nullptr;
  WheelRunner runner = nullptr;
};

GdnFlashInferWheelGemm::GdnFlashInferWheelGemm() : impl_(new Impl) {}

GdnFlashInferWheelGemm::~GdnFlashInferWheelGemm() {
  if (impl_) {
    cudaFree(impl_->workspace);
    if (impl_->library) dlclose(impl_->library);
  }
  delete impl_;
}

void GdnFlashInferWheelGemm::init(
    std::string_view shared_object, int m, int n, int k,
    const std::uint8_t* packed_a, const std::uint8_t* sfa,
    const std::uint8_t* packed_b, const std::uint8_t* sfb, const float* alpha,
    __nv_bfloat16* output) {
  if (!impl_ || shared_object.empty() || m <= 0 || n <= 0 || k <= 0 ||
      !packed_a || !sfa || !packed_b || !sfb || !alpha || !output ||
      impl_->library) {
    throw std::invalid_argument("FlashInfer wheel GDN contract changed");
  }
  const std::string path(shared_object);
  ScopedFd artifact(open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
  if (artifact.get() < 0) {
    throw std::runtime_error("open FlashInfer wheel shared object: " +
                             std::string(std::strerror(errno)));
  }
  if (sha256sum(artifact.get()) != kGdnFlashInferWheelSha256) {
    throw std::runtime_error("FlashInfer wheel shared object identity mismatch");
  }
  const char* symbol = gdn_flashinfer_cutlass_fallback_symbol();
  const std::string fd_path = "/proc/self/fd/" +
                              std::to_string(artifact.get());
  // The wheel's unused outer registration layer leaves TVM-FFI functions
  // unresolved. Lazy binding keeps that layer out while the hash-pinned raw
  // runner and its CUDA dependencies are resolved through this local handle.
  impl_->library = dlopen(fd_path.c_str(), RTLD_LAZY | RTLD_LOCAL);
  if (!impl_->library) {
    throw std::runtime_error(std::string("load FlashInfer wheel shared object: ") +
                             dlerror());
  }
  dlerror();
  impl_->runner = reinterpret_cast<WheelRunner>(dlsym(impl_->library, symbol));
  const char* symbol_error = dlerror();
  if (symbol_error || !impl_->runner) {
    throw std::runtime_error("FlashInfer wheel fallback runner ABI mismatch");
  }
  Dl_info loaded{};
  if (dladdr(reinterpret_cast<const void*>(impl_->runner), &loaded) == 0 ||
      !loaded.dli_fname || !same_file(artifact.get(), loaded.dli_fname)) {
    throw std::runtime_error("FlashInfer wheel fallback symbol interposed");
  }
  impl_->m = m;
  impl_->n = n;
  impl_->k = k;
  impl_->packed_a = packed_a;
  impl_->sfa = sfa;
  impl_->packed_b = packed_b;
  impl_->sfb = sfb;
  impl_->alpha = alpha;
  impl_->output = output;
  impl_->workspace_bytes = impl_->runner(
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, m, n, k, 1,
      nullptr, 0, nullptr, "");
  if (impl_->workspace_bytes != 0) {
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&impl_->workspace),
                          impl_->workspace_bytes),
               "malloc FlashInfer wheel GDN workspace");
  }
}

void GdnFlashInferWheelGemm::run(cudaStream_t stream) {
  if (!impl_ || !impl_->runner || !impl_->packed_a || !stream) {
    throw std::invalid_argument("FlashInfer wheel GDN launch changed");
  }
  impl_->runner(impl_->output, impl_->packed_a, impl_->packed_b, impl_->sfa,
                impl_->sfb, impl_->alpha, impl_->m, impl_->n, impl_->k, 1,
                impl_->workspace, impl_->workspace_bytes, stream, "");
}

}  // namespace rocket::qwen38::linear_attention
