#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

source = (Path(__file__).parents[1] /
          "src/moe/target_moe_n640_device_stage.cu").read_text()
enqueue = source.split("TargetMoeOutcome TargetMoeN640DeviceStage::enqueue", 1)[1]
enqueue = enqueue.split("TargetMoeB12xWeights target_moe_staged_weights", 1)[0]
for forbidden in ("cudaMalloc", "cudaMemcpy", "cudaStreamSynchronize",
                  "cudaDeviceSynchronize", "cudaMemcpyDeviceToHost",
                  "torch", "Python"):
    assert forbidden not in enqueue, forbidden
for required in ("select_experts<<<", "stage_packed<<<", "clear_scales<<<",
                 "stage_scales_and_scalars<<<", "cudaPeekAtLastError"):
    assert required in enqueue, required

header = (Path(__file__).parents[1] /
          "src/moe/target_moe_n640_device_stage.h").read_text()
assert "kTargetMoeStagedExperts = kTargetMoeC1TopK" in header
assert "counter(4) x outcome(4) x rank(2) x layer(48) = 1,536" in header
assert "source_wait_enqueued_" in header
assert "launch.stream != source_stream_" in enqueue
assert "if (source_wait_enqueued_ || !stream" in source
assert "cudaFree(device_experts_)" in source
print("target_moe_n640_device_stage graph_safe=1 compact_experts=10")
