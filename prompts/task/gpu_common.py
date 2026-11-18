"""Hardware and evaluator context shared by the GPU tasks."""


ARCH_NOTES = {
    "l40s": """Target hardware: NVIDIA L40S (Ada Lovelace, sm_89, 48 GB GDDR6, no NVLink).
Ada is not Hopper. The following do NOT exist on this device and will fail to
compile or silently fall back:
- TMA / `tl.make_tensor_descriptor` and descriptor-based async copies
- warpgroup MMA (`wgmma`), and anything assuming a 128-thread warpgroup
- thread-block clusters and distributed shared memory
- the 228 KB shared-memory-per-SM budget; Ada gives you 100 KB per SM
What Ada does have and is worth using: 142 SMs' worth of 4th-gen tensor cores
(bf16 and fp8), `cp.async` style pipelining, and a large 96 MB L2. Memory
bandwidth is 864 GB/s, roughly a quarter of an H100's, so this part is far more
bandwidth-bound than the same kernel on Hopper: favour fewer passes over the
data, fusion, and reuse through L2 over strategies that assume you can stream.""",
    "a100": """Target hardware: NVIDIA A100 (Ampere, sm_80). No TMA, no wgmma, no
thread-block clusters. 164 KB shared memory per SM, 1.5-2.0 TB/s HBM.""",
    "h100": """Target hardware: NVIDIA H100 (Hopper, sm_90). TMA, wgmma, and
thread-block clusters are available. 228 KB shared memory per SM, ~3.3 TB/s HBM.""",
    "h200": """Target hardware: NVIDIA H200 (Hopper, sm_90). As H100 but with 141 GB
HBM3e at ~4.8 TB/s.""",
}


PREVIOUS_OUTPUT = "\n\n--- Previous Program Output ---\n{stdout}\n--- End Output ---"

TRUNCATED_OUTPUT = "\n\n\t\t ...(TRUNCATED)...\n{stdout}"
