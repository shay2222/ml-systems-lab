Experiment 03: a Triton kernel for RMSNorm, checked for correctness and speed against PyTorch
## Results

| Rows x Cols | PyTorch (ms) | Triton (ms) | Speedup |
|---|---|---|---|
| 1024 x 896 | 0.233 | 0.025 | 9.23x |
| 4096 x 896 | 0.848 | 0.064 | 13.34x |
| 16384 x 896 | 2.962 | 0.246 | 12.03x |
| 4096 x 4096 | 3.304 | 0.295 | 11.19x |

Correctness: [PASS / FAIL from Step 28, with the max difference]

Method: RMSNorm in float16, computed in float32 inside the kernel. Each row is handled by one GPU program. Timings are averages over 100 calls after warm-up, on a Tesla T4 (Colab, free).

Why it's faster: the PyTorch version runs several separate operations, each reading and writing memory. The Triton kernel fuses them into one pass over each row.

Limitations: one GPU, one dtype, and no comparison yet against torch.compile. The speedup is against unfused PyTorch, not the fastest possible baseline.

## Results

Three baselines, same inputs, float16, Tesla T4 (Colab, free). Times are averages over 100 calls after warm-up.

| Rows x Cols | PyTorch eager (ms) | torch.compile (ms) | Triton (ms) | Triton vs eager | Triton vs compiled |
|---|---|---|---|---|---|
| 1024 x 896 | 0.233 | 0.127 | 0.025 | 9.23x | 4.24x |
| 4096 x 896 | 0.848 | 0.153 | 0.064 | 13.34x | 2.02x |
| 16384 x 896 | 2.962 | 0.251 | 0.246 | 12.03x | 1.06x |
| 4096 x 4096 | 3.304 | 0.479 | 0.295 | 11.19x | 1.66x |

Correctness: PASS. Max absolute difference from the PyTorch reference: 0.0039 (float16, random inputs, 4096 x 896).

Finding: the hand-written kernel beats unfused PyTorch by 9 to 13x, and beats torch.compile by 1.1 to 4.2x. Most of the eager-mode gap is operator fusion, which the compiler also does. The remaining advantage is largest at smaller sizes and nearly disappears at 16384 x 896.

Limitations: one GPU, one dtype, single benchmark session, and one kernel. The compiled numbers vary between runs, so the 1.06x result at 16384 x 896 should be read as roughly a tie. Only the forward pass was tested.
