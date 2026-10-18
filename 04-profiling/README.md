Experiment 04: where time goes in one forward pass, measured with the PyTorch profiler.

## Profile of one forward pass

Model: Qwen2.5-0.5B-Instruct (fp16), batch of 16 prompts processed together, Tesla T4 (Colab, free). torch.profiler, measured after warm-up.

Totals: 64.3 ms CPU, 62.3 ms CUDA.

| Operation | Self CUDA (ms) | Share of CUDA time |
|---|---|---|
| aten::mm | 34.2 | 54.9% |
| aten::addmm | 6.0 | 9.6% |
| aten::mul | 5.5 | 8.8% |
| aten::copy_ | 4.8 | 7.7% |
| scaled_dot_product_attention (total, incl. children) | 9.6 | 15.5% |

Notes: the GEMM kernels under mm (24.9 ms and 9.3 ms) are its internal work, not extra time. aten::linear and aten::matmul are wrappers and are not added separately.

Finding: matrix multiplies take more than half of GPU time, as expected for a transformer. Attention takes about 15%, running on the unfused math path. copy_ and mul run hundreds of times per pass, so they are the first candidates for reducing overhead.

Limitations: one forward pass over a full prompt, not the one-token-at-a-time decode loop used in generation, so the breakdown would differ there. One batch, one GPU, and the profiler adds overhead, so use the proportions rather than the absolute numbers.
