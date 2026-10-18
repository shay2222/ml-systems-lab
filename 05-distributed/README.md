Experiment 05: data-parallel training with torch.distributed, two processes on CPU (Colab free tier has one GPU).

## Results

Setup: Qwen-free toy MLP (1.59M parameters), batch 64 per process, SGD, gloo backend, 50 timed steps after warm-up. Environment: Google Colab CPU runtime with [N] cores. No GPU was used, so this is not a multi-GPU result.

| Processes | Step time (ms) | All-reduce (ms) | Samples/second | Throughput vs 1 process |
|---|---|---|---|---|
| 1 | 23.99 | 0.07 | 2668 | 1.00x |
| 2 | 37.82 | 6.86 | 3384 | 1.27x |

Finding: two data-parallel processes raised throughput by 27%, about 63% efficiency. Gradient all-reduce took 6.86 ms per step (about 18% of step time). The remaining slowdown is likely from the two processes sharing the same CPU cores. A multi-GPU run would be needed to measure real scaling.

Limitations: CPU only, one run per row, toy model, and the core count was not controlled.
