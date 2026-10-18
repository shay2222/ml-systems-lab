Experiment 02: how throughput changes with batch size.

| Batch size | Total tokens/second | Speedup vs batch 1 |
|---|---|---|
| 1 | 22.2 | 1.0x |
| 4 | 104.0 | 4.7x |
| 16 | 439.0 | 19.8x |

Hardware: Tesla T4 (Colab, free). Model: Qwen2.5-0.5B-Instruct. 200 new tokens per prompt, single run per row.

Finding: total throughput scales close to linearly at these batch sizes, because one request leaves most of the GPU idle. Batching is the main lever for serving throughput.
