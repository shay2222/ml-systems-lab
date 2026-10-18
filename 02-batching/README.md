Experiment 02: how throughput changes with batch size.

| Batch size | Total tokens/second | Speedup vs batch 1 |
|---|---|---|
| 1 | 22.2 | 1.0x |
| 4 | 104.0 | 4.7x |
| 16 | 439.0 | 19.8x |

Hardware: Tesla T4 (Colab, free). Model: Qwen2.5-0.5B-Instruct. 200 new tokens per prompt, single run per row.

Finding: total throughput scales close to linearly at these batch sizes, because one request leaves most of the GPU idle. Batching is the main lever for serving throughput.

## Latency vs throughput

| Batch size | Total tokens/second | Seconds per user (200 tokens) | Tokens per user per second |
|---|---|---|---|
| 1 | 22.2 | 7.57 | 26.4 |
| 4 | 104.0 | 6.95 | 28.8 |
| 16 | 439.0 | 8.05 | 24.8 |

Finding: at batch size 16, total throughput is about 20x higher, while each user waits about 6% longer than at batch size 1. Batching is very cheap for latency at these sizes because a single request leaves the GPU mostly idle. Batch-4 wait being slightly lower than batch-1 is within run-to-run noise.

Limitations: one model, one GPU, and single runs per row. Real serving adds queueing and uneven prompt lengths, which this test doesn't cover.
