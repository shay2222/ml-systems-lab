# ml-systems-lab

Five small experiments on LLM inference and GPU systems, measured on a free Google Colab Tesla T4 (CPU for experiment 05). Each folder has a notebook that reruns the experiment and a README with the results.

| # | Experiment | Key result |
|---|---|---|
| 01 | KV cache | 1.51x faster at 700 tokens; about the same at 200 tokens |
| 02 | Batching | 22 to 439 tokens/s at batch 1 to 16; each user waits about 6% longer at batch 16 |
| 03 | Triton RMSNorm kernel | 9 to 13x faster than PyTorch eager, 1.1 to 4.2x faster than torch.compile; correct to 0.004 |
| 04 | Profiling | Matrix multiplies take 55% of GPU time; attention 15%; copy and mul are frequent |
| 05 | Data-parallel training | 1.27x throughput with 2 CPU processes (63% efficiency); all-reduce 18% of step time |

Hardware: Tesla T4 (Colab, free) for 01 to 04; CPU for 05. Model: Qwen2.5-0.5B-Instruct for 01 to 04.

Limitations: single runs per row, one model, one GPU, free-tier Colab. Results vary between runs, so treat small differences as noise.
