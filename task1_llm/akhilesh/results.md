# Task 1 results: character-level GPT on TinyStories

Member: Akhilesh
Checkpoint used for all reported numbers: `checkpoints/best.pt` (epoch 10)
Hardware: NVIDIA GeForce RTX 5090, Windows, PyTorch 2.11.0 + CUDA 12.8, bf16 autocast

## What I built

A decoder-only Transformer written from scratch in `src/gpt_char.py`. Attention, the causal mask, LayerNorm, the feed-forward blocks and the language-model head are all implemented by hand; no `nn.Transformer*`, `nn.MultiheadAttention`, `nn.LayerNorm` or `scaled_dot_product_attention` is used.

**Data.** 110,000 stories were sampled from the TinyStories train split with seed 266 and split into 100,000 for training and 10,000 for validation. Stories are joined into one character stream with an end-of-story token between them. The vocabulary is built from the training text only, keeping characters that appear at least 50 times, which gives 79 symbols (including an end-of-story token and an unknown token). The mappings are stored as `char_to_idx` and `idx_to_char` in `data_processed/vocab.json`. The validation set had no unknown characters. The streams were cut into non-overlapping 256-character windows, where the target is the input shifted by one character: 349,971 training sequences (89.6M characters) and 35,086 validation sequences (9.0M characters).

**Model.** Token embeddings plus learned positional embeddings, followed by 6 pre-LayerNorm Transformer blocks and a final LayerNorm and linear head over the 79 characters. Each block computes `x + MHSA(LN(x))` and then `x + FFN(LN(x))`. Attention uses 6 heads of size 64, scores are masked with a lower-triangular matrix so position t can only attend to positions 0..t, and softmax is computed in fp32. The FFN is 384 → 1536 → 384 with GELU. The output head shares its weights with the token embedding.

## Architecture and hyperparameters

| setting | value |
|---|---|
| layers / heads / embedding size | 6 / 6 / 384 (head size 64) |
| FFN inner size | 1536 (4 × 384), GELU |
| context length (block size) | 256 characters |
| vocabulary | 79 characters (min frequency 50) |
| dropout | 0.1 (embeddings, attention weights, residual outputs) |
| weight tying | on |
| parameters | 10,776,192 |
| optimizer | AdamW, β = (0.9, 0.95), weight decay 0.1 on weight matrices only |
| learning rate | 6e-4 peak, 1,000-step linear warmup, cosine decay to 6e-5 |
| batch | 64 sequences × 256 characters = 16,384 characters per step |
| training length | 10 epochs = 54,690 steps ≈ 896M characters |
| gradient clipping | 1.0 (global L2 norm) |
| precision | bf16 autocast, fp32 master weights |
| decoding | temperature 0.8 sampling (main samples), plus greedy for comparison |
| seed | 266 |

## Why these choices

**Character-level tokens with a frequency cutoff.** The task asks for character-level modelling. The cutoff of 50 removes a tail of rare Unicode symbols that appear a handful of times in TinyStories and would only waste embedding rows. It cost nothing on validation, where no character was mapped to unknown.

**Model size.** The TinyStories paper shows that models with only a few million parameters can already write fluent, grammatical stories on this dataset, so I did not need a large model. At 10.8M parameters and about 896M training characters, the model sees roughly 83 characters per parameter, which is enough for it to keep improving through all 10 epochs without overfitting. Six heads of size 64 is the standard head size from GPT-2 scaled down, and it divides 384 evenly.

**Context of 256.** Attention cost grows with the square of the context length, and 256 characters covers two or three sentences, which is where most of the local grammar lives. The average story is about 900 characters, so the model sees less than a third of a story at once. This is a deliberate trade-off for speed, and it shows up directly in the failure analysis: the model loses track of characters and objects that were introduced earlier in the story.

**Pre-LayerNorm and initialisation.** Putting LayerNorm before attention and the FFN (as in GPT-2) keeps the residual stream well scaled and makes training much less sensitive to the learning rate than the post-norm layout in "Attention Is All You Need". Weights are initialised with standard deviation 0.02, and the output projections of each residual branch are scaled down by 1/√(2 × layers) so the residual stream does not grow with depth.

**Optimizer and schedule.** AdamW with decoupled weight decay is the standard choice for Transformers. I only decay weight matrices, not biases, LayerNorm gains or positional parameters, because decaying those tends to hurt without regularising anything useful. β₂ = 0.95 reacts faster to changes in gradient scale than the default 0.999, which is common for GPT-style training. The 1,000-step warmup (about 18% of the first epoch) avoids large, poorly-estimated Adam updates at the start, and the log shows why it helps: the gradient norm was 16.7 at step 0 and had settled below 1 within 200 steps. Cosine decay to 10% of the peak lets the model settle into a lower loss at the end; the validation loss was still falling in epoch 10.

**Dropout 0.1 and weight decay.** With 100K stories the model is not data-starved, so light regularisation was enough. The final generalisation gap is only 0.020 nats per character, which suggests the regularisation could even be reduced slightly.

**Weight tying.** Sharing the embedding and output matrices is standard in GPT models. With a 79-character vocabulary it only saves about 30K parameters, so the main reason here is that input and output use the same representation for each character.

**bf16 autocast.** The RTX 5090 supports bf16 natively. bf16 has the same exponent range as fp32, so no loss scaling is needed, and training ran at about 530K characters per second when the GPU was not shared. Attention scores and the loss are computed in fp32 for numerical safety.

**Temperature 0.8.** Greedy decoding repeats itself (repeated 4-gram rate 6.2% against 0.9% for sampling), while temperature 1.0 makes spelling mistakes more likely. 0.8 is a common middle ground and gave readable, varied stories.

## Metrics

| metric | value |
|---|---|
| training cross-entropy loss (eval mode, 35,086 train sequences) | 0.5373 |
| training cross-entropy loss (running average during epoch 10, dropout on) | 0.5708 |
| validation cross-entropy loss | 0.5568 |
| training perplexity | 1.711 |
| validation perplexity | 1.745 |
| validation bits per character | 0.803 |
| generalisation gap (val − train loss) | 0.0195 |
| top-1 next-character accuracy (val / train) | 82.17% / 82.69% |
| distinct-1 / distinct-2 / distinct-3 (words, 5 sampled stories) | 0.344 / 0.799 / 0.935 |
| repeated 4-gram rate (temperature 0.8 / greedy) | 0.94% / 6.16% |
| gradient norm, mean / 99th percentile / max (pre-clip) | 0.272 / 1.048 / 16.81 |
| loss spikes (> 1.5 × running average) | 0 |
| NaN / Inf loss steps, non-finite gradient steps | 0, 0 |
| parameter count | 10,776,192 |
| training throughput | 440,474 characters/s overall |
| generation throughput | 120.6 characters/s |
| peak GPU memory | 3.48 GB |
| total training time | 33.9 min |

### Per-epoch progress

| epoch | train loss (eval) | val loss | val perplexity | val bits/char | val accuracy |
|---|---|---|---|---|---|
| 1 | 0.700 | 0.703 | 2.020 | 1.014 | 77.7% |
| 2 | 0.649 | 0.655 | 1.924 | 0.944 | 79.2% |
| 3 | 0.622 | 0.629 | 1.875 | 0.907 | 80.0% |
| 4 | 0.603 | 0.611 | 1.843 | 0.882 | 80.5% |
| 5 | 0.588 | 0.598 | 1.819 | 0.863 | 80.9% |
| 6 | 0.573 | 0.586 | 1.797 | 0.845 | 81.3% |
| 7 | 0.561 | 0.575 | 1.777 | 0.830 | 81.6% |
| 8 | 0.550 | 0.566 | 1.761 | 0.817 | 81.9% |
| 9 | 0.542 | 0.560 | 1.750 | 0.808 | 82.1% |
| 10 | 0.537 | 0.557 | 1.745 | 0.803 | 82.2% |

Validation loss improved every epoch, so the best checkpoint is the last one.

### How each metric was computed

- **Cross-entropy losses** are averaged over every predicted character. The validation loss uses all 35,086 validation sequences. The training loss is measured the same way in eval mode (dropout off) on a fixed random set of 35,086 training sequences, so the two numbers are directly comparable. The running training loss is the average over the epoch while training, with dropout on, which is why it is higher.
- **Perplexity** is exp(cross-entropy). **Bits per character** is cross-entropy / ln 2.
- **Generalisation gap** is validation loss minus eval-mode training loss.
- **Top-1 accuracy** is the fraction of positions where the most likely character is the true next character.
- **Distinct-n** is the number of unique word n-grams divided by the total number of word n-grams across the five temperature-0.8 stories. **Repeated 4-gram rate** is, per story, the fraction of word 4-grams that already appeared earlier in the same story, averaged over stories.
- **Gradient norm** is the global L2 norm of all gradients before clipping, logged at every step in `logs/train_steps_gpt_char.csv`. A loss spike is a step whose loss exceeds 1.5 times the exponential moving average after warmup.
- **Throughput** counts training characters per second of training time only, excluding validation passes. Generation throughput covers 3,000 sampled characters (5 prompts × 600).

## Observations

- The largest max gradient norm (16.8) is from step 0, before any learning. After warmup the norm stayed between about 0.2 and 0.3 for the rest of training, with no spikes or NaNs.
- Throughput dropped from about 535K to about 359K characters per second from late in epoch 6 onward, because the CycleGAN run for Task 3 started on the same GPU at that point. The model and results are unaffected; only the timing changed.
- Generation is much slower than training (121 vs 440K characters per second) because each new character needs a full forward pass over up to 256 previous characters, one sample at a time, with no key/value cache. Training processes whole sequences in parallel.
- The model writes grammatical sentences and correct dialogue punctuation, but it loses track of who is who over a longer story and can contradict details it stated earlier. The three failure cases are in `failure_analysis.md`.

## What I would try next

A subword tokenizer or a longer context would let the model see more of each story at once, which should help with coherence. A key/value cache would make generation much faster. Since the model was not overfitting, a slightly larger model or a few more epochs would likely lower the validation loss further.

## Files

- Code: `src/gpt_char.py`, config: `src/config.yaml`, notebook with outputs: `src/task1_gpt.ipynb`
- Metrics: `metrics_report.csv`, per-epoch history: `outputs/history.json`
- Loss curves: `outputs/loss_curves.png`
- Samples: `outputs/samples.txt`, `outputs/samples.json`
- Raw logs and manifest: `logs/`
- Checkpoint: `checkpoints/best.pt`
