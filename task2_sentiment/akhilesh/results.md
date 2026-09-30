# Task 2 results: Yelp Polarity sentiment classification

Member: Akhilesh
Hardware for all three models: NVIDIA GeForce RTX 5090, Windows, PyTorch 2.11.0 + CUDA 12.8
Checkpoints: `checkpoints/baseline_best.pt`, `checkpoints/exp1_cnn_best.pt`, `checkpoints/exp2_bilstm_best.pt`

## Data

Yelp Polarity labels 1–2 star reviews as negative (0) and 3–4 star reviews as positive (1). The full train split has 560,000 reviews, exactly balanced (280,000 per class), and the test split has 38,000 (19,000 per class).

| item | value |
|---|---|
| training subsample (seed 266) | 120,000 reviews: 59,992 negative, 60,008 positive |
| missing or empty reviews | 0 in train, 0 in test |
| exact duplicate reviews | 0 |
| reviews empty after cleaning (dropped from train) | 11 |
| final split | 107,991 train / 11,998 validation / 38,000 test |
| review length, negative (words) | mean 152, median 112, 95th percentile 424, max 1,052 |
| review length, positive (words) | mean 116, median 85, 95th percentile 316, max 943 |
| tokens after cleaning | mean 61, median 45, 95th percentile 169 |
| reviews longer than the 256-token limit | 1.4% |
| vocabulary (min frequency 3, max 30,000) | 26,895 words |
| share of training tokens covered by the vocabulary | 99.3% |

The classes are balanced, so accuracy is a fair headline metric and no reweighting or resampling was needed. Negative reviews are clearly longer than positive ones (median 112 against 85 words): people explain in detail what went wrong. Length alone is therefore a weak sentiment signal, which is one reason I report performance separately for short, medium and long reviews. I used a 120,000-review subsample of the training split to keep all three models trainable within one lab session; it is still more than 25 times the size of the validation set and gives tight confidence intervals.

## Preprocessing

1. Lowercase, and replace Yelp's escaped `\n` and `\"` with spaces.
2. Turn "n't" into " not" before removing punctuation, so "wasn't" becomes "was not" instead of "wasn t".
3. Remove punctuation, digits and every other non-letter character.
4. Remove English stopwords (scikit-learn's list), **but keep negation words**: no, not, nor, never, none, nothing, nobody, neither, nowhere, cannot. Removing "not" from "not good" would flip the meaning of the review.
5. Porter stemming, so "loved", "loving" and "loves" share one token and one embedding.
6. Split on whitespace, map tokens to vocabulary indices (0 = padding, 1 = unknown word), and pad or cut to 256 tokens. The limit affects only 1.4% of reviews.

## Embeddings

Every model learns its own 128-dimensional word embeddings from scratch, starting from random values, as part of training on the sentiment labels. No pretrained vectors (word2vec, GloVe) or pretrained language models are used. The padding index has a fixed zero vector and receives no gradient. Because the embeddings are trained only on the sentiment objective, words end up close together when they predict the same label, not necessarily when they mean the same thing. The embedding table (26,895 × 128 = 3.44M weights) makes up most of every model's parameters.

## The three models and why I chose them

| | Baseline | Experimental 1 | Experimental 2 |
|---|---|---|---|
| architecture | bag of words: masked mean of embeddings → 128-unit ReLU MLP → 1 logit | CNN: 1D convolutions, widths 3/4/5 × 128 filters, ReLU, max-pool over time → 1 logit | BiLSTM: 1 bidirectional layer, 128 units per direction, max-pool over time → 1 logit |
| word order | ignored | local phrases up to 5 words | whole review, both directions |
| embedding size | 128 | 128 | 128 |
| dropout | 0.3 | 0.5 | 0.3 |
| peak learning rate | 2e-3 | 1e-3 | 1e-3 |
| batch size | 256 | 256 | 128 |
| max epochs | 8 | 8 | 6 |
| parameters | 3,459,201 | 3,639,937 | 3,707,009 |

Shared settings: binary cross-entropy on one logit, AdamW with weight decay 1e-4, one-cycle learning-rate schedule (10% warmup, then cosine decay), gradient clipping at 1.0, and early stopping on validation loss with patience 2, keeping the best epoch.

**Baseline: bag of words.** Averaging word embeddings is the simplest neural text classifier (the fastText idea). It learns which words are positive or negative but cannot see order, so "good, not bad" and "bad, not good" look identical to it. That makes it a strong but clearly limited reference: any gain from the other two models shows what word order adds.

**Experimental 1: CNN.** Following Kim (2014), convolution filters of widths 3, 4 and 5 act as learned phrase detectors ("not very good", "highly recommend this place"), and max-pooling keeps the strongest match anywhere in the review. It changes the architecture from the baseline by adding local word order at almost no extra cost. I used dropout 0.5, as in Kim's paper, because the 384-dimensional pooled feature vector feeds a single linear layer and overfits easily.

**Experimental 2: BiLSTM.** A bidirectional LSTM reads the review forwards and backwards, so each position's representation depends on the whole review, not just a short window. This is the model that should handle negation scope and contrast ("I expected it to be bad, but ..."). I used max-pooling over all time steps rather than only the final hidden state, so a strong signal in the middle of a long review is not lost. The smaller batch and lower maximum epochs reflect that it is about 10 times slower per epoch and reached its best validation loss sooner.

## Training

| | Baseline | CNN | BiLSTM |
|---|---|---|---|
| epochs run (early stopping) | 6 | 8 | 5 |
| best epoch (lowest validation loss) | 4 | 6 | 3 |
| best validation loss / accuracy | 0.2006 / 92.31% | 0.1972 / 92.19% | 0.1799 / 93.04% |
| training time | 20.3 s | 33.0 s | 173.9 s |
| training throughput | 31,955 examples/s | 26,167 examples/s | 3,106 examples/s |
| peak GPU memory | 0.35 GB | 0.62 GB | 0.86 GB |

All three models started to overfit within a few epochs: training loss kept falling while validation loss flattened or rose, and early stopping kept the best epoch in each case.

## Test results (38,000 reviews)

| metric | Baseline (BoW) | Exp 1 (CNN) | Exp 2 (BiLSTM) |
|---|---|---|---|
| accuracy | 0.9229 | 0.9246 | **0.9309** |
| accuracy, 95% bootstrap CI | 0.9202–0.9253 | 0.9222–0.9271 | 0.9285–0.9335 |
| precision / recall / F1, macro | 0.9229 / 0.9229 / 0.9229 | 0.9246 / 0.9246 / 0.9246 | 0.9309 / 0.9309 / 0.9309 |
| precision / recall / F1, micro | 0.9229 / 0.9229 / 0.9229 | 0.9246 / 0.9246 / 0.9246 | 0.9309 / 0.9309 / 0.9309 |
| precision / recall / F1, weighted | 0.9229 / 0.9229 / 0.9229 | 0.9246 / 0.9246 / 0.9246 | 0.9309 / 0.9309 / 0.9309 |
| macro-F1, 95% bootstrap CI | 0.9202–0.9253 | 0.9222–0.9271 | 0.9285–0.9335 |
| confusion matrix (TN / FP / FN / TP) | 17,568 / 1,432 / 1,499 / 17,501 | 17,541 / 1,459 / 1,405 / 17,595 | 17,673 / 1,327 / 1,298 / 17,702 |
| ROC-AUC | 0.9756 | 0.9781 | **0.9804** |
| PR-AUC | 0.9757 | 0.9786 | **0.9807** |
| MCC | 0.8457 | 0.8493 | **0.8618** |
| MCC, 95% bootstrap CI | 0.8403–0.8506 | 0.8445–0.8542 | 0.8569–0.8670 |
| Brier score (lower is better) | 0.0575 | 0.0558 | **0.0522** |
| expected calibration error, 15 bins | **0.0048** | 0.0094 | 0.0109 |

Macro and weighted averages are identical because the test set is exactly balanced, and micro-averaged precision, recall and F1 always equal accuracy for single-label classification. Confusion matrices, ROC and precision-recall curves and reliability diagrams for each model are in `outputs/<model>/eval_plots.png`.

### Robustness by slice

| slice | n | Baseline macro-F1 / error | CNN macro-F1 / error | BiLSTM macro-F1 / error |
|---|---|---|---|---|
| short (≤ 65 words) | 12,776 | 0.9181 / 7.95% | 0.9182 / 7.94% | 0.9301 / 6.78% |
| medium (65–142 words) | 12,649 | 0.9243 / 7.57% | 0.9274 / 7.26% | 0.9319 / 6.81% |
| long (> 142 words) | 12,575 | 0.9212 / 7.62% | 0.9234 / 7.41% | 0.9262 / 7.14% |
| contains a negation word | 28,322 | 0.9156 / 8.16% | 0.9188 / 7.86% | 0.9256 / 7.20% |
| no negation word | 9,678 | 0.9111 / 6.42% | 0.9083 / 6.60% | 0.9165 / 6.04% |

Length slices use the training set's terciles. The "no negation" slice has a lower error rate but also a lower macro-F1 than the negation slice because its two classes are not balanced, and macro-F1 weights the smaller class as heavily as the larger one.

### McNemar tests (baseline vs each experimental model, same 38,000 reviews)

| comparison | baseline right, model wrong (b) | baseline wrong, model right (c) | χ² (continuity corrected) | exact p-value |
|---|---|---|---|---|
| baseline vs CNN | 916 | 983 | 2.29 | 0.13 |
| baseline vs BiLSTM | 813 | 1,119 | 48.15 | 3.6 × 10⁻¹² |

## Comparison and discussion

**The BiLSTM is the best model, and the gain is real.** It is ahead on every accuracy-type metric, its confidence interval does not overlap the baseline's, and McNemar's test finds the difference highly significant: it fixes 1,119 of the baseline's errors while introducing 813 new ones, a net 306 reviews.

**The CNN is not reliably better than the baseline.** Its accuracy is 0.17 points higher, but the confidence intervals overlap and McNemar gives p = 0.13: it fixes 983 baseline errors and makes 916 new ones, a difference that could easily be chance. Local phrase detectors of 3–5 words recover little that averaged word embeddings do not already capture on this dataset.

**Word order matters most where negation is involved.** On reviews containing a negation word, the error rate falls from 8.16% (bag of words) to 7.86% (CNN) to 7.20% (BiLSTM). The bag-of-words model cannot tell "good" from "not good"; the CNN can if the negation is within its window; the BiLSTM can across the whole sentence. The BiLSTM also improves most on short reviews, where every word counts, and least on long reviews (7.14% error), where the verdict may sit in one or two sentences among many.

**Accuracy and calibration pull in different directions.** The baseline is the best calibrated model (ECE 0.0048): when it says 80% positive, about 80% of those reviews really are positive. The BiLSTM is the most accurate but more than twice as badly calibrated (ECE 0.0109), which is the usual pattern of larger models becoming overconfident. Its Brier score is still the best, because better accuracy outweighs the calibration loss. If the probabilities were used directly, for example to route uncertain reviews to a person, temperature scaling on the validation set would fix most of this.

**Cost.** The BiLSTM's 0.8-point gain costs about 8.6× the training time of the baseline (174 s against 20 s) and about 10× lower throughput, because the recurrence has to process each review step by step. The CNN runs at nearly the baseline's speed. All three have similar parameter counts because the shared-size embedding table dominates.

**Strengths.** All three models reach over 92% accuracy with no pretrained knowledge, are well calibrated in absolute terms (ECE ≤ 0.011) and are stable across length slices (error rates within about one percentage point).

**Weaknesses and limitations.**
- **Mixed reviews** are the main failure: 7 of the 20 errors I reviewed by hand mix praise and complaint, and the label depends on how they are weighed (`failure_analysis.md`).
- **The stopword list removes contrast words.** Negations are kept, but "but", "however", "although", "too", "very", "only" and "enough" are removed, and these are exactly the words that decide mixed reviews. This is my main proposed fix.
- **Label noise.** Some confident errors are reviews whose text contradicts the star rating (for example an edited review), so part of the measured error is not the model's fault.
- **Non-English reviews** lose their accented letters in cleaning and become mostly unknown words.
- **Training subset.** I trained on 120,000 of the 560,000 available reviews; the full set would likely add a few tenths of a point.

## What I would try next

1. **Keep contrast and intensity words** during stopword removal, retrain the BiLSTM with the same seed, and compare with McNemar's test and a new slice of reviews containing those words. This is the testable fix proposed in `failure_analysis.md`.
2. **Attention pooling** over the BiLSTM outputs, so the model can weight the few sentences that carry the verdict in long reviews.
3. **Train on the full 560,000 reviews**, which is affordable for the baseline and CNN and a few minutes per epoch for the BiLSTM.
4. **Temperature scaling** on the validation set to bring the BiLSTM's calibration back to the baseline's level without changing its accuracy.

## Files

- Code: `src/sentiment.py`; configs: `src/cfg_baseline.yaml`, `src/cfg_exp1.yaml`, `src/cfg_exp2.yaml`; notebook with outputs: `src/task2_sentiment.ipynb`
- All metrics for all models, plus McNemar results: `metrics_report.csv`; side-by-side table: `outputs/comparison_table.csv`
- Per model (`outputs/baseline/`, `outputs/exp1_cnn/`, `outputs/exp2_bilstm/`): `eval_plots.png`, `slices.csv`, `history.json`, `test_preds.npz`, `error_review.csv`
- Data analysis: `outputs/eda/eda.json`, `outputs/eda/eda.png`
- Error review: `failure_analysis.md`
- Raw logs and manifests: `logs/`, copied unedited to `reproducibility/`
