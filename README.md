# DATA 266 Lab 1: LLM from scratch, Yelp sentiment, CycleGAN

Each member has a folder under every task (`task*/<member_name>/`). All runs are config-driven, and every script writes an unedited raw log, a per-step CSV, a manifest (args, library versions, GPU, git commit) and a `pip freeze` into that member's `logs/` folder.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

For Task 3, download the Kaggle dataset and put `monet_jpg/` and `photo_jpg/` in `task3_gan/data/`, with `real_stats.npz` next to them.

## Smoke test (one command, about 2 minutes on a GPU)

```bash
cd task3_gan/akhilesh && python src/cyclegan.py --config src/config.yaml --data_dir ../data --out_dir smoke \
  --n_epochs 1 --n_epochs_decay 0 --steps_per_epoch 50 && cd ../..
```

## Full runs

Launch each run from its member folder. Use `nohup` or `tmux` so the job survives a closed notebook tab.

```bash
# Task 3 (longest; start first)
cd task3_gan/akhilesh
nohup python src/cyclegan.py --config src/config.yaml --data_dir ../data --out_dir . > logs/nohup_train.txt 2>&1 &
# resume after a session ends:
python src/cyclegan.py --config src/config.yaml --data_dir ../data --out_dir . --resume checkpoints/last_full.pt
# evaluate + Kaggle submission + audit export:
python src/evaluate_local.py --ckpt checkpoints/generators_final.pt --data_dir ../data --out_dir . --real_stats ../data/real_stats.npz

# Task 1
cd task1_llm/akhilesh
nohup python src/gpt_char.py --config src/config.yaml --out_dir . > logs/nohup_train.txt 2>&1 &
# resume: add --resume checkpoints/last.pt

# Task 2 (three models, then McNemar + comparison table)
cd task2_sentiment/akhilesh
python src/sentiment.py --config src/cfg_baseline.yaml --out_dir .
python src/sentiment.py --config src/cfg_exp1.yaml --out_dir .
python src/sentiment.py --config src/cfg_exp2.yaml --out_dir .
python src/sentiment.py --mode compare --out_dir . --baseline_preds outputs/baseline/test_preds.npz \
  --exp_preds outputs/exp1_cnn/test_preds.npz outputs/exp2_bilstm/test_preds.npz
```

## Where results live (per member, per task)

| Task | Metrics | Plots and samples | Checkpoint |
|---|---|---|---|
| 1 | `metrics_report.csv` | `outputs/loss_curves.png`, `outputs/samples.txt` | `checkpoints/best.pt` |
| 2 | `metrics_report.csv`, `outputs/comparison_table.csv` | `outputs/<run>/eval_plots.png`, `outputs/<run>/error_review.csv`, `outputs/eda/` | `checkpoints/<run>_best.pt` |
| 3 | `full_metrics_report.csv`, `submission.csv` | `outputs/loss_curves.png`, `outputs/train_samples/`, `outputs/pred_A2B/`, `outputs/pred_B2A/` | `checkpoints/generators_final.pt` |

Raw logs and manifests are copied to `reproducibility/raw_logs/` and `reproducibility/manifests/` without edits.

## Notes

Task 1 uses no prebuilt Transformer, attention or LayerNorm modules; everything is written out in `gpt_char.py`. Task 2 uses no pretrained embeddings or language models. Task 3 images are produced only by the trained generators; the Inception and LPIPS networks are used only to measure them.

Checkpoints over 50 MB should go through Git LFS (`git lfs track "*.pt"`).
