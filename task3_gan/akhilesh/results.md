# Task 3 results: CycleGAN, Monet <-> Photo

Member: Akhilesh
Checkpoint used for all reported numbers: `checkpoints/generators_final.pt` (epoch 30)
Hardware: NVIDIA GeForce RTX 5090, Windows, PyTorch 2.11.0 + CUDA 12.8, fp32 training

## What I built

A CycleGAN written in `src/cyclegan.py`, trained from scratch on the competition data. Domain A is the 300 Monet paintings and domain B is the 7,038 photos, all 256 × 256 RGB. The two domains are sampled independently at every step, so no image is ever paired with another.

- **Generators (×2).** `G_AB` maps Monet to photo and `G_BA` maps photo to Monet (the Kaggle direction). Each is a ResNet generator: a 7 × 7 convolution, two stride-2 downsampling convolutions, 9 residual blocks at 256 channels, two upsampling stages and a final 7 × 7 convolution with tanh. Instance normalisation and reflection padding are used throughout. For upsampling I used nearest-neighbour resize followed by a 3 × 3 convolution instead of transposed convolutions.
- **Discriminators (×2).** `D_A` judges Monet images and `D_B` judges photos. Each is a 70 × 70 PatchGAN: it outputs a grid of real/fake scores, one per overlapping 70 × 70 patch, rather than one score per image.
- **Losses.** Least-squares adversarial loss for both generators and both discriminators, cycle-consistency loss `L1(G_BA(G_AB(a)), a) + L1(G_AB(G_BA(b)), b)` weighted by 10, and an identity loss `L1(G_BA(a), a) + L1(G_AB(b), b)` weighted by 0.3 × 10 = 3. The discriminators are trained on a history buffer of 50 previously generated images, as in the CycleGAN paper.

## Architecture and hyperparameters

| setting | value |
|---|---|
| generator | ResNet, 64 base filters, 9 residual blocks, resize-convolution upsampling |
| discriminator | 70 × 70 PatchGAN, 64 base filters, 3 downsampling layers |
| normalisation | instance norm |
| adversarial loss | least squares (LSGAN) |
| cycle weight λ_cyc | 10 |
| identity weight | 0.3 × λ_cyc = 3 |
| image history buffer | 50 |
| optimizer | Adam, lr 2e-4, β = (0.5, 0.999), separate optimizers for G and D |
| schedule | constant lr for 15 epochs, then linear decay over 15 epochs |
| training length | 30 epochs × 1,000 steps = 30,000 steps |
| batch size | 1 |
| augmentation | resize to 286, random 256 crop, random horizontal flip |
| parameters | 11,378,179 per generator, 2,764,737 per discriminator, 28,285,832 in total |
| seed | 2026 |

## Why these choices

**ResNet generator with 9 blocks and a PatchGAN discriminator.** This is the configuration the CycleGAN paper uses for 256 × 256 images. The residual blocks keep the image layout intact while changing texture and colour, which is exactly what style transfer needs. A patch discriminator judges local texture (brush strokes, colour patches) rather than global layout, and style mostly lives at that local scale.

**Resize-convolution upsampling.** Transposed convolutions tend to leave a checkerboard pattern in generated images, because the kernel overlaps unevenly. Upsampling first and then convolving avoids that pattern. This is the main architectural change I made from the paper.

**Least-squares GAN loss.** The CycleGAN paper replaces the standard log loss with least squares because it gives more stable training and fewer vanishing gradients. My run had no NaN or non-finite steps in 30,000 iterations.

**Cycle weight 10.** Without the cycle loss, a generator could produce any Monet-looking image regardless of the input. Weighting it at 10, as in the paper, forces the translation to keep enough of the input that it can be translated back.

**Identity weight 0.3 instead of 0.5.** The identity loss stops the generators from shifting colours when they don't need to. The paper uses 0.5 for painting ↔ photo. I lowered it to 0.3 to give `G_BA` more freedom to change the palette towards Monet's, accepting some risk of colour drift in return.

**Image history buffer.** Showing the discriminator a mix of current and older fakes stops it from overfitting to the generator's latest output and reduces oscillation between the two networks.

**Batch size 1 with instance norm and random crops.** Instance norm normalises each image on its own, which works well for style transfer and doesn't depend on batch statistics. The random 286 → 256 crops and flips matter most for the Monet side: with only 300 paintings, augmentation is the main thing stopping the Monet discriminator from memorising the training set.

**Training length.** The paper trains for 200 epochs over its datasets. I trained for 30,000 steps (30 epochs of 1,000 steps each) to fit the lab time on one GPU. The loss curves show the cycle loss still falling at the end, so the model had not fully converged.

## Training behaviour, convergence and stability

Per-epoch means from the unedited step log (`logs/train_steps_cyclegan.csv`). Cycle and identity values are the weighted terms.

| epoch | G adv (A→B / B→A) | cycle A / B | identity A / B | D_A / D_B | G grad norm |
|---|---|---|---|---|---|
| 1 | 0.454 / 0.497 | 2.90 / 3.14 | 0.82 / 0.87 | 0.315 / 0.335 | 65.5 |
| 5 | 0.475 / 0.504 | 2.14 / 2.31 | 0.64 / 0.64 | 0.191 / 0.213 | 36.5 |
| 10 | 0.498 / 0.621 | 2.00 / 2.14 | 0.61 / 0.60 | 0.141 / 0.187 | 33.3 |
| 15 | 0.499 / 0.428 | 1.92 / 1.91 | 0.57 / 0.55 | 0.183 / 0.178 | 35.8 |
| 20 | 0.526 / 0.471 | 1.76 / 1.71 | 0.54 / 0.50 | 0.154 / 0.155 | 31.6 |
| 25 | 0.546 / 0.507 | 1.59 / 1.58 | 0.51 / 0.47 | 0.130 / 0.135 | 40.2 |
| 30 | 0.548 / 0.601 | 1.36 / 1.42 | 0.48 / 0.44 | 0.106 / 0.132 | 72.0 |

What the log shows:

- **Convergence.** The weighted cycle loss fell by more than half (about 3.0 → 1.4), and most of the improvement in the second half came after the learning rate started decaying at epoch 16. The identity loss fell from about 0.85 to 0.46. Both were still going down at epoch 30.
- **Adversarial balance.** The generators' adversarial losses stayed between about 0.43 and 0.62 for the whole run, so neither network collapsed. Between epochs 1 and 12 the Monet discriminator grew steadily stronger (D_A loss 0.315 → 0.128) and the photo → Monet generator's adversarial loss rose to about 0.62. Around epochs 13–15 this reversed: D_A's loss rose back to about 0.18 and the generator's fell to about 0.43, meaning the generator caught up. In the last ten epochs both discriminators sharpened again (D_A 0.106, D_B 0.132 at epoch 30).
- **Stability.** 0 NaN or non-finite steps out of 30,000. Generator gradient norms averaged 40.0 (99th percentile 135.8, max 634.2) and discriminator gradient norms averaged 19.6 (99th percentile 51.9, max 604.1). The large maxima are rare single-step spikes and never affected the losses. The one trend worth noting is that the generator's average gradient norm roughly doubled over the last five epochs (about 36 → 72), at the same time as the discriminators became stronger. Training ended before this became a problem, but a longer run would need watching here, for example with spectral normalisation or a lower discriminator learning rate.
- **Step-to-step noise.** Individual steps are noisy (for example, a total generator loss of 14.5 at step 600) because each step uses one image per domain. The 200-step averages in `outputs/loss_curves.png` are smooth.

## Verifying cycle-consistency

The cycle constraint is working in both directions. Over all 7,038 photos, translating to Monet and back gives a mean L1 reconstruction error of 0.146 (on the −1 to 1 pixel scale, about 7% of the pixel range). For all 300 Monet paintings going through photo and back, it is 0.142. The standard deviations are small (0.051 and 0.030), so this holds for nearly every image, not just on average. Section 3.2 of the notebook puts this next to the distance between each input and its translation, on 200 images per direction, along with an identity check showing how much each generator changes an image that is already in its output domain.

## Visual quality

I looked at the translations in `src/task3_cyclegan.ipynb` and at the photo | Monet pairs exported for the human audit in `outputs/audit/`. Four patterns stand out.

**What works.** The layout of the photo survives in almost every case: tree silhouettes, a stone archway, rock pinnacles, a statue and shorelines all stay where they were and keep their outlines. The colours move towards Monet's palette of soft greens, yellows and lavender-blues, strong photographic contrast is flattened, and skies are broken into short dabs of colour. Cloudy skies over fields and lakes come out best; there the cloud masses turn into clusters of brush-like marks that look close to Monet's river and field scenes. The content-preservation cosine similarity of 0.74 between each photo and its translation agrees with this: the scene survives, but the image changes clearly (LPIPS 0.48 between input and output).

**Green vegetation painted over the wrong surfaces.** The generator often covers ground with grass-like green texture whatever the ground actually is: a sandy beach, an icy lake, bare coastal rock and desert ground all came out green. Most of the 300 Monet paintings are gardens, fields and riverbanks, so "the lower part of the picture is green vegetation" is a strong pattern in the training data, and the generator applies it even where it changes the content.

**Night and dark scenes turned into daylight.** A foggy street at night, a star-filled night sky and a deep blue dusk sky all became pale, daytime-looking colours. Monet painted very few night scenes, so the generator has almost nothing to map a dark photo onto and pushes it towards the bright palette it knows.

**Texture artifacts.** Large flat areas, such as clear or night skies and calm water, often get a fine speckled or grid-like texture instead of smooth brush strokes, and in a few images there are horizontal streaks across the background. Watermarks and small text in the photos are partly kept and smeared rather than removed. The reconstructions in the notebook keep the layout and main shapes but come back slightly brighter and blurrier than the originals.

The precision of 0.34 fits these observations: only about a third of the generated images fall inside the region of feature space occupied by real Monet paintings, while recall (0.51) and coverage (0.91) show that the outputs spread across most of that region.

## Metrics

Computed by `src/evaluate_local.py` on all 7,038 photo → Monet translations and all 300 Monet → photo translations. Features come from torchvision's Inception-v3, which reproduces the competition's `real_stats.npz` almost exactly (FID 0.074 between the real Monet images and the provided statistics, against 341 with the pytorch-fid weights), so the FID values are on the same scale as Kaggle's.

| metric | photo → Monet (B→A) | Monet → photo (A→B) |
|---|---|---|
| FID | 93.54 (93.56 against the competition's stats) | 113.56 |
| KID (mean ± std over 100 subsets) | 0.0168 ± 0.0011 | 0.0478 ± 0.0016 |
| precision / recall (k = 3) | 0.339 / 0.510 | 0.337 / 0.189 |
| density / coverage | 0.286 / 0.907 | 0.290 / 0.028 |
| MiFID (as described on the competition page) | 0.410 | 0.434 |
| mean nearest-real cosine distance | 0.260 | 0.210 |
| cycle reconstruction L1 (mean ± std) | 0.146 ± 0.051 | 0.142 ± 0.030 |
| LPIPS, input vs translation | 0.477 | 0.392 |
| content-preservation cosine similarity | 0.742 | 0.768 |
| inference speed | 444 images/s | 362 images/s |

| training and cost | value |
|---|---|
| final weighted cycle loss (A / B, epoch 30 mean) | 1.361 / 1.421 |
| final weighted identity loss (A / B, epoch 30 mean) | 0.478 / 0.442 |
| final generator loss / D_A / D_B (epoch 30 mean) | 4.851 / 0.106 / 0.132 |
| generator gradient norm (mean / 99th pct / max) | 40.0 / 135.8 / 634.2 |
| discriminator gradient norm (mean / 99th pct / max) | 19.6 / 51.9 / 604.1 |
| NaN / non-finite steps | 0 of 30,000 |
| parameters | 28,285,832 total (2 × 11,378,179 G, 2 × 2,764,737 D) |
| training time | 56.6 min |
| training throughput | 8.8 steps/s overall (about 4.5 while sharing the GPU with Task 1 in epochs 1–5, about 11 after) |
| peak GPU memory | 3.34 GB training, 1.97 GB evaluation |
| Kaggle public score | 46.9872 (mean of FID 93.564 and MiFID 0.410), rank 2 at submission |

Notes on reading these numbers:

- **Monet → photo coverage of 0.028** is low because coverage measures how much of the real set the generated set spans, and 300 generated photos cannot span 7,038 real ones. It says more about the sample sizes than about the model.
- **Monet → photo FID is higher** than photo → Monet because that generator learned from only 300 paintings and has to produce the much wider variety of real photographs.
- **Memorisation.** The mean cosine distance from each generated Monet to its closest real Monet is 0.26, so the outputs are not near-copies of training paintings.
- **Kaggle integrity.** The submitted FID and MiFID come directly from this checkpoint's inference on the competition photos. No images were edited, selected or taken from anywhere else, and no pretrained model was used to generate or modify them; the Inception and LPIPS networks were used only to measure them.

## Human audit

30 fixed photo → Monet samples (chosen with seed 266, so every member of the team rates translations of the same input photos) were exported to `outputs/audit/` as blinded input | output pairs. Two raters scored each pair from 1 to 5 for Monet style, content preservation and absence of artifacts, without seeing each other's ratings. Mean scores, percentage agreement and Cohen's kappa are added to `full_metrics_report.csv` by `evaluate_local.py --mode audit_kappa`.

## Shortcomings and what I would try next

- **Training length.** 30,000 steps is short for CycleGAN, and the cycle loss was still falling. A longer run is the single change most likely to lower FID.
- **Realism of the Monet style.** Precision of 0.34 means many translations still look like filtered photos. With only 300 Monet paintings, the Monet discriminator sees the same images constantly; differentiable augmentation (DiffAugment) on both real and fake Monet images is designed for exactly this low-data situation.
- **Late-training discriminator strength.** Generator gradient norms doubled in the last five epochs as the discriminators sharpened. Spectral normalisation in the discriminators or a lower discriminator learning rate would make a longer run safer.
- **Content changes from the Monet data's bias.** Green grass painted over sand, ice and rock, and night scenes turned into day, both come from what the 300 Monet paintings contain. A higher identity weight would hold colours closer to the input, at the cost of weaker style; I lowered it to 0.3, and these errors suggest 0.5 may be the better balance.
- **Speckle and grid texture in flat areas.** This is the most visible artifact. Longer training usually smooths it, and a discriminator with spectral normalisation or a larger receptive field would penalise it more.
- **Monet → photo.** This direction is weaker (FID 113.6), which is expected with 300 source paintings but would matter if the reverse direction were ever the goal.

## Files

- Code: `src/cyclegan.py`, `src/evaluate_local.py`, config: `src/config.yaml`, notebook with outputs: `src/task3_cyclegan.ipynb`
- Metrics: `full_metrics_report.csv`, Kaggle file: `submission.csv`
- Translations: `outputs/pred_B2A/` (photo → Monet), `outputs/pred_A2B/` (Monet → photo)
- Training samples per epoch: `outputs/train_samples/`, loss curves: `outputs/loss_curves.png`
- Human audit: `outputs/audit/`
- Raw logs and manifest: `logs/`, copied unedited to `reproducibility/`
- Checkpoints: `checkpoints/generators_final.pt` (plus every 5 epochs)
