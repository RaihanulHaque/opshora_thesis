# Reconstruction quality: quantitative evaluation

- Design: `skeleton_silhouette_partset_v7`
- Checkpoint: `runs/skeleton_silhouette_partset_v7/partset_rank1_001/best_rank1_model.pt`
- Test cache: `runs/fusion_rank1_002/local_processed_cache_full` (split_mode=subject)
- Mask ratio used for this evaluation: 0.3
- Sequences evaluated: 1197
- Frames total: 35910, masked (hidden from encoder): 10773 (30.0%)

## Masked frames (true generative-quality test: encoder never saw these frames)

| Channel | Metric | Value |
|---|---|---|
| Skeleton (binary occupancy) | Dice | 0.3199 |
| Skeleton | IoU | 0.1904 |
| Skeleton | Precision | 0.2018 |
| Skeleton | Recall | 0.7709 |
| Skeleton | Pixel accuracy | 0.8677 |
| Skeleton | BCE | 0.2133 |
| Motion (continuous heatmap) | MSE | 0.015560 |
| Motion | MAE | 0.058567 |
| Motion | PSNR (dB) | 18.08 |
| Motion | SSIM | 0.6486 |

## All frames (includes frames the encoder saw directly -- NOT an upper bound, see note below)

| Channel | Metric | Value |
|---|---|---|
| Skeleton (binary occupancy) | Dice | 0.1954 |
| Skeleton | IoU | 0.1083 |
| Skeleton | Precision | 0.1113 |
| Skeleton | Recall | 0.8010 |
| Skeleton | Pixel accuracy | 0.7339 |
| Skeleton | BCE | 0.4124 |
| Motion (continuous heatmap) | MSE | 0.073442 |
| Motion | MAE | 0.209036 |
| Motion | PSNR (dB) | 11.34 |
| Motion | SSIM | 0.2340 |

## How to read this

- **Dice / IoU** (0 to 1, higher is better) treat the skeleton map as a binary
  segmentation problem: does the decoder put a body-part pixel where one
  actually is? Dice above ~0.7 is considered a strong overlap in medical/
  gait segmentation literature; below ~0.3 means the reconstruction is close
  to guessing.
- **PSNR** (dB, higher is better) is the standard image-fidelity metric used in
  autoencoder/GAN reconstruction papers; 20-30 dB is typical for lossy but
  recognizable reconstructions of small (64x64) maps.
- **SSIM** (-1 to 1, higher is better) captures structural similarity rather than
  raw pixel error, so it does not reward a decoder that outputs a blurry mean
  frame -- it rewards actually recovering the pose's shape and motion pattern.
- These are computed strictly on the **held-out test split**, using the same
  temporal masking the model saw during training, so they are not inflated by
  memorization of the training set.
- **Why 'all frames' looks worse than 'masked frames':** the training loss only
  ever back-propagates through masked (hidden) frames -- the decoder is not
  directly supervised to reconstruct frames the encoder could already see. So
  'masked frames' is the metric that reflects what the model was actually
  trained to do, and it is the one that should be quoted as "reconstruction
  quality"; 'all frames' is reported only for completeness.
