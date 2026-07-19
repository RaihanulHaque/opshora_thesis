# Reconstruction quality: quantitative evaluation

- Design: `skeleton_silhouette_partset_v7`
- Checkpoint: `runs/skeleton_silhouette_partset_v7/clopgait_domain_split_001/best_rank1_model.pt`
- Test cache: `runs/clopgait_local_processed_cache` (split_mode=domain)
- Mask ratio used for this evaluation: 0.3
- Sequences evaluated: 58
- Frames total: 1740, masked (hidden from encoder): 522 (30.0%)

## Masked frames (true generative-quality test: encoder never saw these frames)

| Channel | Metric | Value |
|---|---|---|
| Skeleton (binary occupancy) | Dice | 0.3902 |
| Skeleton | IoU | 0.2424 |
| Skeleton | Precision | 0.2522 |
| Skeleton | Recall | 0.8610 |
| Skeleton | Pixel accuracy | 0.8440 |
| Skeleton | BCE | 0.2902 |
| Motion (continuous heatmap) | MSE | 0.024236 |
| Motion | MAE | 0.085245 |
| Motion | PSNR (dB) | 16.16 |
| Motion | SSIM | 0.3818 |

## All frames (includes frames the encoder saw directly -- NOT an upper bound, see note below)

| Channel | Metric | Value |
|---|---|---|
| Skeleton (binary occupancy) | Dice | 0.2141 |
| Skeleton | IoU | 0.1199 |
| Skeleton | Precision | 0.1209 |
| Skeleton | Recall | 0.9357 |
| Skeleton | Pixel accuracy | 0.6035 |
| Skeleton | BCE | 0.5693 |
| Motion (continuous heatmap) | MSE | 0.127677 |
| Motion | MAE | 0.298975 |
| Motion | PSNR (dB) | 8.94 |
| Motion | SSIM | 0.1212 |

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
