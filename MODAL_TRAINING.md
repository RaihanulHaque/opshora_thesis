# Clean Trial-and-Error Workflow on Modal

## The short version

You do **not** need to run preprocessing and training as two separate jobs.

For every experiment, use one command:

```bash
python submit_modal.py run --design hj_topogait_v1 --run baseline_001
python submit_modal.py run --design skeleton_contrastive_v3 --run baseline_001
python submit_modal.py run --design skeleton_rank1_v4 --run rank1_001
python submit_modal.py run --design skeleton_rank1_v5 --run rank1_multigallery_001
python submit_modal.py run --design skeleton_silhouette_fusion_v6 --run fusion_rank1_001
```

That remote job performs this flow automatically:

```text
Find shared preprocessing cache
        |
        +-- exists --> reuse it
        |
        +-- missing -> preprocess silhouette-C.zip once and save cache
        |
        v
Train selected design
        |
        v
Save config, exact model snapshot, metrics and checkpoints
```

Preprocessing exists because the original 320×240 silhouette frames must be cleaned, aligned, sampled to 30 frames, and converted into distance/topology maps before entering the model. Caching means later model designs reuse that expensive work instead of repeating it.

The separate `preprocess` command still exists as an optional cost optimization, but it is not required.

## One-time setup

The dataset has already been uploaded with:

```bash
modal volume put gait-datasets-store silhouette-C.zip /silhouette-C.zip
```

Deploy the application after cloning it, and redeploy whenever code or a design changes:

```bash
modal deploy modal_app.py
```

Deployment is setup, not a training step. After deployment, experiment submission uses the single command shown above.

## Running an experiment

```bash
python submit_modal.py run --design hj_topogait_v1 --run baseline_001
```

The submission returns immediately and prints a Modal call ID. The job is spawned on Modal and is independent of the local process, so the terminal can close and the laptop can be turned off.

Monitor it later:

```bash
modal app logs hj-topogait-training
```

## Design history

Each architecture lives in its own folder:

```text
designs/
  hj_topogait_v1/
    README.md       # hypothesis and architecture notes
    config.json     # default hyperparameters
    model.py        # model factory for this design
  hj_topogait_v2/   # future alternative
    README.md
    config.json
    model.py
  skeleton_contrastive_v3/
    README.md       # skeleton-first verification design
    config.json
    model.py
  skeleton_rank1_v4/
    README.md       # retrieval-focused follow-up
    config.json
    model.py
  skeleton_rank1_v5/
    README.md       # metric-heavy multi-gallery retrieval follow-up
    config.json
    model.py
  skeleton_silhouette_fusion_v6/
    README.md       # fused silhouette + Hamilton skeleton design
    config.json
    model.py
```

The `gait/` directory contains shared infrastructure such as dataset loading, preprocessing, losses, and the reusable base layers. The `designs/` directory represents the experimental ideas being compared.

Use these history rules:

1. Do not overwrite a design after using it for reported results.
2. Create `hj_topogait_v2`, `silhouette_baseline_v1`, or another descriptive folder for a changed architecture.
3. Give each training attempt a unique run name.
4. Change only hyperparameters with a per-run override file when the architecture remains identical.

Examples:

```bash
python submit_modal.py run --design hj_topogait_v1 --run baseline_001
python submit_modal.py run --design hj_topogait_v1 --run lower_lr_001 --config experiment.example.json
python submit_modal.py run --design hj_topogait_v2 --run baseline_001
python submit_modal.py run --design skeleton_contrastive_v3 --run baseline_001
python submit_modal.py run --design skeleton_rank1_v4 --run rank1_001
python submit_modal.py run --design skeleton_rank1_v5 --run rank1_multigallery_001
python submit_modal.py run --design skeleton_silhouette_fusion_v6 --run fusion_rank1_001
```

`hj_topogait_v2` is the first improvement experiment after the v1 best Rank-1 result of 61.22%. It reuses the same preprocessing cache and evaluation protocol, while changing condition sampling, sparse reconstruction loss, spatial part preservation, temporal modeling, update ratio, and learning-rate schedule. Its results are stored separately under `/experiments/hj_topogait_v2/`.

`skeleton_contrastive_v3` is the advisor-aligned skeleton-first verification experiment. It uses the Hamilton skeleton dataset, learns temporal motion with masked generative reconstruction, and uses contrastive/triplet distance learning without an ID classifier (`lambda_ce = 0`). Its main evidence metrics are `same_distance`, `different_distance`, `distance_gap`, `verification_auc`, and `verification_accuracy`, plus Rank-1/Rank-5 for retrieval comparison.

`skeleton_rank1_v4` is the Rank-1-focused follow-up. It keeps distance-based test embeddings, but adds a stronger part-aware temporal model and an auxiliary normalized identity head during training. It monitors `rank1` for early stopping and is the design to try if the target is Rank-1 closer to 0.70.

`skeleton_rank1_v5` is the safer follow-up after V4 underperformed. It removes the auxiliary identity classifier, strengthens metric learning, and uses a 3-gallery evaluation protocol. If its Rank-1 is higher, report it as **3-gallery Rank-1**, not directly equivalent to the earlier single-gallery Rank-1.

`skeleton_silhouette_fusion_v6` is the next step if skeleton-only Rank-1 remains weak. It pairs CASIA-B silhouettes with Hamilton skeleton sequences, then trains a two-stream fusion model. It requires an additional Modal upload:

```bash
zip -r GaitDatasetB-silh.zip datasets/GaitDatasetB-silh
modal volume put gait-datasets-store GaitDatasetB-silh.zip /GaitDatasetB-silh.zip
```

`experiment.example.json` now contains only per-run overrides. The selected design's `config.json` supplies the normal defaults.

## Saved history in the Modal Volume

Results are separated by design and run:

```text
/data/experiments/
  hj_topogait_v1/
    baseline_001/
      config.json
      model_snapshot.py
      metrics.jsonl
      latest.pt
      best_model.pt
      result_summary.json
      visuals/
        preprocessing_preview.png
        training_curves.png
        reconstruction_epoch_001.png
    lower_lr_001/
      ...
  hj_topogait_v2/
    baseline_001/
      ...
  skeleton_contrastive_v3/
    baseline_001/
      ...
```

`model_snapshot.py` is copied into every run directory, so the exact selected design remains visible with its results. `latest.pt` allows interrupted jobs to resume.

The visual directory contains:

- a preprocessing panel showing silhouette, topology, radius, and flux;
- topology/radius reconstruction comparisons every few epochs;
- continuously updated training-loss and Rank-1/Rank-5 curves.

For `skeleton_contrastive_v3`, those preview panels correspond to skeleton, structural skeleton field, temporal motion field, and reconstructed skeleton/motion channels.

All of these files use the same persistent Modal Volume. A second Volume is unnecessary because keeping weights, metrics, configuration, and figures together makes each run self-contained.

`baseline_001` and `lower_lr_001` are ordinary run names. `baseline_001` conventionally uses the design defaults; `lower_lr_001` conventionally uses the same model with a lower learning-rate override. The software does not attach special behavior to either name.

## Early stopping

Early stopping monitors Rank-1 accuracy on the held-out subject split. The v1 defaults are:

```json
"early_stopping_patience": 8,
"early_stopping_min_delta": 0.002
```

After generative warm-up, training stops when Rank-1 has failed to improve by at least 0.002 for eight evaluated epochs. `best_model.pt` preserves the best model, while `latest.pt` preserves the resumable training state. Set patience to `0` to disable early stopping.

Download all experiment history:

```bash
modal volume get gait-datasets-store /experiments ./modal-results
```

## Resources and persistence

- 2 physical CPU cores
- 8 GiB RAM
- GPU preference: L4, then A10, then T4
- persistent Volume: `gait-datasets-store`
- checkpoint and Volume commit after every epoch
- automatic retry and resume after interruption
- maximum 24 hours per function attempt

The remote **job** survives the laptop turning off. A particular cloud container is not guaranteed to live forever: Modal can preempt it or reach the 24-hour limit. The checkpoint/retry mechanism is what safely continues the experiment.

## Optional preprocessing-only command

Normally, ignore this section. If you want to avoid spending GPU time on the first cache build, submit preprocessing on the cheaper CPU container:

```bash
python submit_modal.py preprocess --design hj_topogait_v1 --run cache_build
```

After it finishes, all compatible designs using the same `cache_dir` reuse that cache. If a future experiment changes preprocessing settings such as image dimensions, sequence length, or topology extraction, assign that design a new `cache_dir`.

## Running the Hamilton skeleton verification design

The skeleton dataset should be present in the Modal Volume. Your upload command placed it under a Hamilton-skeleton path; the V3 config expects:

```text
/data/CASIA_B_Hamilton_Skeleton/CASIA_B_Hamilton_Skeleton.zip
```

The code also tries to resolve nearby Hamilton skeleton zip/folder paths automatically if Modal stored it slightly differently.

Run:

```bash
modal deploy modal_app.py
python submit_modal.py run --design skeleton_contrastive_v3 --run baseline_001
```

In the logs, the most thesis-relevant fields are:

```json
{
  "same_distance": 0.31,
  "different_distance": 0.62,
  "distance_gap": 0.31,
  "verification_auc": 0.88,
  "verification_accuracy": 0.81
}
```

The exact numbers above are illustrative. The desired behavior is: `same_distance` decreases, `different_distance` stays higher, `distance_gap` increases, and `verification_auc` moves toward 1.0.

## Scientific limitation of v1

`hamilton_jacobi_topology()` is a discrete HJ-inspired prototype. It combines a Euclidean distance field, average-flux shock strength, and Zhang-Suen topology-preserving thinning. It is suitable for the first falsification experiment but is not yet a complete reproduction of Siddiqi et al.'s flux-ordered thinning algorithm.
