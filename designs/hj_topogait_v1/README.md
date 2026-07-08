# HJ-TopoGait v1

Initial prototype described in `THESIS_METHOD_PROPOSAL.md`.

## Design

- silhouette stream plus topology/radius/flux stream;
- gated fusion;
- bidirectional GRU temporal encoder;
- masked topology reconstruction;
- supervised contrastive, identity, and triplet objectives;
- alternating generative and recognition mini-steps.

## History rule

Once this design has produced a recorded run, do not repurpose it for another architecture. Copy the folder to a new name such as `hj_topogait_v2`, change that model/config, and submit it with a new run name.

Example:

```bash
python submit_modal.py run --design hj_topogait_v1 --run baseline_001
```

`baseline_001` is simply the first run using this design's default configuration. A name such as `lower_lr_001` means the architecture is unchanged while that run overrides the learning rate. These names have no hidden behavior.

Training uses Rank-1 retrieval accuracy for early stopping. The default patience is eight recognition epochs with a minimum improvement of 0.002.
