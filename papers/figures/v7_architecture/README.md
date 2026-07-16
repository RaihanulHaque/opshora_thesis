# V7 thesis figures (working-principle flow diagram + parameter budget)

Research-paper-style figures for the thesis book and defense slides, built
with LaTeX/TikZ. Every number in them is real and traceable:

- Architecture dimensions come from `designs/skeleton_silhouette_partset_v7/model.py`.
- Metrics (Rank-1 67.9%, Rank-5 91.3%, AUC 91.2%) come from
  `runs/skeleton_silhouette_partset_v7/partset_rank1_001/metrics.jsonl`
  (CASIA-B, 3-gallery, 50 unseen subjects, at the official best-Rank-1 epoch).
- Parameter counts were measured on the built model (see the comment block
  at the top of `v7_parameter_budget.tex` for the exact per-module numbers).
- The input frames in the flow diagram are real CASIA-B silhouette/Hamilton-
  skeleton frames (subject 001, nm-01, 090); the embedding scatter is the
  real PCA plot from the run's post-training analysis.

## Files

```text
v7_flow_diagram.tex / .pdf / .png        main working-principle diagram
v7_parameter_budget.tex / .pdf / .png    stacked parameter-budget bars
*_preview.png                            low-res previews (for quick viewing)
assets/                                  embedded images (frames + PCA plot)
```

Use the `.pdf` in the thesis book (vector, scales cleanly):

```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/v7_flow_diagram.pdf}
  \caption{Working principle of the Part-Set Fusion V7 model. ...}
\end{figure*}
```

Use the `.png` (300 dpi) in the defense slides.

## Rebuild

```bash
pdflatex v7_flow_diagram.tex
pdflatex v7_parameter_budget.tex
magick -density 300 v7_flow_diagram.pdf -background white -flatten v7_flow_diagram.png
```

If the model changes, re-measure the parameter groups before editing the
budget figure:

```bash
python -c "
import json
from gait.config import ExperimentConfig
from designs.skeleton_silhouette_partset_v7 import model as v7
config = ExperimentConfig.from_dict(json.loads(open('designs/skeleton_silhouette_partset_v7/config.json').read()))
m = v7.build_model(config, 74)
from collections import defaultdict
groups = defaultdict(int)
for k, p in m.named_parameters(): groups[k.split('.')[0]] += p.numel()
for k, v in sorted(groups.items(), key=lambda kv: -kv[1]): print(f'{k}: {v:,}')
"
```
