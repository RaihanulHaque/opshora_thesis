# Thesis Context & Engine Specification: Cross-Domain Gait Recognition
**System Intent:** Deep Learning Architecture Context for Coding Agents
**Target Framework:** PyTorch + Modal.com (Serverless GPU Deployment)

---

## 1. Project Overview & Core Objectives
The objective of this thesis is to construct a robust cross-domain gait recognition framework capable of identifying human subjects via walking dynamics across severe environmental transitions.

## 1. Formal Academic Objectives
Any code architecture, evaluation script, or dataset pipeline generated must explicitly fulfill and map back to these three core objectives:

1.  **Objective 1:** To construct a cross-domain gait dataset covering indoor-outdoor settings under varying day-night illumination & clothing conditions.
2.  **Objective 2:** To design a generative and contrastive paradigm framework for robust representation accuracy.
3.  **Objective 3:** To evaluate the performance of our dataset and model based on Existing work.

### Core Targets
*   **Cross-Domain Transitions:** Seamless transition modeling between Indoor/Outdoor, Day/Night illumination, and baseline vs. heavy clothing variations.
*   **Angle Invariance:** Robust feature identification across specific viewing camera angles: $0^\circ$, $90^\circ$, and $180^\circ$.
*   **Primary Benchmarks:** CASIA-B (Indoor, Daylight, Multi-Clothing) and CASIA-C (Outdoor, Night Vision).
*   **Target Domain Evaluation:** Validation on a newly curated 20-subject custom dataset covering edge combinations missing from legacy datasets.

---

## 2. Preprocessing & The Mathematical Novelty
Most contemporary architectures rely heavily on joint-based skeletons (e.g., OpenPose, HRNet). These pipelines break down catastrophically when tracking subjects wearing heavy clothing (e.g., winter coats distorting torso/limb keypoints) or under low-light night-vision noise where joint visibility drops to zero.

### The Innovation: Hamilton Medial Axis Skeleton
To solve this, a custom preprocessing pipeline was developed based on geometric topology principles:
1.  **Input:** Raw binary walking silhouettes extracted from video frames.
2.  **Transformation:** Application of a continuous distance field calculation where every interior point maps its distance to the closest silhouette boundary point:
    $$D(p) = \min_{b \in \partial \Omega} \|p - b\|_2$$
3.  **Axis Extraction:** The Medial Axis (Hamilton Skeleton) is defined mathematically as the set of all interior points that possess two or more distinct closest points on the silhouette boundary ($\partial \Omega$):
    $$M = \{p \in \Omega \mid \exists b_1, b_2 \in \partial \Omega, b_1 \neq b_2 \text{ s.t. } \|p - b_1\|_2 = \|p - b_2\|_2 = D(p)\}$$


```

[Raw Silhouette] ──> [Euclidean Distance Map] ──> [Medial Axis Extraction]
(Noisy)              (Continuous Field)             (Pure Topology)

```

### Why this is the "Novelty" Pitch:
*   **Shape Over Joint Stability:** Unlike structural joints, the Medial Axis extracts a sparse, continuous topological skeleton representing the *pure geometric symmetry* of the walking entity.
*   **Robustness:** Even if a heavy coat expands the body silhouette or night-shadows blur fine details, the medial axis smoothly shifts to find the updated continuous center of mass, filtering out structural noise.

---

## 3. Current Architecture Engineering Bottlenecks
The model has been breaking down and crashing local systems and Kaggle instances due to structural memory leaks in the training graph.

### The Advisor's Constraint
The architecture must tightly couple a **Generative Paradigm** and a **Contrastive Paradigm** simultaneously. 

### Why the Code is Crashing (Memory Heap Leak):
The training loop is currently passing full temporal video frame tensor sequences into a generative encoder-decoder step, calculating a generative/adversarial loss, and passing those high-dimensional internal hidden state graphs directly into a contrastive clustering step. 

Because both backpropagation graphs are alive in GPU VRAM simultaneously, memory usage spikes exponentially, throwing `CUDA Out of Memory (OOM)` errors or triggering hardware engine crashes.

---

## 4. Proposed Architectural Solution for Coding Agents
The implementation must build a joint framework that dynamically splits or sequentially detaches graphs to optimize execution.


```

```
   [Skeletal Motion Dynamics Sequence]
                   │
                   ▼
     [Feature Extraction Encoder]
                   │
     ┌─────────────┴─────────────┐
     ▼ (Generative Track)         ▼ (Contrastive Track)

```

[Reconstruction G]           [Projection Head]
│                            │
Compute: L_gen               Compute: L_contrastive
└─────────────┬──────────────┘
▼
Joint Loss Optimizations

```

### Dual-Paradigm Logic
1.  **Generative Track (Self-Supervised):** An Encoder-Generator-Discriminator sub-network trains on the skeletal motion sequence to reconstruct frame posture dynamics, forcing the encoder to learn detailed spatial-temporal patterns of a human walking cycle.
2.  **Contrastive Track (Subject Separation):** A projection head maps the encoder’s features into a lower-dimensional embedding space. It applies a Contrastive Loss (e.g., InfoNCE or SupCon) to pull embeddings from the same person close together (even if angle/clothing changes) and push embeddings from different subjects far apart.
3.  **The Optimization Objective:**
    $$\mathcal{L}_{\text{Total}} = \alpha \mathcal{L}_{\text{Gen}} + \beta \mathcal{L}_{\text{Contrastive}}$$
    *Agent instruction: Implement graph detaching or sequential backpropagation (`loss.backward()` with distinct optimizer steps) to prevent VRAM aggregation.*

---

## 5. Modal.com Cloud Infrastructure Specification
To run massive coordinate grids safely and unlock high-throughput scaling, the solution must run via `modal.com` serverless environments.

### Target Deployment Script Design
Use the configuration pattern below to define the remote execution runtime for this architecture:

```python
import modal

app = modal.App("gait-recognition-engine")

# Image Definition with required deep learning packages
gait_image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "torchvision",
        "numpy",
        "pandas",
        "scikit-learn",
        "opencv-python-headless"
    )
)

# Remote Volume for persistent dataset storage (CASIA-B, CASIA-C, and Custom Dataset)
volume = modal.Volume.from_name("gait-datasets-store", create_if_missing=True)

@app.function(
    image=gait_image,
    gpu="A10G", # Scalable high-performance VRAM instance to handle dual loss tracking
    volumes={"/data": volume},
    timeout=7200
)
def train_gait_model():
    # Agent: Implement memory-efficient training loops here
    print("Initializing Generative-Contrastive Joint Pipeline...")
    pass

```

---

## 6. Prompt Engineering Directives for AI Agents

When generating code or debugging using this context file, adhere to these strict constraints:

* **Memory Management:** Always break the joint backpropagation loop into decoupled gradient paths or use explicit tracking parameters (`with torch.no_grad():` where applicable) to guarantee execution under 12GB/24GB VRAM.
* **Input Assumption:** Assume inputs are tensor structures derived from the mathematical Hamilton Medial Axis calculations, not raw pixel imagery or traditional skeleton coordinate formats.
* **Deliverables Needed:** A clean dataset loader parsing split zipped sequences, a dual-objective model architecture, and a streamlined serverless orchestrator script ready for running via Modal.

```
