**Pipeline**
------------------------------------------------------

### 1.  Dataset & Batching

The dataset annotation follows a standard structure:

* **`grefs(unc).json`**: Contains all natural language descriptions (grounding references).

* **`instances.json`**: Contains all bounding box numbers.

The batch sampler ensures a balanced representation of **three negative sentence types** and **one empty** image, with a minimum batch size of 82 images.

* * *

### 2.  Model Architecture

The model processes inputs in three main stages.

#### 2.1. Frozen GroundingDINO Encoder (Stage 1: Proposals)

The frozen GroundingDINO encoder (using a Swin-T backbone) generates initial object proposals and rich visual features.

* **Input**: Images (e.g., 1920×1088, resized to 800×453) + a generic `"plant"` prompt.

* **Output**:
  
  1. **Multi-scale Visual Features** (1024-dim)
  
  2. **Initial Proposal Boxes** with low IoU threshold (noisy object proposals).

#### 2.2. Query Initialization (Stage 2: Concatenation)

Instance queries are initialized by extracting visual features at each proposal location and concatenating them with corresponding text embeddings (assigned during matching).

* Process: Direct concatenation of features:
  $\mathbf{Q}_{\text{instance}} = [\mathbf{V}_{\text{features (1024-dim)}}, \mathbf{T}_{\text{embeddings (768-dim)}}]$

* **Shape**: `[Batch Size, Num_Queries, 1792]`

#### 2.3. Trainable Decoder (Stage 3: Refinement)

The **trainable** decoder (using deformable attention and self-attention) refines the initial queries.

* **Input**:
  
  * $\mathbf{Q}_{\text{instance}}$: The 1792-dim multimodal features.
  
  * $\mathbf{P}_r$: **Proposal box centers** (used as initial reference points).
  
  * Encoder Features: For deformable attention.

* **Output**:
  
  * $\mathbf{Q}_d$: **Refined embeddings** (256-dim) for contrastive loss.
  
  * $\mathbf{B}_{\text{pred}}$: **Refined boxes** (proposal centers + predicted deltas).

* * *

### 3.  Training Pipeline

#### 3.1. Hungarian Matching (Supervision Assignment)

This process matches decoder queries (anchored at proposals) to Ground Truth (GT) boxes to assign supervision.

| **Output**                        | **Matched Queries (IoU > 0.1)** | **Unmatched Queries**                       |
| --------------------------------- | ------------------------------- | ------------------------------------------- |
| **Mask** ($\mathbf{valid\_mask}$) | **True** (Supervised)           | **False**                                   |
| **Text Assignment**               | Specific GT sentence embeddings | Generic pooled negative sentence embeddings |
| **Losses Applied**                | HMCE + L1 + GIoU                | None                                        |

* **Key Point**: Only matched queries contribute to both semantic (HMCE) and spatial (L1/GIoU) loss calculations.

#### 3.2. Hierarchical Label Construction (for HMCE)

The HMCE loss uses a two-level hierarchy to learn semantics:

* **Level 0 (Image-level)**: Uses **negative sentences** to determine the _existence_ of crops/weeds. All queries in an image share this label.

* **Level 1 (Instance-level)**: Uses **positive sentences** for _instance discrimination_.

Adaptive penalty weights ($\lambda_0, \lambda_1$) are applied based on the image type to manage emphasis:

| **Image Type**          | **λ0​ (Image)** | **λ1​ (Instance)** | **Emphasis**                   |
| ----------------------- | --------------- | ------------------ | ------------------------------ |
| Mixed-category          | 1.0             | 2.5                | Strong instance discrimination |
| Single-category         | 1.0             | 2.0                | Standard emphasis              |
| Negative (no instances) | 2.0             | 1.0                | Focus on absence detection     |

#### 3.3. Loss Computation

The final training objective combines the semantic (HMCE) and spatial (L1, GIoU) losses.

Final Loss Combination

$\mathcal{L}_{\text{total}} = \lambda_{\text{hmce}} \mathcal{L}_{\text{HMCE}} + \lambda_{\text{l1}} \mathcal{L}_{\text{L1}} + \lambda_{\text{giou}} \mathcal{L}_{\text{GIoU}}$

Hierarchical Multi-label Constraint Enforcing Contrastive Loss ($\mathcal{L}_{\text{HMCE}}$)

This loss averages the weighted contributions from both levels, enforcing the constraint that Level 1 loss (instance) must be at least as high as the max Level 0 loss (image).

$\mathcal{L}_{\text{HMCE}} = \frac{1}{2} \sum_{i \in \mathcal{I}} \left[ \left( \frac{- \lambda_0}{|P_0(i)|} \sum_{p_0 \in P_0(i)} \mathcal{L}^{\text{pair}}(i, p_0) \right) + \left( \frac{- \lambda_1}{|P_1(i)|} \sum_{p_1 \in P_1(i)} \max \left( \mathcal{L}^{\text{pair}}(i, p_1), \mathcal{L}^{\text{pair}}_{\text{max}}(0) \right) \right) \right]$

* * *

### 4.  Training Dynamics

* **What is Trained**: **Only the decoder parameters** (self-attention, deformable-attention, FFN, and box embedding predictors).

* **What is Frozen**: The **GroundingDINO encoder** and the **BERT text encoder**.


