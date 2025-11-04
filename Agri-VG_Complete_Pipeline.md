**Agri-VG: Agricultural Visual Grounding Pipeline**
============================================================

**1. Dataset Preparation**
--------------------------

### **1.1 Dataset Structure**

The **CropAndWeed** dataset follows the **gRefCOCO** format with two main annotation files:

#### `grefs(unc).json` — Grounding References

Contains natural language descriptions (grounding references) for each instance in the images.

#### `instances.json` — Instance Annotations

Includes bounding box annotations and category labels for all instances.

* * *

### **1.2 Data Loading Pipeline**

The dataset loading process:

* Maps images to their corresponding annotations

* Filters valid images

* Prepares data for batch processing during training

* * *

**2. Model Architecture**
-------------------------

### **2.1 GroundingDINO (Frozen)**

#### **Feature Extraction**

* Multi-scale visual features are extracted using the **frozen GroundingDINO**.

* Text embeddings are generated using **BERT**.

**Example:**  
An image (1920×1088, resized to 800×453) is processed by the **Swin-T backbone** at four scales, producing **1024-dimensional multi-level visual features**.  
The text prompt _“plant”_ is encoded into a **768-dimensional [CLS] embedding**.  
Through **cross-attention**, visual and textual features are aligned to produce **grounded (text-aware) visual representations**.

* * *

### **2.2 Agri-VG Decoder (Trainable)**

The **Agri-VG decoder** applies multiple **deformable attention layers** to refine joint visual–text features.

**Real Example Processing:**

* **Input:** 206 queries × 1792 dimensions (1024 visual + 768 text)

* **Reference Points:** Ground truth box centers

* **Deformable Attention:**
  
  * Query 0 attends to _“medium maize crop in the top left”_ region
  
  * Samples **16 points** (4×4) across **4 feature levels** around the GT center
  
  * Aggregates localized visual features

* **Output:** 206 queries × 256 dimensions (refined embeddings)

* **Training Effect:** Learns to focus attention on semantically relevant regions

* * *

### **2.3 Text Encoder (Frozen)**

The **BERT text encoder** converts natural language descriptions into embeddings.  
Embeddings are cached for efficient reuse during training.

* * *

**3. Training Pipeline**
------------------------

### **3.1 Data Batching Strategy**

The batch sampler ensures balanced representation of various **negative sentence types** across batches.

* * *

### **3.2 Forward Pass (Two-Stage Process)**

Each training batch is processed through two main stages to transform raw inputs into refined visual-language queries.

#### **Stage 1: Feature Extraction**

* Extracts multi-scale visual features using the **frozen GroundingDINO encoder**.

* Generates initial object proposals.

#### **Stage 2: Instance Query Refinement**

* Refines instance queries via text encoding, **Hungarian matching**, and **decoder processing**.

* * *

### **3.3 Hungarian Matching Algorithm**

#### **Purpose**

Assigns ground truth (GT) boxes to decoder queries for **HMLC loss supervision**.

#### **Outputs**

1. **valid_query_mask:** Boolean tensor marking which queries are matched (IoU > 0.1).
   
   * **True:** Matched queries → supervised with HMLC + spatial losses
   
   * **False:** Unmatched queries → spatial losses only

2. **Text assignment strategy:**
   
   * Matched queries → receive specific GT sentence embeddings
   
   * Unmatched queries → receive generic pooled sentence embeddings

#### **Process**

* Assign GT box centers to matched queries

* Zero out unmatched embeddings

* Include all queries for forward pass (transformer requires fixed size)

* Apply **L1** and **GIoU** losses to all queries

#### **Training Strategy**

* **Matched queries (IoU > 0.1):**  
  Supervised using hierarchical contrastive (HMLC) and spatial losses

* **Unmatched queries:**  
  Used only for spatial localization learning

#### **Impact on Training**

* All **206 queries** are processed

* **HMLC loss** applied only to matched queries

* **Spatial losses (L1, GIoU)** applied to all queries

* * *

### **3.4 Hierarchical Label Construction**

#### **Level Definitions**

* **Level 0 (Image-level):** Determines existence of crops/weeds
  
  * Negative sentences: _“no crops/weeds visible”_

* **Level 1 (Instance-level):** Fine-grained discrimination
  
  * Positive sentences: _“large corn crop in center”_, etc.

* * *

#### **Example: Image vwg-0286-0002.jpg**

**Input Data**

* **Positive Sentences:**
  
  * “medium maize crop in the top left”
  
  * “small maize crop in the bottom left”
  
  * “large weed in the middle center”

* **Negative Sentence:** “no crops or weeds are visible in this image”

* **Categories:** `["crop", "weed"]`

* **Queries:** 206

**Level 0 (Image-level):**

* All queries share the same label indicating crop/weed presence.

**Level 1 (Instance-level):**

* Queries are cyclically assigned to positive sentences for instance-specific learning.

* * *

#### **Adaptive Penalty Calculation**

| Image Type              | $\lambda_0$ (Image-level) | $\lambda_1$ (Instance-level) | Emphasis                       |
| ----------------------- | ------------------------- | ---------------------------- | ------------------------------ |
| Mixed-category          | 1.0                       | 2.5                          | Strong instance discrimination |
| Single-category         | 1.0                       | 2.0                          | Standard emphasis              |
| Negative (no instances) | 2.0                       | 1.0                          | Focus on absence detection     |

* * *

###### **3.5 Loss Computation**

The total training loss combines the **HMCE** and the spatial localization losses ($\text{L1}$ and $\text{GIoU}$).

#### **Hierarchical Multi-label Constraint Enforcing Contrastive Loss (HMCE)**

With two levels, $\mathcal{L} = \{0, 1\}$, the total $\mathcal{L}_{\text{HMCE}}$ is the average of the loss from Level 0 and the constrained loss from Level 1.

$\mathcal{L}_{\text{HMCE}} = \frac{1}{2} \left[ \mathcal{L}_{\text{Level 0}} + \mathcal{L}_{\text{Level 1, constrained}} \right]$

$\mathcal{L}_{\text{HMCE}} = \frac{1}{2} \sum_{i \in \mathcal{I}} \left[ \left( \frac{- \lambda_0}{|P_0(i)|} \sum_{p_0 \in P_0(i)} \mathcal{L}^{\text{pair}}(i, p_0) \right) + \left( \frac{- \lambda_1}{|P_1(i)|} \sum_{p_1 \in P_1(i)} \max \left( \mathcal{L}^{\text{pair}}(i, p_1), \mathcal{L}^{\text{pair}}_{\text{max}}(0) \right) \right) \right]$

* * *

### **Explanation of Components**

This formula is the average (due to the $\frac{1}{2}$ term, where $|\mathcal{L}|=2$) of the contributions from the two levels:

1. **Level 0 Term (Image-level):**
   
   * Uses the **independent penalty** $\lambda_0$.

2. **Level 1 Term (Instance-level):**
   
   * Uses the **independent penalty** $\lambda_1$. 

Note: $\mathcal{L}^{\text{pair}}_{\text{max}}(0)$ is calculated as:

$\mathcal{L}^{\text{pair}}_{\text{max}}(0) = \max_{(i, p_0)} \mathcal{L}^{\text{pair}}(i, p_0)$

* * *

#### **Final Loss Combination**

The final training objective is a weighted sum of the hierarchical contrastive loss and the spatial localization losses:

$\mathcal{L}_{\text{total}} = \lambda_{\text{hmce}} \mathcal{L}_{\text{HMCE}} + \lambda_{\text{l1}} \mathcal{L}_{\text{L1}} + \lambda_{\text{giou}} \mathcal{L}_{\text{GIoU}}$

* $\lambda_{\text{hmlc}}, \lambda_{\text{l1}}, \lambda_{\text{giou}}$ are the respective loss weights.
