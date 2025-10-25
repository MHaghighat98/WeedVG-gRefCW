### 1. Proposal Output Transformation (Pre-Decoder) in Grounding DINO (GDINO) architecture

This stage generates coarse object proposals using **GDINO**.

1️⃣ **Inputs to GDINO**

* Input Image: $I \in \mathbb{R}^{H \times W \times 3}$
* Input Text: $T_{\text{pos}}$ (positive instance-level sentences)
* Dual Encoding:
  * Image $I \rightarrow$ embeddings from Swin-T.
  * Text $T_{\text{pos}} \rightarrow$ embedding from BERT.
  * Fuse embeddings basd on GDINO's architecture to produce multi-modal features to create up to 900 proposals.

2️⃣ **Top Proposal Selection and Output**

* Selection: Select top-$N_q$ proposals (the number of selected proposals depend on the maximum number of instances in our images which is currently 206, we can remove images that contain high number of instances):
    $
    I_{N_q} = \text{Top-}N_q \big(\max^{-1}(X_I X_T^\top)\big)
    $
* Proposal Output: Bounding boxes and scores from GDINO:
    $
    P_{\text{GDINO}} = \{(b_i, s_i)\}_{i=1}^{N_q}
    $

3️⃣ **Instance Query Initialization**

* Calculate box centers: $P_r = \{p_i \in \mathbb{R}^2\}_{i=1}^{N_q}$.
* Create Instance Queries: Combine visual features at proposals ($F_{\text{GDINO}}$) with respective sentence embeddings ($T_{\text{pos,i}}$):
    $
    Q_{\text{instance}} = \text{MLP}(\text{Concat}(F_{\text{GDINO}}, T_{\text{pos,i}})) \in \mathbb{R}^{N_q \times C}
    $

---

### 2. Decoder Refinement

Purpose: Refine coarse proposals into instance-aware queries using **Multi-Scale Deformable Attention (MSDeformAttn)**.

1️⃣ **MSDeformAttn Mechanism**

For each query $q_i$ at reference point $p_i$ which we will replace them with $Q_{\text{instance}}$ and $P_r$ respectively:

$
\text{MSDeAttn}(q_i, p_i, \{x^{(l)}\}) = \sum_{m=1}^{M} W_m \sum_{l=1}^{L} \sum_{k=1}^{K} A_{mlik} \, W_m^0 \, x^{(l)} \big(\phi_l(p_i) + \Delta p_{mlik}\big)
$

where

* $m$ indexes the attention head, and $M$ represents the total number of attention heads.

* $k$ indexes the number of sampling point for attention, and $K$ denotes the total number of points ($K \ll H_l W_l$).

* $l$ indexes the input feature level, and $L$ refers to the total number of input feature levels (for handling different sizes of instances).

* $\Delta p_{mlik}$ and $A_{mlik}$ denote the sampling offset and attention weight of the $k^{\text{th}}$ sampling point in the $m^{\text{th}}$ attention head, respectively.

* The query feature $q_i$ is used to compute the attention weights $A_{mlik}$.

* The function $\phi_l(p_i)$ re-scales the normalized coordinates $p_i$ to the spatial coordinates on the input feature map of the $l^{\text{th}}$ level.

* $W_m^0$ and $W_m$ are learnable projection matrices applied to the sampled features and to the output of each head, respectively.

2️⃣ **Output**

* Refined Queries (by sampling attention around box centers in different scales): $Q_d = \{q_i \in \mathbb{R}^C\}_{i=1}^{N_q}$.

---

### 3. Negative-Aware Instance Contrastive Learning (Sentence Supervision)

1️⃣ **Inputs**

For each image:

* Refined instance queries from the decoder: $Q_d$
* Positive instance sentences embedding: $T_{\text{pos,i}}$
* Negative image-level sentences embedding: $T_{\text{neg}}$

2️⃣ **Supervised Contrastive Loss**

contrastive pull apart negative and positive samples

supervised version extends that by pulling samples from the same class and pushing samples from other class

hierarchical version puts label levels and penalties for each level

For each instance query $q_i \in Q_d$:



## Hierarchical Multi-label Contrastive Loss

Let $L = \{0, 1\}$ be the set of label levels in the multi-label hierarchy, with $l \in L$ representing a specific level. For an anchor instance query $q_i \in Q_d$ (from the decoder, Section 2), and a positive text embedding $T_{\text{pos,i}}^l$ at level $l$, the pairwise contrastive loss is defined as:

$$
\text{L}^{\mathrm{pair}}(q_i, T_{\text{pos,i}}^l) = -\log \left( \frac{\exp(q_i \cdot T_{\text{pos,i}}^l / \tau)}{\sum_{a \in A \setminus \{i\}} \exp(q_i \cdot f_a / \tau)} \right)
$$

where:

* $q_i \in \mathbb{R}^C$ is the refined instance query for anchor $i$ from $Q_d$.
* $T_{\text{pos,i}}^l \in \mathbb{R}^C$ is the text embedding for the positive instance at level $l$:
  * For $l = 0$: $T_{\text{pos,i}}^0$ represents image-level label embeddings similar to $q_i$.
  * For $l = 1$: $T_{\text{pos,i}}^1$ represents instance-level label embeddings similar to $q_i$.
* $A$ is the set of all instance queries and text embeddings in the batch (including positive and negative samples).
* $f_a$ represents the embedding of sample $a \in A$ :
  * For $l = 0$: $f_a$ represents other image-level label embeddings in a batch dissimilar to $q_i$.
  * For $l = 1$: $f_a$ represents other instance-level label embeddings in a batch dissimilar to $q_i$.
* $\tau > 0$ is a temperature parameter controlling the softness of the softmax.

The Hierarchical Multi-label Contrastive Loss (HiMulCon) is defined as:

$$
\text{L}^{\mathrm{HMC}} = \sum_{l \in L} \frac{1}{|L|} \sum_{i \in I_{\text{anchor}}} \frac{\lambda_l}{|P_l(i)|} \sum_{p_l \in P_l(i)} \text{L}^{\mathrm{pair}}(q_i, T_{\text{pos,i}}^l)
$$

where:

* $I_{\text{anchor}}$ is the set of anchor instance indices.
* $\lambda_l = F(l)$ is a controlling parameter applying a fixed penalty for level $l$.
* $P_l(i)$ is the set of positive text embeddings for anchor $i$ at level $l$:
  * For $l = 0$: $P_0(i) = \{ T_{\text{pos,j}}^0 \mid \text{represents image-level label embeddings similar to } q_i \}$.
  * For $l = 1$: $P_1(i) = \{ T_{\text{pos,j}}^1 \mid \text{represents instance-level label embeddings similar to } q_i \}$.

---

## 

## Hierarchical Multi-label Constraint Enforcing Contrastive Loss

Define the maximum pairwise loss for positive pairs at level $l$:

$$
\text{L}^{\mathrm{pair}}_{\mathrm{max}}(l) = \max_{(i, p_l^i)} \text{L}^{\mathrm{pair}}(q_i, T_{\text{pos,i}}^l)
$$

The combined loss, Hierarchical Multi-label Constraint Enforcing Contrastive Loss ($\text{L}_{\text{HMCE}}$), is defined as:

$$
\text{L}_{\text{HMCE}} = \sum_{l \in L} \frac{1}{|L|} \sum_{i \in I_{\text{anchor}}} \frac{\lambda_l}{|P_l(i)|} \sum_{p_l \in P_l(i)} \max\left(\text{L}^{\mathrm{pair}}(q_i, T_{\text{pos,i}}^l), \text{L}^{\mathrm{pair}}_{\mathrm{max}}(l-1)\right)
$$

This loss combines the level-specific penalty $\lambda_l$ from Eq. (4) with the hierarchical constraint from Eq. (6), ensuring both multi-label alignment and hierarchical consistency.



3️⃣ **Loss Integration**

* Total Loss: $L_\text{total} = \lambda_{\text{L1}} L_\text{L1} + \lambda_{\text{GIoU}} L_\text{GIoU} + \lambda_{\text{HMCE}}L_{\text{HMCE}}​$

---

### Training Procedure

* Visual Forward Pass: Image $x \rightarrow$ instance queries $Q_d$.
* Text Forward Pass: Positive and negative sentences $\rightarrow$ text encoder $\rightarrow T_{\text{pos.i}}, T_{\text{neg}}$.
* Calculate Losses
* Backpropagation: Update queries and align them with sentences jointly, $q_i$ is trained to be close to $T_{\text{pos,i}}$ (and far from negatives considering the hierarchy).

### Inference

For each decoded query $q_i \in Q_d$:

* Scoring: $S_{\text{exist}} = \text{sim}(q_i, T_{\text{pos,i}})$
* Decision Rule:
  * Instance exists if $S_{\text{exist}} > \tau_{\text{exist}} \rightarrow$ keep prediction.
  * Instance absent or misidentified if $S_{\text{exist}} < \tau_{\text{exist}} \rightarrow$ suppress prediction.
