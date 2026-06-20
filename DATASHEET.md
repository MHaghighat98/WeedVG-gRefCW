# Datasheet — gRef-CW

A datasheet for the **gRef-CW** dataset (Generalised Referring expressions for Crops and Weeds),
following Gebru et al., *"Datasheets for Datasets"* (2021).

> **Items marked `TODO` must be confirmed/filled by the authors before release.**

gRef-CW accompanies the paper *"Multi-label Instance-level Generalised Visual Grounding in
Agriculture"* (Haghighat, Saleh, Rahimi Azghadi; ECCV 2026) and the Weed-VG model in this
repository.

---

## ⚠️ Read first — what is and isn't distributed

gRef-CW is **derived from the CropOrWeed9 subset of the CropAndWeed dataset** (Steininger et al.,
WACV 2023; AIT). The CropAndWeed licence permits distributing **derivative annotations** but
**prohibits redistributing the images** ("You do not distribute this dataset or modified
versions"). Therefore:

| Component | Distributed by us | Where | Licence |
|---|---|---|---|
| Annotations (`grefs(unc).json`, `instances.json`: referring expressions, boxes, masks) | ✅ Yes | Zenodo + this repo | CC BY-NC-SA 4.0 |
| Weed-VG checkpoints (`stage_one.pth`, `stage_two.pth`) | ✅ Yes | Zenodo | CC BY-NC-SA 4.0 |
| Source/code (model + eval) | ✅ Yes | This repo | Apache-2.0 |
| **Images** (`data/images/`) | ❌ **No** | Obtained by the user from CropAndWeed | CropAndWeed (non-commercial) |

A helper script (`scripts/build_dataset.py` — **TODO: add**) downloads the CropAndWeed CropOrWeed9
images from the official source and arranges them to match `instances.json`.

---

## Motivation

- **For what purpose was the dataset created?** To enable and benchmark *generalised* visual
  grounding (gVG) in agricultural field imagery — localising crop and weed instances from natural
  language, including **multi-target** and **no-target (negative)** expressions. Existing referring
  datasets (RefCOCO/+/g, ReferItGame) are natural-image, single-target, and lack negative
  expressions; no prior VG benchmark exists for agriculture.
- **Who created it / funded it?** Mohammadreza Haghighat, Alzayat Saleh, and Mostafa Rahimi Azghadi,
  College of Science and Engineering & Centre for AI and Data Science Innovation, James Cook
  University, Australia. Funding: **TODO**.

## Composition

- **What do the instances represent?** High-resolution agricultural field images, each with
  instance-level crop/weed objects (bounding boxes + segmentation masks) and natural-language
  referring expressions at both the **image level** and **instance level**.
- **How many instances are there in total?**
  - **8,034** images.
  - **~78,000** crop/weed object instances across **9 categories** (8 crops: Maize, Sugar beet,
    Bean, Pea, Sunflower, Soy, Potato, Pumpkin; + 1 Weed class).
  - **82,592** language annotations total = **78,288** instance-level expressions + **4,304**
    image-level expressions. Average expression length **6.34** words.
- **Negative (no-target) expressions.** Included at both levels — a distinguishing feature of
  gRef-CW. In the **test set**, negatives are generated from 11,997 candidates per change type,
  yielding **9,186** negative instance-level expressions (**3,706** negative-category, **3,294**
  negative-size, **2,186** negative-position) via *Replace* and *Swap* strategies; image-level
  negatives assert the absence of crops/weeds/both.
- **Attributes per instance.** Category; position (3×3 grid, e.g. "top-left" … "bottom-center");
  size bucket — **tiny** (<2k px²), **small** (2k–20k px²), **medium** (20k–208k px²), **large**
  (≥208k px²); bounding box; segmentation mask. Expressions follow the template
  *"(Size) (Category) in the (Position)"*.
- **Is any data missing?** Instances smaller than **16×16 px** are excluded (not human-identifiable).
- **Distributions.** Instance scale: **tiny 50.7%, small 34.0%, medium 13.7%, large 1.7%** (so
  **84.7%** are tiny/small). Scene density: 1–10 instances = **69.3%** of images, 11–20 = **18.7%**,
  21–30 = **6.4%**, >30 = **5.6%** (so **30.7%** have >10). Instance area ranges ~**0.01%–0.97%** of
  the image; square-root instance size **16–1,402 px** (Table 1, Fig. 3).
- **Splits.** Train/Val/Test = **70/15/15**, balanced across images containing only-crops,
  only-weeds, both, or neither.
- **Image resolution.** High-resolution; square-root image-area **≈1,445 px** (Table 1). Native
  pixel resolution (W×H) is inherited from CropAndWeed — **TODO: state exact dimensions.**

## Collection process

- **Source.** Images and base masks come from the CropAndWeed dataset (CropOrWeed9 subset). gRef-CW
  does **not** collect new imagery.
- **Annotation pipeline.** Images were pre-segmented into soil vs. vegetation (initially colour
  thresholding, later a CNN-based segmentation model). Bounding boxes were generated from masks and
  **manually refined**; ambiguous or densely populated images were validated by **multi-annotator
  voting**. Attributes (size/position/category) are computed programmatically; referring expressions
  are generated from templates; test-set negatives are produced by category/size/position
  Replace/Swap.
- **Who annotated, and how were they compensated?** **TODO.**
- **Over what timeframe was the data collected/annotated?** **TODO.**
- **Ethical review.** Subject matter is plants/soil imagery with **no human subjects, faces, or
  PII**, so human-research-ethics/IRB review is not applicable. (arXiv v1 has no acknowledgements
  section — **TODO: add funding sources for camera-ready.**)

## Preprocessing / cleaning / labelling

- See *Annotation pipeline* above. Annotations are stored in a **gRefCOCO-compatible** format:
  - `instances.json` — COCO-style: `images`, `annotations` (per-instance `bbox`, `segmentation`,
    `category_id`, `area`), `categories`.
  - `grefs(unc).json` — referring expressions: per-ref `ref_id`, `image_id`, `split`,
    `ann_id`(s) (empty list ⇒ no-target/negative), `sentences`, and category/size/position metadata.
  - **TODO: confirm the exact field names/schema and document each field here.**

## Uses

- **Intended use.** Training and evaluation of generalised visual grounding / generalised referring
  expression comprehension models for precision agriculture; benchmarking existence-aware,
  instance-level grounding. Benchmark metrics: **Recall@0.5, Top-k Accuracy, mIoU, Neg-Acc**
  (threshold-free GIoU-based negative accuracy). Reference baselines in the paper: **MDETR,
  GroundingDINO-T, GroundingDINO-L, SAM3**.
- **Out-of-scope / discouraged uses.** **Commercial use is prohibited** (non-commercial licence,
  see below). Not validated for deployment-grade weed-control decisions without further testing.
- **Known limitations / biases.** Template-generated expressions (limited linguistic diversity);
  crop set limited to 9 CropOrWeed9 categories; geographic/seasonal coverage inherited from
  CropAndWeed; heavy tiny/small-object skew.

## Distribution

- **How is it distributed?** Annotations + checkpoints + build script on **Zenodo**
  (DOI: **TODO — `10.5281/zenodo.XXXXXXX`**); code on GitHub
  (https://github.com/MHaghighat98/WeedVG-gRefCW). Images are **not** redistributed — users fetch
  them from CropAndWeed.
- **Licence.**
  - Annotations & checkpoints: **CC BY-NC-SA 4.0** — https://creativecommons.org/licenses/by-nc-sa/4.0/
  - Code: **Apache-2.0** (see `LICENSE`).
  - Images: governed by the **CropAndWeed licence** (non-commercial); see
    https://github.com/cropandweed/cropandweed-dataset.
- **Third-party IP / restrictions.** Use of gRef-CW requires compliance with the upstream CropAndWeed
  licence (non-commercial). Cite both gRef-CW and CropAndWeed.

## Maintenance

- **Maintainer / contact.** Mohammadreza Haghighat (reza.haghighat@my.jcu.edu.au),
  Alzayat Saleh (alzayat.saleh@my.jcu.edu.au). **TODO: confirm primary contact.**
- **Versioning / updates.** Versioned via the Zenodo record (DOI per version). **TODO: erratum/update
  policy.**

## Citation

```bibtex
@inproceedings{haghighat2026multi,
  title     = {Multi-label Instance-level Generalised Visual Grounding in Agriculture},
  author    = {Haghighat, Mohammadreza and Saleh, Alzayat and Rahimi Azghadi, Mostafa},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

Also cite the source dataset:

```bibtex
@inproceedings{steininger2023cropandweed,
  title     = {The CropAndWeed Dataset: A Multi-Modal Learning Approach for Efficient Crop and Weed Manipulation},
  author    = {Steininger, Daniel and Trondl, Andreas and Croonen, Gerardus and Simon, Julia and Widhalm, Verena},
  booktitle = {IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2023}
}
```
