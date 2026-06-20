<h1 align="center">Multi-label Instance-level Generalised Visual Grounding in Agriculture</h1>

<p align="center">  <strong>Mohammadreza Haghighat</strong><sup>1,2</sup>   <strong>Alzayat Saleh</strong><sup>1,2</sup>   <strong>Mostafa Rahimi Azghadi</strong><sup>1,2</sup></p>

<p align="center">  <sup>1</sup> College of Science and Engineering, James Cook University, Townsville, QLD, Australia <br>  <sup>2</sup> Centre for AI and Data Science Innovation, James Cook University, Townsville, QLD, Australia</p>

<p align="center">  <a href="https://arxiv.org/abs/2603.06699">    <img src="https://img.shields.io/badge/arXiv-2603.06699-b31b1b.svg" alt="arXiv">  </a>     <a href="https://doi.org/10.5281/zenodo.XXXXXXX">    <img src="https://img.shields.io/badge/Dataset-gRef--CW-blue.svg" alt="Dataset">  </a></p>

## Abstract

gRef-CW is a generalized visual grounding benchmark for crop and weed instances
in field imagery, including multi-target and no-target expressions. Weed-VG
ranks detector proposals with hierarchical relevance scoring and refines boxes
with interpolation-driven regression.

📄 **Dataset documentation:** see [`DATASHEET.md`](DATASHEET.md).

## Installation

```bash
conda create -n weedvg python=3.11 -y
conda activate weedvg
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .
```

## Data and Weights

The gRef-CW **annotations**, the **Weed-VG checkpoints**, and a dataset **build script** are
released on Zenodo (a stable, citable archive with a DOI):

**Zenodo (DOI):** https://doi.org/10.5281/zenodo.XXXXXXX  ← _TODO: replace with the real DOI once the record is published_

> **Note on images.** gRef-CW is derived from the **CropAndWeed** dataset, whose licence does **not**
> permit redistributing the images. We therefore release only the annotations. The images are
> obtained from the original source — run `scripts/build_dataset.py` (see [`DATASHEET.md`](DATASHEET.md)),
> which downloads the CropOrWeed9 images from CropAndWeed and arranges them to match `instances.json`.

After building, the data layout is:

```text
data/
  images/                # from CropAndWeed (not redistributed here)
  grefs(unc).json        # referring expressions (CC BY-NC-SA 4.0)
  instances.json         # boxes / masks / categories (CC BY-NC-SA 4.0)
```

Include Weed-VG checkpoints:

```text
Weed-VG/
  checkpoints/
    stage_one.pth
    stage_two.pth
```

Place GroundingDINO backbones under:

```text
Weed-VG/
  weights/
    groundingdino_swint_ogc.pth
    groundingdino_swinb_cogcoor.pth
```

## Evaluation

```bash
bash gRef-CW/eval_weedvg.sh
bash gRef-CW/run_all_evals.sh
```

```bash
python gRef-CW/eval_weedvg.py --help
python gRef-CW/eval_baselines.py --help
```

Baselines:
 Follow their README for installation and checkpoint access.
 
- MDETR: [ashkamath/mdetr](https://github.com/ashkamath/mdetr).
- GroundingDINO: [idea-research/groundingdino](https://github.com/idea-research/groundingdino).
- SAM3: [facebookresearch/sam3](https://github.com/facebookresearch/sam3).
  

## Licence

This repository uses **two** licences, and use of the images is governed by a **third** (upstream):

- **Code** (Weed-VG model + gRef-CW evaluation scripts) — **Apache-2.0**, see [`LICENSE`](LICENSE).
  Builds on GroundingDINO (Apache-2.0); original notices are retained.
- **gRef-CW annotations & checkpoints** — **CC BY-NC-SA 4.0** (non-commercial, share-alike):
  https://creativecommons.org/licenses/by-nc-sa/4.0/
- **Images** — part of the **CropAndWeed** dataset, governed by its **non-commercial** licence;
  not redistributed here. See https://github.com/cropandweed/cropandweed-dataset and
  [`DATASHEET.md`](DATASHEET.md).

**Commercial use is not permitted.** When using gRef-CW, please cite both this work and CropAndWeed.

## Citation

```bibtex
@article{haghighat2026multi,
  title={Multi-label Instance-level Generalised Visual Grounding in Agriculture},
  author={Haghighat, Mohammadreza and Saleh, Alzayat and Azghadi, Mostafa Rahimi},
  journal={arXiv preprint arXiv:2603.06699},
  year={2026}
}
```
