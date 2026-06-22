<h1 align="center">Multi-label Instance-level Generalised Visual Grounding in Agriculture</h1>

<p align="center">  <strong>Mohammadreza Haghighat</strong><sup>1,2</sup>   <strong>Alzayat Saleh</strong><sup>1,2</sup>   <strong>Mostafa Rahimi Azghadi</strong><sup>1,2</sup></p>

<p align="center">  <sup>1</sup> College of Science and Engineering, James Cook University, Townsville, QLD, Australia <br>  <sup>2</sup> Centre for AI and Data Science Innovation, James Cook University, Townsville, QLD, Australia</p>

<p align="center">  <a href="https://arxiv.org/abs/2603.06699">    <img src="https://img.shields.io/badge/arXiv-2603.06699-b31b1b.svg" alt="arXiv">  </a>     <a href="https://mhaghighat98.github.io/WeedVG-gRefCW/">    <img src="https://img.shields.io/badge/%F0%9F%8C%90%20Project%20Page-WeedVG-2e7d32.svg" alt="Project Page">  </a>     <a href="https://huggingface.co/datasets/Mhaghighat98/gRef-CW">    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-gRef--CW-yellow.svg" alt="Dataset on Hugging Face">  </a>     <a href="https://doi.org/10.57967/hf/9244">    <img src="https://img.shields.io/badge/DOI-10.57967%2Fhf%2F9244-blue.svg" alt="DOI">  </a></p>

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

The gRef-CW **annotations** and the **Weed-VG checkpoints** are released on Hugging Face
(stable, with a DOI):

- **Dataset:** https://huggingface.co/datasets/Mhaghighat98/gRef-CW
- **DOI:** https://doi.org/10.57967/hf/9244

Download them, e.g. with the Hugging Face CLI:

```bash
pip install -U "huggingface_hub[cli]"
hf download Mhaghighat98/gRef-CW --repo-type dataset --local-dir gref-cw-hf
```

Then arrange the files so the evaluation scripts can find them:

```text
data/
  images/                # NOT distributed — built locally (see below)
  grefs(unc).json        # from Hugging Face Annotations/
  instances.json         # from Hugging Face Annotations/

Weed-VG/
  checkpoints/
    stage_one.pth        # from Hugging Face checkpoints/
    stage_two.pth        # from Hugging Face checkpoints/
```

> **Images are not distributed.** gRef-CW is derived from the **CropAndWeed** dataset, whose licence
> does **not** permit redistributing the images. Reconstruct `data/images/` from the original source:
>
> ```bash
> python scripts/build_dataset.py --data-dir data --auto
> ```
>
> This downloads the CropOrWeed9 images from CropAndWeed and links them to match `instances.json`
> (see [`DATASHEET.md`](DATASHEET.md)).

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
- GroundingDINO-T and GroundingDINO-L: [idea-research/groundingdino](https://github.com/idea-research/groundingdino).
- SAM3: [facebookresearch/sam3](https://github.com/facebookresearch/sam3).
  

## Licence

This repository uses **two** licences, and use of the images is governed by a **third** (upstream):

- **Code** (Weed-VG model + gRef-CW evaluation scripts) — **Apache-2.0**, see [`LICENSE`](LICENSE).
  Builds on GroundingDINO (Apache-2.0); original notices are retained.
- **gRef-CW annotations & checkpoints** — **CC BY 4.0**:
  https://creativecommons.org/licenses/by/4.0/
- **Images** — part of the **CropAndWeed** dataset, governed by its **non-commercial** licence;
  not redistributed here. You must obtain them from CropAndWeed and comply with that licence,
  which restricts commercial use. See https://github.com/cropandweed/cropandweed-dataset and
  [`DATASHEET.md`](DATASHEET.md).

When using gRef-CW, please cite both this work and CropAndWeed.

## Citation

```bibtex
@article{haghighat2026multi,
  title={Multi-label Instance-level Generalised Visual Grounding in Agriculture},
  author={Haghighat, Mohammadreza and Saleh, Alzayat and Rahimi Azghadi, Mostafa},
  journal={arXiv preprint arXiv:2603.06699},
  year={2026}
}
```

Dataset (Hugging Face):

```bibtex
@dataset{haghighat2026grefcw,
  title     = {gRef-CW: Generalised Referring Expressions for Crops and Weeds},
  author    = {Haghighat, Mohammadreza and Saleh, Alzayat and Rahimi Azghadi, Mostafa},
  year      = {2026},
  publisher = {Hugging Face},
  doi       = {10.57967/hf/9244},
  url       = {https://huggingface.co/datasets/Mhaghighat98/gRef-CW}
}
```
