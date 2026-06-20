<h1 align="center">Multi-label Instance-level Generalised Visual Grounding in Agriculture</h1>

<p align="center">  <strong>Mohammadreza Haghighat</strong><sup>1,2</sup>   <strong>Alzayat Saleh</strong><sup>1,2</sup>   <strong>Mostafa Rahimi Azghadi</strong><sup>1,2</sup></p>

<p align="center">  <sup>1</sup> College of Science and Engineering, James Cook University, Townsville, QLD, Australia <br>  <sup>2</sup> Centre for AI and Data Science Innovation, James Cook University, Townsville, QLD, Australia</p>

<p align="center">  <a href="https://arxiv.org/abs/2603.06699">    <img src="https://img.shields.io/badge/arXiv-2603.06699-b31b1b.svg" alt="arXiv">  </a>     <a href="https://myjcuedu-my.sharepoint.com/personal/reza_haghighat_my_jcu_edu_au/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Freza%5Fhaghighat%5Fmy%5Fjcu%5Fedu%5Fau%2FDocuments%2FgRef%2DCW&ga=1">    <img src="https://img.shields.io/badge/Dataset-gRef--CW-blue.svg" alt="Dataset">  </a></p>

## Abstract

gRef-CW is a generalized visual grounding benchmark for crop and weed instances
in field imagery, including multi-target and no-target expressions. Weed-VG
ranks detector proposals with hierarchical relevance scoring and refines boxes
with interpolation-driven regression.

## Installation

```bash
conda create -n weedvg python=3.11 -y
conda activate weedvg
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .
```

## Data

[Dataset and checkpoint](https://myjcuedu-my.sharepoint.com/:f:/g/personal/reza_haghighat_my_jcu_edu_au/IgC-WxqTt28fT49wwQNklsXqAVqm8XE9-bFJwHuYYshw4-A)

```text
data/
  images/
  grefs(unc).json
  instances.json
```

## Weights

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

This project is licensed under the Creative Commons Attribution 4.0
International (CC BY 4.0) licence:
https://creativecommons.org/licenses/by/4.0/

## Citation

```bibtex
@article{haghighat2026multi,
  title={Multi-label Instance-level Generalised Visual Grounding in Agriculture},
  author={Haghighat, Mohammadreza and Saleh, Alzayat and Azghadi, Mostafa Rahimi},
  journal={arXiv preprint arXiv:2603.06699},
  year={2026}
}
```
