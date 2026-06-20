# Multi-label Instance-level Generalised Visual Grounding in Agriculture

Mohammadreza Haghighat, Alzayat Saleh, Mostafa Rahimi Azghadi

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

- MDETR: [ashkamath/mdetr](https://github.com/ashkamath/mdetr), pass
  `--mdetr-root /path/to/mdetr`.
- SAM3: [facebookresearch/sam3](https://github.com/facebookresearch/sam3),
  follow its README for installation and checkpoint access.

## Structure

```text
gRef-CW/        Dataset notes and evaluation
Weed-VG/        Model code, configs, checkpoints, weights
```

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
