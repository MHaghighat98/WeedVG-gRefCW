# gRef-CW

Generalized crop/weed visual grounding benchmark with multi-target and
no-target expressions.

## Data

https://myjcuedu-my.sharepoint.com/:f:/g/personal/reza_haghighat_my_jcu_edu_au/IgC-WxqTt28fT49wwQNklsXqAVqm8XE9-bFJwHuYYshw4-A

```text
data/
  images/
  grefs(unc).json
  instances.json
```

## Evaluation

```bash
python gRef-CW/eval_weedvg.py --help
python gRef-CW/eval_baselines.py --help
bash gRef-CW/run_all_evals.sh
```
