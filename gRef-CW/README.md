## Notes

### Prerequisites and Dependencies

Required for all runs:

* GroundingDINO weights in `weights/`
* Weed-VG checkpoint in `checkpoints/`
* gRef-CW data in `data/`

Additional for MDETR:

* clone `https://github.com/ashkamath/mdetr`
* set `MDETR_ROOT` in `gRef-CW/run_all_evals.sh`
* allow the MDETR RefCOCO checkpoint download, or pre-cache it

Additional for SAM3:

* install SAM3 separately
* ensure `import sam3` works in the active environment

---

### Scripts and Reproduction

* The shell scripts are canonical for paper reproduction. Paper numbers use `--text-threshold 0.95 --box-threshold 0.01`.
* `gRef-CW/run_all_evals.sh` is an all-in-one reproduction script. It assumes all optional baseline dependencies are installed before launch. Because it runs with `set -e`, any missing external dependency or failed baseline stops the script.

---

### Metrics and Evaluation

* `N_Acc` is computed as `n_correct / n_model_eligible`, where model eligibility means the model first grounds the original positive sentence correctly.
* For paper Table 5 compatibility, the scripts also report `N_Acc_avg_paper`, which uses a fixed denominator of 3616. This value is the Weed-VG model-eligible manipulated-query count on the released test split under the paper evaluation settings. It is included only to reproduce the paper’s cross-model Table 5 average. For new models or changed thresholds, report `N_Acc` and `N_Acc_n`, not `N_Acc_avg_paper`.
* MDETR baseline boxes are ranked with the same objectness score used by MDETR’s RefCOCO postprocessor: `1 - P(no-object)`, where `P(no-object)` is the final token/no-object probability from `pred_logits`. This matches the upstream MDETR `PostProcess` behavior for RefCOCO-style referring-expression evaluation. The score is used only for ranking Top-1/Top-5 proposals; boxes come from `pred_boxes`.
* The default baseline uses the MDETR ResNet-101 RefCOCO checkpoint. Other variants can be selected with `--mdetr-variant`.
* Evaluation uses CUDA FP16 autocast by default for speed. For exact FP32 reproducibility, pass `--no-amp` to `eval_weedvg.py`. `eval_baselines.py` currently runs CUDA evaluation under autocast. To compare exact FP32 numbers, disable autocast manually or run on CPU.

---

### Checkpoints

* `stage_one.pth` is not used by the released evaluation scripts. The paper evaluation path uses the original GroundingDINO backbone checkpoint as `--base-detector-checkpoint` and the final Weed-VG checkpoint (`stage_two.pth`) as both `--detector-checkpoint` and `--projector-checkpoint`.
* If provided, `stage_one.pth` is a training/intermediate checkpoint only for validation/debugging in the first training stage. It is not required to reproduce the reported Weed-VG evaluation numbers unless a separate training/resume workflow explicitly references it.
