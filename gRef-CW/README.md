`N_Acc` is computed as `n_correct / n_model_eligible`, where model eligibility means the model first grounds the original positive sentence correctly.

For paper Table 5 compatibility, the scripts also report `N_Acc_avg_paper`, which uses a fixed denominator of 3616. This value is the Weed-VG model-eligible manipulated-query count on the released test split under the paper evaluation settings. It is included only to reproduce the paper’s cross-model Table 5 average. For new models or changed thresholds, report `N_Acc` and `N_Acc_n`, not `N_Acc_avg_paper`.

MDETR baseline boxes are ranked with the same objectness score used by MDETR’s RefCOCO postprocessor: `1 - P(no-object)`, where `P(no-object)` is the final token/no-object probability from `pred_logits`. This matches the upstream MDETR `PostProcess` behavior for RefCOCO-style referring-expression evaluation. The score is used only for ranking Top-1/Top-5 proposals; boxes come from `pred_boxes`.
The default baseline uses the MDETR ResNet-101 RefCOCO checkpoint. Other variants can be selected with `--mdetr-variant`.
