## Model Summary

The current setup is a **bounding box regressor** that combines **ResNet-50** for image encoding and **BERT** for text encoding, used in an **inference-only** pipeline.

- **Image features** are extracted from the `avgpool` layer of ResNet-50 (2048-dim).
- **Text features** use the `[CLS]` token from the last layer of BERT (768-dim), representing the entire sentence.
- Both image and text embeddings are projected to a shared 512-dimensional space via a **linear layer**.
- The model uses these projected features for bounding box regression.

No training is involved—only inference is performed.

 TO DO:
  Finalise the annotation file.
  Modify current architecture.
  
