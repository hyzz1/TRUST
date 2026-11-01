# Method overview

TRUST applies reliable source selection and graph residual correction to native NACLIP features.

| Operation | Implementation |
|---|---|
| Native CLIP patch responses | `base_model.py`, `clip/` |
| External visual features | `trust/dino.py` |
| Candidate associations | `trust/graph.py` |
| Source reliability and selection | `trust/rgss.py` |
| Selected-feature aggregation | `trust/model.py` |
| Graph residual correction | `trust/grc.py` |
| Segmentation decoding | `base_model.py` |

Ground truth is used for evaluation after inference.
