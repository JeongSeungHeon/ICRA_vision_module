# External perception assets

Place the two unmodified external checkouts here. They are intentionally not
tracked because the SAM3D checkout and checkpoints are very large.

```text
external/
├── sam-3d-objects/
│   ├── FastSAM-s.pt
│   └── checkpoints/hf/pipeline.yaml
└── hands23_detector/
    ├── hodetector/modeling/roi_heads/__init__.py
    ├── Base-RCNN-FPN.yaml
    ├── faster_rcnn_X_101_32x8d_FPN_3x_Hands23.yaml
    └── model_weights/model_hands23.pth
```

The `hodetector/` package is required source code. A directory containing only
the YAML and weight file is not a usable Hands23 checkout and is rejected by
the startup preflight before SAM3D capture begins.

Validated source revisions:

- `sam-3d-objects`: `f91db411c50efee93d8db7aeb323885650f6f722`
- `hands23_detector`: `1bc0f919ffa7a9f375e7e8042e1d26f3743d5819`

Validated model hashes:

- `FastSAM-s.pt`: `sha256:c9f78716a81c7aff0d608ccc73e1b82ab3aaad86005049f6a92106a0be6d0844`
- `model_hands23.pth`: `sha256:b3d2ff966d8a19b991a3ce170e2d8f86040f614ef2edf4a55423288b278f723c`

The runtime paths and conda interpreters are configurable under `runtime` in
`configs/handover.yaml`. Do not copy integration code into either checkout.
