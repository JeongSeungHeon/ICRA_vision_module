# External perception assets

Place the two unmodified external checkouts and the HOI-DETR checkpoint here.
They are intentionally not tracked because the repositories and model weights
are large.

```text
external/
├── sam-3d-objects/
│   ├── FastSAM-s.pt
│   └── checkpoints/hf/pipeline.yaml
├── HOI-DETR/
│   ├── mmdet/__init__.py
│   ├── projects/__init__.py
│   └── projects/configs/co_dino_vit/
│       └── co_dino_5scale_vit_large_coco_with_relation_only_all_losses_custom.py
└── checkpoints/
    └── epoch_5.pth
```

The bundled `mmdet/` and `projects/` packages are required source code. A
weights-only directory is rejected by startup preflight before SAM3D capture.

Validated source revisions:

- `sam-3d-objects`: `f91db411c50efee93d8db7aeb323885650f6f722`
- `HOI-DETR`: `1b367292f3833afd64a204bd4d9d84519541d035`

Validated model hashes:

- `FastSAM-s.pt`: `sha256:c9f78716a81c7aff0d608ccc73e1b82ab3aaad86005049f6a92106a0be6d0844`
- `epoch_5.pth`: `sha256:4708fd0ddc5c3d386bad67c31152de58676840b911f6d01219e0092a603277d3`

Create the dedicated environment and install the CUDA extensions with:

```bash
conda env create -f environment/hoi_detr.yml
bash environment/install_hoi_detr.sh
```

The runtime paths and conda interpreters are configurable under `runtime` in
`configs/handover.yaml`. Do not copy integration code into either checkout.
