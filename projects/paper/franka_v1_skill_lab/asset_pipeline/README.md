# asset_pipeline/ — SAPIEN → USD assets (pointer)

## 1. What this module does
Converts SAPIEN / PartNet-Mobility assets (URDF + textured mesh) into Isaac USD
used by the V1 scene (cabinet, microwave, coffee machine, knife, fridge). This
is a **pointer** module — the real tool is `SapienAssetPipeline/` at the repo
root; nothing is duplicated here.

## 2. Upstream dependencies
- `SapienAssetPipeline/tools/convert_sapien_asset.py`, IsaacLab `convert_urdf` tool.

## 3. Downstream consumers
- `SapienAssetPipeline/usd_assets/` → the V1 base task cfg + `layout_editor`.

## 4. Common commands
```bash
python SapienAssetPipeline/tools/convert_sapien_asset.py list
python SapienAssetPipeline/tools/convert_sapien_asset.py convert 44853 --name Cabinet
```

## 5. Current status
**wrapper** (notes only). See `sapien_asset_pipeline_notes.md`.

## 6. Troubleshooting
- *USD not found by layout editor* — confirm the output landed in
  `SapienAssetPipeline/usd_assets/<Name_id>/<name>.usd`.
