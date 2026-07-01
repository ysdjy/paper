# SAPIEN asset pipeline notes (V1)

Known V1 assets (`SapienAssetPipeline/usd_assets/`):
- `Cabinet_44853/cabinet.usd` — drawer articulation (open/close_drawer).
- `Microwave_7320/microwave_flattened.usd` (and `microwave_referenceable.usd`) —
  door articulation (open/close_door).
- `CoffeeMachine_103046/coffeemachine.usd` — static prop.
- `Knife_101054/knife.usd` — graspable articulation.
- `Fridge_12252/fridge.usd` — available, not in the V1 object set.

Conversion steps (in `convert_sapien_asset.py`): sanitize mesh/URDF names →
`mobility_isaac.urdf` → call IsaacLab `convert_urdf` → emit USD.

The V1 base task cfg (`stack_joint_pos_env_cfg.py`) references these USDs by path
and applies the V0 fixed-scene poses. To add a new appliance to V1: convert it
here, then add it in the layout editor and Save V1.
