# Masks & meshes for FoundationPose (V1)

FoundationPose needs, per tracked object:
- a **CAD mesh in meters** (`.obj`/`.ply`) — `franka_d435_foundationpose/assets/meshes/`
- an **initial binary mask** for the first frame — `franka_d435_foundationpose/assets/masks/`
  (or a bbox passed at runtime).

V1 mapping is declared in `configs/foundationpose.yaml` (`meshes_dir`,`masks_dir`).
Stage 1 (sim-GT) needs neither (it reads pose from the sim), so this only
matters once the real runner is wired up. Reuse the existing assets and the
mask loader in `franka_d435_foundationpose/franka_d435_foundationpose/foundationpose/`.
