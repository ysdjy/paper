# environment_snapshot.md

Text captures only (no environment body committed).

## python --version
```
Python 3.11.15
```
## which python
```
/home1/banghai/miniconda3/envs/env_isaaclab/bin/python
```
## git --version
```
git version 2.34.1
```
## uname -a
```
Linux SPRL6201012107U 6.8.1-1052-realtime #53~22.04.1-Ubuntu SMP PREEMPT_RT Tue May 26 19:37:49 UTC 2026 x86_64 x86_64 x86_64 GNU/Linux
```
## nvidia-smi
```
Sun Jul  5 14:53:15 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.159.03             Driver Version: 580.159.03     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Quadro RTX 8000                Off |   00000000:15:00.0 Off |                  Off |
| 33%   30C    P8             13W /  260W |   14453MiB /  49152MiB |      1%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  Quadro RTX 8000                Off |   00000000:2D:00.0 Off |                  Off |
| 33%   45C    P8             22W /  260W |    8667MiB /  49152MiB |     14%      Default |
|                                         |                        |                  N/A |
```
## key package versions (pip freeze subset)
```
isaacsim==5.1.0.0
isaacsim-app==5.1.0.0
isaacsim-asset==5.1.0.0
isaacsim-benchmark==5.1.0.0
isaacsim-code-editor==5.1.0.0
isaacsim-core==5.1.0.0
isaacsim-cortex==5.1.0.0
isaacsim-example==5.1.0.0
isaacsim-extscache-kit==5.1.0.0
isaacsim-extscache-kit-sdk==5.1.0.0
isaacsim-extscache-physics==5.1.0.0
isaacsim-gui==5.1.0.0
isaacsim-kernel==5.1.0.0
isaacsim-replicator==5.1.0.0
isaacsim-rl==5.1.0.0
isaacsim-robot==5.1.0.0
isaacsim-robot-motion==5.1.0.0
isaacsim-robot-setup==5.1.0.0
isaacsim-ros1==5.1.0.0
isaacsim-ros2==5.1.0.0
isaacsim-sensor==5.1.0.0
isaacsim-storage==5.1.0.0
isaacsim-template==5.1.0.0
isaacsim-test==5.1.0.0
isaacsim-utils==5.1.0.0
numpy==1.26.0
numpy-quaternion==2024.0.13
pytest==9.0.3
pytest-mock==3.15.1
scipy==1.15.3
torch==2.7.0+cu128
torchaudio==2.7.0
torchvision==0.22.0+cu128
```
## conda env
```
env: env_isaaclab
name: env_isaaclab
  - python=3.11.15
      - gitpython==3.1.46
      - ipython==9.10.1
      - numpy==1.26.0
      - numpy-quaternion==2024.0.13
      - scipy==1.15.3
```
## Isaac Sim / Isaac Lab
- Isaac Sim version: NOT_AVAILABLE (not needed for offline phase)
- Isaac Lab commit: repository ysdjy/paper (see commit_lineage.md); offline analysis does not import Isaac
