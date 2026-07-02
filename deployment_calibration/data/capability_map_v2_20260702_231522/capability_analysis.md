# Stage-1 capability map — v2

Run: `capability_map_v2_20260702_231522`  (100 episodes)


## middle_drawer  (n=50, success=0.58)
- terminal position: [0.000, 0.328], std=0.108
- failure modes: {'NONE': 29, 'POSITION_TIMEOUT': 14, 'HANDLE_DETACHED': 7}
- grasp_offset_local_y: [-0.06,-0.02]=0.312(16) [-0.02,0.02]=1.0(18) [0.02,0.06]=0.375(16)
- max_pos_step: [0.008,0.017]=0.625(16) [0.017,0.026]=0.526(19) [0.026,0.035]=0.6(15)
- pull_lead: [0.03,0.0667]=0.647(17) [0.0667,0.1033]=0.588(17) [0.1033,0.14]=0.5(16)
- proposed band per dim: {'grasp_offset_local_y': (-0.06, 0.06), 'max_pos_step': (0.008, 0.035), 'pull_lead': (0.03, 0.14)}
- verdict: MIX (usable)

## sektion_top_drawer  (n=50, success=0.94)
- terminal position: [0.000, 0.286], std=0.076
- failure modes: {'NONE': 47, 'HANDLE_DETACHED': 3}
- grasp_offset_local_y: [-0.06,-0.02]=0.812(16) [-0.02,0.02]=1.0(18) [0.02,0.06]=1.0(16)
- max_pos_step: [0.008,0.017]=1.0(16) [0.017,0.026]=0.895(19) [0.026,0.035]=0.933(15)
- pull_lead: [0.03,0.0667]=1.0(17) [0.0667,0.1033]=0.882(17) [0.1033,0.14]=0.938(16)
- proposed band per dim: {'grasp_offset_local_y': (-0.06, 0.06), 'max_pos_step': (0.008, 0.035), 'pull_lead': (0.03, 0.14)}
- verdict: ALL-SUCCESS