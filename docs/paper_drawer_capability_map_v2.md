# Stage-1 drawer capability map — v2

Run `capability_map_v2_20260702_231522`: Sobol over (grasp_offset_local_y, max_pos_step, pull_lead)
× {0.12, 0.20, 0.28}, 50 points/drawer, baseline damping (3.0). middle_drawer + sektion_top_drawer.

## Result

| drawer | success | terminal pos [min,max] (std) | failure modes | verdict |
|---|---|---|---|---|
| **middle_drawer** | **0.58** | [0.000, 0.328] (0.108) | NONE 29 / POSITION_TIMEOUT 14 / HANDLE_DETACHED 7 | **MIX — usable** |
| sektion_top_drawer | 0.94 | [0.000, 0.286] (0.076) | NONE 47 / HANDLE_DETACHED 3 | ALL-SUCCESS — too easy |

## middle_drawer — per-dim marginal success (terciles)
- **grasp_offset_local_y**: [-0.06,-0.02] = **0.31** · [-0.02,0.02] = **1.00** · [0.02,0.06] = **0.38**
  → **dominant driver**: centered grasp always works; large |offset| = edge-miss → failure.
- max_pos_step: 0.625 / 0.526 / 0.60 → weak effect.
- pull_lead: 0.647 / 0.588 / 0.50 → mild (larger pull_lead slightly harder).

This is exactly the candidate space we want: a continuous terminal-position signal (std 0.108, range
0.00–0.33), a 20–80% overall band, multiple failure modes, and **orderable** candidates (grasp offset
monotonically separates success from failure).

## Frozen formal candidate theta space
Written to `configs/drawer_theta_frozen_v2.yaml` and adopted by `damping_pilot_v1.yaml`:

    grasp_offset_local_y: [-0.06, 0.06]     # spans center-success to edge-fail boundary
    max_pos_step:         [0.008, 0.035]
    pull_lead:            [0.03, 0.14]
    targets:              [0.12, 0.20, 0.28]

## Drawer selection for round-1
- **Pilot / core validation: middle_drawer** — the only drawer with a genuine success/fail mix at
  baseline damping, so candidates are differentiable and the hidden-damping effect will be visible.
- **sektion_top_drawer**: kept for later cross-mechanism generalization, but at baseline damping it is
  near-saturated (0.94). It will become informative once higher damping is applied (harder to pull), so
  it is a good *cross-mechanism* probe in the formal stage, not a standalone candidate space now.
- top_drawer (Stage-0 stable) not mapped here; available if a second cabinet drawer is needed.

## Next
Damping micro-sweep on middle_drawer → pick 3 non-degenerate hidden damping levels → pilot
(9 sessions × 8 eps) → B0/B1/B2/Oracle eval → D1–D5 → Go/Modify/Stop.
