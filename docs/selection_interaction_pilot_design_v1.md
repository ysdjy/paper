# Selection-interaction pilot — design (v1)

**Question.** In the damping pilot we showed the hidden per-session drawer DAMPING changes outcome
*probabilities* (D1) and is recoverable from probes (D3–D4). This pilot asks the stronger, selection-
relevant question: **does the hidden damping change which candidate θ is *optimal*** — i.e. is a
state-aware selector (reads probe history → picks θ) genuinely better than a single state-agnostic best
θ? If the best candidate never moves with damping, calibration buys prediction but not *decisions*.

## Three problems in the damping pilot this design fixes
1. **Unmatched candidates.** Each session drew its own Sobol candidates, so candidate sets differed across
   damping levels → no strict paired comparison. → **Fixed matched candidate bank** (below).
2. **grasp_offset dominance.** `grasp_offset_local_y` was the dominant main effect (capability map:
   `[-0.02,0.02]`=1.00 success, `|y|` large = edge-miss fail), masking the `max_pos_step`/`pull_lead` ×
   damping interaction. → **Confine grasp offset to the reliable zone `[-0.02,0.02]`** and vary the
   candidates along `max_pos_step × pull_lead` instead.
3. **Too few groups.** One target, one candidate group per session → only 3 test groups → regret (D5)
   was quantized and inconclusive. → **3 targets × 3 replicates = 9 matched groups**, appearing at all 3
   damping levels; **27 within-session candidate groups** total.

## Matched candidate bank (`configs/selection_interaction_candidate_bank_v1.json`)
- 5 archetypes on a conservative→aggressive gradient, **the same θ for every target** (matched
  archetypes), with `candidate_id` = `<target>_<archetype>`:

  | archetype | grasp_offset_y | max_pos_step | pull_lead | intent |
  |---|---|---|---|---|
  | c0_conservative | 0.000 | 0.010 | 0.040 | slow, gentle → expected most robust at high damping |
  | c1_steady | −0.015 | 0.015 | 0.060 | modest |
  | c2_balanced | +0.015 | 0.020 | 0.080 | nominal |
  | c3_brisk | −0.010 | 0.028 | 0.110 | brisk |
  | c4_aggressive | +0.010 | 0.034 | 0.140 | fast at low damping → expected to detach/timeout at high |

- **Frozen guarantees** (enforced by `tests/test_matched_candidate_bank_v1.py`):
  - `candidate_id → θ` is a pure function written in the JSON; **never** redrawn from a session seed,
    replicate, or the hidden damping/state.
  - grasp offsets stay in `[-0.02,0.02]` and their ordering ≠ the aggressiveness ordering, so no single
    grasp offset explains the candidate ranking.
  - differentiation lives on `max_pos_step`/`pull_lead` (≥4 distinct values each per target); no
    duplicate/exact-copy candidates; all θ within the frozen capability-map ranges.
- **The interaction we expect** (physics prior from capability map + micro-sweep, set *before* observing
  outcomes; not tuned to a result): at low damping the drawer follows easily so the **aggressive** c4 is
  fastest and best under `U = p_succ − 1.0·err − 0.02·time`; at high damping (≈40, near the detach
  threshold) the aggressive long-carrot pull races ahead of the lagging drawer → handle detach /
  position timeout, while the **conservative** c0 keeps the handle attached and still reaches the target.
  If that holds, the argmax candidate switches with damping → state-aware selection wins.

## Run structure (`configs/selection_interaction_pilot_v1.yaml`)
- drawer = `middle_drawer`; damping levels `{3, 20, 40}` (frozen micro-sweep band).
- 3 replicate sessions per level → **9 sessions**. `replicate_id` spans damping (matched).
- per session: **3 fixed probes** (`probe_library_v2` P0/P1/P2) + **3 targets × 5 candidates** = 18 eps.
- **162 episodes** total. Within-session candidate group = `{session}__{target}` (27 groups). Cross-
  damping matched group = `{drawer}__{target}__r{replicate}` (9 groups, **no damping in the id**), each
  appearing at all 3 damping levels.

## Provenance / invariants (per episode)
Every episode: independent **full reset** (`full_reset_verified` computed from reset invariants), damping
**set + verified** (`damping_eff`) before and **re-read** (`damping_post`) after, then skill execution.
New fields on top of the frozen `open_drawer_v2` contract (purely additive — no contract change):
`candidate_bank_version`, `candidate_bank_sha256`, `candidate_id`, `candidate_archetype`, `target_id`,
`matched_group_id`, `replicate_id`, `runtime_design_commit`. `git_info()` is captured **before** the
output dir is created so a clean committed tree yields `dirty_worktree=false`. `metadata.json` carries the
full bank snapshot + sha256, the condition assignment, and the per-target `candidate_id` lists.

## Not in scope (Claude B owns)
No offline model, AUROC, held-out D3, bootstrap CI, or formal regret here — this pilot only produces the
matched data and runs raw integrity + physical-validity + a raw optimal-candidate-switch statistic. Formal
evaluation is Claude B's under `offline_v2/`.
