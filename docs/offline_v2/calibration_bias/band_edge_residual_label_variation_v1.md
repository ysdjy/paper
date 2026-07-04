# Band-edge residual → block-level label variation v1 (read-only)

Claude B. Does the v3 block residual genuinely change success LABELS (not just take nonzero values), and
does it do so **at the future confirmatory operating region**? This is the decisive question for whether
the confirmatory block-bootstrap CI would be non-degenerate. Plot: `band_edge_per_offset_blocks_v1.png`.

## Verdict: **RESIDUAL_INDUCES_BLOCK_LABEL_VARIATION (operating region) = false**
The residual flips labels for **edge-adjacent offsets** (mechanism present), but at the **v3 future test
operating region** every cell is deterministic (0 or 1), so the residual is **inert where H3 needs
variance**.

## 8.1 Observed: same offset across the 18 blocks
Because nominal bias = 0, `|eff| = |offset + residual|`. Offsets whose |offset| sits near the measured
0.034 edge cross it as the block residual varies, flipping the label:

| offset | success blocks | failure blocks | |eff| range | mixed |
|---|---|---|---|---|
| ±0.030 | 13 / 14 | 5 / 4 | 0.021–0.039 | **yes** |
| ±0.035 | 7 / 10 | 11 / 8 | 0.026–0.044 | **yes** |
| ±0.040 | 2 / 3 | 16 / 15 | 0.031–0.049 | **yes** |
| |offset| ≤ 0.025 | 18 | 0 | ≤ ~0.032 | no (all success) |

So the residual is **not merely numerically nonzero** — it demonstrably changes the discrete outcome for
offsets whose |eff| is near the edge. Mechanism confirmed.

## 8.2 Future v3 operating region mapping (frozen residual support, offsets from v3 — not re-picked)
Map the v3 confirmatory **test nominal biases** `±0.03` + the FROZEN residual support `[−0.01, +0.01]` to
`|eff|`, then to the empirical edge:

| nominal bias | policy | offset | |eff| range | empirical P(success) |
|---|---|---|---|---|
| −0.03 | state-aware compensating | +0.02 | [0.00, 0.02] | 1 → 1 (deterministic success) |
| +0.03 | state-aware compensating | −0.02 | [0.00, 0.02] | 1 → 1 (deterministic success) |
| −0.03 | v3 best-single (−0.02) | −0.02 | [0.04, 0.06] | 0 → 0 (deterministic fail) |
| +0.03 | v3 best-single (−0.02) | −0.02 | [0.00, 0.02] | 1 → 1 (deterministic success) |

**Every operating cell is deterministic** — none straddles the 0.034 edge under the frozen residual. The
state-aware compensated |eff| stays ≤ 0.02 (deep in-band) for all residuals, because the 0.02-spaced offset
grid compensates ±0.03 to within ±0.01 and the residual (≤0.01) cannot push it across the 0.034 edge.

## Why this drives MODIFY
For H3 the block-bootstrap CI resamples nuisance blocks of the **selected** episodes. At the operating
cells the selected outcomes are deterministic across blocks → the CI is **degenerate** (zero width) — the
exact "residual present but inert at the operating points" failure the pre-run audit flagged and the
band-edge study was built to detect. The residual induces label variation for edge-adjacent offsets, but
those are not the offsets a compensating selector uses. Hence the residual/design must change before the
confirmatory run (recommendation only; not implemented here).
