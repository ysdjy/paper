# Stage-1 Pilot / Sensitivity Analysis

## Default regression

| run | skill | n | success | rate |
|---|---|---|---|---|
| regression_drawer_middle | open_drawer | 10 | 0 | 0% |
| regression_drawer_top | open_drawer | 10 | 10 | 100% |
| regression_place | place | 10 | 10 | 100% |

## open_drawer :: parameter `close_duration`

| level | n | succ | rate | elapsed mean(s) | err mean(m) | maxTrack(m) |
|---|---|---|---|---|---|---|
| 0.5 | 5 | 5 | 100% | 11.17 | 0.0177 | 0.2463 |
| 1.0 | 5 | 5 | 100% | 10.05 | 0.0180 | 0.2038 |
| 1.5 | 5 | 5 | 100% | 10.46 | 0.0176 | 0.2038 |

- **elapsed_time**: effective=False spread/noise=1.062 corr=-0.626
- **primary_error**: effective=False spread/noise=1.062 corr=-0.243
- **max_tcp_tracking_error**: effective=False spread/noise=1.486 corr=-0.865
- success-rate spread across levels: 0.000 (affects_failure=False)
- **VERDICT: NO MEASURABLE EFFECT**

## open_drawer :: parameter `grasp_offset_local_xyz`

| level | n | succ | rate | elapsed mean(s) | err mean(m) | maxTrack(m) |
|---|---|---|---|---|---|---|
| [0.0, -0.03, 0.0] | 5 | 5 | 100% | 8.77 | 0.0140 | 0.1870 |
| [0.0, 0.0, 0.0] | 5 | 5 | 100% | 11.67 | 0.0173 | 0.2463 |
| [0.0, 0.03, 0.0] | 5 | 5 | 100% | 10.82 | 0.0687 | 0.2708 |

- **elapsed_time**: effective=True spread/noise=2.500 corr=-
- **primary_error**: effective=True spread/noise=49.569 corr=-
- **max_tcp_tracking_error**: effective=True spread/noise=2.037 corr=-
- success-rate spread across levels: 0.000 (affects_failure=False)
- **VERDICT: EFFECTIVE**

## open_drawer :: parameter `max_pos_step`

| level | n | succ | rate | elapsed mean(s) | err mean(m) | maxTrack(m) |
|---|---|---|---|---|---|---|
| 0.01 | 5 | 5 | 100% | 16.60 | 0.0219 | 0.2468 |
| 0.02 | 5 | 5 | 100% | 9.69 | 0.0172 | 0.2060 |
| 0.03 | 5 | 5 | 100% | 8.68 | 0.0089 | 0.2055 |

- **elapsed_time**: effective=True spread/noise=5.020 corr=-0.919
- **primary_error**: effective=True spread/noise=33.125 corr=-0.987
- **max_tcp_tracking_error**: effective=False spread/noise=1.444 corr=-0.872
- success-rate spread across levels: 0.000 (affects_failure=False)
- **VERDICT: EFFECTIVE**

## open_drawer :: parameter `pull_lead`

| level | n | succ | rate | elapsed mean(s) | err mean(m) | maxTrack(m) |
|---|---|---|---|---|---|---|
| 0.04 | 5 | 5 | 100% | 11.67 | 0.0182 | 0.2462 |
| 0.08 | 5 | 5 | 100% | 10.05 | 0.0181 | 0.2038 |
| 0.12 | 5 | 5 | 100% | 9.94 | 0.0172 | 0.2037 |

- **elapsed_time**: effective=True spread/noise=1.648 corr=-0.893
- **primary_error**: effective=True spread/noise=4.467 corr=-0.909
- **max_tcp_tracking_error**: effective=False spread/noise=1.487 corr=-0.867
- success-rate spread across levels: 0.000 (affects_failure=False)
- **VERDICT: EFFECTIVE**

## open_drawer :: parameter `target_open_position`

| level | n | succ | rate | elapsed mean(s) | err mean(m) | maxTrack(m) |
|---|---|---|---|---|---|---|
| 0.1 | 5 | 5 | 100% | 10.81 | 0.0181 | 0.2830 |
| 0.2 | 5 | 5 | 100% | 10.11 | 0.0177 | 0.2133 |
| 0.3 | 5 | 5 | 100% | 10.37 | 0.0166 | 0.2093 |

- **elapsed_time**: effective=False spread/noise=0.580 corr=-0.622
- **primary_error**: effective=True spread/noise=3.210 corr=-0.968
- **max_tcp_tracking_error**: effective=True spread/noise=2.479 corr=-0.889
- success-rate spread across levels: 0.000 (affects_failure=False)
- **VERDICT: EFFECTIVE**

## place :: parameter `descend_max_position_step`

| level | n | succ | rate | elapsed mean(s) | err mean(m) | maxTrack(m) |
|---|---|---|---|---|---|---|
| 0.003 | 3 | 2 | 67% | 5.15 | 0.0027 | 0.2486 |
| 0.006 | 3 | 2 | 67% | 7.22 | 0.0024 | 0.2487 |
| 0.009 | 3 | 2 | 67% | 7.17 | 0.0026 | 0.2490 |

- **elapsed_time**: effective=False spread/noise=0.671 corr=0.855
- **primary_error**: effective=False spread/noise=1.369 corr=-0.438
- **max_tcp_tracking_error**: effective=False spread/noise=0.008 corr=0.945
- **settling_time**: effective=False spread/noise=0.000 corr=-
- success-rate spread across levels: 0.000 (affects_failure=False)
- **VERDICT: NO MEASURABLE EFFECT**

## place :: parameter `move_max_position_step`

| level | n | succ | rate | elapsed mean(s) | err mean(m) | maxTrack(m) |
|---|---|---|---|---|---|---|
| 0.006 | 3 | 2 | 67% | 5.47 | 0.0026 | 0.2490 |
| 0.012 | 3 | 2 | 67% | 4.28 | 0.0023 | 0.2482 |
| 0.018 | 3 | 2 | 67% | 4.00 | 0.0024 | 0.2486 |

- **elapsed_time**: effective=True spread/noise=4.256 corr=-0.943
- **primary_error**: effective=True spread/noise=2.810 corr=-0.748
- **max_tcp_tracking_error**: effective=False spread/noise=0.013 corr=-0.452
- **settling_time**: effective=False spread/noise=0.000 corr=-
- success-rate spread across levels: 0.000 (affects_failure=False)
- **VERDICT: EFFECTIVE**

## place :: parameter `open_duration`

| level | n | succ | rate | elapsed mean(s) | err mean(m) | maxTrack(m) |
|---|---|---|---|---|---|---|
| 0.25 | 3 | 2 | 67% | 4.22 | 0.0024 | 0.2486 |
| 0.45 | 3 | 2 | 67% | 7.30 | 0.0026 | 0.2490 |
| 0.9 | 3 | 2 | 67% | 7.60 | 0.0026 | 0.2490 |

- **elapsed_time**: effective=False spread/noise=1.135 corr=0.789
- **primary_error**: effective=True spread/noise=1.916 corr=0.733
- **max_tcp_tracking_error**: effective=False spread/noise=0.006 corr=0.737
- **settling_time**: effective=False spread/noise=0.000 corr=-
- success-rate spread across levels: 0.000 (affects_failure=False)
- **VERDICT: EFFECTIVE**

## place :: parameter `release_clearance`

| level | n | succ | rate | elapsed mean(s) | err mean(m) | maxTrack(m) |
|---|---|---|---|---|---|---|
| 0.0 | 3 | 2 | 67% | 4.48 | 0.0026 | 0.2490 |
| 0.02 | 3 | 2 | 67% | 4.12 | 0.0030 | 0.2479 |
| 0.04 | 3 | 2 | 67% | 4.12 | 0.0037 | 0.2490 |

- **elapsed_time**: effective=True spread/noise=1.673 corr=-0.866
- **primary_error**: effective=True spread/noise=3.153 corr=0.976
- **max_tcp_tracking_error**: effective=False spread/noise=0.019 corr=0.004
- **settling_time**: effective=False spread/noise=0.000 corr=-
- success-rate spread across levels: 0.000 (affects_failure=False)
- **VERDICT: EFFECTIVE**

## place :: parameter `settle_after_release_duration`

| level | n | succ | rate | elapsed mean(s) | err mean(m) | maxTrack(m) |
|---|---|---|---|---|---|---|
| 0.2 | 3 | 2 | 67% | 4.12 | 0.0024 | 0.2486 |
| 0.5 | 3 | 2 | 67% | 7.30 | 0.0026 | 0.2490 |
| 1.0 | 3 | 2 | 67% | 7.63 | 0.0026 | 0.2490 |

- **elapsed_time**: effective=False spread/noise=1.182 corr=0.836
- **primary_error**: effective=True spread/noise=1.997 corr=0.774
- **max_tcp_tracking_error**: effective=False spread/noise=0.006 corr=0.786
- **settling_time**: effective=False spread/noise=0.000 corr=-
- success-rate spread across levels: 0.000 (affects_failure=False)
- **VERDICT: EFFECTIVE**

## Effective parameters

grasp_offset_local_xyz, max_pos_step, move_max_position_step, open_duration, pull_lead, release_clearance, settle_after_release_duration, target_open_position

## Ineffective parameters (drop from first model inputs)

close_duration, descend_max_position_step
