# Stage-1 数据 schema 与示例

本目录每个 `<run_id>/` 含 `episodes.jsonl` + `trajectories/<id>.npz` + 运行元数据。
大数据（trajectories/*.npz）默认 .gitignore；仓库内 `_examples/` 与各 regression run 保留小样本。

## Episode JSON（四组变量分离：initial_state x / task_target g / execution_parameters θ / outcomes y）

```json
{
  "controller": {
    "action_dim": 8,
    "control_dt": 0.05,
    "control_frequency_hz": 20.0,
    "decimation": 5,
    "ik_command_type": "pose",
    "ik_max_joint_step": 0.2,
    "ik_method": "dls",
    "ik_relative": false,
    "physics_dt": 0.01,
    "task_id": "Isaac-Stack-Cube-Franka-JointPolicy-v0",
    "type": "joint_position_with_dls_ik"
  },
  "env_id": 0,
  "episode_id": "place_000000",
  "execution_parameters": [
    {
      "applied_at": "place_skill",
      "clamped": false,
      "default_value": 0.1,
      "dtype": "float",
      "effective_value": 0.1,
      "maximum": 0.3,
      "minimum": 0.02,
      "name": "pre_place_height",
      "rejected": false,
      "requested_value": null,
      "source": "default"
    },
    {
      "applied_at": "place_skill",
      "clamped": false,
      "default_value": 0.0,
      "dtype": "float",
      "effective_value": 0.0,
      "maximum": 0.1,
      "minimum": 0.0,
      "name": "release_clearance",
      "rejected": false,
      "requested_value": null,
      "source": "default"
    },
    "...(14 traces total)"
  ],
  "failure_reason": null,
  "initial_state": {
    "grasp_setup_attempts": 1,
    "initial_object_pose": [
      0.5012609362602234,
      -0.28225797414779663,
      0.13923785090446472,
      0.020872101187705994,
      -0.0005618383875116706,
      -0.0005788200069218874,
      -0.9997817873954773
    ],
    "initial_robot_joint_position": [
      -0.23486176133155823,
      0.3653542101383209,
      -0.2799076437950134,
      -2.1241328716278076,
      0.15889228880405426,
      2.4685118198394775,
      0.16357927024364471,
      0.01872311346232891,
      0.019688213244080544
    ],
    "initial_tcp_pose": [
      0.5012463331222534,
      -0.2825671136379242,
      0.1392689049243927,
      -0.00038606449379585683,
      0.9999995231628418,
      -7.807096699252725e-05,
      -0.0007832507835701108
    ],
    "object_to_tcp_transform": {
      "pos": [
        2.741878415690735e-05,
        0.0003086297947447747,
        3.111250771326013e-05
      ],
      "quat": [
        0.00021225280943326652,
        0.020949890837073326,
        0.9997798800468445,
        -0.0009803972207009792
      ]
    },
    "object_type": "cube_1"
  },
  "legacy_reached": true,
  "level": "default",
  "outcomes": {
    "collision_available": false,
    "collision_count": null,
    "contact_available": false,
    "elapsed_time": 4.8500000000000565,
    "failure_reason": null,
    "legacy_reached": true,
    "maximum_contact_force": null,
    "maximum_tcp_tracking_error": 0.3911663293838501,
    "mean_tcp_tracking_error": 0.13128453895885062,
    "object_dropped": false,
    "object_final_angular_speed": 0.0041217803955078125,
    "object_final_linear_speed": 0.00013402212061919272,
    "object_orientation_error": 0.001602240139618516,
    "object_out_of_bounds": false,
    "object_position_error": 0.0072817252948880196,
    "settling_time": 0.15000000000000213,
    "success": true,
    "tcp_final_orientation_error": 0.02395082451403141,
    "tcp_final_position_error": 0.01625838130712509
  },
  "requested_parameters": {},
  "seed": 42,
  "setup_failure": false,
  "skill": "place",
  "success": true,
  "task_target": {
    "point_name": "point_a",
    "target_surface_xyz": [
      0.42,
      0.1,
      0.0
    ],
    "task_target_object_pose": [
      0.42,
      0.1,
      0.0223,
      0.020872,
      -0.000562,
      -0.000579,
      -0.999782
    ]
  },
  "trajectory_file": "trajectories/place_000000.npz"
}
```

## Trajectory NPZ 统一 schema（键 / 形状，T=步数；不适用字段填 NaN）

```
sim_time                           [98]
elapsed_time                       [98]
skill_state                        [98]
joint_position                     [98, 9]
joint_velocity                     [98, 9]
tcp_position                       [98, 3]
tcp_orientation                    [98, 4]
target_tcp_position                [98, 3]
target_tcp_orientation             [98, 4]
tcp_position_error                 [98]
tcp_orientation_error              [98]
measured_tcp_linear_speed          [98]
measured_tcp_angular_speed         [98]
gripper_command                    [98]
object_position                    [98, 3]
object_orientation                 [98, 4]
object_linear_velocity             [98, 3]
object_angular_velocity            [98, 3]
object_position_error              [98]
object_orientation_error           [98]
drawer_joint_position              [98]
drawer_joint_velocity              [98]
handle_position                    [98, 3]
target_handle_grasp_position       [98, 3]
handle_relative_position_error     [98]
drawer_progress                    [98]
contact_available                  [98]
contact_force                      [98]
collision_available                [98]
collision_flag                     [98]
```