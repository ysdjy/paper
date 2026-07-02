# gello scripts (V1)

The real GELLO read-only / calibration scripts are NOT duplicated here — they
depend on the isolated `.venv-gello` and Dynamixel SDK. Use the legacy ones:

    projects/gello_franka_teleop/scripts/
      setup_gello_env.sh        # one-shot env + gello_software clone
      detect_gello_port.sh      # find the serial port
      calibrate_gello_offset.sh # joint offset calibration
      read_gello_joints.py      # stage-1 read-only (q[0:7] + gripper)
      diagnose_gello.py         # env sanity check

V1 hardware-free entry points live in `teleop_collection/entries/`.
