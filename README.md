# Welding Arm with Aubo I3
This project implements a Circular Path Planner for the Aubo I3 Robot Arm, using ROS 1 Noetic and MoveIt for motion planning and execution.

### Clone The Repo
```bash
git clone --recurse-submodules https://github.com/Lexciese/welding-arm-aubo-i3.git
```

### Build the Workspace
```bash
cd welding-arm-aubo-i3
```

```bash
catkin_make
```

### Run

Open 2 terminal

terminal 1:
```bash
source devel/setup.bash
```
```bash
roslaunch aubo_i3_moveit_config demo.launch
```

terminal 2:
```bash
source devel/setup.bash
```
```bash
rosrun cornerfolk planner2.py
```

