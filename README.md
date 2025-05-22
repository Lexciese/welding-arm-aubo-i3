# Welding Arm with Aubo I3
This project implements a Circular Path Planner for the Aubo I3 Robot Arm, using ROS 1 Noetic and MoveIt for motion planning and execution.

### Clone The Repo
```bash
git clone --recurse-submodules https://github.com/Lexciese/welding-arm-aubo-i3.git
```

### Requirements

- Ubuntu 20.04
- ROS 1 Noetic
- MoveIt
- Python 3
If you are using Windows 10/11, you can setup WSL with Ubuntu 20.04

### Dependencies

<details>
<summary>Install ROS Noetic</summary>

```bash
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
```

```bash
sudo apt install curl
```

```bash
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
```
```bash
sudo apt update && sudo apt install ros-noetic-desktop-full
```
```bash
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
```
```bash
source ~/.bashrc
```
</details>

Install MoveIt:
```bash
sudo apt install ros-noetic-moveit
```

Install required Python packages:
```bash
pip3 install numpy scipy
```

Install additional ROS dependencies:
```bash
sudo apt install ros-noetic-joint-state-publisher ros-noetic-joint-state-publisher-gui ros-noetic-gazebo-ros-pkgs ros-noetic-gazebo-ros-control ros-noetic-ros-controllers ros-noetic-controller-manager ros-noetic-joint-trajectory-controller
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
rosrun cornerfolk planner6.py
```

