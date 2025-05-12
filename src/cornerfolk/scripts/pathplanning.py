#!/usr/bin/env python3
import rospy
import math
import numpy as np
import tf.transformations as tf
from geometry_msgs.msg import Pose, Point
from moveit_commander import MoveGroupCommander, RobotCommander, PlanningSceneInterface
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from moveit_msgs.msg import DisplayTrajectory

# Initialize MoveIt and ROS node
rospy.init_node('pilz_circular_motion')
robot = RobotCommander()
scene = PlanningSceneInterface()
group_name = "manipulator_i3"  # Replace with your move group name

# Create move group commander and set planner
move_group = MoveGroupCommander(group_name)
move_group.set_planner_id("pilz_industrial_motion_planner/CommandPlanner")

# Set reference frame and tolerance
move_group.set_pose_reference_frame("base_link")  # Adjust to your robot's base frame
move_group.set_goal_position_tolerance(0.01)  # meters
move_group.set_goal_orientation_tolerance(0.1)  # radians

# Create a DisplayTrajectory publisher for RViz visualization
display_trajectory_publisher = rospy.Publisher(
    "/move_group/display_planned_path", DisplayTrajectory, queue_size=20
)

# Function to visualize trajectory in RViz
def visualize_trajectory_in_rviz(trajectory):
    if trajectory is None:
        rospy.logwarn("No trajectory to visualize")
        return
        
    display_trajectory = DisplayTrajectory()
    display_trajectory.trajectory_start = robot.get_current_state()
    display_trajectory.trajectory.append(trajectory)
    display_trajectory_publisher.publish(display_trajectory)
    rospy.loginfo("Trajectory visualized in RViz")

# Function to extract and visualize trajectory in matplotlib
def visualize_trajectory_in_matplotlib(trajectory):
    if trajectory is None or not hasattr(trajectory, 'joint_trajectory') or not trajectory.joint_trajectory.points:
        rospy.logwarn("No valid trajectory points to visualize")
        return
        
    # Extract trajectory data
    points = []
    for point in trajectory.joint_trajectory.points:
        points.append(point.positions)
    
    # Convert to numpy array for easier manipulation
    points = np.array(points)
    
    # Create a new figure for 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get the end effector positions by forward kinematics
    # This is a simplified approach - in practice, you'd use tf or KDL
    # Here we're just using the trajectory points as a representation
    
    # For demonstration, we'll plot the first 3 joint values in 3D space
    # In a real implementation, you'd compute the end effector position
    joint_indices = min(3, points.shape[1])
    
    ax.plot(points[:, 0], points[:, 1], points[:, 2], 'b-', linewidth=2)
    ax.scatter(points[0, 0], points[0, 1], points[0, 2], color='g', s=100, label='Start')
    ax.scatter(points[-1, 0], points[-1, 1], points[-1, 2], color='r', s=100, label='End')
    
    ax.set_xlabel('Joint 1')
    ax.set_ylabel('Joint 2')
    ax.set_zlabel('Joint 3')
    ax.set_title('Robot Trajectory Visualization')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('/home/wsl20/dev/trajectory_plot.png')
    rospy.loginfo("Trajectory plot saved to /home/wsl20/dev/trajectory_plot.png")
    plt.show()

# Circular path parameters
center = Point(1, 0.0, -0.5)  # Circle center in base frame
radius = 0.1  # meters
num_points = 100  # Number of discrete points in the circle

# Create circular waypoints using PILZ CIRC command
current_pose = move_group.get_current_pose().pose

# Define orientation (90° perpendicular to surface - adjust based on your needs)
quaternion = tf.quaternion_from_euler(math.pi, 0, 0)  # Pointing downward
current_pose.orientation.x = quaternion[0]
current_pose.orientation.y = quaternion[1]
current_pose.orientation.z = quaternion[2]
current_pose.orientation.w = quaternion[3]

# Calculate intermediate and target positions for CIRC command
waypoints = []
all_trajectories = []  # Store all trajectories for visualization

for i in range(num_points + 1):
    theta = 2 * math.pi * i / num_points
    
    # Create auxiliary and target poses for CIRC motion
    if i == 0:
        start_pose = current_pose
    else:
        aux_pose = Pose()
        target_pose = Pose()
        
        # Intermediate point (auxiliary pose)
        aux_theta = theta - math.pi/num_points
        aux_pose.position.x = center.x + radius * math.cos(aux_theta)
        aux_pose.position.y = center.y + radius * math.sin(aux_theta)
        aux_pose.position.z = center.z
        aux_pose.orientation = current_pose.orientation
        
        # Target point
        target_pose.position.x = center.x + radius * math.cos(theta)
        target_pose.position.y = center.y + radius * math.sin(theta)
        target_pose.position.z = center.z
        target_pose.orientation = current_pose.orientation
        
        # Configure CIRC motion: current pose -> aux pose -> target pose
        plan_result = move_group.plan()
        
        # Handle different return types from different MoveIt versions
        if isinstance(plan_result, tuple):
            # Newer MoveIt version might return (success, trajectory, planning_time, error_code)
            success = plan_result[0]
            trajectory = plan_result[1] if len(plan_result) > 1 else None
        else:
            # Older versions might return just the trajectory with success implied
            success = bool(plan_result)
            trajectory = plan_result
        
        if success and trajectory and hasattr(trajectory, 'joint_trajectory') and trajectory.joint_trajectory.points:
            # Visualize in RViz
            visualize_trajectory_in_rviz(trajectory)
            
            # Store trajectory for later visualization
            all_trajectories.append(trajectory)
            
            move_group.execute(trajectory, wait=True)
            rospy.sleep(0.5)
        else:
            rospy.logerr("Failed to plan CIRC motion between points")

# Return to start position
move_group.set_pose_target(start_pose)
plan_result = move_group.plan()

# Handle different return types from different MoveIt versions
if isinstance(plan_result, tuple):
    # Newer MoveIt version might return (success, trajectory, planning_time, error_code)
    success = plan_result[0]
    trajectory = plan_result[1] if len(plan_result) > 1 else None
else:
    # Older versions might return just the trajectory with success implied
    success = bool(plan_result)
    trajectory = plan_result

if success and trajectory and hasattr(trajectory, 'joint_trajectory') and trajectory.joint_trajectory.points:
    move_group.execute(trajectory, wait=True)
else:
    rospy.logerr("Failed to return to start position")

rospy.loginfo("Circular motion complete!")

# Visualize the complete trajectory in matplotlib
if all_trajectories:
    # For simplicity, visualize just the last trajectory
    # A more complex implementation would combine all trajectories
    visualize_trajectory_in_matplotlib(all_trajectories[-1])
else:
    rospy.logwarn("No trajectories to visualize in matplotlib")