#!/usr/bin/env python3
import math
import numpy as np
import rospy
from moveit_commander import MoveGroupCommander, RobotCommander, PlanningSceneInterface
from geometry_msgs.msg import Pose, PoseStamped
from std_msgs.msg import Header
from tf.transformations import quaternion_from_euler

# Initialize ROS node
rospy.init_node('ik_moveit_interface', anonymous=True)

# Robot link lengths
l1 = 0.197
l2 = 0.201
l3 = 0.07

# Initialize MoveIt interfaces
robot = RobotCommander()
scene = PlanningSceneInterface()
group_name = "armor"  # Replace with your robot's move group name
move_group = MoveGroupCommander(group_name)

# Set planning parameters
move_group.set_planning_time(5.0)
move_group.set_num_planning_attempts(10)
move_group.set_max_velocity_scaling_factor(0.3)
move_group.set_max_acceleration_scaling_factor(0.1)

# Function to send joint values to MoveIt
def send_joint_values_to_moveit(joint_values):
    """
    Send joint values to MoveIt for execution
    
    Args:
        joint_values: List of joint angles in radians
        
    Returns:
        bool: Success status
    """
    try:
        # Print the joint values being sent
        rospy.loginfo(f"Sending joint values to MoveIt: {[round(math.degrees(v), 2) for v in joint_values]}")
        
        # Get joint names from the move group
        joint_names = move_group.get_active_joints()
        rospy.loginfo(f"Robot joints: {joint_names}")
        
        if len(joint_names) != len(joint_values):
            rospy.logwarn(f"Joint count mismatch: {len(joint_names)} joints in group, {len(joint_values)} values provided")
            
            # Handle the mismatch by adding default values for missing joints
            if len(joint_names) > len(joint_values):
                # Add default value (0.0) for additional joints
                additional_joints = len(joint_names) - len(joint_values)
                joint_values = list(joint_values) + [0.0] * additional_joints
                rospy.loginfo(f"Added {additional_joints} default joint values (0.0)")
            else:
                # Truncate if we have too many values
                joint_values = joint_values[:len(joint_names)]
                rospy.loginfo("Truncated extra joint values")
        
        # Clear previous targets
        move_group.clear_pose_targets()
        
        # Create a dictionary mapping joint names to values
        joint_targets = dict(zip(joint_names, joint_values))
        rospy.loginfo(f"Setting joint targets: {joint_targets}")
        
        # Set joint target
        move_group.set_joint_value_target(joint_targets)
        
        # Plan and execute
        plan = move_group.plan()
        
        # Handle different return values from different MoveIt versions
        if isinstance(plan, tuple):
            success = plan[0]
            trajectory = plan[1] if len(plan) > 1 else None
        else:
            success = bool(plan)
            trajectory = plan
            
        if success:
            rospy.loginfo("Planning successful, executing trajectory...")
            move_group.execute(trajectory if isinstance(plan, tuple) else plan, wait=True)
            return True
        else:
            rospy.logerr("Planning failed!")
            return False
            
    except Exception as e:
        rospy.logerr(f"Error in send_joint_values_to_moveit: {e}")
        return False

# Function to send pose target to MoveIt
def send_pose_to_moveit(tx, ty, tz, telev, tazim):
    """
    Send end-effector pose to MoveIt
    
    Args:
        tx, ty, tz: Target position
        telev, tazim: Target orientation (elevation and azimuth angles)
        
    Returns:
        bool: Success status
    """
    try:
        # Convert angles to quaternion
        quat = quaternion_from_euler(0, telev, tazim)  # Adjust axes as needed
        
        # Create pose target
        target_pose = Pose()
        target_pose.position.x = tx
        target_pose.position.y = ty
        target_pose.position.z = tz + 0.113  # Add back the offset
        target_pose.orientation.x = quat[0]
        target_pose.orientation.y = quat[1]
        target_pose.orientation.z = quat[2]
        target_pose.orientation.w = quat[3]
        
        # Set pose target
        move_group.set_pose_target(target_pose)
        
        # Plan and execute
        plan = move_group.plan()
        
        # Handle different return values
        if isinstance(plan, tuple):
            success = plan[0]
            trajectory = plan[1] if len(plan) > 1 else None
        else:
            success = bool(plan)
            trajectory = plan
            
        if success:
            rospy.loginfo("Planning successful, executing trajectory...")
            move_group.execute(trajectory if isinstance(plan, tuple) else plan, wait=True)
            return True
        else:
            rospy.logerr("Planning failed!")
            return False
            
    except Exception as e:
        rospy.logerr(f"Error in send_pose_to_moveit: {e}")
        return False

def forkin(theta1, theta2, theta3):
    rot_theta1 = np.array([
        [math.cos(theta1), -math.sin(theta1), 0],
        [math.sin(theta1), math.cos(theta1), 0],
        [0, 0, 1]
    ])
    rot_theta2 = np.array([
        [math.cos(theta2), 0, math.sin(theta2)],
        [0, 1, 0],
        [-math.sin(theta2), 0, math.cos(theta2)]
    ])
    rot_theta3 = np.array([
        [math.cos(theta3), 0, math.sin(theta3)],
        [0, 1, 0],
        [-math.sin(theta3), 0, math.cos(theta3)]
    ])
    return rot_theta1 @ rot_theta2 @ rot_theta3

def invkin(wx, wy, wz, telev, tazim):
    theta1_1 = math.atan2(wy,wx)
    theta1_2 = theta1_1 + math.radians(180)
    print((l1**2 + l2**2 - (math.sqrt(wx**2 + wy**2))**2 - wz**2) / (2*l1*l2))
    y = math.sqrt(abs(((2*l1*l2)**2) - ((l1**2 + l2**2 - (math.sqrt(wx**2 + wy**2))**2 - wz**2)**2)))
    theta3_1 = math.radians(180) - math.atan2(y, (l1**2 + l2**2 - (math.sqrt(wx**2 + wy**2))**2 - wz**2))
    theta3_2 = -math.radians(180) + math.atan2(y, (l1**2 + l2**2 - (math.sqrt(wx**2 + wy**2))**2 - wz**2))
    theta3_3 = -theta3_1
    theta3_4 = -theta3_2
    theta2_1 = math.radians(90) - (math.atan2(l2*math.sin(theta3_1), (l1 + l2*math.cos(theta3_1))) + math.atan2(wz, (math.sqrt(wx**2 + wy**2))))
    theta2_2 = math.radians(90) - (math.atan2(l2*math.sin(theta3_2), (l1 + l2*math.cos(theta3_2))) + math.atan2(wz, (math.sqrt(wx**2 + wy**2))))
    theta2_3 = -theta2_1
    theta2_4 = -theta2_2
    rot_wrist_1 = forkin(theta1_1, theta2_1, theta3_1)
    rot_wrist_2 = forkin(theta1_1, theta2_2, theta3_2)
    rot_wrist_3 = forkin(theta1_2, theta2_3, theta3_3)
    rot_wrist_4 = forkin(theta1_2, theta2_4, theta3_4)
    
    target_vector_global = np.array([
        math.sin(telev)*math.cos(tazim),
        math.sin(telev)*math.sin(tazim),
        math.cos(telev)
    ])
    
    target_vector_local_1 = np.linalg.inv(rot_wrist_1) @ target_vector_global
    target_vector_local_2 = np.linalg.inv(rot_wrist_2) @ target_vector_global
    target_vector_local_3 = np.linalg.inv(rot_wrist_3) @ target_vector_global
    target_vector_local_4 = np.linalg.inv(rot_wrist_4) @ target_vector_global

    theta4_1 = math.atan2(target_vector_local_1[1], target_vector_local_1[0])
    theta4_3 = math.atan2(target_vector_local_2[1], target_vector_local_2[0])
    theta4_5 = math.atan2(target_vector_local_3[1], target_vector_local_3[0])
    theta4_7 = math.atan2(target_vector_local_4[1], target_vector_local_4[0])
    theta5_1 = math.radians(90) - math.atan2(target_vector_local_1[2], math.sqrt(abs(target_vector_local_1[0]**2 + target_vector_local_1[1]**2)))
    theta5_3 = math.radians(90) - math.atan2(target_vector_local_2[2], math.sqrt(abs(target_vector_local_2[0]**2 + target_vector_local_2[1]**2)))
    theta5_5 = math.radians(90) - math.atan2(target_vector_local_3[2], math.sqrt(abs(target_vector_local_3[0]**2 + target_vector_local_3[1]**2)))
    theta5_7 = math.radians(90) - math.atan2(target_vector_local_4[2], math.sqrt(abs(target_vector_local_4[0]**2 + target_vector_local_4[1]**2)))
    theta4_2 = theta4_1 + math.radians(180)
    theta4_4 = theta4_3 + math.radians(180)
    theta4_6 = theta4_5 + math.radians(180)
    theta4_8 = theta4_7 + math.radians(180)
    theta5_2 = -theta5_1
    theta5_4 = -theta5_3
    theta5_6 = -theta5_5
    theta5_8 = -theta5_7

    angle_1 = [theta1_1, theta2_1, theta3_1, theta4_1, theta5_1]
    angle_2 = [theta1_1, theta2_1, theta3_1, theta4_2, theta5_2]
    angle_3 = [theta1_1, theta2_2, theta3_2, theta4_3, theta5_3]
    angle_4 = [theta1_1, theta2_2, theta3_2, theta4_4, theta5_4]
    angle_5 = [theta1_2, theta2_3, theta3_3, theta4_5, theta5_5]
    angle_6 = [theta1_2, theta2_3, theta3_3, theta4_6, theta5_6]
    angle_7 = [theta1_2, theta2_4, theta3_4, theta4_7, theta5_7]
    angle_8 = [theta1_2, theta2_4, theta3_4, theta4_8, theta5_8]
    for joint in range(5):
        angle_1[joint] = round(angle_1[joint] % (2*np.pi), 2)
        angle_2[joint] = round(angle_2[joint] % (2*np.pi), 2)
        angle_3[joint] = round(angle_3[joint] % (2*np.pi), 2)
        angle_4[joint] = round(angle_4[joint] % (2*np.pi), 2)
        angle_5[joint] = round(angle_5[joint] % (2*np.pi), 2)
        angle_6[joint] = round(angle_6[joint] % (2*np.pi), 2)
        angle_7[joint] = round(angle_7[joint] % (2*np.pi), 2)
        angle_8[joint] = round(angle_8[joint] % (2*np.pi), 2)
        if joint == 1 or joint == 2 or joint == 4:
            if angle_1[joint] > math.pi:
                angle_1[joint] = round(angle_1[joint] - 2*math.pi, 2)
            if angle_2[joint] > math.pi:
                angle_2[joint] = round(angle_2[joint] - 2*math.pi, 2)
            if angle_3[joint] > math.pi:
                angle_3[joint] = round(angle_3[joint] - 2*math.pi, 2)
            if angle_4[joint] > math.pi:
                angle_4[joint] = round(angle_4[joint] - 2*math.pi, 2)
            if angle_5[joint] > math.pi:
                angle_5[joint] = round(angle_5[joint] - 2*math.pi, 2)
            if angle_6[joint] > math.pi:
                angle_6[joint] = round(angle_6[joint] - 2*math.pi, 2)
            if angle_7[joint] > math.pi:
                angle_7[joint] = round(angle_7[joint] - 2*math.pi, 2)
            if angle_8[joint] > math.pi:
                angle_8[joint] = round(angle_8[joint] - 2*math.pi, 2)

    low1 = low2 = low3 = low4 = -135
    high1 = high2 = high3 = high4 = 135
    if -1.57 <= theta2_1 <= 0:
        low1 = -(3*theta2_1/2) - 135
        high1 = 135
    elif 0 <= theta2_1 <= 1.57:
        low1 = -135
        high1 = -(3*theta2_1/2) + 135
    if -1.57 <= theta2_2 <= 0:
        low2 = -(3*theta2_2/2) - 135
        high2 = 135
    elif 0 <= theta2_2 <= 1.57:
        low2 = -135
        high2 = -(3*theta2_2/2) + 135
    if -1.57 <= theta2_3 <= 0:
        low3 = -(3*theta2_3/2) - 135
        high3 = 135
    elif 0 <= theta2_3 <= 1.57:
        low3 = -135
        high3 = -(3*theta2_3/2) + 135
    if -1.57 <= theta2_4 <= 0:
        low4 = -(3*theta2_4/2) - 135
        high4 = 135
    elif 0 <= theta2_4 <= 1.57:
        low4 = -135
        high4 = -(3*theta2_4/2) + 135
    limit1 = [(0, 360), (-90, 90), (low1, high1), (0, 360), (-90, 90)]
    limit2 = [(0, 360), (-90, 90), (low2, high2), (0, 360), (-90, 90)]
    limit3 = [(0, 360), (-90, 90), (low3, high3), (0, 360), (-90, 90)]
    limit4 = [(0, 360), (-90, 90), (low4, high4), (0, 360), (-90, 90)]
    for joint in range(5):
        if not (limit1[joint][0] <= math.degrees(angle_1[joint]) <= limit1[joint][1]):
            check_angle_1 = False
            break
        else:
            check_angle_1 = True
    for joint in range(5):
        if not (limit1[joint][0] <= math.degrees(angle_2[joint]) <= limit1[joint][1]):
            check_angle_2 = False
            break
        else:
            check_angle_2 = True
    for joint in range(5):
        if not (limit2[joint][0] <= math.degrees(angle_3[joint]) <= limit2[joint][1]):
            check_angle_3 = False
            break
        else:
            check_angle_3 = True
    for joint in range(5):
        if not (limit2[joint][0] <= math.degrees(angle_4[joint]) <= limit2[joint][1]):
            check_angle_4 = False
            break
        else:
            check_angle_4 = True
    for joint in range(5):
        if not (limit3[joint][0] <= math.degrees(angle_5[joint]) <= limit3[joint][1]):
            check_angle_5 = False
            break
        else:
            check_angle_5 = True
    for joint in range(5):
        if not (limit3[joint][0] <= math.degrees(angle_6[joint]) <= limit3[joint][1]):
            check_angle_6 = False
            break
        else:
            check_angle_6 = True
    for joint in range(5):
        if not (limit4[joint][0] <= math.degrees(angle_7[joint]) <= limit4[joint][1]):
            check_angle_7 = False
            break
        else:
            check_angle_7 = True
    for joint in range(5):
        if not (limit4[joint][0] <= math.degrees(angle_8[joint]) <= limit4[joint][1]):
            check_angle_8 = False
            break
        else:
            check_angle_8 = True

    return [
        (angle_1, check_angle_1),
        (angle_2, check_angle_2),
        (angle_3, check_angle_3),
        (angle_4, check_angle_4),
        (angle_5, check_angle_5),
        (angle_6, check_angle_6),
        (angle_7, check_angle_7),
        (angle_8, check_angle_8)
    ]

def main():
    try:
        # Get input from the user
        tx = float(input("Target coordinate (x): "))
        ty = float(input("Target coordinate (y): "))
        tz = float(input("Target coordinate (z): ")) - 0.113
        telev = math.radians(float(input("Target elevation angle (degrees): ")))
        tazim = math.radians(float(input("Target azimuth angle (degrees): ")))
        
        # Calculate wrist position
        wx = tx - l3*math.sin(telev)*math.cos(tazim)
        wy = ty - l3*math.sin(telev)*math.sin(tazim)
        wz = tz - l3*math.cos(telev)
        
        print(f"Wrist position: ({wx}, {wy}, {wz})")
        
        # Calculate inverse kinematics
        configurations = invkin(wx, wy, wz, telev, tazim)
        
        # Find valid configurations
        valid_configs = [(i+1, angles) for i, (angles, valid) in enumerate(configurations) if valid]
        
        if not valid_configs:
            print("No valid configurations found!")
            return
        
        # Print valid configurations
        print("\nValid configurations:")
        for i, (config_num, angles) in enumerate(valid_configs):
            print(f"{i+1}. Configuration {config_num}: {[round(math.degrees(a), 2) for a in angles]}")
        
        # Let user choose configuration or pose target
        print("\nOptions:")
        print("1-{}: Send specific configuration".format(len(valid_configs)))
        print("p: Send pose target directly")
        print("q: Quit")
        
        choice = input("Choose an option: ")
        
        if choice.lower() == 'q':
            print("Exiting...")
            return
        elif choice.lower() == 'p':
            print("Sending pose target to MoveIt...")
            if send_pose_to_moveit(tx, ty, tz+0.113, telev, tazim):
                print("Pose target executed successfully!")
            else:
                print("Failed to execute pose target")
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(valid_configs):
                    config_num, joint_values = valid_configs[idx]
                    print(f"Sending configuration {config_num} to MoveIt...")
                    if send_joint_values_to_moveit(joint_values):
                        print("Joint values executed successfully!")
                    else:
                        print("Failed to execute joint values")
                else:
                    print("Invalid configuration number")
            except ValueError:
                print("Invalid input")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass

