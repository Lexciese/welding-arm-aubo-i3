#!/usr/bin/env python3
import math
import numpy as np
import rospy
from sensor_msgs.msg import JointState
from moveit_commander import MoveGroupCommander, RobotCommander, PlanningSceneInterface
from geometry_msgs.msg import Pose, PoseStamped
from std_msgs.msg import Header
from std_srvs.srv import Empty

# Initialize ROS node
rospy.init_node('secondesign_inverse_kinematics', anonymous=True)

# Robot link lengths
l1 = 0.4
l2 = 0.5
l3 = 0.2

# Setup MoveIt interfaces
robot = RobotCommander()
scene = PlanningSceneInterface()
group_name = "armor"  # Replace with your actual move group name
move_group = MoveGroupCommander(group_name)

# Set planning parameters
move_group.set_planning_time(5.0)
move_group.set_num_planning_attempts(10)
move_group.set_max_velocity_scaling_factor(0.3)
move_group.set_max_acceleration_scaling_factor(0.1)

# Function to get target coordinates and orientation from user
def get_target_from_user():
    tx = float(input("Target coordinate (x): "))
    ty = float(input("Target coordinate (y): "))
    tz = float(input("Target coordinate (z): ")) - 0.25
    telev = math.radians(float(input("Target elevation angle (degrees): ")))
    tazim = math.radians(float(input("Target azimuth angle (degrees): ")))
    
    return tx, ty, tz, telev, tazim

# Function to send joint angles to MoveIt
def send_to_moveit(joint_values):
    try:
        # Convert radians to appropriate joint order expected by your robot
        # Adjust this to match your robot's joint names and order
        joint_names = move_group.get_active_joints()
        
        rospy.loginfo(f"Setting joint target: {joint_values}")
        rospy.loginfo(f"Joint names: {joint_names}")
        
        # Clear previous targets
        move_group.clear_pose_targets()
        
        # Set joint target
        move_group.set_joint_value_target(dict(zip(joint_names, joint_values)))
        
        # Plan trajectory
        plan = move_group.plan()
        
        # Handle different return types from plan() in different MoveIt versions
        success = False
        if isinstance(plan, tuple):
            # Newer MoveIt versions return (success, trajectory, planning_time, error_code)
            success = plan[0]
            trajectory = plan[1] if len(plan) > 1 else None
        else:
            # Older versions return just the trajectory
            success = bool(plan)
            trajectory = plan
        
        if success:
            rospy.loginfo("Planning succeeded! Executing motion...")
            move_group.execute(trajectory if isinstance(plan, tuple) else plan, wait=True)
            return True
        else:
            rospy.logerr("Planning failed!")
            return False
    except Exception as e:
        rospy.logerr(f"Error sending to MoveIt: {e}")
        return False

# Function to create a pose target
def send_pose_target(tx, ty, tz, telev, tazim):
    try:
        # Convert Euler angles to quaternion
        from tf.transformations import quaternion_from_euler
        
        quat = quaternion_from_euler(0, telev, tazim)  # Adjust axes order as needed
        
        # Create pose target
        target_pose = Pose()
        target_pose.position.x = tx
        target_pose.position.y = ty
        target_pose.position.z = tz + 0.25  # Adding back the offset
        target_pose.orientation.x = quat[0]
        target_pose.orientation.y = quat[1]
        target_pose.orientation.z = quat[2]
        target_pose.orientation.w = quat[3]
        
        # Set pose target
        move_group.set_pose_target(target_pose)
        
        # Plan and execute
        plan = move_group.plan()
        
        # Handle different return types
        success = False
        if isinstance(plan, tuple):
            success = plan[0]
            trajectory = plan[1] if len(plan) > 1 else None
        else:
            success = bool(plan)
            trajectory = plan
        
        if success:
            rospy.loginfo("Planning succeeded! Executing motion...")
            move_group.execute(trajectory if isinstance(plan, tuple) else plan, wait=True)
            return True
        else:
            rospy.logerr("Planning failed!")
            return False
    except Exception as e:
        rospy.logerr(f"Error sending pose target: {e}")
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
    theta1_1 = math.atan(wy/wx)
    theta1_2 = theta1_1 + math.radians(180)
    print((l1**2 + l2**2 - (wx/math.cos(math.atan2(wy,wx)))**2 - wz**2) / (2*l1*l2))
    theta3_1 = math.radians(180) - math.acos(max(-1.0, min(1.0, (l1**2 + l2**2 - (wx/math.cos(math.atan2(wy,wx)))**2 - wz**2) / (2*l1*l2))))
    theta3_2 = -math.radians(180) + math.acos(max(-1.0, min(1.0, (l1**2 + l2**2 - (wx/math.cos(math.atan2(wy,wx)))**2 - wz**2) / (2*l1*l2))))
    theta3_3 = -theta3_1
    theta3_4 = -theta3_2
    theta2_1 = math.radians(90) - (math.atan(l2*math.sin(theta3_1) / (l1 + l2*math.cos(theta3_1))) + math.atan(wz/(wx/math.cos(math.atan2(wy,wx)))))
    theta2_2 = math.radians(90) - (-math.atan(l2*math.sin(theta3_2) / (l1 + l2*math.cos(theta3_2))) + math.atan(wz/(wx/math.cos(math.atan2(wy,wx)))))
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
        angle_1[joint] = round(angle_1[joint] % (2 * np.pi), 2)
        angle_2[joint] = round(angle_2[joint] % (2 * np.pi), 2)
        angle_3[joint] = round(angle_3[joint] % (2 * np.pi), 2)
        angle_4[joint] = round(angle_4[joint] % (2 * np.pi), 2)
        angle_5[joint] = round(angle_5[joint] % (2 * np.pi), 2)
        angle_6[joint] = round(angle_6[joint] % (2 * np.pi), 2)
        angle_7[joint] = round(angle_7[joint] % (2 * np.pi), 2)
        angle_8[joint] = round(angle_8[joint] % (2 * np.pi), 2)

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

    all_configs = [
        (angle_1, check_angle_1),
        (angle_2, check_angle_2), 
        (angle_3, check_angle_3),
        (angle_4, check_angle_4),
        (angle_5, check_angle_5),
        (angle_6, check_angle_6),
        (angle_7, check_angle_7),
        (angle_8, check_angle_8)
    ]
    
    for i, (angles, valid) in enumerate(all_configs):
        print(f"Configuration {i+1}: {[round(math.degrees(a), 2) for a in angles]} -> Valid: {valid}")
    
    valid_configs = [(angles, i+1) for i, (angles, valid) in enumerate(all_configs) if valid]
    
    if valid_configs:
        best_config = valid_configs[0]
        print(f"Selected configuration {best_config[1]}: {[round(math.degrees(a), 2) for a in best_config[0]]}")
        return best_config[0]
    else:
        print("No valid configurations found!")
        return None

def main():
    tx, ty, tz, telev, tazim = get_target_from_user()
    
    wx = tx - l3*math.sin(telev)*math.cos(tazim)
    wy = ty - l3*math.sin(telev)*math.sin(tazim)
    wz = tz - l3*math.cos(telev)
    
    print(f"Wrist position: ({wx}, {wy}, {wz})")
    
    joint_values = invkin(wx, wy, wz, telev, tazim)
    
    if joint_values:
        print("Joint values to send to MoveIt:")
        print([round(math.degrees(angle), 2) for angle in joint_values])
        
        method = input("Send as (1) Joint angles or (2) Pose target? [1/2]: ")
        
        if method == "1":
            if send_to_moveit(joint_values):
                print("Motion executed successfully!")
            else:
                print("Failed to execute motion.")
        else:
            if send_pose_target(tx, ty, tz, telev, tazim):
                print("Motion executed successfully!")
            else:
                print("Failed to execute motion.")
    else:
        print("No valid joint configuration found.")

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        print("\nProgram terminated by user")

