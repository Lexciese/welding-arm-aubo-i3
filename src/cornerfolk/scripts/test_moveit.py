#!/usr/bin/env python

import sys
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list

def main():
    # Initialize moveit_commander and rospy
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('move_arm_to_position', anonymous=True)
    
    # Instantiate a RobotCommander object
    robot = moveit_commander.RobotCommander()
    
    # Instantiate a PlanningSceneInterface object
    scene = moveit_commander.PlanningSceneInterface()
    
    # Instantiate a MoveGroupCommander object for the arm group
    arm_group = moveit_commander.MoveGroupCommander("armor")
    
    # Create a DisplayTrajectory publisher for visualization
    display_trajectory_publisher = rospy.Publisher(
        '/move_group/display_planned_path',
        moveit_msgs.msg.DisplayTrajectory,
        queue_size=20
    )
    
    # Print detailed information about robot and planning group
    rospy.loginfo("=" * 50)
    rospy.loginfo("ROBOT INFORMATION:")
    rospy.loginfo("Reference frame: %s" % arm_group.get_planning_frame())
    rospy.loginfo("End effector link: %s" % arm_group.get_end_effector_link())
    rospy.loginfo("Available planning groups: %s" % robot.get_group_names())
    rospy.loginfo("Joint names: %s" % arm_group.get_joints())
    rospy.loginfo("Current joint values: %s" % arm_group.get_current_joint_values())
    current_pose = arm_group.get_current_pose().pose
    rospy.loginfo("Current pose: x=%s, y=%s, z=%s" % 
                 (current_pose.position.x, current_pose.position.y, current_pose.position.z))
    rospy.loginfo("Robot state: %s" % robot.get_current_state())
    rospy.loginfo("=" * 50)
    
    # Define waypoints to explore the workspace
    waypoints = [
        # First waypoint - the originally requested position
        {"x": -1.2000, "y": -0.0980, "z": 0.3000, "name": "Requested Position"},
        
        # Additional waypoints to explore the workspace
        {"x": -0.8, "y": 0.3, "z": 0.5, "name": "Front Right Top"},
        {"x": -0.8, "y": -0.3, "z": 0.5, "name": "Front Left Top"},
        {"x": -0.8, "y": 0.0, "z": 0.2, "name": "Front Center Bottom"},
        {"x": -1.0, "y": 0.0, "z": 0.4, "name": "Middle Center"},
        {"x": -1.2, "y": 0.3, "z": 0.3, "name": "Back Right"},
        {"x": -1.2, "y": -0.3, "z": 0.3, "name": "Back Left"},
    ]
    
    # Move to each waypoint
    for waypoint in waypoints:
        rospy.loginfo("Moving to: %s" % waypoint["name"])
        
        # Create a pose target
        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.position.x = waypoint["x"]
        pose_goal.position.y = waypoint["y"]
        pose_goal.position.z = waypoint["z"]
        pose_goal.orientation.w = 1.0  # Default orientation (no rotation)
        
        # Set the pose target
        arm_group.set_pose_target(pose_goal)
        
        # Plan and execute
        rospy.loginfo("Planning and moving to %s..." % waypoint["name"])
        plan = arm_group.go(wait=True)
        
        # Clear targets after planning
        arm_group.clear_pose_targets()
        
        # Print result
        current_pose = arm_group.get_current_pose().pose
        rospy.loginfo("Movement executed. Current position: x=%s, y=%s, z=%s" % 
                     (current_pose.position.x, current_pose.position.y, current_pose.position.z))
        
        # Pause briefly to observe the position
        rospy.sleep(1.0)
    
    # Return to home position
    rospy.loginfo("Returning to home position...")
    arm_group.set_named_target("home")  # If 'home' is defined in your SRDF
    arm_group.go(wait=True)
    
    # Shutdown
    moveit_commander.roscpp_shutdown()
    rospy.loginfo("Motion exploration complete!")

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr("An error occurred: %s" % e)

