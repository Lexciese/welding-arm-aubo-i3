#!/usr/bin/env python

import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi, sin, cos, asin, acos, atan2, sqrt
import numpy as np
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list

class PipeWeldingPlanner:
    def __init__(self):
        # Initialize ROS node
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('pipe_welding_planner', anonymous=True)
        
        # Setup robot and scene
        robot = moveit_commander.RobotCommander()
        scene = moveit_commander.PlanningSceneInterface()
        group_name = "manipulator_i3"  # This should match your robot's move group
        move_group = moveit_commander.MoveGroupCommander(group_name)
        
        # Store variables
        self.robot = robot
        self.scene = scene
        self.move_group = move_group
        self.planning_frame = move_group.get_planning_frame()
        self.eef_link = move_group.get_end_effector_link()
        self.group_names = robot.get_group_names()
        
        # Setup Publisher
        self.display_trajectory_publisher = rospy.Publisher(
            '/move_group/display_planned_path',
            moveit_msgs.msg.DisplayTrajectory,
            queue_size=20)
            
        # Pipe parameters (in meters)
        self.pipe_diameter = 4 * 0.0254  # 4 inch to meters
        self.pipe_radius = self.pipe_diameter / 2
        self.pipe_center = [0.4, 0.0, 0.2]  # Position of the center of the pipe joint in robot base frame
        self.pipe_orientation = [0, 1, 0]   # The pipe runs along the y-axis
        
        # Welding torch parameters
        self.torch_length = 0.15   # Length of welding torch from end effector
        self.torch_angle = pi/4    # Angle of torch to pipe surface
        
        # Welding parameters
        self.weld_speed = 0.005    # meters per second
        self.tack_duration = 3.0   # seconds
        self.tack_positions = 4    # number of tack positions
        
        # For each layer, specify the distance from the joint seam and the number of passes
        self.weld_layers = [
            {"name": "root", "distance": 0.0, "passes": 1},        # Root pass
            {"name": "hot", "distance": 0.001, "passes": 1},       # Hot pass (1mm from root)
            {"name": "fill", "distance": 0.003, "passes": 2},      # Fill passes (3mm)
            {"name": "cap", "distance": 0.005, "passes": 1}        # Cap pass (5mm)
        ]
        
        rospy.loginfo("PipeWeldingPlanner initialized")
        rospy.loginfo("Planning frame: %s", self.planning_frame)
        rospy.loginfo("End effector link: %s", self.eef_link)
        rospy.loginfo("Available Planning Groups: %s", self.group_names)

    def go_to_home_position(self):
        """Move the robot to a safe home position"""
        joint_goal = self.move_group.get_current_joint_values()
        joint_goal[0] = 0
        joint_goal[1] = -pi/4
        joint_goal[2] = 0
        joint_goal[3] = -pi/2
        joint_goal[4] = 0
        joint_goal[5] = 0
        
        self.move_group.go(joint_goal, wait=True)
        self.move_group.stop()
        
        rospy.loginfo("Robot moved to home position")
        return True

    def calculate_weld_pose(self, angle, layer_distance):
        """
        Calculate a pose for the end effector at the given angle on the pipe joint.
        
        Args:
            angle: Angle around the pipe (0 to 2*pi)
            layer_distance: Distance from the center joint seam
        
        Returns:
            geometry_msgs.msg.Pose: The calculated pose
        """
        pose = geometry_msgs.msg.Pose()
        
        # Calculate position on pipe
        x = self.pipe_center[0] + (self.pipe_radius + layer_distance) * cos(angle)
        y = self.pipe_center[1]
        z = self.pipe_center[2] + (self.pipe_radius + layer_distance) * sin(angle)
        
        # Calculate orientation - torch points inward to pipe center at an angle
        # This is a simplified orientation calculation for demonstration
        direction_to_center = [self.pipe_center[0] - x, 
                              self.pipe_center[1] - y,
                              self.pipe_center[2] - z]
        norm = sqrt(sum([d*d for d in direction_to_center]))
        direction_to_center = [d/norm for d in direction_to_center]
        
        # Adjust for torch angle
        adjusted_direction = [
            direction_to_center[0] * cos(self.torch_angle),
            direction_to_center[1],
            direction_to_center[2] * cos(self.torch_angle)
        ]
        
        # Convert direction to quaternion (simplified)
        # For a real robot, you would need a more sophisticated orientation calculation
        qx = 0
        qy = sin(angle/2)
        qz = 0
        qw = cos(angle/2)
        
        # Set pose
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation.x = qx
        pose.orientation.y = qy
        pose.orientation.z = qz
        pose.orientation.w = qw
        
        return pose

    def execute_tack_welds(self):
        """Perform tack welds at 4 positions around the pipe"""
        rospy.loginfo("Starting tack welding")
        
        # Calculate tack weld positions (equally spaced around the pipe)
        for i in range(self.tack_positions):
            angle = i * 2 * pi / self.tack_positions
            
            # Move to position
            tack_pose = self.calculate_weld_pose(angle, 0)
            
            # Approach position (slightly away from the pipe)
            approach_pose = copy.deepcopy(tack_pose)
            approach_pose.position.x += 0.05 * cos(angle)
            approach_pose.position.z += 0.05 * sin(angle)
            
            rospy.loginfo(f"Moving to tack position {i+1}/{self.tack_positions}")
            self.move_group.set_pose_target(approach_pose)
            self.move_group.go(wait=True)
            self.move_group.stop()
            self.move_group.clear_pose_targets()
            
            # Move to weld position
            self.move_group.set_pose_target(tack_pose)
            self.move_group.go(wait=True)
            self.move_group.stop()
            self.move_group.clear_pose_targets()
            
            # Simulate welding
            rospy.loginfo(f"Performing tack weld {i+1}")
            rospy.sleep(self.tack_duration)
            
            # Retreat after welding
            self.move_group.set_pose_target(approach_pose)
            self.move_group.go(wait=True)
            self.move_group.stop()
            self.move_group.clear_pose_targets()
        
        rospy.loginfo("Tack welding completed")
        return True

    def execute_circular_weld(self, layer_name, layer_distance, passes=1):
        """
        Execute a circular weld around the pipe at the specified distance from the joint.
        
        Args:
            layer_name: Name of this weld pass (for logging)
            layer_distance: Distance from the center joint seam
            passes: Number of passes to make at this layer
        """
        rospy.loginfo(f"Starting {layer_name} weld, distance: {layer_distance}m, passes: {passes}")
        
        for p in range(passes):
            rospy.loginfo(f"Executing {layer_name} weld pass {p+1}/{passes}")
            
            # Number of waypoints for a smooth circular motion
            num_waypoints = 20
            waypoints = []
            
            # Generate waypoints for a circular path
            for i in range(num_waypoints + 1):
                angle = i * 2 * pi / num_waypoints
                
                # Get pose at this angle
                waypoint_pose = self.calculate_weld_pose(angle, layer_distance)
                waypoints.append(copy.deepcopy(waypoint_pose))
            
            # First move to the approach position for the first waypoint
            approach_pose = copy.deepcopy(waypoints[0])
            approach_angle = 0
            approach_pose.position.x += 0.05 * cos(approach_angle)
            approach_pose.position.z += 0.05 * sin(approach_angle)
            
            self.move_group.set_pose_target(approach_pose)
            self.move_group.go(wait=True)
            self.move_group.stop()
            self.move_group.clear_pose_targets()
            
            # Then move to the first waypoint
            self.move_group.set_pose_target(waypoints[0])
            self.move_group.go(wait=True)
            self.move_group.stop()
            self.move_group.clear_pose_targets()
            
            # Plan and execute path through all waypoints
            (plan, fraction) = self.move_group.compute_cartesian_path(
                waypoints[1:],   # Skip the first waypoint since we're already there
                0.01,            # Step size of 1cm
                0.0)             # No jump threshold
            
            # Calculate duration based on weld speed
            # Circumference of pipe = 2 * pi * radius
            path_distance = 2 * pi * (self.pipe_radius + layer_distance)
            duration = path_distance / self.weld_speed
            
            # Scale the plan for the weld speed
            plan = self.move_group.retime_trajectory(
                self.robot.get_current_state(),
                plan,
                velocity_scaling_factor=1.0,
                acceleration_scaling_factor=1.0,
                algorithm="iterative_time_parameterization")
            
            rospy.loginfo(f"Executing {layer_name} circular weld")
            self.move_group.execute(plan, wait=True)
            
            # Return to approach position
            self.move_group.set_pose_target(approach_pose)
            self.move_group.go(wait=True)
            self.move_group.stop()
            self.move_group.clear_pose_targets()
        
        rospy.loginfo(f"{layer_name.capitalize()} weld completed")
        return True

    def execute_complete_welding_sequence(self):
        """Execute the complete pipe welding sequence"""
        rospy.loginfo("Starting complete welding sequence")
        
        try:
            # Go to home position
            self.go_to_home_position()
            
            # Execute tack welds
            self.execute_tack_welds()
            
            # Execute each weld layer
            for layer in self.weld_layers:
                self.execute_circular_weld(
                    layer["name"], 
                    layer["distance"], 
                    layer["passes"]
                )
            
            # Return to home position
            self.go_to_home_position()
            
            rospy.loginfo("Complete welding sequence finished successfully")
            return True
            
        except Exception as e:
            rospy.logerr(f"Error during welding sequence: {e}")
            return False

def main():
    try:
        weld_planner = PipeWeldingPlanner()
        weld_planner.execute_complete_welding_sequence()
    except rospy.ROSInterruptException:
        return
    except Exception as e:
        rospy.logerr(f"Unexpected error: {e}")
        return

if __name__ == '__main__':
    main()
