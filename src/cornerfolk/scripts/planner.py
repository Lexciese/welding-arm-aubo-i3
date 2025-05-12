#!/usr/bin/env python

# Software License Agreement (BSD License)
# Copyright (c) 2017-2018, Aubo Robotics
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and/or other materials provided
#    with the distribution.
#  * Neither the name of SRI International nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import sys
import copy
import rospy
import threading
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseArray
import math
import numpy as np

class AuboRobotPlannerNode():
    """
    Constructor of aubo robot planner
    """
    def __init__(self,  update_rate = 10):
        rospy.init_node('aubo_ros_plan')

        # Class lock
        self.lock = threading.Lock()
        self.plan_flag = 0

        # Instantiate a MoveGroupCommander object. This object is an interface to one group of joints.
        self.group = moveit_commander.MoveGroupCommander("manipulator_i3")

        moveit_commander.roscpp_initialize(sys.argv)
        # Instantiate a RobotCommander object.  This object is an interface to the robot as a whole.
        self.robot = moveit_commander.RobotCommander()

        # Instantiate a PlanningSceneInterface object.  This object is an interface to the world surrounding the robot.
        self.scene = moveit_commander.PlanningSceneInterface()

        # We create this DisplayTrajectory publisher which is used below to publish trajectories for RVIZ to visualize.
        self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path', moveit_msgs.msg.DisplayTrajectory)

        # Planning to a Pose goal
        self.target_pose_subs = rospy.Subscriber('/aubo_driver/target_pose', Pose, self.set_target_pose)

        # Planning to joint-space goal
        self.target_joints_subs = rospy.Subscriber('/aubo_driver/target_joint_value', Float32MultiArray, self.set_target_joint_value)

        # Planning to Cartesian path goal
        self.cartesian_path_subs = rospy.Subscriber('/aubo_driver/cartesian_path_list', PoseArray, self.set_cartesian_path_list)
        
        # Subscribe to circular motion parameters
        self.circular_motion_subs = rospy.Subscriber('/aubo_driver/circular_motion', Float32MultiArray, self.generate_circular_motion)
        
        # Publisher for circular motion commands
        self.circular_motion_pub = rospy.Publisher('/aubo_driver/circular_motion', Float32MultiArray, queue_size=10)

        # Wait for RVIZ to initialize. This sleep is ONLY to allow Rviz to come up.
        rospy.sleep(10)

        rospy.loginfo('The name of the reference frame for this robot: %s', str(self.group.get_planning_frame()))
        rospy.loginfo('The name of the end-effector link for this group: %s', str(self.group.get_end_effector_link()))
        rospy.loginfo('A list of all the groups in the robot: %s', str(self.robot.get_group_names()))
        rospy.loginfo('The entire state of the robot: %s', str(self.robot.get_current_state()))

        self.execute = False
        self.pose_target = geometry_msgs.msg.Pose()
        self.group_variable_values = self.group.get_current_joint_values()
        self.num_joint = len(self.group_variable_values)

        # The resolution of the cartesian path to be interpolated
        self.eef_step = 0.01
        self.jump_threshold = 0.0

        self.update_rate = update_rate
        rospy.logdebug("ros planner update rate (hz): %f", self.update_rate)

        # Motion thread
        self.motion_thread = threading.Thread(target=self.ros_planner)
        self.motion_thread.daemon = True
        self.motion_thread.start()
        
        # Add a timer to demonstrate circular motion (uncomment to auto-trigger)
        # rospy.Timer(rospy.Duration(15), self.publish_demo_circular_motion)

    """
    """
    def ros_planner(self):
        rospy.spin()
        # self.moveit_commander.roscpp_shutdown()

    """
    Plan a motion for this group to a desired pose for the end-effector
    """
    def set_target_pose(self, msg_in):
        if self.plan_flag == 0:
            self.plan_flag = 1
            self.pose_target.orientation.w = msg_in.orientation.w
            self.pose_target.orientation.x = msg_in.orientation.x
            self.pose_target.orientation.y = msg_in.orientation.y
            self.pose_target.orientation.z = msg_in.orientation.z
            self.pose_target.position.x = msg_in.position.x
            self.pose_target.position.y = msg_in.position.y
            self.pose_target.position.z = msg_in.position.z
            self.group.set_pose_target(self.pose_target)
            # self.plan_type = 0
            if self.execute == True:
                success = self.group.go(wait=True)
            else:
                plan = self.group.plan()
            self.plan_flag = 0
        else:
            rospy.logdebug('There is a planning already!')

    """
    Plan a motion for this group to a joint-space goal
    """
    def set_target_joint_value(self, msg_in):
        if self.plan_flag == 0:
            self.plan_flag = 1
            self.group.clear_pose_targets()
            for i in range(0, self.num_joint):
                self.group_variable_values[i] = msg_in.data[i]
            self.group.set_joint_value_target(self.group_variable_values)

            if self.execute == True:
                success = self.group.go(wait=True)
            else:
                plan = self.group.plan()
            self.plan_flag = 0
        else:
            rospy.logdebug('There is a planning already!')

    """
    Plan a motion for this group to a Cartesian path goal
    """
    def set_cartesian_path_list(self, msg_in):
        if self.plan_flag == 0:
            self.plan_flag = 1
            waypoints = []
            waypoints.append(self.group.get_current_pose().pose)

            for i in range(0, len(msg_in)):
                waypoints.append(msg_in[i])
            (plan, fraction) = self.group.compute_cartesian_path(waypoints, self.eef_step)
            # self.group.
            # if self.execute == True:
            #     success = self.group.go(wait=True)
            # else:
            #     plan = self.group.plan()

            self.plan_flag = 0
        else:
            rospy.logdebug('There is a planning already!')

    """
    Generate and execute circular motion
    msg_in format: [center_x, center_y, center_z, radius, start_angle, end_angle, num_points, execute_flag]
    """
    def generate_circular_motion(self, msg_in):
        if self.plan_flag == 0:
            self.plan_flag = 1
            
            # Extract parameters from message
            center_x = msg_in.data[0]
            center_y = msg_in.data[1]
            center_z = msg_in.data[2]
            radius = msg_in.data[3]
            start_angle = msg_in.data[4]
            end_angle = msg_in.data[5]
            num_points = int(msg_in.data[6])
            execute_flag = int(msg_in.data[7]) == 1
            
            # Generate waypoints
            waypoints = []
            current_pose = self.group.get_current_pose().pose
            waypoints.append(current_pose)
            
            # Calculate angle step
            angle_step = (end_angle - start_angle) / (num_points - 1) if num_points > 1 else 0
            
            for i in range(num_points):
                pose = Pose()
                angle = start_angle + i * angle_step
                
                # Position on circle
                pose.position.x = center_x + radius * math.cos(angle)
                pose.position.y = center_y + radius * math.sin(angle)
                pose.position.z = center_z
                
                # Calculate orientation to keep end effector perpendicular to circle
                # Vector from center to point on circle
                direction_vector = [math.cos(angle), math.sin(angle), 0]
                
                # Using quaternion to represent orientation
                # This makes the end effector point inward toward circle center
                # For outward orientation, negate the direction vector
                
                # Assuming Z-axis of end-effector should be pointing along direction vector
                # and Y-axis should be pointing up
                z_axis = [-direction_vector[0], -direction_vector[1], -direction_vector[2]]  # Inward facing
                z_norm = np.linalg.norm(z_axis)
                if z_norm > 0:
                    z_axis = [z_axis[0]/z_norm, z_axis[1]/z_norm, z_axis[2]/z_norm]
                
                # Up vector (typically Z in world frame)
                up = [0, 0, 1]
                
                # X-axis is perpendicular to Z and up
                x_axis = np.cross(up, z_axis)
                x_norm = np.linalg.norm(x_axis)
                if x_norm > 0:
                    x_axis = [x_axis[0]/x_norm, x_axis[1]/x_norm, x_axis[2]/x_norm]
                
                # Y-axis completes the right-handed coordinate system
                y_axis = np.cross(z_axis, x_axis)
                
                # Rotation matrix
                rotation_matrix = np.array([x_axis, y_axis, z_axis]).T
                
                # Convert to quaternion (simplified method)
                # Using the fact that the trace of the rotation matrix is related to the scalar part of the quaternion
                trace = rotation_matrix[0, 0] + rotation_matrix[1, 1] + rotation_matrix[2, 2]
                
                if trace > 0:
                    s = 0.5 / math.sqrt(trace + 1.0)
                    pose.orientation.w = 0.25 / s
                    pose.orientation.x = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) * s
                    pose.orientation.y = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) * s
                    pose.orientation.z = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) * s
                else:
                    if rotation_matrix[0, 0] > rotation_matrix[1, 1] and rotation_matrix[0, 0] > rotation_matrix[2, 2]:
                        s = 2.0 * math.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2])
                        pose.orientation.w = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
                        pose.orientation.x = 0.25 * s
                        pose.orientation.y = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
                        pose.orientation.z = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
                    elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
                        s = 2.0 * math.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2])
                        pose.orientation.w = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
                        pose.orientation.x = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
                        pose.orientation.y = 0.25 * s
                        pose.orientation.z = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
                    else:
                        s = 2.0 * math.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1])
                        pose.orientation.w = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
                        pose.orientation.x = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
                        pose.orientation.y = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
                        pose.orientation.z = 0.25 * s
                
                waypoints.append(pose)
            
            # Compute cartesian path
            (plan, fraction) = self.group.compute_cartesian_path(
                waypoints,      # waypoints to follow
                self.eef_step)  # jump_threshold
            
            rospy.loginfo("Circular path planning: path computed with %s success rate", fraction)
            
            # Execute the trajectory if requested
            if execute_flag:
                self.group.execute(plan, wait=True)
                rospy.loginfo("Circular motion executed")
            
            self.plan_flag = 0
        else:
            rospy.logdebug('There is a planning already!')

    """
    Publish a circular motion command
    """
    def publish_circular_motion(self, center_x, center_y, center_z, radius, start_angle, end_angle, num_points, execute_flag):
        msg = Float32MultiArray()
        msg.data = [center_x, center_y, center_z, radius, start_angle, end_angle, num_points, execute_flag]
        rospy.loginfo("Publishing circular motion command: %s", str(msg.data))
        self.circular_motion_pub.publish(msg)
    
    """
    Demo function to publish a predefined circular motion
    Can be called manually or triggered by a timer
    """
    def publish_demo_circular_motion(self, event=None):
        # Get current pose
        current_pose = self.group.get_current_pose().pose
        
        # Define circle parameters
        center_x = current_pose.position.x      # Center X relative to current position
        center_y = current_pose.position.y      # Center Y
        center_z = current_pose.position.z-1      # Same Z height
        radius = 0.1                           # 10cm radius
        start_angle = 0.0                      # Start at 0 radians
        end_angle = 2.0 * math.pi              # Complete circle (2π radians)
        num_points = 36                        # Number of waypoints (10° increments)
        execute_flag = 1                       # Execute the motion (1 = yes, 0 = plan only)
        
        self.publish_circular_motion(
            center_x, center_y, center_z, 
            radius, start_angle, end_angle, 
            num_points, execute_flag
        )

if __name__ == '__main__':
    try:
        rospy.loginfo('Starting aubo ros plan!')
        planner = AuboRobotPlannerNode()
        
        # Wait for initialization
        rospy.sleep(2)
        
        # Call the demo function to execute a circular motion
        planner.publish_demo_circular_motion()
        
        rospy.spin()
    except rospy.ROSInterruptException:
        pass