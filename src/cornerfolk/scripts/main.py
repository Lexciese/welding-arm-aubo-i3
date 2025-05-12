#!/usr/bin/env python

import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from std_msgs.msg import Float64
import numpy as np
import tf.transformations
import sys
import copy
from math import pi, cos, sin
import planner
import planner_lurus  # Import the straight line planner

class CircularWeldingNode:
    def __init__(self):
        # Initialize moveit_commander and node
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('circular_welding_node', anonymous=True)
        
        # Initialize robot and scene
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        
        # Initialize the move group for the arm
        self.move_group = moveit_commander.MoveGroupCommander("armor")  # For Aubo i5
        
        # Set reference frame
        self.move_group.set_pose_reference_frame("base_link")
        
        # Configure the move group for best Cartesian planning
        self.move_group.set_max_velocity_scaling_factor(0.1)  # Adjust as needed
        self.move_group.set_max_acceleration_scaling_factor(0.1)  # Adjust as needed
        self.move_group.set_goal_position_tolerance(0.001)  # 1mm tolerance
        self.move_group.set_goal_orientation_tolerance(0.01)  # ~0.6 degrees
        
        # Initialize welding parameter publishers
        self.speed_pub = rospy.Publisher('/welding/speed', Float64, queue_size=10)
        self.torch_angle_pub = rospy.Publisher('/welding/torch_angle', Float64, queue_size=10)
        
        # Publisher for displaying trajectory in RViz
        self.display_trajectory_publisher = rospy.Publisher(
            '/move_group/display_planned_path',
            moveit_msgs.msg.DisplayTrajectory,
            queue_size=20
        )
        
        # Initialize pipe parameters (can be set via ROS params)
        self.cylinder_center = rospy.get_param('~cylinder_center', [1, 0.2, 1])
        self.cylinder_axis = rospy.get_param('~cylinder_axis', [0.0, 0.0, 1.0])
        self.cylinder_radius = rospy.get_param('~cylinder_radius', 0.1)  # Meters
        self.welding_height = rospy.get_param('~welding_height', 0.1)  # Meters from base
        self.torch_offset = rospy.get_param('~torch_offset', 0.1)  # Meters from surface
        self.welding_speed = rospy.get_param('~welding_speed', 0.02)  # m/s
        self.torch_angle = rospy.get_param('~torch_angle', 1)  # Radians offset
        
        # Planning parameters
        self.min_acceptable_fraction = rospy.get_param('~min_acceptable_fraction', 0.7)  # Lower threshold for path completeness
        self.cartesian_step = rospy.get_param('~cartesian_step', 0.01)  # Step size for Cartesian planning
        self.waypoint_count = rospy.get_param('~waypoint_count', 24)  # Reduced number of waypoints
        
        # Add the cylinder to the planning scene
        self._add_cylinder_to_scene()
        
        rospy.loginfo("Circular welding node initialized")
    
    def _add_cylinder_to_scene(self):
        """Add cylinder to the planning scene for collision avoidance"""
        cylinder_pose = geometry_msgs.msg.PoseStamped()
        cylinder_pose.header.frame_id = "base_link"
        cylinder_pose.pose.position.x = self.cylinder_center[0]
        cylinder_pose.pose.position.y = self.cylinder_center[1]
        cylinder_pose.pose.position.z = self.cylinder_center[2]
        cylinder_pose.pose.orientation.w = 1.0
        
        # Get normalized axis
        axis = np.array(self.cylinder_axis)
        axis = axis / np.linalg.norm(axis)
        
        # Calculate cylinder height (make it sufficiently tall)
        height = 0.5  # Can be adjusted as needed
        
        self.scene.add_cylinder("workpiece", cylinder_pose, height, self.cylinder_radius)
        rospy.sleep(1.0)  # Wait for the scene to update
    
    def generate_waypoints(self, num_points=None):
        """Generate waypoints for a circular path around the cylinder"""
        if num_points is None:
            num_points = self.waypoint_count
            
        # Compute total radius (cylinder + offset)
        total_radius = self.cylinder_radius + self.torch_offset
        
        rospy.loginfo(f"Generating {num_points} waypoints for circular path with radius {total_radius}m")
        
        # Generate waypoints around the cylinder
        waypoints = planner.generate_circular_waypoints(
            self.cylinder_center, 
            self.cylinder_axis, 
            total_radius,
            self.welding_height, 
            num_points
        )
        
        # Log the first and last waypoints for debugging
        if waypoints:
            first = waypoints[0].position
            last = waypoints[-1].position
            rospy.loginfo(f"First waypoint: ({first.x:.3f}, {first.y:.3f}, {first.z:.3f})")
            rospy.loginfo(f"Last waypoint: ({last.x:.3f}, {last.y:.3f}, {last.z:.3f})")
        
        return waypoints
    
    def plan_and_execute_path(self):
        """Plan and execute the circular welding path"""
        waypoints = self.generate_waypoints()
        
        # Try different planning approaches if needed
        fraction = 0.0
        plan = None
        
        # First try: Standard approach
        rospy.loginfo("Planning circular path with standard settings...")
        (plan, fraction) = self.move_group.compute_cartesian_path(
            waypoints,      # waypoints to follow
            self.cartesian_step  # eef_step
        )
        
        rospy.loginfo(f"Path planning first attempt: {fraction * 100:.1f}% achieved")
        
        # Second try: If first attempt didn't work well, try with fewer waypoints
        if fraction < self.min_acceptable_fraction:
            rospy.loginfo("First attempt insufficient, trying with fewer waypoints...")
            reduced_waypoints = self.generate_waypoints(int(self.waypoint_count / 2))
            (plan, fraction) = self.move_group.compute_cartesian_path(
                reduced_waypoints,
                self.cartesian_step * 2  # Larger step size
            )
            rospy.loginfo(f"Path planning second attempt: {fraction * 100:.1f}% achieved")
        
        # Check if we got a valid plan with acceptable coverage
        if plan and len(plan.joint_trajectory.points) > 0:
            # Visualize the planned trajectory in RViz
            self._visualize_trajectory(plan)
            
            if fraction >= self.min_acceptable_fraction:
                # Set custom velocity scaling for the welding speed
                plan = self._adjust_trajectory_speed(plan, self.welding_speed)
                
                rospy.loginfo(f"Circular path planning successful: {fraction * 100:.1f}% of trajectory achieved")
                
                # Execute the plan and monitor
                self._start_welding_process()
                
                # Execute the path
                result = self.move_group.execute(plan, wait=True)
                
                # Stop welding process
                self._stop_welding_process()
                
                return result
            else:
                rospy.logerr(f"Path completeness {fraction * 100:.1f}% below threshold {self.min_acceptable_fraction * 100:.1f}%")
                return False
        else:
            rospy.logerr("Failed to generate any valid plan")
            rospy.logerr("Please check the following:")
            rospy.logerr("1. Ensure the cylinder is within the robot's workspace")
            rospy.logerr("2. Verify the cylinder radius and torch offset are appropriate")
            rospy.logerr("3. Check that the welding height is reachable")
            rospy.logerr("4. Inspect the robot's joint limits and collision objects")
            return False
    
    def plan_and_execute_straight_path(self, start_point, end_point, num_points=10):
        """Plan and execute a straight line welding path"""
        rospy.loginfo(f"Planning straight path from {start_point} to {end_point}")
        
        # Generate waypoints
        waypoints = planner_lurus.generate_straight_waypoints(
            start_point, 
            end_point, 
            num_points, 
            orientation_mode="path_direction"
        )
        
        # Log the waypoints for debugging
        if waypoints:
            first = waypoints[0].position
            last = waypoints[-1].position
            rospy.loginfo(f"First waypoint: ({first.x:.3f}, {first.y:.3f}, {first.z:.3f})")
            rospy.loginfo(f"Last waypoint: ({last.x:.3f}, {last.y:.3f}, {last.z:.3f})")
        
        # Plan Cartesian path
        rospy.loginfo("Planning straight line path...")
        (plan, fraction) = self.move_group.compute_cartesian_path(
            waypoints,      # waypoints to follow
            self.cartesian_step  # eef_step
        )
        
        rospy.loginfo(f"Path planning: {fraction * 100:.1f}% achieved")
        
        # Check if we got a valid plan with acceptable coverage
        if plan and len(plan.joint_trajectory.points) > 0:
            # Visualize the planned trajectory in RViz
            self._visualize_trajectory(plan)
            
            if fraction >= self.min_acceptable_fraction:
                # Set custom velocity scaling for the welding speed
                plan = self._adjust_trajectory_speed(plan, self.welding_speed)
                
                # Execute the plan and monitor
                self._start_welding_process()
                
                # Execute the path
                result = self.move_group.execute(plan, wait=True)
                
                # Stop welding process
                self._stop_welding_process()
                
                return result
            else:
                rospy.logerr(f"Path completeness {fraction * 100:.1f}% below threshold")
                return False
        else:
            rospy.logerr("Failed to generate a valid plan for straight line path")
            return False
    
    def _visualize_trajectory(self, plan):
        """Visualize the planned trajectory in RViz"""
        rospy.loginfo("Publishing trajectory for visualization in RViz...")
        
        # Create a DisplayTrajectory message
        display_trajectory = moveit_msgs.msg.DisplayTrajectory()
        display_trajectory.trajectory_start = self.robot.get_current_state()
        display_trajectory.trajectory.append(plan)
        
        # Publish the trajectory for visualization
        self.display_trajectory_publisher.publish(display_trajectory)
        
        # Brief pause to ensure the trajectory is visible
        rospy.sleep(0.5)
    
    def _adjust_trajectory_speed(self, plan, speed):
        """Adjust the trajectory speed for consistent welding"""
        # Get the distance of the path
        total_path_length = self._calculate_path_length(plan)
        
        # Check for zero speed to avoid division by zero
        if speed <= 0.001:  # Small threshold to avoid very small speeds
            rospy.logwarn("Specified welding speed is too small, using default")
            speed = 0.01  # Default minimal speed
            
        # Calculate desired time based on speed
        desired_time = total_path_length / speed
        
        # Get current trajectory duration
        current_time = plan.joint_trajectory.points[-1].time_from_start.to_sec()
        
        # Avoid division by zero when scaling time
        if current_time < 0.001:  # Small threshold for near-zero values
            rospy.logwarn("Trajectory duration is too short, using default scaling")
            time_scaling = 1.0  # Default no scaling
        else:
            time_scaling = desired_time / current_time
            
        # Ensure time_scaling is reasonable
        if time_scaling <= 0.001:
            rospy.logwarn("Time scaling too small, using minimum value")
            time_scaling = 0.001
        
        # Adjust the timing of all trajectory points
        new_plan = copy.deepcopy(plan)
        for i, point in enumerate(new_plan.joint_trajectory.points):
            point.time_from_start = rospy.Duration(point.time_from_start.to_sec() * time_scaling)
            
            # Scale velocities and accelerations if needed
            for j in range(len(point.velocities)):
                point.velocities[j] = point.velocities[j] / time_scaling
                point.accelerations[j] = point.accelerations[j] / (time_scaling * time_scaling)
        
        return new_plan
    
    def _calculate_path_length(self, plan):
        """Calculate the approximate path length"""
        # Get the waypoints from the trajectory
        points = []
        for point in plan.joint_trajectory.points:
            joint_positions = point.positions
            # Forward kinematics to get cartesian position
            self.robot.get_group(self.move_group.get_name()).get_current_state().joint_state.position = joint_positions
            pose = self.move_group.get_current_pose().pose
            points.append([pose.position.x, pose.position.y, pose.position.z])
        
        # Calculate length as sum of segments
        length = 0.0
        for i in range(1, len(points)):
            length += np.linalg.norm(np.array(points[i]) - np.array(points[i-1]))
        
        return length
    
    def _start_welding_process(self):
        """Initialize welding process"""
        # Publish initial welding parameters
        self.speed_pub.publish(Float64(self.welding_speed))
        self.torch_angle_pub.publish(Float64(self.torch_angle))
        
        # Additional welding start logic would go here
        rospy.loginfo("Starting welding process")
    
    def _stop_welding_process(self):
        """Stop welding process"""
        # Publish zero speed to stop welding
        self.speed_pub.publish(Float64(0.0))
        
        # Additional welding stop logic would go here
        rospy.loginfo("Stopping welding process")
    
    def update_parameters(self, cylinder_radius, welding_height=None, torch_offset=None, 
                         welding_speed=None, torch_angle=None):
        """Update the welding parameters dynamically"""
        # Update parameters that are provided
        self.cylinder_radius = cylinder_radius
        
        if welding_height is not None:
            self.welding_height = welding_height
        
        if torch_offset is not None:
            self.torch_offset = torch_offset
        
        if welding_speed is not None:
            self.welding_speed = welding_speed
            # Update welding speed if we're currently welding
            self.speed_pub.publish(Float64(self.welding_speed))
        
        if torch_angle is not None:
            self.torch_angle = torch_angle
            # Update torch angle if we're currently welding
            self.torch_angle_pub.publish(Float64(self.torch_angle))
        
        # Update scene with new cylinder
        self.scene.remove_world_object("workpiece")
        rospy.sleep(0.5)
        self._add_cylinder_to_scene()
        
        rospy.loginfo("Updated welding parameters")

def main():
    try:
        # Initialize the node
        welding_node = CircularWeldingNode()
        
        # For demonstration, perform the welding
        success = welding_node.plan_and_execute_path()
        
        if success:
            rospy.loginfo("Welding completed successfully")
        else:
            rospy.logerr("Welding failed")
            
        # Keep node alive
        rospy.spin()
        
    except rospy.ROSInterruptException:
        return
    except Exception as e:
        rospy.logerr(f"An error occurred: {e}")

if __name__ == '__main__':
    main()