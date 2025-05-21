#!/usr/bin/env python
import rospy
import math
import numpy as np
from geometry_msgs.msg import Pose, Point, Quaternion
from moveit_msgs.msg import DisplayTrajectory, RobotTrajectory
from moveit_commander import MoveGroupCommander, RobotCommander, PlanningSceneInterface
from tf.transformations import quaternion_from_euler

class ThermalWeldingPathPlanner:
    def __init__(self):
        rospy.init_node('thermal_welding_path_planner', anonymous=True)
        
        # Load parameters
        self.load_parameters()
        
        # Initialize MoveIt components
        self.robot = RobotCommander()
        self.scene = PlanningSceneInterface()
        self.group = MoveGroupCommander("manipulator_i3")
        
        # Set planning parameters
        self.group.set_planning_time(10.0)
        self.group.set_num_planning_attempts(10)
        self.group.allow_replanning(True)
        
        # Thermal model variables
        self.thermal_map = np.zeros(360)  # 1 degree resolution
        self.current_pass = 0
        
        # Setup the environment
        self.setup_environment()
        
        # Execute welding sequence
        self.execute_welding_sequence()

    def load_parameters(self):
        """Load welding parameters from ROS parameter server"""
        self.pipe_diameter = rospy.get_param('~pipe/diameter', 0.1143)
        self.pipe_length = rospy.get_param('~pipe/length', 1.0)
        self.pipe_position = rospy.get_param('~pipe/position', [0.5, 0.0, 0.3])
        self.pipe_orientation = rospy.get_param('~pipe/orientation', [0, 0, 0])
        
        self.tack_points = rospy.get_param('~welding/tack_points', 4)
        self.base_speed = rospy.get_param('~welding/base_speed', 0.005)
        self.min_speed = rospy.get_param('~welding/min_speed', 0.003)
        self.max_speed = rospy.get_param('~welding/max_speed', 0.007)
        self.weld_offset = rospy.get_param('~welding/weld_offset', 0.002)
        self.interpass_offset = rospy.get_param('~welding/interpass_offset', 0.0015)
        
        # Thermal parameters
        self.heat_sensitivity_angles = rospy.get_param('~welding/thermal/heat_sensitivity_angles', [45, 135, 225, 315])
        self.sensitivity_radius = rospy.get_param('~welding/thermal/sensitivity_radius', 20)
        self.cooling_time = rospy.get_param('~welding/thermal/cooling_time_per_pass', 30)
        self.segment_size = rospy.get_param('~welding/thermal/segment_size', 90)
        
        # Calculate pipe radius
        self.pipe_radius = self.pipe_diameter / 2.0

    def setup_environment(self):
        """Add the pipe to the planning scene"""
        rospy.loginfo("Setting up welding environment with thermal management...")

    def calculate_speed_profile(self, angle_deg):
        """Calculate adaptive speed based on thermal conditions"""
        # Find minimum distance to any heat-sensitive point
        min_dist = 180
        for sensitive_angle in self.heat_sensitivity_angles:
            dist = min(abs(angle_deg - sensitive_angle), 
                      360 - abs(angle_deg - sensitive_angle))
            if dist < min_dist:
                min_dist = dist
        
        # If within sensitivity radius, reduce speed
        if min_dist <= self.sensitivity_radius:
            # Linear speed reduction based on proximity
            speed_reduction = 1 - (min_dist / self.sensitivity_radius)
            return self.base_speed * (1 - speed_reduction * 0.4)  # Reduce up to 40%
        else:
            # In cooler areas, we can go slightly faster
            return min(self.base_speed * 1.2, self.max_speed)

    def update_thermal_map(self, angle_deg, duration):
        """Update the thermal model with new heat input"""
        angle_idx = int(angle_deg) % 360
        # Heat input is proportional to duration (simplified model)
        self.thermal_map[angle_idx] += duration * 0.1
        
        # Also affect neighboring areas (heat diffusion)
        for i in range(1, 6):  # 5 degrees each side
            self.thermal_map[(angle_idx + i) % 360] += duration * 0.05
            self.thermal_map[(angle_idx - i) % 360] += duration * 0.05

    def get_segment_order(self, pass_num):
        """Determine optimal segment welding order for thermal distribution"""
        # Alternate segment order based on pass number to distribute heat
        segments = list(range(0, 360, self.segment_size))
        if pass_num % 2 == 0:
            return segments  # 0, 90, 180, 270
        else:
            return [segments[2], segments[3], segments[0], segments[1]]  # 180, 270, 0, 90

    def create_adaptive_path(self, radius, center, offset, num_segments=4):
        """Generate path with adaptive speed and thermal sequencing"""
        segments = self.get_segment_order(self.current_pass)
        all_waypoints = []
        speed_profile = []
        
        for start_angle in segments:
            end_angle = start_angle + self.segment_size
            segment_waypoints = []
            segment_speeds = []
            
            # Create points for this segment
            for angle in range(start_angle, end_angle, 5):  # 5 degree resolution
                angle_rad = math.radians(angle)
                
                # Calculate position
                x = center[0]
                y = center[1] + (radius + offset) * math.cos(angle_rad)
                z = center[2] + (radius + offset) * math.sin(angle_rad)
                
                # Calculate orientation
                q = quaternion_from_euler(math.pi, angle_rad + math.pi/2, 0)
                
                pose = Pose()
                pose.position = Point(x, y, z)
                pose.orientation = Quaternion(*q)
                
                segment_waypoints.append(pose)
                
                # Calculate adaptive speed for this angle
                speed = self.calculate_speed_profile(angle)
                segment_speeds.append(speed)
            
            all_waypoints.extend(segment_waypoints)
            speed_profile.extend(segment_speeds)
        
        return all_waypoints, speed_profile

    def execute_adaptive_trajectory(self, waypoints, speed_profile):
        """Execute trajectory with adaptive speed control"""
        # Plan the Cartesian path
        (plan, fraction) = self.group.compute_cartesian_path(
            waypoints,   # waypoints to follow
            0.01,       # eef_step
            True)       # avoid_collisions
        
        if fraction < 0.9:
            rospy.logwarn(f"Only {fraction*100:.1f}% of the path was planned")
            return False
        
        # Retime the trajectory based on speed profile
        plan = self.retime_trajectory_with_profile(plan, waypoints, speed_profile)
        
        # Execute the plan
        self.group.execute(plan, wait=True)
        
        # Update thermal model
        self.update_thermal_model_from_trajectory(plan, waypoints)
        
        return True

    def retime_trajectory_with_profile(self, plan, waypoints, speed_profile):
        """Retime trajectory based on adaptive speed profile"""
        trajectory = plan.joint_trajectory
        
        # Calculate cumulative distance
        distances = [0]
        for i in range(1, len(waypoints)):
            p1 = waypoints[i-1].position
            p2 = waypoints[i].position
            distances.append(distances[-1] + math.sqrt(
                (p2.x-p1.x)**2 + (p2.y-p1.y)**2 + (p2.z-p1.z)**2))
        
        # Calculate time for each point based on speed profile
        times = [0]
        for i in range(1, len(distances)):
            segment_length = distances[i] - distances[i-1]
            avg_speed = (speed_profile[i] + speed_profile[i-1]) / 2
            times.append(times[-1] + segment_length / avg_speed)
        
        # Update trajectory points
        for i, point in enumerate(trajectory.points):
            point.time_from_start = rospy.Duration(times[i])
        
        return plan

    def update_thermal_model_from_trajectory(self, plan, waypoints):
        """Update thermal model after executing a trajectory"""
        trajectory = plan.joint_trajectory
        
        for i in range(1, len(trajectory.points)):
            # Get angle from waypoint position
            pos = waypoints[i].position
            rel_y = pos.y - self.pipe_position[1]
            rel_z = pos.z - self.pipe_position[2]
            angle_deg = math.degrees(math.atan2(rel_z, rel_y)) % 360
            
            # Calculate time spent at this point
            duration = (trajectory.points[i].time_from_start - 
                       trajectory.points[i-1].time_from_start).to_sec()
            
            # Update thermal model
            self.update_thermal_map(angle_deg, duration)

    def execute_tack_welding(self):
        """Perform tack welding at specified points with thermal management"""
        rospy.loginfo("Starting thermal-optimized tack welding...")
        
        # Generate tack points in optimal order
        tack_order = self.get_segment_order(0)  # First pass order
        tack_angles = [a for a in range(0, 360, 360//self.tack_points)]
        ordered_angles = sorted(tack_angles, 
                              key=lambda x: min(abs(x - s) for s in tack_order))
        
        # Create poses for tack points
        tack_poses = []
        for angle in ordered_angles:
            angle_rad = math.radians(angle)
            x = self.pipe_position[0]
            y = self.pipe_position[1] + (self.pipe_radius + self.weld_offset) * math.cos(angle_rad)
            z = self.pipe_position[2] + (self.pipe_radius + self.weld_offset) * math.sin(angle_rad)
            
            q = quaternion_from_euler(math.pi, angle_rad + math.pi/2, 0)
            
            pose = Pose()
            pose.position = Point(x, y, z)
            pose.orientation = Quaternion(*q)
            tack_poses.append(pose)
        
        # Move to each tack point with speed adjustment
        for i, pose in enumerate(tack_poses):
            angle = ordered_angles[i]
            speed = self.calculate_speed_profile(angle)
            
            rospy.loginfo(f"Moving to tack point {i+1}/{self.tack_points} at {math.degrees(angle):.1f}° (speed: {speed:.4f} m/s)")
            
            # Set max velocity scaling factor based on speed
            self.group.set_max_velocity_scaling_factor(speed / self.base_speed)
            self.group.set_pose_target(pose)
            plan = self.group.go(wait=True)
            self.group.stop()
            self.group.clear_pose_targets()
            
            # Reset speed scaling
            self.group.set_max_velocity_scaling_factor(0.9)
            
            # Simulate tack welding (pause based on thermal conditions)
            tack_duration = 1.0 + (0.5 if angle in self.heat_sensitivity_angles else 0)
            rospy.sleep(tack_duration)
            self.update_thermal_map(angle, tack_duration)
        
        rospy.loginfo("Tack welding completed with thermal management")

    def execute_welding_pass(self, pass_name, offset, pass_num):
        """Execute a complete welding pass with thermal management"""
        rospy.loginfo(f"Starting {pass_name} with thermal optimization...")
        self.current_pass = pass_num
        
        # Generate adaptive path
        waypoints, speed_profile = self.create_adaptive_path(
            self.pipe_radius,
            self.pipe_position,
            offset
        )
        
        # Execute trajectory
        success = self.execute_adaptive_trajectory(waypoints, speed_profile)
        
        if success:
            rospy.loginfo(f"{pass_name} completed successfully")
            
            # Cooling period if not last pass
            if pass_name != "cap pass":
                rospy.loginfo(f"Cooling for {self.cooling_time} seconds...")
                rospy.sleep(self.cooling_time)
                
                # Simulate cooling in thermal model
                self.thermal_map = self.thermal_map * 0.7  # 30% cooling
        else:
            rospy.logwarn(f"{pass_name} encountered planning issues")

    def execute_welding_sequence(self):
        """Execute the complete welding sequence with thermal management"""
        rospy.loginfo("Starting thermal-optimized welding sequence...")
        
        # 1. Tack welding (pass 0)
        self.execute_tack_welding()
        rospy.sleep(self.cooling_time/2)  # Shorter cooling after tacks
        
        # 2. Root pass (pass 1)
        self.execute_welding_pass("root pass", self.weld_offset, 1)
        
        # 3. Hot pass (pass 2)
        self.execute_welding_pass("hot pass", self.weld_offset + self.interpass_offset, 2)
        
        # 4. Fill pass (pass 3)
        self.execute_welding_pass("fill pass", self.weld_offset + 2*self.interpass_offset, 3)
        
        # 5. Cap pass (pass 4)
        self.execute_welding_pass("cap pass", self.weld_offset + 3*self.interpass_offset, 4)
        
        rospy.loginfo("Welding sequence completed with optimal thermal management")

if __name__ == '__main__':
    try:
        ThermalWeldingPathPlanner()
    except rospy.ROSInterruptException:
        pass