import sys
import copy
import rospy
import threading
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import trajectory_msgs.msg  # Add this import for JointTrajectoryPoint
from std_msgs.msg import Float32MultiArray, ColorRGBA
from geometry_msgs.msg import Pose, PoseArray, Point
from visualization_msgs.msg import Marker, MarkerArray
import math
import numpy as np
from tf.transformations import quaternion_from_euler, euler_from_quaternion

class AuboWeldingPlanner():
    """
    Enhanced Aubo robot planner for multilayer pipe welding with thermal management
    """
    def __init__(self, update_rate=10):
        rospy.init_node('aubo_welding_planner')

        # Class lock
        self.lock = threading.Lock()
        self.plan_flag = 0

        # Instantiate MoveIt interfaces
        self.group = moveit_commander.MoveGroupCommander("manipulator_i3")
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()

        # Setup publishers
        self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path', 
                                                          moveit_msgs.msg.DisplayTrajectory, 
                                                          queue_size=20)
        self.marker_publisher = rospy.Publisher('/welding_markers', MarkerArray, queue_size=10)
        self.thermal_pub = rospy.Publisher('/thermal_markers', Marker, queue_size=10)
        self.speed_pub = rospy.Publisher('/speed_markers', Marker, queue_size=10)

        # Get current end effector position and set pipe position relative to it
        try:
            current_pose = self.group.get_current_pose().pose
            self.pipe_position = [
                current_pose.position.x + 0.2,  # Offset in x direction
                current_pose.position.y,        # Same y as end effector
                current_pose.position.z - 0.2         # Same z as end effector
            ]
            rospy.loginfo(f"Pipe position set at: {self.pipe_position}")
        except Exception as e:
            # Fallback to default position if we can't get current pose
            self.pipe_position = [0.5, 0.0, 0.3]
            rospy.logwarn(f"Could not get current end effector position: {str(e)}")
            rospy.logwarn(f"Using default pipe position: {self.pipe_position}")

        # Welding parameters
        self.pipe_radius = 0.05715  # 4-inch diameter pipe (114.3mm/2)
        self.weld_offset = 0.002  # offset from pipe surface
        self.interpass_offset = 0.0015  # offset between layers
        
        # Thermal management parameters
        self.heat_sensitivity_angles = [45, 135, 225, 315]  # Degrees where heat builds up
        self.sensitivity_radius = 20  # Degrees around sensitive points to slow down
        self.cooling_time = 30  # Seconds between passes
        self.base_speed = 0.005  # m/s
        self.min_speed = 0.003  # m/s in heat-sensitive areas
        self.max_speed = 0.007  # m/s in cooler areas
        self.thermal_map = np.zeros(360)  # 1 degree resolution thermal model
        
        # Initialize velocity scaling
        self.velocity_scaling = 0.5
        self.group.set_max_velocity_scaling_factor(self.velocity_scaling)
        
        # Wait for RVIZ to initialize
        rospy.sleep(5)
        rospy.loginfo("Welding planner initialized")

    def set_velocity_scaling(self, scaling_factor):
        """Set velocity scaling with validation"""
        if 0.0 <= scaling_factor <= 1.0:
            self.velocity_scaling = scaling_factor
            self.group.set_max_velocity_scaling_factor(scaling_factor)
            rospy.loginfo(f"Velocity scaling set to {scaling_factor*100}%")
        else:
            rospy.logwarn("Velocity scaling must be between 0.0 and 1.0")

    def calculate_speed_profile(self, angle_deg):
        """Calculate adaptive speed based on thermal conditions"""
        min_dist = min(abs(angle_deg - a) % 360 for a in self.heat_sensitivity_angles)
        
        if min_dist <= self.sensitivity_radius:
            speed_reduction = 1 - (min_dist / self.sensitivity_radius)
            return max(self.base_speed * (1 - speed_reduction * 0.4), self.min_speed)
        return min(self.base_speed * 1.2, self.max_speed)

    def update_thermal_map(self, angle_deg, duration):
        """Update thermal model with heat input"""
        angle_idx = int(angle_deg) % 360
        self.thermal_map[angle_idx] += duration * 0.1
        for i in range(1, 6):  # Heat diffusion to neighbors
            self.thermal_map[(angle_idx + i) % 360] += duration * 0.05
            self.thermal_map[(angle_idx - i) % 360] += duration * 0.05

    def generate_welding_waypoints(self, offset, num_segments=4):
        """Generate welding path with thermal-optimized segmentation"""
        waypoints = []
        speed_profile = []
        segment_size = 360 // num_segments
        
        # Alternate segment order based on pass number
        segments = [i*segment_size for i in range(num_segments)]
        if self.current_pass % 2 == 1:
            segments = segments[2:] + segments[:2]  # Rotate order for even heat
        
        for start_angle in segments:
            end_angle = start_angle + segment_size
            for angle in range(start_angle, end_angle, 5):  # 5° resolution
                angle_rad = math.radians(angle)
                
                # Calculate position with offset
                pose = Pose()
                pose.position.x = self.pipe_position[0]
                pose.position.y = self.pipe_position[1] + (self.pipe_radius + offset) * math.cos(angle_rad)
                pose.position.z = self.pipe_position[2] + (self.pipe_radius + offset) * math.sin(angle_rad)
                
                # Orientation tangent to pipe surface
                q = quaternion_from_euler(math.pi, angle_rad + math.pi/2, 0)
                pose.orientation.x = q[0]
                pose.orientation.y = q[1]
                pose.orientation.z = q[2]
                pose.orientation.w = q[3]
                
                waypoints.append(pose)
                speed_profile.append(self.calculate_speed_profile(angle))
        
        return waypoints, speed_profile

    def visualize_welding_path(self, waypoints, speed_profile):
        """Visualize welding path with thermal and speed information"""
        # Path visualization
        path_marker = Marker()
        path_marker.header.frame_id = "world"
        path_marker.header.stamp = rospy.Time.now()
        path_marker.ns = "welding_path"
        path_marker.id = 0
        path_marker.type = Marker.LINE_STRIP
        path_marker.action = Marker.ADD
        path_marker.scale.x = 0.005
        
        # Speed visualization
        speed_marker = Marker()
        speed_marker.header.frame_id = "world"
        speed_marker.header.stamp = rospy.Time.now()
        speed_marker.ns = "speed_profile"
        speed_marker.id = 0
        speed_marker.type = Marker.POINTS
        speed_marker.action = Marker.ADD
        speed_marker.scale.x = 0.02
        speed_marker.scale.y = 0.02
        
        # Thermal visualization
        thermal_marker = Marker()
        thermal_marker.header.frame_id = "world"
        thermal_marker.header.stamp = rospy.Time.now()
        thermal_marker.ns = "thermal_map"
        thermal_marker.id = 0
        thermal_marker.type = Marker.LINE_STRIP
        thermal_marker.action = Marker.ADD
        thermal_marker.scale.x = 0.01
        
        # Populate markers
        for i, (pose, speed) in enumerate(zip(waypoints, speed_profile)):
            # Path line
            path_marker.points.append(pose.position)
            
            # Speed points
            speed_marker.points.append(pose.position)
            color = ColorRGBA()
            speed_ratio = (speed - self.min_speed) / (self.max_speed - self.min_speed)
            color.r = 1.0 - speed_ratio
            color.g = speed_ratio
            color.b = 0.0
            color.a = 1.0
            speed_marker.colors.append(color)
        
        # Thermal ring
        for angle in range(0, 360, 5):
            angle_rad = math.radians(angle)
            point = Point()
            point.x = self.pipe_position[0]
            point.y = self.pipe_position[1] + (self.pipe_radius + 0.05) * math.cos(angle_rad)
            point.z = self.pipe_position[2] + (self.pipe_radius + 0.05) * math.sin(angle_rad)
            thermal_marker.points.append(point)
            
            color = ColorRGBA()
            heat = min(self.thermal_map[angle] / 10.0, 1.0)
            color.r = heat
            color.g = 0
            color.b = 1 - heat
            color.a = 0.8
            thermal_marker.colors.append(color)
        
        # Publish markers
        self.marker_publisher.publish(MarkerArray(markers=[path_marker, speed_marker]))
        self.thermal_pub.publish(thermal_marker)

    def compute_ik_trajectory(self, waypoints, speed_profile):
        """Generate a trajectory using inverse kinematics for each waypoint"""
        rospy.loginfo("Computing IK solution for trajectory...")
        joint_trajectory = []
        successful_waypoints = []
        successful_speeds = []
        
        # Get the current joint values as a starting point
        start_state = self.group.get_current_joint_values()
        current_state = copy.deepcopy(start_state)
        
        for i, pose in enumerate(waypoints):
            # Set the target pose and calculate IK
            self.group.set_pose_target(pose)
            plan_success = self.group.plan()
            
            # Check if we got a valid plan (MoveIt returns different values in different versions)
            if isinstance(plan_success, tuple):
                success = plan_success[0]
                plan = plan_success[1]
            else:
                success = plan_success
                plan = self.group.get_plan()
            
            if success:
                # Extract the last joint state from the plan
                if plan.joint_trajectory.points:
                    joint_state = plan.joint_trajectory.points[-1].positions
                    joint_trajectory.append(joint_state)
                    successful_waypoints.append(pose)
                    successful_speeds.append(speed_profile[i])
                    current_state = joint_state
                else:
                    rospy.logwarn(f"Empty trajectory at waypoint {i}")
            else:
                rospy.logwarn(f"Failed to find IK solution for waypoint {i}")
            
            # Clear target for next iteration
            self.group.clear_pose_targets()
        
        rospy.loginfo(f"IK solutions found for {len(joint_trajectory)}/{len(waypoints)} waypoints")
        return joint_trajectory, successful_waypoints, successful_speeds

    def visualize_trajectory(self, joint_trajectory, waypoints):
        """Visualize the planned trajectory in RViz"""
        if not joint_trajectory:
            rospy.logwarn("No trajectory to visualize")
            return
            
        display_trajectory = moveit_msgs.msg.DisplayTrajectory()
        display_trajectory.trajectory_start = self.robot.get_current_state()
        
        # Create a RobotTrajectory message
        robot_traj = moveit_msgs.msg.RobotTrajectory()
        robot_traj.joint_trajectory.header.frame_id = "world"
        robot_traj.joint_trajectory.joint_names = self.group.get_active_joints()
        
        # Add waypoints to trajectory with timing based on distance
        dt = 0.5  # time step between waypoints
        for i, joint_state in enumerate(joint_trajectory):
            point = trajectory_msgs.msg.JointTrajectoryPoint()
            point.positions = joint_state
            point.time_from_start = rospy.Duration(i * dt)
            robot_traj.joint_trajectory.points.append(point)
        
        # Add trajectory to display message and publish
        display_trajectory.trajectory.append(robot_traj)
        self.display_trajectory_publisher.publish(display_trajectory)
        rospy.loginfo("Published trajectory visualization to RViz")

    def execute_welding_pass(self, pass_name, offset, pass_num):
        """Execute a complete welding pass with thermal management using IK"""
        rospy.loginfo(f"Starting {pass_name} with thermal optimization using IK...")
        self.current_pass = pass_num
        
        # Generate adaptive path
        waypoints, speed_profile = self.generate_welding_waypoints(offset)
        self.visualize_welding_path(waypoints, speed_profile)
        
        # Plan with IK instead of cartesian path
        try:
            # Compute trajectory using IK
            joint_trajectory, successful_waypoints, successful_speeds = self.compute_ik_trajectory(waypoints, speed_profile)
            
            if len(joint_trajectory) < 0.9 * len(waypoints):
                rospy.logwarn(f"Only found IK solutions for {len(joint_trajectory)}/{len(waypoints)} waypoints")
                return False
            
            # Visualize the planned trajectory in RViz
            self.visualize_trajectory(joint_trajectory, successful_waypoints)
            
            # Give the user time to see the visualization
            rospy.sleep(2.0)
            
            # Execute each waypoint
            for i, joint_state in enumerate(joint_trajectory):
                # Set joint values target and move
                self.group.set_joint_value_target(joint_state)
                
                # Adjust velocity based on thermal conditions
                if i < len(successful_speeds):
                    speed_factor = successful_speeds[i] / self.base_speed
                    self.set_velocity_scaling(min(max(speed_factor, 0.3), 1.0))
                
                # Execute motion to this waypoint
                success = self.group.go(wait=True)
                if not success:
                    rospy.logwarn(f"Failed to move to waypoint {i}")
                    continue
                
                # Update thermal model if successful
                if i < len(successful_waypoints):
                    pose = successful_waypoints[i]
                    speed = successful_speeds[i]
                    angle = math.degrees(math.atan2(
                        pose.position.z - self.pipe_position[2],
                        pose.position.y - self.pipe_position[1])) % 360
                    self.update_thermal_map(angle, 0.1/speed if speed > 0 else 0)
            
            self.group.clear_pose_targets()
            rospy.loginfo(f"{pass_name} completed")
            return True
            
        except Exception as e:
            rospy.logerr(f"IK planning failed: {str(e)}")
            self.group.clear_pose_targets()
            return False

    def execute_tack_welding(self):
        """Perform thermal-optimized tack welding"""
        rospy.loginfo("Starting tack welding...")
        
        tack_angles = [i*(360//4) for i in range(4)]  # 4 tacks at 90°
        tack_poses = []
        
        for angle in tack_angles:
            angle_rad = math.radians(angle)
            pose = Pose()
            pose.position.x = self.pipe_position[0]
            pose.position.y = self.pipe_position[1] + (self.pipe_radius + self.weld_offset) * math.cos(angle_rad)
            pose.position.z = self.pipe_position[2] + (self.pipe_radius + self.weld_offset) * math.sin(angle_rad)
            
            q = quaternion_from_euler(math.pi, angle_rad + math.pi/2, 0)
            pose.orientation.x = q[0]
            pose.orientation.y = q[1]
            pose.orientation.z = q[2]
            pose.orientation.w = q[3]
            
            tack_poses.append(pose)
        
        # Execute tacks with speed adjustment
        for i, pose in enumerate(tack_poses):
            angle = tack_angles[i]
            speed = self.calculate_speed_profile(angle)
            self.set_velocity_scaling(speed/self.base_speed)
            
            self.group.set_pose_target(pose)
            if self.group.go(wait=True):
                rospy.loginfo(f"Tack {i+1} completed")
                rospy.sleep(1.0)  # Simulate tack duration
                self.update_thermal_map(angle, 1.0)
            else:
                rospy.logwarn(f"Failed to complete tack {i+1}")
        
        self.group.clear_pose_targets()
        rospy.loginfo("Tack welding completed")

    def execute_full_welding_sequence(self):
        """Execute complete multilayer welding sequence"""
        rospy.loginfo("Starting thermal-optimized welding sequence")
        
        # 1. Tack welding
        self.execute_tack_welding()
        rospy.sleep(self.cooling_time/2)
        
        # 2. Root pass
        self.execute_welding_pass("root pass", self.weld_offset, 1)
        rospy.sleep(self.cooling_time)
        
        # 3. Hot pass
        self.execute_welding_pass("hot pass", self.weld_offset + self.interpass_offset, 2)
        rospy.sleep(self.cooling_time)
        
        # 4. Fill pass
        self.execute_welding_pass("fill pass", self.weld_offset + 2*self.interpass_offset, 3)
        rospy.sleep(self.cooling_time)
        
        # 5. Cap pass
        self.execute_welding_pass("cap pass", self.weld_offset + 3*self.interpass_offset, 4)
        
        rospy.loginfo("Welding sequence completed")

if __name__ == '__main__':
    try:
        welder = AuboWeldingPlanner()
        rospy.sleep(2)  # Initialization delay
        
        # Execute full welding sequence
        welder.execute_full_welding_sequence()
        
        rospy.spin()
    except rospy.ROSInterruptException:
        pass