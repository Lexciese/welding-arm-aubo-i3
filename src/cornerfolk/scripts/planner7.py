import sys
import copy
import rospy
import threading
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import shape_msgs.msg  # Import for collision object primitives
from std_msgs.msg import Float32MultiArray, ColorRGBA
from geometry_msgs.msg import Pose, PoseArray, Point
from visualization_msgs.msg import Marker, MarkerArray
import math
import numpy as np
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from matplotlib import pyplot as plt

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
        self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path', moveit_msgs.msg.DisplayTrajectory, queue_size=20)
        
        # Add marker publisher for waypoint visualization
        self.marker_publisher = rospy.Publisher('/waypoint_markers', MarkerArray, queue_size=10)
        self.waypoint_frame_id = "world" # Use your robot's base frame

        # Wait for RVIZ to initialize. This sleep is ONLY to allow Rviz to come up.
        rospy.sleep(5)

        rospy.loginfo('The name of the reference frame for this robot: %s', str(self.group.get_planning_frame()))
        rospy.loginfo('The name of the end-effector link for this group: %s', str(self.group.get_end_effector_link()))
        rospy.loginfo('A list of all the groups in the robot: %s', str(self.robot.get_group_names()))

        self.execute = False
        self.pose_target = geometry_msgs.msg.Pose()
        self.group_variable_values = self.group.get_current_joint_values()
        self.num_joint = len(self.group_variable_values)

        # The resolution of the cartesian path to be interpolated
        self.eef_step = 0.01
        
        # Set velocity scaling factor (default: 0.5 = 50% of maximum speed)
        self.velocity_scaling = 0.5
        self.group.set_max_velocity_scaling_factor(self.velocity_scaling)

        self.update_rate = update_rate
        rospy.logdebug("ros planner update rate (hz): %f", self.update_rate)

        # Motion thread
        self.motion_thread = threading.Thread(target=self.ros_planner)
        self.motion_thread.daemon = True
        self.motion_thread.start()
        
        # Add a timer to demonstrate circular motion (uncomment to auto-trigger)
        # rospy.Timer(rospy.Duration(15), self.publish_demo_circular_motion)

    def set_velocity_scaling(self, scaling_factor):
        """
        Set the velocity scaling factor for robot movements (0.0 to 1.0)
        """
        if 0.0 <= scaling_factor <= 1.0:
            self.velocity_scaling = scaling_factor
            self.group.set_max_velocity_scaling_factor(scaling_factor)
            rospy.loginfo(f"Velocity scaling set to {scaling_factor*100}% of maximum speed")
        else:
            rospy.logwarn("Velocity scaling factor must be between 0.0 and 1.0")
    
    def generate_circular_waypoints(self, center, radius, num_points, start_angle=0, end_angle=2*math.pi, height=0, axis='z', pipe_inclination=0):
        """
        Generate waypoints in a circular pattern with end effector X-axis pointing toward circle center
        
        Parameters:
        - center: [x, y, z] coordinates of circle center
        - radius: circle radius
        - num_points: number of waypoints to generate
        - start_angle, end_angle: define the portion of circle to trace (radians)
        - height: z-height of the circle
        - axis: axis perpendicular to circle plane ('x', 'y', or 'z')
        - pipe_inclination: inclination angle of the pipe in degrees (0=horizontal, 45=6G position)
        
        Returns array of geometry_msgs.msg.Pose
        """
        waypoints = []
        angle_step = (end_angle - start_angle) / num_points
        
        # Convert inclination to radians
        inclination_rad = math.radians(pipe_inclination)
        
        for i in range(num_points + 1):
            angle = start_angle + i * angle_step
            pose = Pose()
            
            # Generate base position and orientation first (assuming horizontal pipe)
            if axis == 'z':
                # Base position on horizontal circle
                base_x = center[0] + radius * math.cos(angle)
                base_y = center[1] + radius * math.sin(angle)
                base_z = center[2] + height
                
                # Calculate vector from center to waypoint (outward radial direction)
                x_axis = [
                    base_x - center[0],
                    base_y - center[1],
                    0
                ]
                
                # Normalize x_axis
                x_norm = math.sqrt(x_axis[0]**2 + x_axis[1]**2 + x_axis[2]**2)
                x_axis = [x/x_norm for x in x_axis]
                
                # Z axis is along the circle axis (global Z) - pointing outward
                z_axis = [0, 0, 1]
                
                # Y axis is the cross product of Z and X (right-hand rule)
                y_axis = [
                    z_axis[1]*x_axis[2] - z_axis[2]*x_axis[1],
                    z_axis[2]*x_axis[0] - z_axis[0]*x_axis[2],
                    z_axis[0]*x_axis[1] - z_axis[1]*x_axis[0]
                ]
                
            elif axis == 'x':
                # Base position on YZ plane circle
                base_x = center[0] + height
                base_y = center[1] + radius * math.cos(angle)
                base_z = center[2] + radius * math.sin(angle)
                
                x_axis = [
                    0,
                    base_y - center[1],
                    base_z - center[2]
                ]
                
                x_norm = math.sqrt(x_axis[0]**2 + x_axis[1]**2 + x_axis[2]**2)
                x_axis = [x/x_norm for x in x_axis]
                
                z_axis = [1, 0, 0]
                
                y_axis = [
                    z_axis[1]*x_axis[2] - z_axis[2]*x_axis[1],
                    z_axis[2]*x_axis[0] - z_axis[0]*x_axis[2],
                    z_axis[0]*x_axis[1] - z_axis[1]*x_axis[0]
                ]
                
            elif axis == 'y':
                # Base position on XZ plane circle
                base_x = center[0] + radius * math.cos(angle)
                base_y = center[1] + height
                base_z = center[2] + radius * math.sin(angle)
                
                x_axis = [
                    base_x - center[0],
                    0,
                    base_z - center[2]
                ]
                
                x_norm = math.sqrt(x_axis[0]**2 + x_axis[1]**2 + x_axis[2]**2)
                x_axis = [x/x_norm for x in x_axis]
                
                z_axis = [0, 1, 0]
                
                y_axis = [
                    z_axis[1]*x_axis[2] - z_axis[2]*x_axis[1],
                    z_axis[2]*x_axis[0] - z_axis[0]*x_axis[2],
                    z_axis[0]*x_axis[1] - z_axis[1]*x_axis[0]
                ]
            
            # Apply pipe inclination if specified
            if pipe_inclination != 0:
                # Apply rotation based on the circle axis
                if axis == 'z':
                    # For Z-axis: rotate around Y-axis (pipe runs along X-axis)
                    cos_inc = math.cos(inclination_rad)
                    sin_inc = math.sin(inclination_rad)
                    
                    # Rotate the position relative to center
                    rel_pos = [base_x - center[0], base_y - center[1], base_z - center[2]]
                    
                    # Apply Y-axis rotation to position
                    rotated_pos = [
                        rel_pos[0] * cos_inc + rel_pos[2] * sin_inc,
                        rel_pos[1],
                        -rel_pos[0] * sin_inc + rel_pos[2] * cos_inc
                    ]
                    
                    # Apply inclination rotation to orientation vectors
                    rotated_x = [
                        x_axis[0] * cos_inc + x_axis[2] * sin_inc,
                        x_axis[1],
                        -x_axis[0] * sin_inc + x_axis[2] * cos_inc
                    ]
                    
                    rotated_y = [
                        y_axis[0] * cos_inc + y_axis[2] * sin_inc,
                        y_axis[1],
                        -y_axis[0] * sin_inc + y_axis[2] * cos_inc
                    ]
                    
                    rotated_z = [
                        z_axis[0] * cos_inc + z_axis[2] * sin_inc,
                        z_axis[1],
                        -z_axis[0] * sin_inc + z_axis[2] * cos_inc
                    ]
                    
                elif axis == 'y':
                    # For Y-axis: rotate around X-axis (pipe runs along Z-axis when inclined)
                    cos_inc = math.cos(inclination_rad)
                    sin_inc = math.sin(inclination_rad)
                    
                    # Rotate the position relative to center
                    rel_pos = [base_x - center[0], base_y - center[1], base_z - center[2]]
                    
                    # Apply X-axis rotation to position
                    rotated_pos = [
                        rel_pos[0],
                        rel_pos[1] * cos_inc - rel_pos[2] * sin_inc,
                        rel_pos[1] * sin_inc + rel_pos[2] * cos_inc
                    ]
                    
                    # Apply inclination rotation to orientation vectors
                    rotated_x = [
                        x_axis[0],
                        x_axis[1] * cos_inc - x_axis[2] * sin_inc,
                        x_axis[1] * sin_inc + x_axis[2] * cos_inc
                    ]
                    
                    rotated_y = [
                        y_axis[0],
                        y_axis[1] * cos_inc - y_axis[2] * sin_inc,
                        y_axis[1] * sin_inc + y_axis[2] * cos_inc
                    ]
                    
                    rotated_z = [
                        z_axis[0],
                        z_axis[1] * cos_inc - z_axis[2] * sin_inc,
                        z_axis[1] * sin_inc + z_axis[2] * cos_inc
                    ]
                    
                elif axis == 'x':
                    # For X-axis: rotate around Z-axis (pipe runs along Y-axis when inclined)
                    cos_inc = math.cos(inclination_rad)
                    sin_inc = math.sin(inclination_rad)
                    
                    # Rotate the position relative to center
                    rel_pos = [base_x - center[0], base_y - center[1], base_z - center[2]]
                    
                    # Apply Z-axis rotation to position
                    rotated_pos = [
                        rel_pos[0] * cos_inc - rel_pos[1] * sin_inc,
                        rel_pos[0] * sin_inc + rel_pos[1] * cos_inc,
                        rel_pos[2]
                    ]
                    
                    # Apply inclination rotation to orientation vectors
                    rotated_x = [
                        x_axis[0] * cos_inc - x_axis[1] * sin_inc,
                        x_axis[0] * sin_inc + x_axis[1] * cos_inc,
                        x_axis[2]
                    ]
                    
                    rotated_y = [
                        y_axis[0] * cos_inc - y_axis[1] * sin_inc,
                        y_axis[0] * sin_inc + y_axis[1] * cos_inc,
                        y_axis[2]
                    ]
                    
                    rotated_z = [
                        z_axis[0] * cos_inc - z_axis[1] * sin_inc,
                        z_axis[0] * sin_inc + z_axis[1] * cos_inc,
                        z_axis[2]
                    ]
                
                # Set final position
                pose.position.x = center[0] + rotated_pos[0]
                pose.position.y = center[1] + rotated_pos[1]
                pose.position.z = center[2] + rotated_pos[2]
                
                # Build rotation matrix with rotated axes
                rotation_matrix = [
                    rotated_x[0], rotated_y[0], rotated_z[0],
                    rotated_x[1], rotated_y[1], rotated_z[1],
                    rotated_x[2], rotated_y[2], rotated_z[2]
                ]
                
            else:
                # No inclination - use original position and orientation
                pose.position.x = base_x
                pose.position.y = base_y
                pose.position.z = base_z
                
                rotation_matrix = [
                    x_axis[0], y_axis[0], z_axis[0],
                    x_axis[1], y_axis[1], z_axis[1],
                    x_axis[2], y_axis[2], z_axis[2]
                ]
            
            # Convert to quaternion
            q = self.rotation_matrix_to_quaternion(rotation_matrix)
            
            pose.orientation.x = q[0]
            pose.orientation.y = q[1]
            pose.orientation.z = q[2]
            pose.orientation.w = q[3]
            
            # Log orientation angles for debugging
            rpy = euler_from_quaternion([q[0], q[1], q[2], q[3]])
            rospy.logdebug(f"Waypoint {i} (axis={axis}, inclination {pipe_inclination}°): Roll={math.degrees(rpy[0]):.1f}°, "
                          f"Pitch={math.degrees(rpy[1]):.1f}°, "
                          f"Yaw={math.degrees(rpy[2]):.1f}°")
            
            waypoints.append(pose)
            
        return waypoints

    def rotation_matrix_to_quaternion(self, R):
        """
        Convert a 3x3 rotation matrix to a quaternion
        Expects R as a flattened list [r11, r12, r13, r21, r22, r23, r31, r32, r33]
        Returns quaternion as [x, y, z, w]
        """
        # Reshape R to access elements easily
        r11, r12, r13 = R[0], R[1], R[2]
        r21, r22, r23 = R[3], R[4], R[5]
        r31, r32, r33 = R[6], R[7], R[8]
        
        # Algorithm from "Quaternion from rotation matrix" in Wikipedia
        trace = r11 + r22 + r33
        
        if trace > 0:
            S = math.sqrt(trace + 1.0) * 2
            qw = 0.25 * S
            qx = (r32 - r23) / S
            qy = (r13 - r31) / S
            qz = (r21 - r12) / S
        elif (r11 > r22) and (r11 > r33):
            S = math.sqrt(1.0 + r11 - r22 - r33) * 2
            qw = (r32 - r23) / S
            qx = 0.25 * S
            qy = (r12 + r21) / S
            qz = (r13 + r31) / S
        elif r22 > r33:
            S = math.sqrt(1.0 + r22 - r11 - r33) * 2
            qw = (r13 - r31) / S
            qx = (r12 + r21) / S
            qy = 0.25 * S
            qz = (r23 + r32) / S
        else:
            S = math.sqrt(1.0 + r33 - r11 - r22) * 2
            qw = (r21 - r12) / S
            qx = (r13 + r31) / S
            qy = (r23 + r32) / S
            qz = 0.25 * S
            
        # Return as [x, y, z, w]
        return [qx, qy, qz, qw]
    
    def visualize_waypoints(self, waypoints, clear_previous=True):
        """
        Visualize waypoints in RViz using markers
        """
        marker_array = MarkerArray()
        
        # Clear previous markers if requested
        if clear_previous:
            clear_marker = Marker()
            clear_marker.action = Marker.DELETEALL
            marker_array.markers.append(clear_marker)
            self.marker_publisher.publish(marker_array)
            marker_array = MarkerArray()
        
        # Create a marker for each waypoint
        for i, waypoint in enumerate(waypoints):
            # Sphere marker for position
            sphere_marker = Marker()
            sphere_marker.header.frame_id = self.waypoint_frame_id
            sphere_marker.header.stamp = rospy.Time.now()
            sphere_marker.ns = "waypoint_positions"
            sphere_marker.id = i
            sphere_marker.type = Marker.SPHERE
            sphere_marker.action = Marker.ADD
            
            # Set position from waypoint
            sphere_marker.pose.position = waypoint.position
            sphere_marker.pose.orientation.w = 1.0
            
            # Set marker scale and color
            sphere_marker.scale.x = 0.02
            sphere_marker.scale.y = 0.02
            sphere_marker.scale.z = 0.02
            
            # Green color
            sphere_marker.color.r = 0.2
            sphere_marker.color.g = 0.8
            sphere_marker.color.b = 0.2
            sphere_marker.color.a = 0.7
            
            marker_array.markers.append(sphere_marker)
            
            # Visualize the X, Y, Z axes for better orientation understanding
            axis_length = 0.05  # Length of each axis arrow
            axis_width = 0.005   # Width of each axis arrow
            
            # X axis (red) - points forward in robot tool frame
            x_marker = Marker()
            x_marker.header.frame_id = self.waypoint_frame_id
            x_marker.header.stamp = rospy.Time.now()
            x_marker.ns = "waypoint_x_axis"
            x_marker.id = i
            x_marker.type = Marker.ARROW
            x_marker.action = Marker.ADD
            x_marker.pose = waypoint
            x_marker.scale.x = axis_length
            x_marker.scale.y = axis_width
            x_marker.scale.z = axis_width
            x_marker.color.r = 1.0
            x_marker.color.g = 0.0
            x_marker.color.b = 0.0
            x_marker.color.a = 0.8
            marker_array.markers.append(x_marker)
            
            # Y axis (green) - points left in robot tool frame
            y_marker = Marker()
            y_marker.header.frame_id = self.waypoint_frame_id
            y_marker.header.stamp = rospy.Time.now()
            y_marker.ns = "waypoint_y_axis"
            y_marker.id = i
            y_marker.type = Marker.ARROW
            y_marker.action = Marker.ADD
            y_marker.pose = waypoint
            
            # For Y-axis, we need to rotate the arrow 90 degrees around Z
            # We'll use the original orientation and apply an additional rotation
            q_orig = [waypoint.orientation.x, waypoint.orientation.y, 
                      waypoint.orientation.z, waypoint.orientation.w]
            q_y = quaternion_from_euler(0, 0, math.pi/2)  # 90 degrees around Z
            
            # Multiply quaternions: q_result = q_orig * q_y
            # This gives us the Y axis orientation
            q_result = [
                q_orig[3]*q_y[0] + q_orig[0]*q_y[3] + q_orig[1]*q_y[2] - q_orig[2]*q_y[1],
                q_orig[3]*q_y[1] - q_orig[0]*q_y[2] + q_orig[1]*q_y[3] + q_orig[2]*q_y[0],
                q_orig[3]*q_y[2] + q_orig[0]*q_y[1] - q_orig[1]*q_y[0] + q_orig[2]*q_y[3],
                q_orig[3]*q_y[3] - q_orig[0]*q_y[0] - q_orig[1]*q_y[1] - q_orig[2]*q_y[2]
            ]
            
            y_marker.pose.orientation.x = q_result[0]
            y_marker.pose.orientation.y = q_result[1]
            y_marker.pose.orientation.z = q_result[2]
            y_marker.pose.orientation.w = q_result[3]
            
            y_marker.scale.x = axis_length
            y_marker.scale.y = axis_width
            y_marker.scale.z = axis_width
            y_marker.color.r = 0.0
            y_marker.color.g = 1.0
            y_marker.color.b = 0.0
            y_marker.color.a = 0.8
            marker_array.markers.append(y_marker)
            
            # Z axis (blue) - points up in robot tool frame
            z_marker = Marker()
            z_marker.header.frame_id = self.waypoint_frame_id
            z_marker.header.stamp = rospy.Time.now()
            z_marker.ns = "waypoint_z_axis"
            z_marker.id = i
            z_marker.type = Marker.ARROW
            z_marker.action = Marker.ADD
            z_marker.pose = waypoint
            
            # For Z-axis, we need to rotate 90 degrees around Y
            q_z = quaternion_from_euler(0, -math.pi/2, 0)  # -90 degrees around Y
            
            # Multiply quaternions: q_result = q_orig * q_z
            q_result = [
                q_orig[3]*q_z[0] + q_orig[0]*q_z[3] + q_orig[1]*q_z[2] - q_orig[2]*q_z[1],
                q_orig[3]*q_z[1] - q_orig[0]*q_z[2] + q_orig[1]*q_z[3] + q_orig[2]*q_z[0],
                q_orig[3]*q_z[2] + q_orig[0]*q_z[1] - q_orig[1]*q_z[0] + q_orig[2]*q_z[3],
                q_orig[3]*q_z[3] - q_orig[0]*q_z[0] - q_orig[1]*q_z[1] - q_orig[2]*q_z[2]
            ]
            
            z_marker.pose.orientation.x = q_result[0]
            z_marker.pose.orientation.y = q_result[1]
            z_marker.pose.orientation.z = q_result[2]
            z_marker.pose.orientation.w = q_result[3]
            
            z_marker.scale.x = axis_length
            z_marker.scale.y = axis_width
            z_marker.scale.z = axis_width
            z_marker.color.r = 0.0
            z_marker.color.g = 0.0
            z_marker.color.b = 1.0
            z_marker.color.a = 0.8
            marker_array.markers.append(z_marker)
            
            # Text marker for waypoint number
            text_marker = Marker()
            text_marker.header.frame_id = self.waypoint_frame_id
            text_marker.header.stamp = rospy.Time.now()
            text_marker.ns = "waypoint_numbers"
            text_marker.id = i
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            
            # Position slightly above the waypoint
            text_marker.pose.position.x = waypoint.position.x
            text_marker.pose.position.y = waypoint.position.y
            text_marker.pose.position.z = waypoint.position.z + 0.05
            text_marker.pose.orientation.w = 1.0
            
            # Set text and scale
            text_marker.text = str(i+1)
            text_marker.scale.z = 0.03  # text height
            
            # White color
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 0.9
            
            marker_array.markers.append(text_marker)
        
        # Add line strip connecting all waypoints
        line_marker = Marker()
        line_marker.header.frame_id = self.waypoint_frame_id
        line_marker.header.stamp = rospy.Time.now()
        line_marker.ns = "waypoint_path"
        line_marker.id = 0
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        
        # Add points for each waypoint
        for waypoint in waypoints:
            point = Point()
            point.x = waypoint.position.x
            point.y = waypoint.position.y
            point.z = waypoint.position.z
            line_marker.points.append(point)
        
        # Close the loop if it's a full circle
        if len(waypoints) > 2 and all(abs(waypoints[0].position.__getattribute__(attr) - 
                                         waypoints[-1].position.__getattribute__(attr)) < 0.01 
                                     for attr in ['x', 'y', 'z']):
            point = Point()
            point.x = waypoints[0].position.x
            point.y = waypoints[0].position.y
            point.z = waypoints[0].position.z
            line_marker.points.append(point)
        
        # Set line properties
        line_marker.scale.x = 0.005  # line width
        line_marker.color.r = 0.8
        line_marker.color.g = 0.4
        line_marker.color.b = 0.0
        line_marker.color.a = 0.6
        
        marker_array.markers.append(line_marker)
        
        # Publish the marker array
        self.marker_publisher.publish(marker_array)
        rospy.loginfo(f"Published {len(waypoints)} waypoint markers with full coordinate frames to RViz")

    def get_welding_position_inclination(self, position_name):
        """
        Get the pipe inclination angle for standard welding positions
        
        Parameters:
        - position_name: '1G', '2G', '5G', '6G' (standard AWS welding positions)
        
        Returns:
        - inclination angle in degrees
        """
        welding_positions = {
            '1G': 0,     # Flat position (horizontal pipe, horizontal weld)
            '2G': 0,     # Horizontal position (vertical pipe, horizontal weld)
            '5G': 0,     # Horizontal fixed position (horizontal pipe, vertical weld)
            '6G': 45,    # Inclined position (45° inclined pipe)
            '6GR': 45    # Restricted 6G position
        }
        
        return welding_positions.get(position_name.upper(), 0)

    def execute_circular_motion(self, center, radius, num_points=20, axis='x', velocity_scale=None, max_retries=3, position_tolerance=0.01, orientation_tolerance=0.1, pipe_inclination=0):
        """
        Execute motion in a circular pattern with end effector perpendicular to circle
        Default axis is now 'x' instead of 'z'
        
        Parameters:
        - center: [x, y, z] coordinates of circle center
        - radius: circle radius
        - num_points: number of waypoints to generate
        - axis: axis perpendicular to circle plane ('x', 'y', or 'z')
        - velocity_scale: robot movement speed (0.0 to 1.0)
        - max_retries: maximum number of attempts for each waypoint
        - position_tolerance: tolerance for position error (meters)
        - orientation_tolerance: tolerance for orientation error (radians)
        - pipe_inclination: inclination angle of the pipe in degrees (0=horizontal, 45=6G position)
        """
        # Set velocity scaling if provided
        if velocity_scale is not None:
            self.set_velocity_scaling(velocity_scale)
            
        # Generate waypoints with pipe inclination
        waypoints = self.generate_circular_waypoints(
            center=center,
            radius=radius,
            num_points=num_points,
            axis=axis,
            pipe_inclination=pipe_inclination
        )
        
        # Log welding position information
        if pipe_inclination == 0:
            rospy.loginfo("Executing circular motion in flat/horizontal position")
        elif pipe_inclination == 45:
            rospy.loginfo("Executing circular motion in 6G welding position (45° inclined)")
        else:
            rospy.loginfo(f"Executing circular motion with {pipe_inclination}° pipe inclination")
        
        # Visualize waypoints in RViz
        self.visualize_waypoints(waypoints)
        
        # Give time to view the waypoints in RViz before execution
        rospy.loginfo("Waypoints have been visualized in RViz. Starting motion in 3 seconds...")
        rospy.sleep(3)

        position_errors = []
        joint_angle_history = [[] for _ in range(self.num_joint)]
        time_stamps = []

        # Move through each waypoint using IK
        for i, waypoint in enumerate(waypoints):
            retry_count = 0
            success = False
            
            while not success and retry_count < max_retries:
                if retry_count > 0:
                    rospy.logwarn(f"Retrying waypoint {i+1}/{len(waypoints)}, attempt {retry_count+1}/{max_retries}")
                else:
                    rospy.loginfo(f"Moving to waypoint {i+1}/{len(waypoints)}")
                
                # Set the pose target
                self.group.set_pose_target(waypoint)
                
                # Plan and execute
                plan_success, plan, planning_time, error_code = self.group.plan()
                
                if not plan_success:
                    rospy.logwarn(f"Failed to plan path to waypoint {i+1}, error code: {error_code}")
                    retry_count += 1
                    continue
                
                # Print out joint values from inverse kinematics solution
                joint_values = self.group.get_current_joint_values()
                rospy.loginfo("Joint values for waypoint %d/%d (in degrees):", i+1, len(waypoints))
                for j, value in enumerate(joint_values):
                    # Convert radians to degrees
                    degrees = math.degrees(value)
                    rospy.loginfo("  Joint %d: %.2f degrees", j+1, degrees)
                
                # Execute the plan
                execution_success = self.group.execute(plan, wait=True)
                
                if not execution_success:
                    rospy.logwarn(f"Failed to execute plan to waypoint {i+1}")
                    retry_count += 1
                    continue

                if plan.joint_trajectory.points:
                    joint_vals = plan.joint_trajectory.points[-1].positions
                    for j in range(self.num_joint):
                        joint_angle_history[j].append(math.degrees(joint_vals[j]))
                    time_stamps.append(i)
                
                # Verify the robot has reached the target position
                reached_position, pos_error = self.verify_target_reached(waypoint, position_tolerance, orientation_tolerance, return_error=True)
                position_errors.append(pos_error)

                if reached_position:
                    rospy.loginfo(f"Successfully reached waypoint {i+1}/{len(waypoints)}")
                    success = True
                else:
                    rospy.logwarn(f"Robot did not reach waypoint {i+1} within tolerance")
                    retry_count += 1
            
            # Clear targets for next iteration
            self.group.clear_pose_targets()
            
            # If we couldn't reach this waypoint after max retries, ask user what to do
            if not success:
                rospy.logwarn(f"Failed to reach waypoint {i+1} after {max_retries} attempts")
                user_input = input("Continue to next waypoint? (y/n): ").strip().lower()
                if user_input != 'y':
                    rospy.logwarn("Circular motion aborted by user")
                    return False

        plt.figure()
        plt.plot(time_stamps, joint_angle_history[0], label="Joint1_JointValue (degree)")
        plt.plot(time_stamps, joint_angle_history[1], label="Joint2_JointValue (degree)")
        plt.plot(time_stamps, joint_angle_history[2], label="Joint3_JointValue (degree)")
        plt.title("Joint 1, 2, 3 angles produced in IK")
        plt.xlabel("Time (s)")
        plt.ylabel("Value")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Plot joint 4, 5, 6
        plt.figure()
        plt.plot(time_stamps, joint_angle_history[3], label="Joint4_JointValue (degree)")
        plt.plot(time_stamps, joint_angle_history[4], label="Joint5_JointValue (degree)")
        plt.plot(time_stamps, joint_angle_history[5], label="Joint6_JointValue (degree)")
        plt.title("Joint 4, 5, 6 angles produced in IK")
        plt.xlabel("Time (s)")
        plt.ylabel("Value")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Plot error
        plt.figure()
        plt.plot(range(1, len(position_errors)+1), position_errors, marker='o')
        plt.title('Position Error vs Waypoint Index')
        plt.xlabel('Waypoint Index')
        plt.ylabel('Position Error (m)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        rospy.loginfo("Circular motion completed")
        
        # Return to home position (all joints at 0)
        rospy.loginfo("Returning to home position...")
        
        # Create a named joint target for home position
        home_position = [0.0] * len(self.group.get_current_joint_values())
        self.group.set_joint_value_target(home_position)
        
        # Plan and execute move to home
        home_plan_success, home_plan, home_planning_time, home_error_code = self.group.plan()
        
        if not home_plan_success:
            rospy.logwarn(f"Failed to plan path to home position, error code: {home_error_code}")
            return True  # Still return success for the circular motion itself
        
        # Print the joint values for the home position
        rospy.loginfo("Joint values for home position (in degrees):")
        for j, value in enumerate(home_position):
            degrees = math.degrees(value)
            rospy.loginfo("  Joint %d: %.2f degrees", j+1, degrees)
        
        # Execute the plan to home position
        home_success = self.group.execute(home_plan, wait=True)
        
        if home_success:
            rospy.loginfo("Successfully returned to home position")
        else:
            rospy.logwarn("Failed to return to home position")
        
        # Clear targets
        self.group.clear_pose_targets()
        
        return True
    
    def verify_target_reached(self, target_pose, position_tolerance=0.01, orientation_tolerance=0.1, return_error=False):
        """
        Verify that the robot has reached the target pose within tolerance
        
        Parameters:
        - target_pose: The target Pose
        - position_tolerance: Maximum allowed position error (meters)
        - orientation_tolerance: Maximum allowed orientation error (radians)
        
        Returns:
        - True if target is reached within tolerance, False otherwise
        """
        # Wait a short time for the robot to settle
        rospy.sleep(0.5)
        
        # Get current pose
        current_pose = self.group.get_current_pose().pose
        
        # Check position error
        position_error = math.sqrt(
            (current_pose.position.x - target_pose.position.x) ** 2 +
            (current_pose.position.y - target_pose.position.y) ** 2 +
            (current_pose.position.z - target_pose.position.z) ** 2
        )
        
        # Check orientation error (simplified - just checking quaternion dot product)
        # dot product of 1.0 means identical orientation, -1.0 means opposite
        # We use 1.0 - abs(dot) as the error, which is 0 for identical orientations
        current_q = [current_pose.orientation.x, current_pose.orientation.y, 
                    current_pose.orientation.z, current_pose.orientation.w]
        target_q = [target_pose.orientation.x, target_pose.orientation.y, 
                   target_pose.orientation.z, target_pose.orientation.w]
        
        dot_product = sum(c*t for c, t in zip(current_q, target_q))
        orientation_error = 1.0 - abs(dot_product)
        
        # Log the errors
        rospy.loginfo(f"Position error: {position_error:.4f}m, Orientation error: {orientation_error:.4f}")
    
        if return_error:
            return position_error <= position_tolerance and orientation_error <= orientation_tolerance, position_error
        else:
            return position_error <= position_tolerance and orientation_error <= orientation_tolerance
        
    def demo_circular_motion(self, event=None):
        """
        Demonstrate a circular motion with the robot
        Performs welding motions in different positions including 6G
        """
        rospy.loginfo("Starting multi-position welding demo")
        
        # Get current end effector position
        current_pose = self.group.get_current_pose().pose
        

        # ######################################################################
        # TUNING/ GANTI PARAMETER DI SINI
        # ######################################################################

        # Base center position
        base_center = [
            current_pose.position.x + 0.3,  # 30cm in front of current position
            current_pose.position.y + 0.1,
            current_pose.position.z - 0.3
        ]
        radius = 0.05
        axis = 'z'

        # INKLINASI PIPA
        inclination = 45  # 45 degrees for 6G position

        # ######################################################################
        # ######################################################################
        
        for i in range(1):
            
            # Execute circular motion with the specified inclination
            success = self.execute_circular_motion_with_collision_avoidance(
                center=base_center,
                radius=radius,
                num_points=20,
                axis=axis,
                velocity_scale=0.3,  # Move at 30% of maximum speed
                pipe_inclination=inclination
            )
            
            # If motion failed, stop the demo
            if not success:
                rospy.logwarn("Multi-position welding demo aborted")
                return
            
            # Pause between positions
            if i < 2:
                rospy.loginfo(f"Completed. Starting next position in 3 seconds...")
                rospy.sleep(3)
        
        rospy.loginfo("Multi-position welding demo completed successfully")

    def add_tube_collision_object(self, name, radius, height, pose, frame_id="world"):
        """
        Add a tube-shaped collision object to the planning scene
        
        Parameters:
        - name: unique identifier for the collision object
        - radius: radius of the tube
        - height: height of the tube
        - pose: geometry_msgs.msg.Pose object with position and orientation
        - frame_id: reference frame for the object
        """
        # Create a collision object
        collision_object = moveit_msgs.msg.CollisionObject()
        collision_object.header.frame_id = frame_id
        collision_object.id = name
        
        # Define the shape as a cylinder (tube)
        cylinder = shape_msgs.msg.SolidPrimitive()
        cylinder.type = shape_msgs.msg.SolidPrimitive.CYLINDER
        cylinder.dimensions = [height, radius]  # [height, radius] for CYLINDER
        
        # Set the pose
        collision_object.primitives = [cylinder]
        collision_object.primitive_poses = [pose]
        collision_object.operation = collision_object.ADD
        
        # Add the collision object to the planning scene
        self.scene.add_object(collision_object)
        rospy.loginfo(f"Added tube collision object '{name}' to planning scene")
        
        return collision_object
    
    def visualize_tube(self, name, radius, height, pose, color=None, frame_id="world"):
        """
        Visualize a tube in RViz using markers
        
        Parameters:
        - name: unique identifier for the marker
        - radius: radius of the tube
        - height: height of the tube
        - pose: geometry_msgs.msg.Pose object with position and orientation
        - color: (r, g, b, a) tuple for color, defaults to orange if None
        - frame_id: reference frame for the visualization
        """
        if color is None:
            color = (1.0, 0.5, 0.0, 0.6)  # Orange semi-transparent by default
        
        # Create a marker for the tube
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "collision_objects"
        marker.id = hash(name) % 10000  # Convert name to a numeric ID
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        
        # Set the pose
        marker.pose = pose
        
        # Set the scale (dimensions)
        marker.scale.x = radius * 2  # Diameter in x
        marker.scale.y = radius * 2  # Diameter in y
        marker.scale.z = height      # Height in z
        
        # Set the color
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]
        
        # Set marker to last indefinitely
        marker.lifetime = rospy.Duration(0)
        
        # Create a marker array and publish
        marker_array = MarkerArray()
        marker_array.markers.append(marker)
        self.marker_publisher.publish(marker_array)
        
        rospy.loginfo(f"Published visualization marker for tube '{name}'")
    
    def setup_collision_environment(self, center, radius, axis='z', pipe_inclination=0):
        """
        Set up the collision environment with a tube inside the circular motion path
        
        Parameters:
        - center: [x, y, z] coordinates of circle center
        - radius: circle radius of the motion path
        - axis: axis perpendicular to circle plane ('x', 'y', or 'z')
        - pipe_inclination: inclination angle of the pipe in degrees
        """
        # Clear existing collision objects
        self.scene.remove_world_object()
        rospy.sleep(0.5)  # Wait for the scene to update
        
        # Use smaller radius for the tube so the robot moves around it
        tube_radius = radius * 0.6  # 60% of the motion radius
        tube_height = 0.3  # 50cm height
        
        # Create a tube pose
        tube_pose = Pose()
        tube_pose.position.x = center[0]
        tube_pose.position.y = center[1]
        tube_pose.position.z = center[2]
        
        # Set orientation based on the circle axis and pipe inclination
        if axis == 'z':
            if pipe_inclination != 0:
                # Apply inclination rotation around Y-axis
                q = quaternion_from_euler(0, math.radians(pipe_inclination), 0)
            else:
                # Tube is aligned with z-axis by default
                q = [0, 0, 0, 1]
        elif axis == 'x':
            if pipe_inclination != 0:
                # For X-axis: first align with X, then apply Z-axis rotation for inclination
                q1 = quaternion_from_euler(0, math.pi/2, 0)  # Align with X
                q2 = quaternion_from_euler(0, 0, math.radians(pipe_inclination))  # Apply inclination around Z
                # Multiply quaternions
                q = self.multiply_quaternions(q1, q2)
            else:
                # Rotate 90 degrees around y to align with x-axis
                q = quaternion_from_euler(0, math.pi/2, 0)
        elif axis == 'y':
            if pipe_inclination != 0:
                # For Y-axis: first align with Y, then apply X-axis rotation for inclination
                q1 = quaternion_from_euler(math.pi/2, 0, 0)  # Align with Y
                q2 = quaternion_from_euler(math.radians(pipe_inclination), 0, 0)  # Apply inclination around X
                # Multiply quaternions
                q = self.multiply_quaternions(q1, q2)
            else:
                # Rotate 90 degrees around x to align with y-axis
                q = quaternion_from_euler(math.pi/2, 0, 0)
        
        tube_pose.orientation.x = q[0]
        tube_pose.orientation.y = q[1]
        tube_pose.orientation.z = q[2]
        tube_pose.orientation.w = q[3]
        
        # Add the tube collision object
        self.add_tube_collision_object("motion_tube", tube_radius, tube_height, tube_pose)
        
        # Visualize the tube
        self.visualize_tube("motion_tube", tube_radius, tube_height, tube_pose)
        
        # Wait for the planning scene to update
        rospy.sleep(1.0)

    def multiply_quaternions(self, q1, q2):
        """
        Multiply two quaternions
        
        Parameters:
        - q1, q2: quaternions as [x, y, z, w]
        
        Returns:
        - result quaternion as [x, y, z, w]
        """
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        
        return [
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2
        ]

    def execute_circular_motion_with_collision_avoidance(self, center, radius, num_points=20, axis='x', velocity_scale=None, max_retries=3, position_tolerance=0.01, orientation_tolerance=0.1, pipe_inclination=0):
        """
        Execute motion in a circular pattern with collision avoidance
        
        Parameters:
        - center: [x, y, z] coordinates of circle center
        - radius: circle radius
        - num_points: number of waypoints to generate
        - axis: axis perpendicular to circle plane ('x', 'y', or 'z')
        - velocity_scale: robot movement speed (0.0 to 1.0)
        - max_retries: maximum number of attempts for each waypoint
        - position_tolerance: tolerance for position error (meters)
        - orientation_tolerance: tolerance for orientation error (radians)
        - pipe_inclination: inclination angle of the pipe in degrees (0=horizontal, 45=6G position)
        """
        # Set up the collision environment with a tube inside the circle
        self.setup_collision_environment(center, radius, axis=axis, pipe_inclination=pipe_inclination)
        
        # Now execute the circular motion with collision checking enabled
        return self.execute_circular_motion(center, radius, num_points, axis, velocity_scale, max_retries, position_tolerance, orientation_tolerance, pipe_inclination)

    def ros_planner(self):
        rospy.spin()
        # self.moveit_commander.roscpp_shutdown()


if __name__ == '__main__':
    try:
        rospy.loginfo('Starting aubo ros plan!')
        planner = AuboRobotPlannerNode()
        
        # Wait for initialization
        rospy.sleep(2)

        planner.demo_circular_motion()
        
        rospy.spin()
    except rospy.ROSInterruptException:
        pass