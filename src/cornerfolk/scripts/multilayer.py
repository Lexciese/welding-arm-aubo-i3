#!/usr/bin/env python

import sys
import math
import rospy
import os
import csv
import moveit_commander
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String
import tf.transformations
import numpy as np
import matplotlib.pyplot as plt  # Add matplotlib import
from mpl_toolkits.mplot3d import Axes3D  # Import 3D toolkit

class WeldPathPlanner:
    def __init__(self):
        rospy.init_node('multilayer_weld_planner', anonymous=False)

        # Parameters (with defaults) - Only read once at startup
        self.h = rospy.get_param('~thickness', 0.016)       # plate thickness (m)
        self.beta_deg = rospy.get_param('~angle', 30.0)    # half-angle (degrees)
        self.g = rospy.get_param('~clearance', 0.002)      # assembly clearance (m)
        self.t = rospy.get_param('~bead_thickness', 0.002) # bead thickness (m)
        self.execute = rospy.get_param('~execute', True)   # whether to run MoveIt - set to True by default
        
        # Position adjustment parameters (to fit robot workspace)
        self.x_offset = rospy.get_param('~x_offset', -1.2)  # X offset from origin (m)
        self.y_offset = rospy.get_param('~y_offset', 0.0)  # Y offset from origin (m)
        self.z_offset = rospy.get_param('~z_offset', 0.3)  # Z offset from origin (m)
        self.scale = rospy.get_param('~scale', 1.0)       # Scale factor (default: multiply by 50)
        
        # Welding approach parameters
        self.approach_distance = rospy.get_param('~approach_distance', 0.1)  # approach distance (m)
        self.retreat_distance = rospy.get_param('~retreat_distance', 0.1)    # retreat distance (m)
        self.approach_steps = rospy.get_param('~approach_steps', 5)         # number of steps for approach
        self.retreat_steps = rospy.get_param('~retreat_steps', 5)           # number of steps for retreat
        
        # Welding tool orientation
        self.tool_tilt = rospy.get_param('~tool_tilt', math.radians(-30))  # Tilt angle in radians (-30 degrees default)
        self.tool_roll = rospy.get_param('~tool_roll', 0.0)                # Roll angle in radians
        
        # Waiting time at each point
        self.point_delay = rospy.get_param('~point_delay', 0.5)  # seconds
        
        # Trajectory interpolation parameters
        self.interpolation_resolution = rospy.get_param('~interpolation_resolution', 0.005)  # meters between points
        self.interpolation_enabled = rospy.get_param('~interpolation_enabled', True)  # enable trajectory interpolation
        
        # Output file path
        self.output_path = rospy.get_param('~output_file', os.path.expanduser('~/weld_trajectory.csv'))

        # Publishers
        self.weld_point_pub = rospy.Publisher('weld_point', PoseStamped, queue_size=10)
        self.marker_pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)
        self.marker_array_pub = rospy.Publisher('visualization_marker_array', MarkerArray, queue_size=10)
        
        # Publisher for status messages
        self.status_pub = rospy.Publisher('weld_planner/status', String, queue_size=10)
        
        # Subscriber for start command
        rospy.Subscriber('weld_planner/command', String, self.command_callback)
        
        # MoveIt configuration
        self.setup_moveit()
        
        # Flag to control execution
        self.start_execution = False
        
        # Print initial configuration
        rospy.loginfo("Multilayer Weld Planner initialized with parameters:")
        rospy.loginfo(f"  Thickness: {self.h} m")
        rospy.loginfo(f"  Angle: {self.beta_deg} degrees")
        rospy.loginfo(f"  Clearance: {self.g} m")
        rospy.loginfo(f"  Bead thickness: {self.t} m")
        rospy.loginfo(f"  Execute moves: {self.execute}")
        rospy.loginfo(f"  Position offsets: X={self.x_offset}, Y={self.y_offset}, Z={self.z_offset}")
        rospy.loginfo(f"  Scale factor: {self.scale}")
        rospy.loginfo(f"  Output file: {self.output_path}")
        rospy.loginfo(f"  Interpolation enabled: {self.interpolation_enabled}")
        rospy.loginfo(f"  Interpolation resolution: {self.interpolation_resolution} m")

        # Compute initial path
        self.weld_points = []
        self.compute_weld_path()
        
        # Generate the complete trajectory including approach and retreat
        self.full_trajectory = []
        self.interpolated_trajectory = []
        self.generate_complete_trajectory()
        
        # Save points and publish visualization immediately
        self.save_trajectory_to_file()
        self.publish_markers()
        
        rospy.loginfo(f"Generated {len(self.weld_points)} weld points and {len(self.interpolated_trajectory)} trajectory points")
        rospy.loginfo("Send 'start' to /weld_planner/command topic to begin execution")
        
    def setup_moveit(self):
        """Initialize and configure MoveIt with more detailed setup"""
        try:
            moveit_commander.roscpp_initialize(sys.argv)
            self.robot = moveit_commander.RobotCommander()
            self.scene = moveit_commander.PlanningSceneInterface()
            self.group_name = "armor"
            self.group = moveit_commander.MoveGroupCommander(self.group_name)
            
            # Configure the move group for better performance
            self.group.set_max_velocity_scaling_factor(0.3)  # Increased velocity for faster movement
            self.group.set_max_acceleration_scaling_factor(0.2)  # Increased acceleration
            self.group.set_goal_position_tolerance(0.01)  # 1cm
            self.group.set_goal_orientation_tolerance(0.1)  # ~5.7 degrees
            
            # Set planning parameters
            self.group.set_planning_time(5.0)  # Allow more planning time
            self.group.set_num_planning_attempts(10)  # Try multiple planning attempts
            
            # Get planning reference frame
            planning_frame = self.group.get_planning_frame()
            rospy.loginfo(f"Planning frame: {planning_frame}")
            
            # Get end effector link
            eef_link = self.group.get_end_effector_link()
            rospy.loginfo(f"End effector link: {eef_link}")
            
            # Get group names
            group_names = self.robot.get_group_names()
            rospy.loginfo(f"Available planning groups: {group_names}")
            
            self.moveit_ready = True
            rospy.loginfo("MoveIt! initialized successfully")
        except Exception as e:
            self.moveit_ready = False
            rospy.logerr(f"Failed to initialize MoveIt!: {e}")
            rospy.logerr("Check if the robot is properly launched and MoveIt is running")

    def command_callback(self, msg):
        """Handle command messages"""
        command = msg.data.strip().lower()
        if command == "start":
            self.start_execution = True
            rospy.loginfo("Starting trajectory execution")
        elif command == "stop":
            self.start_execution = False
            rospy.loginfo("Stopping trajectory execution")

    def compute_weld_path(self):
        # Read/update parameters
        beta = math.radians(self.beta_deg)
        # Number of layers
        K = int(math.floor(self.h / self.t))
        # Helper: layer length
        def L(m):
            return self.g + 2 * m * self.t * math.tan(beta)
        # Precompute bead cross-section length
        S = (math.tan(beta) * self.t + self.g) * self.t
        l = S / self.t

        # Orientation vectors (scaled)
        length = l / 2.0
        dl = (math.sin(math.radians(45 - self.beta_deg/2)) * length,
              math.cos(math.radians(45 - self.beta_deg/2)) * length)
        dv = (-math.cos(math.radians(45 + self.beta_deg/2)) * length,
               math.sin(math.radians(45 + self.beta_deg/2)) * length)

        rospy.loginfo("Computing weld path...")
        points = []
        number = 1
        for i in range(1, K+1):
            y = i * self.t
            xr = math.tan(beta) * y + self.g/2.0
            xl = -xr
            Li = L(i)
            Li_prev = L(i-1) if i > 1 else self.g
            N = int(round((Li + Li_prev) / (L(1) + self.g)))
            k = N - 1

            if N <= 1:
                # Center bead only
                temp_x = 0.0
                temp_y = y
                dx, dy = 0.0, length
                points.append((temp_x, temp_y, dx, dy))
                number += 1

            elif N == 2:
                # One left, one center
                x = xl + l
                temp_x = x + self.t * math.tan(beta)
                temp_y = y - self.t
                points.append((temp_x, temp_y, dl[0], dl[1]))
                number += 1
                temp_x = l/2.0
                temp_y = y - self.t
                points.append((temp_x, temp_y, 0.0, length))
                number += 1

            else:
                # Multiple beads: left side, right side, then center
                lp = int(math.floor(N / 2))
                # Left beads
                for j in range(1, lp+1):
                    x = xl + j * l
                    temp_x = x + self.t * math.tan(beta)
                    temp_y = y - self.t
                    points.append((temp_x, temp_y, dl[0], dl[1]))
                    number += 1
                # Right beads
                for j in range(lp+1, N):
                    x = xr - (N - k) * l
                    temp_x = x - self.t * math.tan(beta)
                    temp_y = y - self.t
                    points.append((temp_x, temp_y, dv[0], dv[1]))
                    number += 1
                    k -= 1
                # Center bead
                if lp > 0:
                    # midpoint of last left and last right
                    idx_left = -(lp) 
                    idx_right = -1
                    tempx = (points[idx_left][0] + points[idx_right][0]) / 2.0
                    tempy = (points[idx_left][1] + points[idx_right][1]) / 2.0
                    points.append((tempx, tempy, 0.0, length))
                    number += 1

        self.weld_points = points
        rospy.loginfo(f"Computed {len(self.weld_points)} weld points")

    def generate_complete_trajectory(self):
        """Generate a complete trajectory with approach and retreat paths"""
        self.full_trajectory = []
        
        if not self.weld_points:
            rospy.logwarn("No weld points available to generate trajectory")
            return
            
        # Get first and last point to calculate approach and retreat
        first_point = self.weld_points[0]
        last_point = self.weld_points[-1]
        
        # Calculate approach direction (from first weld point direction)
        first_x, first_y, first_dx, first_dy = first_point
        approach_vec = np.array([first_dx, first_dy, 0])
        if np.linalg.norm(approach_vec) < 0.001:
            approach_vec = np.array([0, 0, -1])  # Default approach from above
        else:
            approach_vec = -approach_vec / np.linalg.norm(approach_vec)  # Reverse and normalize
            
        # Calculate retreat direction (from last weld point direction)
        last_x, last_y, last_dx, last_dy = last_point
        retreat_vec = np.array([last_dx, last_dy, 0])
        if np.linalg.norm(retreat_vec) < 0.001:
            retreat_vec = np.array([0, 0, 1])  # Default retreat upwards
        else:
            retreat_vec = retreat_vec / np.linalg.norm(retreat_vec)  # Normalize
            
        # Generate approach path points
        approach_path = []
        for i in range(self.approach_steps):
            # Interpolate from approach start to first point
            t = float(i) / self.approach_steps if self.approach_steps > 0 else 0
            dist = self.approach_distance * (1.0 - t)
            
            # Calculate position
            x = first_x + dist * approach_vec[0]
            y = first_y + dist * approach_vec[1]
            z = dist * approach_vec[2]  # Z offset
            
            # Use same direction as first point
            approach_path.append((x, y, z, first_dx, first_dy, 'approach'))
            
        # Add main weld points with z value
        weld_path = []
        for x, y, dx, dy in self.weld_points:
            weld_path.append((x, y, 0, dx, dy, 'weld'))
            
        # Generate retreat path points
        retreat_path = []
        for i in range(1, self.retreat_steps + 1):
            # Interpolate from last point to retreat end
            t = float(i) / self.retreat_steps if self.retreat_steps > 0 else 1
            dist = self.retreat_distance * t
            
            # Calculate position
            x = last_x + dist * retreat_vec[0]
            y = last_y + dist * retreat_vec[1] 
            z = dist * retreat_vec[2]  # Z offset
            
            # Use same direction as last point
            retreat_path.append((x, y, z, last_dx, last_dy, 'retreat'))
            
        # Combine all paths
        self.full_trajectory = approach_path + weld_path + retreat_path
        
        # Apply interpolation if enabled
        if self.interpolation_enabled:
            self.interpolate_trajectory()
        else:
            self.interpolated_trajectory = self.full_trajectory
        
        rospy.loginfo(f"Generated complete trajectory with {len(self.full_trajectory)} points")
        rospy.loginfo(f" - Approach: {len(approach_path)} points")
        rospy.loginfo(f" - Weld: {len(weld_path)} points")
        rospy.loginfo(f" - Retreat: {len(retreat_path)} points")
        
        if self.interpolation_enabled:
            rospy.loginfo(f"Interpolated to {len(self.interpolated_trajectory)} points")
        
        return self.full_trajectory
    
    def interpolate_trajectory(self):
        """Interpolate the trajectory for smoother motion"""
        self.interpolated_trajectory = []
        
        if not self.full_trajectory:
            return
        
        # Add the first point
        self.interpolated_trajectory.append(self.full_trajectory[0])
        
        # Interpolate between each pair of consecutive points
        for i in range(len(self.full_trajectory) - 1):
            p1 = self.full_trajectory[i]
            p2 = self.full_trajectory[i+1]
            
            # Extract coordinates and other data
            x1, y1, z1, dx1, dy1, type1 = p1
            x2, y2, z2, dx2, dy2, type2 = p2
            
            # Calculate distance between points
            dist = math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
            
            # Calculate number of interpolation steps
            steps = max(1, int(dist / self.interpolation_resolution))
            
            # Skip if already in interpolated trajectory or only 1 step needed
            if steps <= 1:
                self.interpolated_trajectory.append(p2)
                continue
                
            # Generate interpolated points
            for step in range(1, steps):
                t = float(step) / steps
                
                # Linear interpolation of position
                x = x1 + t * (x2 - x1)
                y = y1 + t * (y2 - y1)
                z = z1 + t * (z2 - z1)
                
                # Linear interpolation of direction (might need normalization)
                dx = dx1 + t * (dx2 - dx1)
                dy = dy1 + t * (dy2 - dy1)
                
                # Use the point type of the segment
                point_type = type1
                
                # Add the interpolated point
                self.interpolated_trajectory.append((x, y, z, dx, dy, point_type))
            
            # Add the end point of the segment
            self.interpolated_trajectory.append(p2)
            
        rospy.loginfo(f"Interpolated {len(self.full_trajectory)} points to {len(self.interpolated_trajectory)} points")

    def save_trajectory_to_file(self):
        """Save the weld trajectory points to a CSV file"""
        try:
            # Plot the trajectory before saving
            self.plot_trajectory()
            
            # First save main weld points
            with open(self.output_path, 'w') as f:
                writer = csv.writer(f)
                # Write header
                writer.writerow(['point_num', 'x', 'y', 'z', 'direction_x', 'direction_y'])
                # Write data
                for i, (x, y, dx, dy) in enumerate(self.weld_points):
                    writer.writerow([i+1, x, y, 0.0, dx, dy])
            
            # Then save the full interpolated trajectory
            full_path = os.path.splitext(self.output_path)[0] + '_full.csv'
            with open(full_path, 'w') as f:
                writer = csv.writer(f)
                # Write header
                writer.writerow(['point_num', 'x', 'y', 'z', 'direction_x', 'direction_y', 'type'])
                # Write data
                for i, (x, y, z, dx, dy, point_type) in enumerate(self.interpolated_trajectory):
                    writer.writerow([i+1, x, y, z, dx, dy, point_type])
            
            rospy.loginfo(f"Saved weld points to {self.output_path}")
            rospy.loginfo(f"Saved full trajectory to {full_path}")
        except Exception as e:
            rospy.logerr(f"Failed to write trajectory file: {e}")

    def plot_trajectory(self):
        """Plot the weld trajectory using matplotlib in 3D"""
        try:
            # Create figure with 3D projection
            plt.figure(figsize=(12, 10))
            ax = plt.subplot(111, projection='3d')
            
            # Extract coordinates from weld points
            x_coords = [x for x, y, dx, dy in self.weld_points]
            y_coords = [y for x, y, dx, dy in self.weld_points]
            z_coords = [0 for _ in self.weld_points]  # Currently all points are at z=0
            
            # Extract coordinates from complete trajectory (including approach and retreat)
            if self.full_trajectory:
                full_x = [x for x, y, z, dx, dy, _ in self.full_trajectory]
                full_y = [y for x, y, z, dx, dy, _ in self.full_trajectory]
                full_z = [z for x, y, z, dx, dy, _ in self.full_trajectory]
                
                # Plot the complete path as a line
                ax.plot(full_x, full_y, full_z, 'b-', alpha=0.5, label='Complete Path')
                
                # Different colors for different path segments
                approach_points = [i for i, (_, _, _, _, _, t) in enumerate(self.full_trajectory) if t == 'approach']
                weld_points = [i for i, (_, _, _, _, _, t) in enumerate(self.full_trajectory) if t == 'weld']
                retreat_points = [i for i, (_, _, _, _, _, t) in enumerate(self.full_trajectory) if t == 'retreat']
                
                if approach_points:
                    ax.plot([full_x[i] for i in approach_points], 
                             [full_y[i] for i in approach_points],
                             [full_z[i] for i in approach_points], 'g-', linewidth=2, label='Approach')
                    
                if weld_points:
                    ax.plot([full_x[i] for i in weld_points], 
                             [full_y[i] for i in weld_points],
                             [full_z[i] for i in weld_points], 'r-', linewidth=2, label='Weld Path')
                    
                if retreat_points:
                    ax.plot([full_x[i] for i in retreat_points], 
                             [full_y[i] for i in retreat_points],
                             [full_z[i] for i in retreat_points], 'y-', linewidth=2, label='Retreat')
            else:
                # If no complete trajectory, just plot the main weld path
                ax.plot(x_coords, y_coords, z_coords, 'r-', alpha=0.7, label='Weld Path')
            
            # Plot points
            ax.scatter(x_coords, y_coords, z_coords, c='r', s=50, depthshade=True, label='Weld Points')
            
            # Add direction vectors in 3D
            for i, (x, y, dx, dy) in enumerate(self.weld_points):
                # Scale the direction for visualization
                arrow_scale = 0.01
                # Plot 3D arrow
                ax.quiver(x, y, 0, dx * arrow_scale, dy * arrow_scale, 0, 
                          color='green', arrow_length_ratio=0.3, label='Direction' if i == 0 else "")
                
                # Add point numbers next to the points
                ax.text(x, y, 0, str(i+1), fontsize=8, ha='center', va='center')
            
            # Plot the weld joint geometry
            self.plot_joint_geometry(ax)
            
            # Set labels and title
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title('3D Weld Trajectory Path')
            
            # Set equal aspect ratio
            # This is a bit tricky in 3D, we'll scale the axes
            max_range = max([
                max(x_coords) - min(x_coords),
                max(y_coords) - min(y_coords),
                0.01  # Minimum z-range
            ])
            mid_x = (max(x_coords) + min(x_coords)) * 0.5
            mid_y = (max(y_coords) + min(y_coords) * 0.5)
            mid_z = 0
            
            ax.set_xlim(mid_x - max_range * 0.6, mid_x + max_range * 0.6)
            ax.set_ylim(mid_y - max_range * 0.6, mid_y + max_range * 0.6)
            ax.set_zlim(mid_z - max_range * 0.3, mid_z + max_range * 0.3)
            
            # Adjust view angle for better 3D perspective
            ax.view_init(elev=30, azim=45)
            
            # Add annotations in 3D space
            info_text = (f"Thickness: {self.h} m\n"
                         f"Angle: {self.beta_deg} degrees\n"
                         f"Clearance: {self.g} m\n"
                         f"Bead thickness: {self.t} m\n"
                         f"Number of points: {len(self.weld_points)}")
            
            ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))
            
            # Add a legend
            ax.legend()
            
            # Save figure to file
            plot_filename = os.path.splitext(self.output_path)[0] + '_3d.png'
            plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
            rospy.loginfo(f"Saved 3D trajectory plot to {plot_filename}")
            
            # Show the plot
            plt.ion()  # Interactive mode on
            plt.show(block=False)
            plt.pause(2)  # Longer pause to view the 3D plot
            
        except Exception as e:
            rospy.logerr(f"Failed to plot 3D trajectory: {e}")
    
    def plot_joint_geometry(self, ax):
        """Plot a schematic of the joint geometry in 3D"""
        try:
            # Convert parameters to float to ensure calculations work
            h = float(self.h) 
            beta = math.radians(float(self.beta_deg))
            g = float(self.g)
            
            # Calculate corner points for the V-groove
            # Bottom of the groove
            x_bottom_left = -g/2
            x_bottom_right = g/2
            y_bottom = 0
            
            # Top edges of the plates
            x_top_left = x_bottom_left - h * math.tan(beta)
            x_top_right = x_bottom_right + h * math.tan(beta)
            y_top = h
            
            # Left plate points
            left_plate_x = [x_top_left, x_bottom_left, x_bottom_left - 0.5 * g, x_top_left - 0.5 * g]
            left_plate_y = [y_top, y_bottom, y_bottom, y_top]
            left_plate_z = [0, 0, 0, 0]
            
            # Right plate points
            right_plate_x = [x_bottom_right, x_top_right, x_top_right + 0.5 * g, x_bottom_right + 0.5 * g]
            right_plate_y = [y_bottom, y_top, y_top, y_bottom]
            right_plate_z = [0, 0, 0, 0]
            
            # Extrude to 3D by adding points with z-depth
            z_depth = max([abs(x) for x in left_plate_x + right_plate_x]) * 0.2
            
            # Plot the plates with some z-depth
            for plate_x, plate_y in [(left_plate_x, left_plate_y), (right_plate_x, right_plate_y)]:
                # Front face
                ax.plot(plate_x, plate_y, [0]*4, 'k-', linewidth=1.5, alpha=0.7)
                
                # Back face
                ax.plot(plate_x, plate_y, [z_depth]*4, 'k-', linewidth=1, alpha=0.5)
                
                # Connect front to back
                for i in range(4):
                    ax.plot([plate_x[i], plate_x[i]], 
                             [plate_y[i], plate_y[i]], 
                             [0, z_depth], 'k-', linewidth=1, alpha=0.3)
            
            # Plot the V-groove outline with stronger lines
            groove_x = [x_top_left, x_bottom_left, x_bottom_right, x_top_right]
            groove_y = [y_top, y_bottom, y_bottom, y_top]
            groove_z = [0, 0, 0, 0]
            
            # Front face of the groove
            ax.plot([x_top_left, x_bottom_left], [y_top, y_bottom], [0, 0], 'k-', linewidth=2)
            ax.plot([x_bottom_left, x_bottom_right], [y_bottom, y_bottom], [0, 0], 'k-', linewidth=2)
            ax.plot([x_bottom_right, x_top_right], [y_bottom, y_top], [0, 0], 'k-', linewidth=2)
            
            # Back face of the groove
            ax.plot([x_top_left, x_bottom_left], [y_top, y_bottom], [z_depth, z_depth], 'k-', linewidth=1.5, alpha=0.5)
            ax.plot([x_bottom_left, x_bottom_right], [y_bottom, y_bottom], [z_depth, z_depth], 'k-', linewidth=1.5, alpha=0.5)
            ax.plot([x_bottom_right, x_top_right], [y_bottom, y_top], [z_depth, z_depth], 'k-', linewidth=1.5, alpha=0.5)
            
            # Connect front to back of the groove
            for i in range(4):
                ax.plot([groove_x[i], groove_x[i]], 
                         [groove_y[i], groove_y[i]], 
                         [0, z_depth], 'k-', linewidth=1.5, alpha=0.4)
                
            rospy.loginfo("Added joint geometry visualization to 3D plot")
            
        except Exception as e:
            rospy.logerr(f"Failed to plot joint geometry: {e}")

    def publish_markers(self):
        # Publish an arrow or line strip marker for the weld path
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "weld_path"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.005  # line width
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        # Add each point to the line strip
        for (x,y,dx,dy) in self.weld_points:
            p = Point()
            p.x = x
            p.y = y
            p.z = 0.0
            marker.points.append(p)
        self.marker_pub.publish(marker)
        
        # Also publish point markers for better visualization
        marker_array = MarkerArray()
        for i, (x, y, dx, dy) in enumerate(self.weld_points):
            # Point marker
            point_marker = Marker()
            point_marker.header.frame_id = "base_link"
            point_marker.header.stamp = rospy.Time.now()
            point_marker.ns = "weld_points"
            point_marker.id = i
            point_marker.type = Marker.SPHERE
            point_marker.action = Marker.ADD
            point_marker.pose.position.x = x
            point_marker.pose.position.y = y
            point_marker.pose.position.z = 0.0
            point_marker.scale.x = 0.01
            point_marker.scale.y = 0.01
            point_marker.scale.z = 0.01
            point_marker.color.r = 1.0
            point_marker.color.g = 0.0
            point_marker.color.b = 0.0
            point_marker.color.a = 1.0
            marker_array.markers.append(point_marker)
            
            # Direction arrow marker
            arrow_marker = Marker()
            arrow_marker.header.frame_id = "base_link"
            arrow_marker.header.stamp = rospy.Time.now()
            arrow_marker.ns = "weld_directions"
            arrow_marker.id = i
            arrow_marker.type = Marker.ARROW
            arrow_marker.action = Marker.ADD
            arrow_marker.pose.position.x = x
            arrow_marker.pose.position.y = y
            arrow_marker.pose.position.z = 0.0
            # Calculate orientation from direction
            yaw = math.atan2(dy, dx)
            q = Quaternion()
            q.w = math.cos(yaw/2.0)
            q.x = 0.0
            q.y = 0.0
            q.z = math.sin(yaw/2.0)
            arrow_marker.pose.orientation = q
            arrow_marker.scale.x = 0.02  # arrow length
            arrow_marker.scale.y = 0.005  # arrow width
            arrow_marker.scale.z = 0.005  # arrow height
            arrow_marker.color.r = 0.0
            arrow_marker.color.g = 0.0
            arrow_marker.color.b = 1.0
            arrow_marker.color.a = 1.0
            marker_array.markers.append(arrow_marker)
        
        # Add a text marker to show execution status
        text_marker = Marker()
        text_marker.header.frame_id = "base_link"
        text_marker.header.stamp = rospy.Time.now()
        text_marker.ns = "weld_status"
        text_marker.id = 999
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        text_marker.pose.position.x = self.x_offset
        text_marker.pose.position.y = self.y_offset
        text_marker.pose.position.z = self.z_offset + 0.2  # Above the model
        text_marker.scale.z = 0.05  # Text height
        text_marker.color.r = 1.0
        text_marker.color.g = 1.0
        text_marker.color.b = 1.0
        text_marker.color.a = 1.0
        text_marker.text = f"Weld Points: {len(self.weld_points)}\nExecution: {'Enabled' if self.execute else 'Disabled'}"
        marker_array.markers.append(text_marker)
            
        # Publish the marker array
        self.marker_array_pub.publish(marker_array)
        rospy.loginfo("Published visualization markers")

    def publish_weld_points(self):
        """Execute trajectory with the robot"""
        # Only execute if explicitly commanded
        if self.execute and self.start_execution and self.moveit_ready:
            rospy.loginfo("Executing weld trajectory...")
            self.status_pub.publish(String("Executing trajectory"))
            
            # Move to a safe starting position first
            try:
                rospy.loginfo("Moving to starting position...")
                success = self.move_to_start()
                if not success:
                    rospy.logerr("Failed to move to starting position. Aborting trajectory execution.")
                    self.status_pub.publish(String("Failed to move to starting position"))
                    return
            except Exception as e:
                rospy.logerr(f"Failed to move to starting position: {e}")
                self.status_pub.publish(String(f"Error: {e}"))
                return
            
            # Execute waypoints as a cartesian path if possible
            rospy.loginfo("Planning cartesian path for trajectory...")
            
            # Split trajectory into smaller segments for better planning
            segment_size = 10  # Number of points per segment
            segments = []
            for i in range(0, len(self.interpolated_trajectory), segment_size):
                segment = self.interpolated_trajectory[i:i+segment_size]
                segments.append(segment)
            
            rospy.loginfo(f"Split trajectory into {len(segments)} segments")
            
            # Process each segment
            for segment_idx, segment in enumerate(segments):
                # Create waypoints array for this segment
                waypoints = []
                for i, (x, y, z, dx, dy, point_type) in enumerate(segment):
                    # Apply scaling and offset to position
                    tx = x * self.scale + self.x_offset
                    ty = y * self.scale + self.y_offset
                    tz = self.z_offset + z * self.scale
                    
                    # Create the pose
                    wpose = Pose()
                    wpose.position.x = tx
                    wpose.position.y = ty
                    wpose.position.z = tz
                    
                    # Convert orientation to quaternion
                    yaw = math.atan2(dy, dx)
                    q = tf.transformations.quaternion_from_euler(self.tool_roll, self.tool_tilt, yaw)
                    wpose.orientation.x = q[0]
                    wpose.orientation.y = q[1]
                    wpose.orientation.z = q[2]
                    wpose.orientation.w = q[3]
                    
                    waypoints.append(wpose)
                
                # Plan cartesian path for this segment
                rospy.loginfo(f"Planning segment {segment_idx+1}/{len(segments)} with {len(waypoints)} waypoints")
                
                # First try cartesian path
                try:
                    (plan, fraction) = self.group.compute_cartesian_path(
                        waypoints,      # waypoints to follow
                        0.01,           # eef_step (meters)
                        avoid_collisions=True
                    )
                    
                    if fraction < 0.8:  # If less than 80% of the path is valid
                        rospy.logwarn(f"Only {fraction*100:.1f}% of the cartesian path was valid. Trying point-to-point.")
                        self.execute_point_to_point(segment)
                    else:
                        rospy.loginfo(f"Executing cartesian path (coverage: {fraction*100:.1f}%)")
                        self.group.execute(plan, wait=True)
                        rospy.loginfo(f"Segment {segment_idx+1} complete")
                except Exception as e:
                    rospy.logerr(f"Cartesian path planning failed: {e}. Trying point-to-point.")
                    self.execute_point_to_point(segment)
                
                # Check if we should continue
                if not self.start_execution or rospy.is_shutdown():
                    rospy.loginfo("Execution stopped")
                    break
            
            # Move back to the starting position
            try:
                rospy.loginfo("Moving back to starting position...")
                self.move_to_start()
                rospy.loginfo("Execution complete")
                self.status_pub.publish(String("Execution complete"))
            except Exception as e:
                rospy.logerr(f"Failed to move to starting position: {e}")
        else:
            # Just publish status
            if not self.execute:
                self.status_pub.publish(String("Execution disabled"))
            elif not self.start_execution:
                self.status_pub.publish(String("Waiting for start command"))
            elif not self.moveit_ready:
                self.status_pub.publish(String("MoveIt not initialized"))
    
    def execute_point_to_point(self, segment):
        """Execute trajectory points one by one"""
        rospy.loginfo(f"Executing {len(segment)} points in point-to-point mode")
        
        for i, (x, y, z, dx, dy, point_type) in enumerate(segment):
            try:
                # Apply scaling and offset to position
                tx = x * self.scale + self.x_offset
                ty = y * self.scale + self.y_offset
                tz = self.z_offset + z * self.scale
                
                # Create the pose
                pose = Pose()
                pose.position.x = tx
                pose.position.y = ty
                pose.position.z = tz
                
                # Convert orientation to quaternion
                yaw = math.atan2(dy, dx)
                q = tf.transformations.quaternion_from_euler(self.tool_roll, self.tool_tilt, yaw)
                pose.orientation.x = q[0]
                pose.orientation.y = q[1]
                pose.orientation.z = q[2]
                pose.orientation.w = q[3]
                
                # For debugging
                if i % 5 == 0 or point_type == 'weld':
                    rospy.loginfo(f"Moving to {point_type} point {i+1}: ({tx:.4f}, {ty:.4f}, {tz:.4f})")
                
                # Set target and move
                self.group.set_pose_target(pose)
                success = self.group.go(wait=True)
                self.group.stop()
                self.group.clear_pose_targets()
                
                if success:
                    if i % 5 == 0 or point_type == 'weld':
                        rospy.loginfo(f"Successfully moved to point {i+1}")
                    
                    # Only pause at weld points
                    if point_type == 'weld':
                        rospy.sleep(self.point_delay)
                else:
                    rospy.logwarn(f"Failed to move to point {i+1}")
                    # Try to continue anyway
                
            except Exception as e:
                rospy.logerr(f"Error executing movement to point {i+1}: {e}")
            
            # Check if we should continue
            if not self.start_execution or rospy.is_shutdown():
                return False
        
        return True
    
    def move_to_start(self):
        """Move to a safe starting position"""
        # Set a known joint configuration for the starting position
        try:
            # Try named target first
            try:
                self.group.set_named_target("home")
                success = self.group.go(wait=True)
                if success:
                    self.group.stop()
                    self.group.clear_pose_targets()
                    rospy.loginfo("Successfully moved to 'home' position")
                    return True
            except Exception as e:
                rospy.logwarn(f"Failed to use named target 'home': {e}")
            
            # Fall back to a hardcoded joint position
            rospy.loginfo("Moving to safe starting joint position")
            joint_goal = self.group.get_current_joint_values()
            
            # Set conservative joint values - adjust these for your robot
            # This sets joints closer to center of range
            for i in range(len(joint_goal)):
                joint_goal[i] = 0.0  # Default to all zeros
            
            # Execute the joint movement
            self.group.set_joint_value_target(joint_goal)
            success = self.group.go(wait=True)
            
            if success:
                self.group.stop()
                self.group.clear_pose_targets()
                rospy.loginfo("Successfully moved to joint position")
                return True
            else:
                rospy.logwarn("Failed to move to joint position")
            
            # Last resort - try to move to a safe pose
            rospy.loginfo("Trying to move to safe pose position")
            pose_goal = Pose()
            pose_goal.position.x = 0.4
            pose_goal.position.y = 0.0
            pose_goal.position.z = 0.4
            q = tf.transformations.quaternion_from_euler(0, 0, 0)
            pose_goal.orientation.x = q[0]
            pose_goal.orientation.y = q[1]
            pose_goal.orientation.z = q[2]
            pose_goal.orientation.w = q[3]
            
            self.group.set_pose_target(pose_goal)
            success = self.group.go(wait=True)
            
            if success:
                rospy.loginfo("Successfully moved to safe pose")
            else:
                rospy.logwarn("Failed to move to safe pose")
            
            self.group.stop()
            self.group.clear_pose_targets()
            return success
            
        except Exception as e:
            rospy.logerr(f"Error in move_to_start: {e}")
            return False

    def run(self):
        rate = rospy.Rate(1.0)  # update once per second
        
        rospy.loginfo("Running weld planner. Press Ctrl+C to stop.")
        
        while not rospy.is_shutdown():
            # Focus on trajectory execution and avoid unnecessary visualization
            self.publish_weld_points()
            
            rate.sleep()

if __name__ == '__main__':
    try:
        planner = WeldPathPlanner()
        planner.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Error in weld planner: {e}")
        import traceback
        traceback.print_exc()
