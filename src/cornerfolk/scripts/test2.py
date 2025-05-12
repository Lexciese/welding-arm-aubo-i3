#!/usr/bin/env python3
import rospy
import math
import moveit_commander
import moveit_msgs.msg
from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped
from tf.transformations import quaternion_from_euler

class WeldingPathPlanner:
    def _init_(self):
        # Initialize MoveIt! commander
        moveit_commander.roscpp_initialize(sys.argv)
        
        # Robot commander and scene
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        
        # Arm move group
        self.arm = moveit_commander.MoveGroupCommander("armor")
        
        # Display trajectory publisher
        self.display_trajectory_publisher = rospy.Publisher(
            '/move_group/display_planned_path',
            moveit_msgs.msg.DisplayTrajectory,
            queue_size=20)
        
        # Planning parameters
        self.arm.set_planning_time(10.0)
        self.arm.set_num_planning_attempts(10)
        self.arm.set_goal_position_tolerance(0.01)  # 1 cm
        self.arm.set_goal_orientation_tolerance(0.1)  # ~5.7 degrees

    def add_pipe_to_scene(self):
        """Add the welding pipe to the planning scene"""
        pipe_pose = PoseStamped()
        pipe_pose.header.frame_id = "base_link"
        pipe_pose.pose.position.x = 0.5
        pipe_pose.pose.position.y = 0.0
        pipe_pose.pose.position.z = 0.2
        pipe_pose.pose.orientation.w = 1.0
        
        # self.scene.add_cylinder("welding_pipe", pipe_pose, height=1.0, radius=0.1016)  # 4-inch pipe
        
        rospy.sleep(1)  # Wait for scene to update
        rospy.loginfo("Added welding pipe to planning scene")

    def plan_circular_welding_path(self):
        """Plan a circular path around the pipe for welding"""
        # Pipe parameters (must match scene)
        pipe_radius = 0.1016  # 4 inches in meters
        pipe_center = [0.5, 0.0, 0.2]  # Center of the pipe
        
        # Welding parameters
        welding_distance = 0.02  # 2 cm from pipe surface
        total_angle = 2 * math.pi  # Full circle
        waypoints = []
        
        # Create waypoints in a circle around the pipe
        num_waypoints = 36  # One every 10 degrees
        for i in range(num_waypoints + 1):
            angle = total_angle * i / num_waypoints
            
            # Position (circle around pipe with offset for welding)
            x = pipe_center[0]
            y = pipe_center[1] + (pipe_radius + welding_distance) * math.cos(angle)
            z = pipe_center[2] + (pipe_radius + welding_distance) * math.sin(angle)
            
            # Orientation (tangent to circle, pointing at pipe)
            # Adjust these values based on your end-effector requirements
            q = quaternion_from_euler(math.pi/2, 0, angle + math.pi/2)
            
            pose = Pose()
            pose.position = Point(x, y, z)
            pose.orientation = Quaternion(*q)
            waypoints.append(pose)
        
        # Start with current pose
        waypoints.insert(0, self.arm.get_current_pose().pose)
        
        # Plan Cartesian path
        (plan, fraction) = self.arm.compute_cartesian_path(
            waypoints,   # List of poses
            0.01,       # Step size (1 cm)
            True)       # Avoid collisions
        
        if fraction == 1.0:
            rospy.loginfo("Full path planned successfully!")
            return plan
        else:
            rospy.logwarn("Only %.2f%% of path could be planned", fraction * 100)
            return None

    def execute_plan(self, plan):
        """Execute the planned trajectory"""
        if plan is not None:
            self.arm.execute(plan, wait=True)
            return True
        return False

    def run(self):
        """Main execution function"""
        rospy.loginfo("=== Starting Welding Path Planning ===")
        
        # Add pipe to planning scene
        self.add_pipe_to_scene()
        
        # Plan welding path
        rospy.loginfo("Planning welding path...")
        welding_plan = self.plan_circular_welding_path()
        
        # Execute plan if successful
        if welding_plan:
            rospy.loginfo("Executing welding path...")
            success = self.execute_plan(welding_plan)
            
            if success:
                rospy.loginfo("Welding completed successfully!")
            else:
                rospy.logerr("Failed to execute welding path")
        else:
            rospy.logerr("Failed to plan complete welding path")

if __name__ == '__main__':
    try:
        rospy.init_node('welding_path_planner', anonymous=True)
        planner = WeldingPathPlanner()
        planner.run()
    except rospy.ROSInterruptException:
        pass