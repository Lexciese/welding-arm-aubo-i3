#!/usr/bin/env python3
import rospy
import math
import copy
from geometry_msgs.msg import Pose
from moveit_msgs.msg import DisplayTrajectory
from moveit_commander import MoveGroupCommander, RobotCommander, PlanningSceneInterface

def Circular_path(self, pos_x, pos_y, pos_z, angle_resolution=5, radius=0.1):
    """
    Generate and execute a circular path for the robot end effector.
    
    Args:
        pos_x, pos_y, pos_z: Starting position
        angle_resolution: Angle step size in degrees
        radius: Radius of the circle
    """
    group = self.group
    scene = self.scene
    eef_link = self.eef_link
    robot = self.robot
    display_trajectory_publisher = self.display_trajectory_publisher
    
    waypoints = []
    
    try:
        x_center, y_center = input("Enter center coordinates (x y): ").split() 
        x_center = float(x_center)
        y_center = float(y_center)
        print("X Coordinate is: ", x_center) 
        print("Y Coordinate is: ", y_center) 
        print() 
    except ValueError:
        rospy.logwarn("Invalid input. Using default center position.")
        x_center = pos_x
        y_center = pos_y
    
    # Define trajectory parameters
    angle1 = angle_resolution * math.pi / 180
    angle = 0
    
    # Create a waypoint at the starting position
    wpose = group.get_current_pose().pose
    
    # Main logic for circular trajectory
    for i in range(0, int(360 / angle_resolution)):
        angle += angle1
        wpose.position.x = x_center + radius * math.cos(angle)
        wpose.position.y = y_center + radius * math.sin(angle)
        wpose.position.z = pos_z
        
        waypoints.append(copy.deepcopy(wpose))
    
    # Plan Cartesian path
    (plan, fraction) = group.compute_cartesian_path(
                               waypoints,   # waypoints to follow
                               0.01,        # eef_step
                               True)        # avoid_collisions (was incorrectly set as jump_threshold)

    rospy.loginfo(f"Path planning completed. {fraction*100:.1f}% of the path was planned.")
    
    if fraction < 0.5:
        rospy.logwarn("Less than 50% of the path was successfully planned. Aborting execution.")
        return False
    
    # Visualize trajectory in RViz
    display_trajectory = DisplayTrajectory()
    display_trajectory.trajectory_start = robot.get_current_state()
    display_trajectory.trajectory.append(plan)
    
    # Publish
    display_trajectory_publisher.publish(display_trajectory)
    
    # Wait for visualization before executing
    rospy.sleep(1)
    
    # Execute the planned path
    result = group.execute(plan, wait=True)
    
    if result:
        rospy.loginfo("The robot moved in a circular path successfully")
        return True
    else:
        rospy.logerr("Failed to execute circular path")
        return False


class CircularPathPlanner:
    def __init__(self, group_name="arm"):
        """Initialize the circular path planner"""
        self.robot = RobotCommander()
        self.scene = PlanningSceneInterface()
        self.group = MoveGroupCommander(group_name)
        self.eef_link = self.group.get_end_effector_link()
        
        # Create a display trajectory publisher
        self.display_trajectory_publisher = rospy.Publisher(
            '/move_group/display_planned_path',
            DisplayTrajectory,
            queue_size=20)
        
        self.radius = 0.1
        self.waypoints = []
        
    def plan_circular_path(self, x_pos=0.5, y_pos=0.0, z_pos=0.5, radius=0.1, angle_res=5):
        """Wrapper method to call the circular path function"""
        return Circular_path(self, x_pos, y_pos, z_pos, angle_res, radius)


if __name__ == "__main__":
    rospy.init_node('circular_path_planner_node')
    try:
        planner = CircularPathPlanner("armor")  # Replace with your move group name
        planner.plan_circular_path()
    except rospy.ROSInterruptException:
        pass