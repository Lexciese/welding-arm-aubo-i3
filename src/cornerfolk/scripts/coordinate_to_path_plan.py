import sys
import rospy
import moveit_commander
from moveit_commander import RobotTrajectory, PlanningSceneInterface
from geometry_msgs.msg import Pose

def move_to_target(target_pose):
    # Initialize moveit_commander and the node
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('moveit_commander_python_interface', anonymous=True)

    # Instantiate the robot, scene, and planning interface
    robot = moveit_commander.RobotCommander()
    scene = PlanningSceneInterface()
    group_name = "armor"  # Replace with your arm group name in MoveIt!
    move_group = moveit_commander.MoveGroupCommander(group_name)
    move_group.set_planning_time(10)  # Sets the planning time to 10 seconds 
    move_group.set_end_effector_link("link_9")  # Use the correct end-effector link
    move_group.set_goal_position_tolerance(0.01)  # 1cm tolerance in position
    move_group.set_goal_orientation_tolerance(0.05)  # 5 degrees tolerance in orientation

    #move_group.set_named_target("Full")

    # Plan and execute the movement
    #success = move_group.go(wait=True)

    # Optional: Reset the pose target after the move
    #move_group.stop()
    #move_group.clear_pose_targets()

    # Set target pose
    move_group.set_pose_target(target_pose)
    
    # Plan and execute the movement
    plan = move_group.go(wait=True)
    move_group.stop()  # Ensure no residual movement
    move_group.clear_pose_targets()

    # Check if the robot reached the target
    current_pose = move_group.get_current_pose().pose
    print("Current Pose:", current_pose)
    
    # Shut down MoveIt! interface
    moveit_commander.roscpp_shutdown()

def main():
    # Define the target pose (for example, x=0.5, y=0.0, z=0.5)
    target_pose = Pose()
    target_pose.position.x = -0.50001
    target_pose.position.y = -0.10001
    target_pose.position.z = 0.60001
    
    # You can also set the orientation of the end effector if necessary
    #target_pose.orientation.x = 0.0
    #target_pose.orientation.y = 0.0
    #target_pose.orientation.z = 0.0
    #target_pose.orientation.w = 1.0
    
    # Move the robot to the target pose
    # Assuming "Full" is a predefined pose in your robot's configuration

    move_to_target(target_pose)

if __name__ == '__main__':
    main()
