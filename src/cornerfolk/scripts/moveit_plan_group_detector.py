import sys, moveit_commander
moveit_commander.roscpp_initialize(sys.argv)
robot = moveit_commander.RobotCommander()
print("Available planning groups:", robot.get_group_names())
