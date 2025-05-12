import sys
import rospy
import moveit_commander
import geometry_msgs.msg


# Inisialisasi moveit_commander dan node ROS
moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node('load_pose_example', anonymous=True)

# Inisialisasi Group
arm_group = moveit_commander.MoveGroupCommander("manipulator_i3")

# Set goal state
arm_group.set_pose_target(geometry_msgs.msg.Pose(
    position=geometry_msgs.msg.Point(-0.46014, -0.00389, 0.58712),
    orientation=geometry_msgs.msg.Quaternion(0, -0.20398, 0, 0.97897)
))

#arm_group.set_pose_target(geometry_msgs.msg.Pose(
#    position=geometry_msgs.msg.Point(4.4, 1.0, 0.8),
#    orientation=geometry_msgs.msg.Quaternion(0, -0.2, 0, 0.98)
#))

# Jalankan perencanaan
# plan = arm_group.plan()
plan_success, plan, planning_time, error_code =  arm_group.plan()
# Eksekusi gerakan
arm_group.execute(plan, wait=True)

# Shutdown moveit_commander
moveit_commander.roscpp_shutdown()

