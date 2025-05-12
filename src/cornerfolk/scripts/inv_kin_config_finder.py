import rospy
from moveit_commander import RobotCommander, MoveGroupCommander

# Inisialisasi node ROS
rospy.init_node('moveit_joint_config_extractor', anonymous=True)

# Inisialisasi MoveIt commander
robot = RobotCommander()
group = MoveGroupCommander("plangro")  # Ganti "arm" dengan nama grup joint robot Anda

# Set target posisi (x, y, z)
target_position = [1.9766, 0.0, 1.255]
group.set_position_target(target_position)

# Jalankan perencanaan gerakan
plan = group.plan()

if plan.joint_trajectory.points:
    print("Trajectory found!")

    # Ambil setiap konfigurasi joint pada setiap titik trajectory
    for idx, point in enumerate(plan.joint_trajectory.points):
        print("\nPoint {}:".format(idx + 1))
        for joint_name, position in zip(plan.joint_trajectory.joint_names, point.positions):
            print("{}: {}".format(joint_name, position))
else:
    print("No valid trajectory found.")
