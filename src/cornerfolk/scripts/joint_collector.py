import rospy
from moveit_msgs.msg import DisplayTrajectory, RobotTrajectory

def planned_path_callback(data):
    # Mengakses joint trajectory dari hasil perencanaan path
    trajectory = data.trajectory[0]  # Ambil trajectory pertama
    joint_trajectory = trajectory.joint_trajectory
    
    # Ekstrak posisi joint untuk setiap titik di path planning
    joint_positions_list = []
    for point in joint_trajectory.points:
        joint_positions = point.positions
        joint_positions_list.append(joint_positions)
        #print("Joint Positions:", joint_positions)

    # Simpan data joint_positions_list ke dalam format yang diinginkan
    #print("Data trajectory lengkap:", joint_positions_list)

    joint_positions_converted = []
    for joint_positions in joint_positions_list:
        # Konversi setiap posisi joint dari radian ke derajat, lalu bulatkan ke tiga angka di belakang koma
        joint_degrees = [round(pos * (180.0 / 3.14159), 3) for pos in joint_positions]
        joint_positions_converted.append(joint_degrees)

    # Tulis data dalam format CSV atau serial string
    for joint_data in joint_positions_converted:
        data_string = ",".join(map(str, joint_data))
        #print("Data untuk Arduino:", data_string)

    print(joint_positions_converted)

def main():
    rospy.init_node('path_extractor', anonymous=True)
    # Dengarkan topik planned path
    rospy.Subscriber("/move_group/display_planned_path", DisplayTrajectory, planned_path_callback)
    rospy.spin()

if __name__ == "__main__":
    main()
