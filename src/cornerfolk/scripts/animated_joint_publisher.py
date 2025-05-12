#!/usr/bin/env python
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import numpy as np

# Inisialisasi node
rospy.init_node('animated_joint_publisher', anonymous=True)
pub = rospy.Publisher('/joint_states', JointState, queue_size=10)

# Tentukan joint dan posisi awal/tujuan
joint_names = ["slider_joint", "arm_joint"]  # Sesuaikan dengan URDF kamu
start_pos = [0.0, 0.0]  # Posisi awal netral
target_pos = [2.0, 1.5]  # Posisi tujuan

# Parameter animasi
steps = 100  # Jumlah langkah untuk transisi
pause_duration = 0.5  # Waktu jeda antar langkah dalam detik
rate = rospy.Rate(5)  # Frekuensi update (10 Hz)

# Fungsi interpolasi posisi
def interpolate(start, target, steps):
    """Menghasilkan array nilai interpolasi dari start ke target."""
    return np.linspace(start, target, steps)

# Membuat lintasan interpolasi untuk setiap joint
trajectories = [interpolate(s, t, steps) for s, t in zip(start_pos, target_pos)]

while not rospy.is_shutdown():
    # Loop melalui setiap langkah lintasan
    for i in range(steps):
        # Buat pesan JointState
        msg = JointState()
        msg.header = Header(stamp=rospy.Time.now())
        msg.name = joint_names
        msg.position = [traj[i] for traj in trajectories]

        # Publish posisi joint saat ini
        pub.publish(msg)

        # Tunggu sejenak sebelum lanjut ke langkah berikutnya
        rospy.sleep(pause_duration)

    # Tukar posisi awal dan tujuan untuk pergerakan bolak-balik
    start_pos, target_pos = target_pos, start_pos
    trajectories = [interpolate(s, t, steps) for s, t in zip(start_pos, target_pos)]
