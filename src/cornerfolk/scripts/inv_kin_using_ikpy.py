import numpy as np
from ikpy.chain import Chain

# Buat kinematic chain dari URDF file
robot_chain = Chain.from_urdf_file(
    urdf_file="/home/rfassholeh/UnlimitedGundamCreator/Subject1/src/cornerfolk/urdf/test2.urdf",
    base_elements=["world"],  # Nama link awal dalam chain
    last_link_vector=np.array([0.13255, -0.20062, 0.2647]),  # Vektor akhir pada camera_link
    base_element_type="link",  # Gunakan 'link' sebagai tipe
    active_links_mask=[False, False, False, True, False, True, True, False, True, True, False, False, False]  # Aktifkan joint yang relevan
)

# Tentukan posisi target (misalnya, di koordinat [1.0, 0.5, 0.3])
target_position = np.array([-0.13, -0.43, 0.59])

# Tentukan orientasi target (misalnya, rotasi dalam arah X)
#target_orientation = np.array([0, -0.20398, 0, 0.97897])

# Hitung konfigurasi joint yang diperlukan untuk mencapai target
joint_angles = robot_chain.inverse_kinematics(
    target_position=target_position,
    #target_orientation=target_orientation,
    #orientation_mode='X'  # Mengarahkan end effector ke sumbu X
)

# Cetak hasil konfigurasi joint
print("Konfigurasi joint yang diperlukan:")
for i, angle in enumerate(joint_angles):
    if i not in [0,1,2,4,7,10,11,12]:
        print(f"Joint {i-1}: {np.rad2deg(angle):.2f} deg")

#np.rad2deg(angle):.2f
