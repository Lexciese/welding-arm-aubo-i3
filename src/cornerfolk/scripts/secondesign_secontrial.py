import numpy as np
import math

# Link lengths
l1 = 0.4
l2 = 0.5
l3 = 0.2

# Target inputs
tx = float(input("Target coordinate (x): "))
ty = float(input("Target coordinate (y): "))
tz = float(input("Target coordinate (z): ")) - 0.25
telev = math.radians(float(input("Target elevation angle (degrees): ")))
tazim = math.radians(float(input("Target azimuth angle (degrees): ")))

# Wrist position
wx = tx - l3 * math.cos(telev) * math.cos(tazim)
wy = ty - l3 * math.cos(telev) * math.sin(tazim)
wz = tz - l3 * math.sin(telev)

# Forward kinematics function
def forward_kinematics(theta1, theta2, theta3, theta4, theta5):
    # Rotation matrices for each joint
    rot_theta1 = np.array([
        [math.cos(theta1), -math.sin(theta1), 0],
        [math.sin(theta1), math.cos(theta1), 0],
        [0, 0, 1]
    ])
    rot_theta2 = np.array([
        [math.cos(theta2), 0, math.sin(theta2)],
        [0, 1, 0],
        [-math.sin(theta2), 0, math.cos(theta2)]
    ])
    rot_theta3 = np.array([
        [math.cos(theta3), 0, math.sin(theta3)],
        [0, 1, 0],
        [-math.sin(theta3), 0, math.cos(theta3)]
    ])
    rot_theta4 = np.array([
        [math.cos(theta4), -math.sin(theta4), 0],
        [math.sin(theta4), math.cos(theta4), 0],
        [0, 0, 1]
    ])
    rot_theta5 = np.array([
        [math.cos(theta5), 0, math.sin(theta5)],
        [0, 1, 0],
        [-math.sin(theta5), 0, math.cos(theta5)]
    ])
    
    # Overall transformation
    R = rot_theta1 @ rot_theta2 @ rot_theta3 @ rot_theta4 @ rot_theta5
    
    # Calculate end-effector position
    position = np.array([0, 0, l1]) + R @ np.array([0, 0, l2 + l3])
    return position

# Inverse kinematics for theta1, theta2, theta3
def invkin(wx, wy, wz):
    theta1 = math.atan2(wy, wx)
    r = math.sqrt(wx**2 + wz**2)
    theta3 = -math.acos((r**2 - l1**2 - l2**2) / (2 * l1 * l2))
    theta2 = math.atan2(wz, wx) - math.atan2(l2 * math.sin(theta3), l1 + l2 * math.cos(theta3))
    return theta1, theta2, theta3

# Brute force for theta4 and theta5
def brute_force(wx, wy, wz, telev, tazim):
    theta1, theta2, theta3 = invkin(wx, wy, wz)
    target_vector = np.array([
        math.cos(telev) * math.cos(tazim),
        math.cos(telev) * math.sin(tazim),
        math.sin(telev)
    ])
    
    min_error = float('inf')
    best_theta4 = 0
    best_theta5 = 0
    
    for theta5 in np.arange(-math.pi / 2, math.pi / 2, 0.1):
        for theta4 in np.arange(0, 2 * math.pi, 0.1):
            end_effector_pos = forward_kinematics(theta1, theta2, theta3, theta4, theta5)
            error = np.linalg.norm(end_effector_pos - target_vector)
            if error < min_error:
                min_error = error
                best_theta4 = theta4
                best_theta5 = theta5
    
    return best_theta4, best_theta5

# Solve for angles
theta1, theta2, theta3 = invkin(wx, wy, wz)
theta4, theta5 = brute_force(wx, wy, wz, telev, tazim)

# Print results
print(f"Theta1: {theta1} radian")
print(f"Theta2: {theta2} radian")
print(f"Theta3: {theta3} radian")
print(f"Theta4: {theta4} radian")
print(f"Theta5: {theta5} radian")
