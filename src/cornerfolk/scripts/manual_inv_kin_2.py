import math
import numpy as np

# Define DH parameters
d = [0.45, 0.205, 0, 0.15, 0.105]
r = [0, 0.2, -0.0475, 0, 0]
alpha = [math.radians(-90), 0, math.radians(90), math.radians(-90), math.radians(90)]

ex = 0.1325
ey = -0.2
ez = 0.2645

# Placeholder for target end-effector position and orientation (example values)
px, py, pz = 0.4, -0.2, 0.8  # Example target position
px -= ex
py -= ey
pz -= ez
r11, r12, r13 = 1, 0, 0  # Part of rotation matrix, representing orientation
r21, r22, r23 = 0, 1, 0
r31, r32, r33 = 0, 0, 1

# Inverse Kinematics Calculation
try:
    # Solve for theta1 (depends on px, py)
    theta1_1 = math.atan2(py, px) - math.atan2(d[2], math.sqrt(abs(px**2 + py**2 - d[2]**2)))
    theta1_2 = math.atan2(py, px) - math.atan2(d[2], -math.sqrt(abs(px**2 + py**2 - d[2]**2)))

    # Solve for theta3 using known DH lengths and px, py, pz (simplified)
    k = (px**2 + py**2 + pz**2 - r[1]**2 - r[2]**2) / (2 * r[1] * r[2])
    theta3_1 = math.atan2(r[2], d[3]) - math.atan2(k, math.sqrt(abs(1 - k**2)))
    theta3_2 = math.atan2(r[2], d[3]) - math.atan2(k, -math.sqrt(abs(1 - k**2)))

    # Solve for theta2 using a combination of positions and solved theta3
    theta2_1 = math.atan2(pz - d[0], math.sqrt(abs(px**2 + py**2))) - math.atan2(r[2] * math.sin(theta3_1), r[1] + r[2] * math.cos(theta3_1))
    theta2_2 = math.atan2(pz - d[0], math.sqrt(abs(px**2 + py**2))) - math.atan2(r[2] * math.sin(theta3_2), r[1] + r[2] * math.cos(theta3_2))

    # Solve for theta4 based on orientation components and previously calculated angles
    theta4_1 = math.atan2(-r13 * math.sin(theta1_1) + r23 * math.cos(theta1_1),
                          r33 - math.sin(theta2_1 + theta3_1))
    theta4_2 = math.atan2(-r13 * math.sin(theta1_2) + r23 * math.cos(theta1_2),
                          r33 - math.sin(theta2_2 + theta3_2))

    # Solve for theta5 (end effector orientation about the z-axis of the final frame)
    theta5_1 = math.atan2(r11 * math.cos(theta1_1) + r21 * math.sin(theta1_1), r31)
    theta5_2 = math.atan2(r11 * math.cos(theta1_2) + r21 * math.sin(theta1_2), r31)

    # Normalize angles to [0, 2π]
    theta1_1 %= 2 * math.pi
    theta1_2 %= 2 * math.pi
    theta3_1 %= 2 * math.pi
    theta3_2 %= 2 * math.pi
    theta2_1 %= 2 * math.pi
    theta2_2 %= 2 * math.pi
    theta4_1 %= 2 * math.pi
    theta4_2 %= 2 * math.pi
    theta5_1 %= 2 * math.pi
    theta5_2 %= 2 * math.pi

    limit = [(0, 360), (0, 180), (0, 225), (0, 360), (-90, 90)]
    angle_1 = [theta1_1, theta2_1, theta3_1, theta4_1, theta5_1]
    angle_2 = [theta1_2, theta2_2, theta3_2, theta4_2, theta5_2]
    print(angle_1)
    print(angle_2)
    for joint in range(3):
        if not (limit[joint][0] <= math.degrees(angle_1[joint]) <= limit[joint][1]):
            check_angle_1 = False
            break
        else:
            check_angle_1 = True
    for joint in range(3):
        if not (limit[joint][0] <= math.degrees(angle_2[joint]) <= limit[joint][1]):
            check_angle_2 = False
            break
        else:
            check_angle_2 = True

    print(str(angle_1) + " ---> " + str(check_angle_1))
    print(str(angle_2) + " ---> " + str(check_angle_2))

except ValueError as e:
    print(f"Error in calculation: {e}")
