import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def iPUMA(nx, ny, nz, ox, oy, oz, ax, ay, az, Px, Py, Pz):
    # Input: end-effector position
    # Output: joint angles at each frame

    # DH parameters
    A = np.radians([-90, 0, 90, -90, 90])  # twist angles in radians
    r = [0, 0.2, -0.0475, 0, 0]           # offset as to x(n)
    d = [0.45, 0.205, 0, 0.15, 0.105]  # offset as to z(n-1)
    
    # End-effector transformation matrix
    T0_6 = np.array([[nx, ox, ax, Px], [ny, oy, ay, Py], [nz, oz, az, Pz], [0, 0, 0, 1]])
    
    # Joint 5 position
    P = np.array([Px - 56.50 * ax, Py - 56.50 * ay, Pz - 56.50 * az])
    
    # Determining joint angles for frames 1, 2, and 3
    C1 = np.sqrt(P[0]**2 + P[1]**2)
    C2 = P[2] - d[0]
    C3 = np.sqrt(C1**2 + C2**2)
    C4 = np.sqrt(r[2]**2 + d[3]**2)
    D1 = d[1] / C1
    D2 = (C3**2 + r[1]**2 - C4**2) / (2 * r[1] * C3)
    D3 = (r[1]**2 + C4**2 - C3**2) / (2 * r[1] * C4)
    
    a1 = np.degrees(np.arctan2(D1, np.sqrt(abs(1 - D1**2))))
    a2 = np.degrees(np.arctan2(np.sqrt(abs(1 - D2**2)), D2))
    b = np.degrees(np.arctan2(np.sqrt(abs(1 - D3**2)), D3))
    p1 = np.degrees(np.arctan2(P[1], P[0]))
    p2 = np.degrees(np.arctan2(C2, C1))
    
    # Joint angles: theta_1, theta_2, and theta_3
    J = [p1 - a1, round(a2 - p2), round(b - 90)]
    
    # Forward kinematics for the first three joints
    T = []
    for n in range(3):
        theta = np.radians(J[n])
        matT = np.array([
            [np.cos(theta), -np.sin(theta) * np.cos(A[n]), np.sin(theta) * np.sin(A[n]), r[n] * np.cos(theta)],
            [np.sin(theta), np.cos(theta) * np.cos(A[n]), -np.cos(theta) * np.sin(A[n]), r[n] * np.sin(theta)],
            [0, np.sin(A[n]), np.cos(A[n]), d[n]],
            [0, 0, 0, 1]
        ])
        T.append(matT)
    
    T0_3 = T[0] @ T[1] @ T[2]
    T3_6 = np.linalg.inv(T0_3) @ T0_6
    
    # Joint angles: theta_4, theta_5, and theta_6
    J4 = round(np.degrees(np.arctan2(T3_6[1, 2], T3_6[0, 2])))
    J5 = round(np.degrees(np.arctan2(np.sqrt(abs(1 - T3_6[2, 2]**2)), T3_6[2, 2])))
    #J6 = np.degrees(np.arctan2(T3_6[2, 1], -T3_6[2, 0]))
    
    J.extend([J4, J5])
    
    # Joint angle limits for the PUMA 560
    joint_limits = [
        (0, 360), (0, 180), (0, 225),
        (0, 360), (-90, 90)
    ]
    
    # Check if the joint angles are within the limits
    if all(joint_limits[i][0] <= J[i] <= joint_limits[i][1] for i in range(5)):
        # Plotting the robot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title("Inverse Kinematics of PUMA 560 Manipulator (6DoF)")
        
        P_list = [np.eye(4)]
        for i in range(6):
            theta = np.radians(J[i])
            matT = np.array([
                [np.cos(theta), -np.sin(theta) * np.cos(A[i]), np.sin(theta) * np.sin(A[i]), r[i] * np.cos(theta)],
                [np.sin(theta), np.cos(theta) * np.cos(A[i]), -np.cos(theta) * np.sin(A[i]), r[i] * np.sin(theta)],
                [0, np.sin(A[i]), np.cos(A[i]), d[i]],
                [0, 0, 0, 1]
            ])
            P_list.append(P_list[-1] @ matT)
        
        x = [P[0, 3] for P in P_list]
        y = [P[1, 3] for P in P_list]
        z = [P[2, 3] for P in P_list]
        
        ax.plot(x, y, z, marker='o', color='k')
        ax.plot(x, y, z, 'o-')
        plt.show()
        
        print("The joint angles are:")
        for i, angle in enumerate(J, start=1):
            print(f"theta{i} = {angle}")
    else:
        print("The joint angles are:")
        for i, angle in enumerate(J, start=1):
            print(f"theta{i} = {angle}")
        print("Resulting joint angles are out of range.")

n = np.arange(-0.1, 0.1, 0.01)

for a in range(20):
    for b in range(20):
        for c in range(20):
            iPUMA(1,0,0,0,1,0,0,0,1,-0.13+n[a-1],-0.43+n[b-1], 0.59+n[c-1])