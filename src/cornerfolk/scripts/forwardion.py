import numpy as np

def dh_transform(a, alpha, d, theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0, np.sin(alpha), np.cos(alpha), d],
        [0, 0, 0, 1]
    ])

def forward_kinematics(theta1, theta2, theta3, theta4, theta5):
    # Define DH parameters for your manipulator
    dh_params = [
        (0, 0, 0.113, theta1),  # Example: replace with actual a, alpha, d values
        (0, 1.57, 0.197, theta2),  # Replace these rows with your robot's DH parameters
        (0, 0, 0.211, theta3),
        (0, -1.57, 0, theta4),
        (0, 1.57, 0.06, theta5),
    ]
    
    # Compute the overall transformation matrix
    T = np.eye(4)  # Start with the identity matrix
    for a, alpha, d, theta in dh_params:
        T = np.dot(T, dh_transform(a, alpha, d, theta))
    
    # Extract end-effector position
    end_effector_position = T[:3, 3]
    return end_effector_position

# Example usage with joint angles in radians
theta1, theta2, theta3, theta4, theta5 = [0.85, -0.21, -1.87, 0.6, 0.61]
position = forward_kinematics(theta1, theta2, theta3, theta4, theta5)
print("End-Effector Position:", position)
