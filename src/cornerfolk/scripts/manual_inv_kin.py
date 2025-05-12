import math
import numpy as np

a2 = 0.2
d3 = 0.1575
a3 = 0
d4 = 0.2
ex = 0.1325
ey = -0.2
ez = 0.2645
eelev = math.asin(2*((1*0)-(0*0)))
eazim = math.atan2(2*((1*0)+(0*0)), 1-(2*((0*0)+(0*0))))
x = float(input("Target coordinate (x): "))
y = float(input("Target coordinate (y): "))
z = float(input("Target coordinate (z): "))
elev =  math.radians(float(input("Target orientation angle from z+ axis: ")))

"""xx = float(input("Target quaternion orientation (x): "))
yy = float(input("Target quaternion orientation (x): "))
zz = float(input("Target quaternion orientation (x): "))
ww = float(input("Target quaternion orientation (x): "))"""

px = x - ex
py = y - ey
pz = z - ez
pelev = elev - eelev

"""r11 = math.cos(gamma)*math.cos(beta)
r12 = (math.cos(gamma)*math.sin(beta)*math.sin(alpha)) - (math.sin(gamma)*math.cos(alpha))
r13 = (math.cos(gamma)*math.sin(beta)*math.cos(alpha)) + (math.sin(gamma)*math.sin(alpha))
r21 = math.sin(gamma)*math.cos(beta)
r22 = (math.sin(gamma)*math.sin(beta)*math.sin(alpha)) + (math.cos(gamma)*math.cos(alpha))
r23 = (math.sin(gamma)*math.sin(beta)*math.cos(alpha)) - (math.cos(gamma)*math.cos(alpha))
r31 = -math.sin(beta)
r32 = math.cos(beta)*math.sin(alpha)
r33 = math.cos(beta)*math.cos(alpha)"""

def trial(px,py,pz,r11,r12,r13,r21,r22,r23,r31,r32,r33):
    check_angle_1 = False
    check_angle_2 = False
    check_angle_3 = False
    check_angle_4 = False

    theta1_1 = math.atan2(py,px) - math.atan2(d3,math.sqrt(abs((px**2)+(py**2)-(d3**2))))
    theta1_2 = math.atan2(py,px) - math.atan2(d3,-math.sqrt(abs((px**2)+(py**2)-(d3**2))))
    theta1_3 = theta1_1
    theta1_4 = theta1_2
    k = ((px**2)+(py**2)+(px**2)-(a2**2)-(a3**2)-(d3**2)-(d4**2))/(2*a2)
    theta3_1 = math.atan2(a3,d4) - math.atan2(k,math.sqrt(abs((a3**2)+(d4**2)-(k**2))))
    theta3_2 = math.atan2(a3,d4) - math.atan2(k,-math.sqrt(abs((a3**2)+(d4**2)-(k**2))))
    theta3_3 = theta3_2
    theta3_4 = theta3_1
    theta2_1 = math.atan2(((-a3-(a2*math.cos(theta3_1)))*pz) - (((math.cos(theta1_1)*px)+(math.sin(theta1_1)*py))*(d4-(a2*math.sin(theta3_1)))), (((a2*math.sin(theta3_1))-d4)*pz) - ((a3+(a2*math.cos(theta3_1)))*((math.cos(theta1_1)*px)+((math.sin(theta1_1)*py))))) - theta3_1
    theta2_2 = math.atan2(((-a3-(a2*math.cos(theta3_2)))*pz) - (((math.cos(theta1_2)*px)+(math.sin(theta1_2)*py))*(d4-(a2*math.sin(theta3_2)))), (((a2*math.sin(theta3_2))-d4)*pz) - ((a3+(a2*math.cos(theta3_2)))*((math.cos(theta1_2)*px)+((math.sin(theta1_2)*py))))) - theta3_2
    theta2_3 = math.atan2(((-a3-(a2*math.cos(theta3_3)))*pz) - (((math.cos(theta1_3)*px)+(math.sin(theta1_3)*py))*(d4-(a2*math.sin(theta3_3)))), (((a2*math.sin(theta3_3))-d4)*pz) - ((a3+(a2*math.cos(theta3_3)))*((math.cos(theta1_3)*px)+((math.sin(theta1_3)*py))))) - theta3_3
    theta2_4 = math.atan2(((-a3-(a2*math.cos(theta3_4)))*pz) - (((math.cos(theta1_4)*px)+(math.sin(theta1_4)*py))*(d4-(a2*math.sin(theta3_4)))), (((a2*math.sin(theta3_4))-d4)*pz) - ((a3+(a2*math.cos(theta3_4)))*((math.cos(theta1_4)*px)+((math.sin(theta1_4)*py))))) - theta3_4
    theta4_1 = math.atan2((-r13*math.sin(theta1_1))+(r23*math.cos(theta1_1)), (-r13*math.cos(theta1_1)*math.cos(theta2_1+theta3_1))-(r23*math.sin(theta1_1)*math.cos(theta2_1+theta3_1))+(r33*math.sin(theta2_1+theta3_1)))
    theta4_2 = math.atan2((-r13*math.sin(theta1_2))+(r23*math.cos(theta1_2)), (-r13*math.cos(theta1_2)*math.cos(theta2_2+theta3_2))-(r23*math.sin(theta1_2)*math.cos(theta2_2+theta3_2))+(r33*math.sin(theta2_2+theta3_2)))
    theta5_1 = math.atan2((-r13*((math.cos(theta1_1)*math.cos(theta2_1+theta3_1)*math.cos(theta4_1))+(math.sin(theta1_1)*math.sin(theta4_1))))-(r23*((math.sin(theta1_1)*math.cos(theta2_1+theta3_1)*math.cos(theta4_1))-(math.cos(theta1_1)*math.sin(theta4_1))))+(r33*(math.sin(theta2_1*theta3_1)*math.cos(theta4_1))), (r13*(-math.cos(theta1_1)*math.sin(theta2_1+theta3_1)))+(r23*(-math.sin(theta1_1)*math.sin(theta2_1+theta3_1)))+(r33*-math.cos(theta2_1+theta3_1)))
    theta5_2 = math.atan2((-r13*((math.cos(theta1_2)*math.cos(theta2_2+theta3_2)*math.cos(theta4_2))+(math.sin(theta1_2)*math.sin(theta4_2))))-(r23*((math.sin(theta1_2)*math.cos(theta2_2+theta3_2)*math.cos(theta4_2))-(math.cos(theta1_2)*math.sin(theta4_2))))+(r33*(math.sin(theta2_2*theta3_2)*math.cos(theta4_2))), (r13*(-math.cos(theta1_2)*math.sin(theta2_2+theta3_2)))+(r23*(-math.sin(theta1_2)*math.sin(theta2_2+theta3_2)))+(r33*-math.cos(theta2_2+theta3_2)))

    limit = [(0, 360), (0, 180), (0, 225), (0, 360), (-90, 90)]
    angle_1 = [theta1_1, theta2_1, theta3_1]
    angle_2 = [theta1_2, theta2_2, theta3_2]
    angle_3 = [theta1_3, theta2_3, theta3_3]
    angle_4 = [theta1_4, theta2_4, theta3_4]
    for joint in range(3):
        angle_1[joint] = angle_1[joint] % (2 * np.pi)
        angle_2[joint] = angle_2[joint] % (2 * np.pi)
        angle_3[joint] = angle_3[joint] % (2 * np.pi)
        angle_4[joint] = angle_4[joint] % (2 * np.pi)
    print(angle_1)
    print(angle_2)
    print(angle_3)
    print(angle_4)
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
    for joint in range(3):
        if not (limit[joint][0] <= math.degrees(angle_3[joint]) <= limit[joint][1]):
            check_angle_3 = False
            break
        else:
            check_angle_3 = True
    for joint in range(3):
        if not (limit[joint][0] <= math.degrees(angle_4[joint]) <= limit[joint][1]):
            check_angle_4 = False
            break
        else:
            check_angle_4 = True

    print(str(angle_1) + " ---> " + str(check_angle_1))
    print(str(angle_2) + " ---> " + str(check_angle_2))
    print(str(angle_3) + " ---> " + str(check_angle_3))
    print(str(angle_4) + " ---> " + str(check_angle_4))

    fx1 = (math.cos(theta1_1)*((a2*math.cos(theta2_1))+(a3*math.cos(theta2_1+theta3_1))-(d4*math.sin(theta2_1+theta3_1)))) - (d3*math.sin(theta1_1))
    fx2 = (math.cos(theta1_2)*((a2*math.cos(theta2_2))+(a3*math.cos(theta2_2+theta3_2))-(d4*math.sin(theta2_2+theta3_2)))) - (d3*math.sin(theta1_2))
    fx3 = (math.cos(theta1_3)*((a2*math.cos(theta2_3))+(a3*math.cos(theta2_3+theta3_3))-(d4*math.sin(theta2_3+theta3_3)))) - (d3*math.sin(theta1_3))
    fx4 = (math.cos(theta1_4)*((a2*math.cos(theta2_4))+(a3*math.cos(theta2_4+theta3_4))-(d4*math.sin(theta2_4+theta3_4)))) - (d3*math.sin(theta1_4))
    fy1 = (math.sin(theta1_1)*((a2*math.cos(theta2_1))+(a3*math.cos(theta2_1+theta3_1))-(d4*math.sin(theta2_1+theta3_1)))) + (d3*math.cos(theta1_1))
    fy2 = (math.sin(theta1_2)*((a2*math.cos(theta2_2))+(a3*math.cos(theta2_2+theta3_2))-(d4*math.sin(theta2_2+theta3_2)))) + (d3*math.cos(theta1_2))
    fy3 = (math.sin(theta1_3)*((a2*math.cos(theta2_3))+(a3*math.cos(theta2_3+theta3_3))-(d4*math.sin(theta2_3+theta3_3)))) + (d3*math.cos(theta1_3))
    fy4 = (math.sin(theta1_4)*((a2*math.cos(theta2_4))+(a3*math.cos(theta2_4+theta3_4))-(d4*math.sin(theta2_4+theta3_4)))) + (d3*math.cos(theta1_4))
    fz1 = (-a3*math.sin(theta2_1+theta3_1)) - (a2*math.sin(theta2_1)) - (d4*math.cos(theta2_1+theta3_1))
    fz2 = (-a3*math.sin(theta2_2+theta3_2)) - (a2*math.sin(theta2_2)) - (d4*math.cos(theta2_2+theta3_2))
    fz3 = (-a3*math.sin(theta2_3+theta3_3)) - (a2*math.sin(theta2_3)) - (d4*math.cos(theta2_3+theta3_3))
    fz4 = (-a3*math.sin(theta2_4+theta3_4)) - (a2*math.sin(theta2_4)) - (d4*math.cos(theta2_4+theta3_4))

    print(fx1, fy1, fz1)
    print(fx2, fy2, fz2)
    print(fx3, fy3, fz3)
    print(fx4, fy4, fz4)

"""n = np.arange(-0.7, 0.7, 0.1)

for a in range(14):
    for b in range(14):
        for c in range(14):
            trial(n[a-1],n[b-1],n[c-1]+1)"""

"""r11 = float(input("r11: "))
r12 = float(input("r12: "))
r13 = float(input("r13: "))
r21 = float(input("r21: "))
r22 = float(input("r22: "))
r23 = float(input("r23: "))
r31 = float(input("r31: "))
r32 = float(input("r32: "))
r33 = float(input("r33: "))"""

"""for pazim in np.radians(np.arange(-180, 181, 1)):"""
pazim = 3.14
r11 = math.cos(pazim)*math.cos(pelev)
r12 = -math.sin(pazim)
r13 = math.cos(pazim)*math.sin(pelev)
r21 = math.sin(pazim)*math.cos(pelev)
r22 = math.cos(pazim)
r23 = math.sin(pazim)*math.sin(pelev)
r31 = -math.sin(pazim)
r32 = 0
r33 = math.cos(pazim)
trial(px,py,pz,r11,r12,r13,r21,r22,r23,r31,r32,r33)


