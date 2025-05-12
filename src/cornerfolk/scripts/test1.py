#!/usr/bin/env python

import rospy
import math
from geometry_msgs.msg import Pose
from moveit_commander import MoveGroupCommander, RobotCommander, PlanningSceneInterface, roscpp_initialize
from tf.transformations import quaternion_from_euler

def circular_welding_path(center_x, center_y, center_z, radius, steps):
    poses = []
    for i in range(steps):
        angle = 2 * math.pi * (i / float(steps))
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        z = center_z

        # Flat orientation (Z axis up)
        quat = quaternion_from_euler(0, math.pi, angle)  # tool points inward toward the pipe

        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation.x = quat[0]
        pose.orientation.y = quat[1]
        pose.orientation.z = quat[2]
        pose.orientation.w = quat[3]
        poses.append(pose)
    return poses

def main():
    roscpp_initialize([])
    rospy.init_node('circular_welding_path_planner', anonymous=True)

    robot = RobotCommander()
    scene = PlanningSceneInterface()
    group = MoveGroupCommander("armor")

    # Pastikan nama end-effector sesuai
    group.set_end_effector_link("link_7")
    group.set_pose_reference_frame("base_link")
    group.set_planning_time(20)

    # Pipe parameters
    radius = 0.09
    center_x = 0.1
    center_y = 0.0
    center_z = 0.2
    steps = 60

    waypoints = circular_welding_path(center_x, center_y, center_z, radius, steps)

    (plan, fraction) = group.compute_cartesian_path(
        waypoints,
        0.02,  # eef_step
        avoid_collisions=False
    )
    

    rospy.loginfo("Path planning success fraction: {:.2f}".format(fraction))

    if fraction > 0.9:
        rospy.loginfo("Executing circular welding path...")
        group.execute(plan, wait=True)
    else:
        rospy.logwarn("Path planning failed or incomplete.")

    group.stop()
    group.clear_pose_targets()

if __name__ == '__main__':
    main()