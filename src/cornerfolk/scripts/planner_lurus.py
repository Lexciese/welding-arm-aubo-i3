#!/usr/bin/env python

import rospy
import geometry_msgs.msg
import numpy as np
import tf.transformations
from math import pi

def generate_straight_waypoints(start_point, end_point, num_points=10, orientation_mode="fixed"):
    """
    Generate waypoints for a straight line path between two points.
    
    Args:
        start_point (list): Starting point [x, y, z]
        end_point (list): Ending point [x, y, z]
        num_points (int): Number of waypoints to generate (including start and end)
        orientation_mode (str): How to set orientation:
                               "fixed" - use the same orientation for all points
                               "path_direction" - orient along the path direction
    
    Returns:
        list: List of geometry_msgs.msg.Pose waypoints
    """
    waypoints = []
    
    # Convert input points to numpy arrays
    start = np.array(start_point)
    end = np.array(end_point)
    
    # Calculate direction vector
    direction = end - start
    path_length = np.linalg.norm(direction)
    
    if path_length < 0.001:
        rospy.logwarn("Start and end points are too close, returning single point")
        pose = geometry_msgs.msg.Pose()
        pose.position.x = start[0]
        pose.position.y = start[1]
        pose.position.z = start[2]
        
        # Default orientation (pointing forward in x direction)
        q = tf.transformations.quaternion_from_euler(0, 0, 0)
        pose.orientation.x = q[0]
        pose.orientation.y = q[1]
        pose.orientation.z = q[2]
        pose.orientation.w = q[3]
        
        return [pose]
    
    # Normalize direction
    direction = direction / path_length
    
    # Calculate orientation quaternion if using path_direction mode
    if orientation_mode == "path_direction":
        # Find rotation from [1,0,0] to our direction vector
        # First handle special case if direction is parallel to [1,0,0]
        if abs(direction[0] - 1.0) < 0.001 and abs(direction[1]) < 0.001 and abs(direction[2]) < 0.001:
            # Direction is already [1,0,0]
            path_quaternion = tf.transformations.quaternion_from_euler(0, 0, 0)
        elif abs(direction[0] + 1.0) < 0.001 and abs(direction[1]) < 0.001 and abs(direction[2]) < 0.001:
            # Direction is [-1,0,0]
            path_quaternion = tf.transformations.quaternion_from_euler(0, pi, 0)
        else:
            # General case - find rotation from [1,0,0] to direction
            # First find the cross product of [1,0,0] and direction
            cross = np.cross([1, 0, 0], direction)
            
            # Calculate the angle between vectors
            angle = np.arccos(np.dot([1, 0, 0], direction))
            
            # Create quaternion - axis-angle representation
            if np.linalg.norm(cross) > 0.001:
                axis = cross / np.linalg.norm(cross)
                path_quaternion = tf.transformations.quaternion_about_axis(angle, axis)
            else:
                # Default if cross product is zero
                path_quaternion = tf.transformations.quaternion_from_euler(0, 0, 0)
    else:
        # Fixed orientation - default to pointing in x direction
        path_quaternion = tf.transformations.quaternion_from_euler(0, 0, 0)
    
    # Generate waypoints
    for i in range(num_points):
        # Calculate interpolation factor
        t = float(i) / (num_points - 1) if num_points > 1 else 0
        
        # Interpolate position
        pos = start + t * (end - start)
        
        # Create pose
        pose = geometry_msgs.msg.Pose()
        pose.position.x = pos[0]
        pose.position.y = pos[1]
        pose.position.z = pos[2]
        
        # Set orientation
        pose.orientation.x = path_quaternion[0]
        pose.orientation.y = path_quaternion[1]
        pose.orientation.z = path_quaternion[2]
        pose.orientation.w = path_quaternion[3]
        
        waypoints.append(pose)
    
    return waypoints

def execute_straight_path(robot, start_point, end_point, num_points=10, 
                          cartesian_step=0.01, orientation_mode="fixed"):
    """
    Create and execute a straight line path using MoveIt.
    
    Args:
        robot: MoveGroupCommander instance
        start_point: Starting point [x, y, z]
        end_point: Ending point [x, y, z]
        num_points: Number of waypoints
        cartesian_step: Step size for Cartesian path computation
        orientation_mode: Orientation mode for waypoints
        
    Returns:
        bool: Success or failure
    """
    # Log the workspace boundaries for debugging
    ws_bounds = robot.get_planning_frame_bounds() if hasattr(robot, 'get_planning_frame_bounds') else None
    if ws_bounds:
        rospy.loginfo(f"Robot workspace bounds: {ws_bounds}")
    
    rospy.loginfo(f"Current robot position: {robot.get_current_pose().pose.position}")
    rospy.loginfo(f"Attempting straight line path from {start_point} to {end_point}")
    
    # Check if the points are in opposite directions from the robot
    # If they are, split the path into two parts
    try:
        current_pose = robot.get_current_pose().pose.position
        current_pos = np.array([current_pose.x, current_pose.y, current_pose.z])
        start_dir = np.array(start_point) - current_pos
        end_dir = np.array(end_point) - current_pos
        
        # If the start and end are in very different directions, plan in segments
        if np.dot(start_dir, end_dir) < 0 and np.linalg.norm(start_dir) > 0.2 and np.linalg.norm(end_dir) > 0.2:
            rospy.loginfo("Start and end points are in opposite directions, planning in segments")
            
            # First move to start point
            start_waypoints = generate_straight_waypoints(
                [current_pose.x, current_pose.y, current_pose.z], 
                start_point, 
                5, 
                orientation_mode)
            
            (start_plan, start_fraction) = robot.compute_cartesian_path(
                start_waypoints,
                cartesian_step
            )
            
            if start_fraction < 0.9:
                rospy.logwarn(f"Could only plan {start_fraction * 100:.1f}% of path to start point")
                return False
            
            rospy.loginfo("Executing move to start point...")
            robot.execute(start_plan, wait=True)
            
            # Then from start to end
            path_waypoints = generate_straight_waypoints(
                start_point, 
                end_point, 
                num_points, 
                orientation_mode)
        else:
            # Generate direct waypoints from current position
            path_waypoints = generate_straight_waypoints(start_point, end_point, num_points, orientation_mode)
    except Exception as e:
        rospy.logerr(f"Error in pre-planning: {e}")
        # Fallback to direct waypoint generation
        path_waypoints = generate_straight_waypoints(start_point, end_point, num_points, orientation_mode)
    
    # Log waypoint information
    rospy.loginfo(f"Generated {len(path_waypoints)} waypoints for path planning")
    
    # Try with different orientation modes if the first attempt fails
    orientation_modes = [orientation_mode, "fixed", "path_direction"]
    
    best_plan = None
    best_fraction = 0.0
    
    # First try with various orientations and Cartesian planning
    for mode in orientation_modes:
        if mode != orientation_mode:
            rospy.loginfo(f"Trying orientation mode: {mode}")
            path_waypoints = generate_straight_waypoints(start_point, end_point, num_points, mode)
        
        # Try different step sizes if needed
        for step_multiplier in [1.0, 2.0, 0.5]:
            current_step = cartesian_step * step_multiplier
            rospy.loginfo(f"Planning with step size: {current_step:.4f}")
            
            # Plan Cartesian path with this orientation mode and step size
            (plan, fraction) = robot.compute_cartesian_path(
                path_waypoints,
                current_step,
            )
            
            rospy.loginfo(f"Path planning with {mode} orientation, step {current_step:.4f}: {fraction * 100:.1f}% achieved")
            
            if fraction > best_fraction:
                best_fraction = fraction
                best_plan = plan
                
            # If we got a good plan, don't try more variations
            if fraction > 0.9:
                break
                
        # If we got a good plan, don't try more variations
        if best_fraction > 0.9:
            break
    
    # If Cartesian planning failed, try regular motion planning
    if best_fraction < 0.7:
        rospy.loginfo("Cartesian planning insufficient, trying regular motion planning")
        
        try:
            # Set the goal pose
            target_pose = geometry_msgs.msg.Pose()
            target_pose.position.x = end_point[0]
            target_pose.position.y = end_point[1]
            target_pose.position.z = end_point[2]
            
            # Use current orientation if at the starting point
            current_pose = robot.get_current_pose().pose
            current_pos = [current_pose.position.x, current_pose.position.y, current_pose.position.z]
            current_point = np.array(current_pos)
            start_np = np.array(start_point)
            
            if np.linalg.norm(current_point - start_np) < 0.05:  # Within 5cm of start
                target_pose.orientation = current_pose.orientation
            else:
                # Otherwise use the orientation from our waypoints
                target_pose.orientation = path_waypoints[0].orientation
            
            # Set the target pose and plan
            robot.set_pose_target(target_pose)
            
            # Plan with various planners if needed
            for planner_id in ["RRTConnect", "RRTstar", "PRM"]:
                robot.set_planner_id(planner_id)
                rospy.loginfo(f"Planning with {planner_id}")
                
                plan = robot.plan()
                
                # In newer MoveIt versions, plan() returns a tuple
                if isinstance(plan, tuple):
                    success = plan[0]
                    plan = plan[1]
                else:
                    # For older MoveIt versions
                    success = len(plan.joint_trajectory.points) > 0
                
                if success:
                    rospy.loginfo(f"Regular planning with {planner_id} successful")
                    robot.execute(plan, wait=True)
                    return True
        
        except Exception as e:
            rospy.logerr(f"Error in regular motion planning: {e}")
            # Continue to try other methods
    
    # Return with the best Cartesian plan if it's good enough
    if best_plan and best_fraction > 0.7:
        rospy.loginfo(f"Executing Cartesian path with {best_fraction * 100:.1f}% coverage")
        robot.execute(best_plan, wait=True)
        return True
    else:
        # Last resort: try to split the path into smaller segments
        if best_fraction < 0.3:
            rospy.loginfo("Path planning largely failed, trying segmented approach")
            try:
                # Split the path into 3 segments
                segments = 3
                success = True
                
                for i in range(segments):
                    t_start = float(i) / segments
                    t_end = float(i + 1) / segments
                    
                    seg_start = [
                        start_point[0] + (end_point[0] - start_point[0]) * t_start,
                        start_point[1] + (end_point[1] - start_point[1]) * t_start,
                        start_point[2] + (end_point[2] - start_point[2]) * t_start
                    ]
                    
                    seg_end = [
                        start_point[0] + (end_point[0] - start_point[0]) * t_end,
                        start_point[1] + (end_point[1] - start_point[1]) * t_end,
                        start_point[2] + (end_point[2] - start_point[2]) * t_end
                    ]
                    
                    rospy.loginfo(f"Planning segment {i+1}/{segments}: {seg_start} to {seg_end}")
                    
                    # Try to plan and execute this segment
                    seg_waypoints = generate_straight_waypoints(seg_start, seg_end, 5, "fixed")
                    (seg_plan, seg_fraction) = robot.compute_cartesian_path(seg_waypoints, cartesian_step * 2)
                    
                    if seg_fraction > 0.8:
                        rospy.loginfo(f"Executing segment {i+1} with {seg_fraction * 100:.1f}% coverage")
                        robot.execute(seg_plan, wait=True)
                    else:
                        rospy.logwarn(f"Segment {i+1} planning failed with only {seg_fraction * 100:.1f}% coverage")
                        success = False
                        break
                
                return success
            except Exception as e:
                rospy.logerr(f"Error in segmented planning: {e}")
                
        if best_fraction > 0:
            rospy.logwarn(f"Could only plan {best_fraction * 100:.1f}% of straight line path. Consider using different start/end points or changing orientation mode.")
        else:
            rospy.logerr("Path planning completely failed. Points may be outside workspace or unreachable.")
        return False

def execute_straight_path_from_current(robot, end_point, num_points=10, 
                                      cartesian_step=0.01, orientation_mode="fixed"):
    """
    Execute a straight line path from the current robot position.
    
    Args:
        robot: MoveGroupCommander instance
        end_point: Ending point [x, y, z]
        num_points: Number of waypoints
        cartesian_step: Step size for Cartesian path computation
        orientation_mode: Orientation mode for waypoints
        
    Returns:
        bool: Success or failure
    """
    # Get current position
    current_pose = robot.get_current_pose().pose.position
    start_point = [current_pose.x, current_pose.y, current_pose.z]
    
    rospy.loginfo(f"Starting straight line path from current position: {start_point}")
    
    # Use the existing function to plan and execute
    return execute_straight_path(robot, start_point, end_point, num_points, cartesian_step, orientation_mode)

# Example usage:
if __name__ == "__main__":
    import moveit_commander
    import sys
    
    # Initialize
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('straight_path_example', anonymous=True)
    
    # Create move group
    robot = moveit_commander.MoveGroupCommander("armor")
    
    # Get current position for reference
    current_pose = robot.get_current_pose().pose
    current_pos = [current_pose.position.x, current_pose.position.y, current_pose.position.z]
    rospy.loginfo(f"Current robot position: {current_pos}")
    
    # Choose which type of path to execute
    path_option = 5  # Change this value to try different paths (1-5)
    
    if path_option == 1:
        # Horizontal path along X-axis (forward/backward)
        start = [1, 0.0, 2.5]
        end = [1.0, 0.0, 2.0]
        rospy.loginfo("Executing straight horizontal path along X-axis (forward)")
        success = execute_straight_path(robot, start, end)
    elif path_option == 2:
        # Horizontal path along Y-axis (left/right)
        start = [0.4, -0.2, 0.3]
        end = [0.4, 0.2, 0.3]
        rospy.loginfo("Executing straight horizontal path along Y-axis (left to right)")
        success = execute_straight_path(robot, start, end)
    elif path_option == 3:
        # Horizontal diagonal path (X and Y change, Z constant)
        start = [0.3, -0.1, 0.3]
        end = [0.5, 0.1, 0.3]
        rospy.loginfo("Executing straight horizontal diagonal path")
        success = execute_straight_path(robot, start, end)
    elif path_option == 4:
        # From current position to a target point - more conservative
        # Move just 10cm in a small diagonal
        end = [current_pos[0] + 0.1, current_pos[1] - 0.05, current_pos[2]]
        rospy.loginfo(f"Executing short straight path from current position to {end}")
        success = execute_straight_path_from_current(robot, end)
    elif path_option == 5:
        # Very small incremental motion (3cm)
        end = [current_pos[0] + 0.03, current_pos[1], current_pos[2]]
        rospy.loginfo(f"Executing minimal straight path from current position")
        # Use much smaller Cartesian step size
        success = execute_straight_path_from_current(robot, end, cartesian_step=0.005)
    
    if success:
        rospy.loginfo("Straight path executed successfully")
    else:
        rospy.logerr("Failed to execute straight path")
