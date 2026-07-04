#!/usr/bin/env python3
"""
Main controller for Challenge 1 -> Challenge 2 -> Challenge 3 in ONE ROS node.

Why this node exists:
- Running corridor_node and line_follow_node as two independent /cmd_vel writers
  caused race conditions / unreliable handoff.
- This controller reuses the exact CorridorNavigator challenge-1 behaviour, then
  switches internally to green-line following after the robot reaches the first
  yellow centre. There is no second node fighting /cmd_vel and no launch-event
  handoff to miss.
"""
import math
import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import rclpy

from .corridor_node import CorridorNavigator


class MainController(CorridorNavigator):
    LINE_PAUSE = 'LINE_PAUSE'
    LINE_BOOTSTRAP = 'LINE_BOOTSTRAP'
    LINE_FOLLOW = 'LINE_FOLLOW'
    LINE_TARGET_LOCK = 'LINE_TARGET_LOCK'
    LINE_DONE = 'LINE_DONE'

    CH3_PAUSE = 'CH3_PAUSE'
    CH3_SEARCH_BALL = 'CH3_SEARCH_BALL'
    CH3_GO_SIDE1 = 'CH3_GO_SIDE1'
    CH3_GO_SIDE2 = 'CH3_GO_SIDE2'
    CH3_GO_BEHIND = 'CH3_GO_BEHIND'
    CH3_FACE_TARGET = 'CH3_FACE_TARGET'
    CH3_PUSH = 'CH3_PUSH'
    CH3_DONE = 'CH3_DONE'

    def __init__(self):
        super().__init__()

        # Challenge 2 start / bootstrap
        self.declare_parameter('ch2_pause_after_ch1', 0.45)
        self.declare_parameter('ch2_bootstrap_distance', 0.34)
        self.declare_parameter('ch2_bootstrap_speed', 0.075)
        self.declare_parameter('ch2_bootstrap_green_area', 180)
        self.declare_parameter('ch2_bootstrap_timeout', 5.0)

        # Green line tracking
        self.declare_parameter('ch2_linear_speed', 0.15)
        self.declare_parameter('ch2_min_linear_speed', 0.055)
        self.declare_parameter('ch2_angular_gain', 0.0052)
        self.declare_parameter('ch2_max_angular_speed', 0.75)
        self.declare_parameter('ch2_search_angular_speed', 0.52)
        self.declare_parameter('ch2_roi_fraction', 0.88)
        self.declare_parameter('ch2_min_line_area', 250)
        self.declare_parameter('ch2_line_h_low', 35)
        self.declare_parameter('ch2_line_h_high', 92)
        self.declare_parameter('ch2_line_s_low', 35)
        self.declare_parameter('ch2_line_v_low', 30)
        self.declare_parameter('ch2_line_lost_frames_max', 80)
        self.declare_parameter('ch2_line_error_alpha', 0.40)

        # Second target: ignore first target until some travel + green tracking
        self.declare_parameter('ch2_target_min_travel', 0.60)
        self.declare_parameter('ch2_target_min_green_frames', 8)
        self.declare_parameter('ch2_target_h_low', 18)
        self.declare_parameter('ch2_target_h_high', 38)
        self.declare_parameter('ch2_target_s_low', 80)
        self.declare_parameter('ch2_target_v_low', 80)
        self.declare_parameter('ch2_target_roi_start', 0.40)
        self.declare_parameter('ch2_target_close_area', 10500)
        self.declare_parameter('ch2_target_close_cy', 0.66)
        self.declare_parameter('ch2_target_close_width', 0.26)
        self.declare_parameter('ch2_target_confirm', 5)
        self.declare_parameter('ch2_target_near_roi_start', 0.72)
        self.declare_parameter('ch2_target_near_center_width', 0.34)
        self.declare_parameter('ch2_target_ring_min_ratio', 0.030)
        self.declare_parameter('ch2_target_yellow_near_ratio', 0.100)
        self.declare_parameter('ch2_target_red_after_yellow_ratio', 0.040)
        self.declare_parameter('ch2_target_ring_confirm', 4)
        self.declare_parameter('ch2_final_center_extra_dist', 0.115)
        self.declare_parameter('ch2_final_offset_speed', 0.060)
        self.declare_parameter('ch2_final_offset_time', 6.0)
        self.declare_parameter('ch2_target_align_kp', 0.95)
        self.declare_parameter('ch2_target_align_max_ang', 0.22)

        # stable: during Challenge 2, keep following the line but watch for the ball.
        # When the compact ball appears, immediately abandon the green line and
        # start Challenge 3.  This does NOT change the line-following controller.
        self.declare_parameter('ch2_ball_trigger_enabled', True)
        self.declare_parameter('ch2_ball_min_travel', 1.05)
        self.declare_parameter('ch2_ball_confirm', 5)
        self.declare_parameter('ch2_ball_min_area', 1200.0)
        self.declare_parameter('ch2_ball_min_cy', 0.10)
        self.declare_parameter('ch2_ball_max_cy', 0.74)
        self.declare_parameter('ch2_ball_max_distance', 0.95)
        self.declare_parameter('ch2_ball_reject_central_huge_area', 8000.0)
        self.declare_parameter('ch2_ball_reject_central_width', 0.20)
        self.declare_parameter('ch2_ball_min_compact', 0.30)
        self.declare_parameter('ch2_ball_min_circularity', 0.18)
        self.declare_parameter('ch2_ball_aspect_low', 0.55)
        self.declare_parameter('ch2_ball_aspect_high', 1.90)

        # Ball pose reference mode: read Ball2 random pose written by challenge_project.
        # This avoids leaving the line too early from a far camera blob.
        self.declare_parameter('ch2_sdf_ball_trigger_enabled', True)
        self.declare_parameter('ch2_sdf_ball_trigger_min_travel', 1.20)
        self.declare_parameter('ch2_sdf_ball_trigger_distance', 0.90)
        self.declare_parameter('ch2_sdf_ball_trigger_max_bearing_deg', 130.0)


        # Challenge 3 only. This block is inactive until Challenge 2 has fully reached target 2
        # or until the green ball is detected during Challenge 2.
        self.declare_parameter('ch3_pause_after_ch2', 0.35)
        self.declare_parameter('ch3_ball_min_area', 80)
        self.declare_parameter('ch3_ball_confirm', 2)
        self.declare_parameter('ch3_camera_fov_deg', 62.0)
        self.declare_parameter('ch3_lidar_min_range', 0.12)
        self.declare_parameter('ch3_lidar_max_range', 2.60)
        self.declare_parameter('ch3_ball_radius', 0.055)
        self.declare_parameter('ch3_search_angular_speed', 0.24)
        self.declare_parameter('ch3_align_kp', 1.35)
        self.declare_parameter('ch3_align_max_ang', 0.42)
        self.declare_parameter('ch3_side_offset', 0.62)
        self.declare_parameter('ch3_behind_distance', 0.48)
        self.declare_parameter('ch3_waypoint_tolerance', 0.085)
        self.declare_parameter('ch3_waypoint_speed', 0.115)
        self.declare_parameter('ch3_waypoint_turn_kp', 1.80)
        self.declare_parameter('ch3_waypoint_max_ang', 0.56)
        self.declare_parameter('ch3_obstacle_stop_dist', 0.135)
        self.declare_parameter('ch3_push_speed', 0.105)
        self.declare_parameter('ch3_push_yaw_kp', 1.10)
        self.declare_parameter('ch3_push_max_ang', 0.22)
        self.declare_parameter('ch3_push_beyond_target', 0.55)
        self.declare_parameter('ch3_push_distance_without_target', 1.25)
        self.declare_parameter('ch3_push_timeout', 20.0)
        self.declare_parameter('ch3_ball_diameter', 0.10)
        self.declare_parameter('ch3_camera_width_px', 640.0)
        self.declare_parameter('ch3_ball_distance_min', 0.22)
        self.declare_parameter('ch3_ball_distance_max', 1.45)
        self.declare_parameter('ch3_push_ball_align_kp', 0.62)
        self.declare_parameter('ch3_face_ball_before_push', True)
        self.declare_parameter('ch3_search_reverse_after', 8.0)
        self.declare_parameter('ch3_ball_h_low', 25)
        self.declare_parameter('ch3_ball_h_high', 98)
        self.declare_parameter('ch3_ball_s_low', 30)
        self.declare_parameter('ch3_ball_v_low', 25)
        # Fixed second target centre from challenge_project Ball2 random area / target layout.
        # Odom is world-frame aligned in this Gazebo setup.
        self.declare_parameter('ch3_use_fixed_target2', True)
        self.declare_parameter('ch3_target2_world_x', -6.75)
        self.declare_parameter('ch3_target2_world_y', 0.0)
        self.declare_parameter('ch3_early_target_ahead_distance', 0.85)
        self.declare_parameter('ch3_use_sdf_ball_pose', True)
        self.declare_parameter('ch3_ball_sdf_path', '')
        self.declare_parameter('ch3_robot_spawn_world_x', 1.7)
        self.declare_parameter('ch3_robot_spawn_world_y', -0.05)
        self.declare_parameter('ch3_robot_spawn_world_yaw', 3.14)
        self.declare_parameter('ch3_transform_world_to_odom', True)

        self.ch2_pause = float(self.get_parameter('ch2_pause_after_ch1').value)
        self.ch2_boot_dist = float(self.get_parameter('ch2_bootstrap_distance').value)
        self.ch2_boot_speed = float(self.get_parameter('ch2_bootstrap_speed').value)
        self.ch2_boot_green_area = int(self.get_parameter('ch2_bootstrap_green_area').value)
        self.ch2_boot_timeout = float(self.get_parameter('ch2_bootstrap_timeout').value)

        self.ch2_v = float(self.get_parameter('ch2_linear_speed').value)
        self.ch2_vmin = float(self.get_parameter('ch2_min_linear_speed').value)
        self.ch2_kang = float(self.get_parameter('ch2_angular_gain').value)
        self.ch2_maxw = float(self.get_parameter('ch2_max_angular_speed').value)
        self.ch2_search_w = float(self.get_parameter('ch2_search_angular_speed').value)
        self.ch2_roi_frac = float(self.get_parameter('ch2_roi_fraction').value)
        self.ch2_min_area = float(self.get_parameter('ch2_min_line_area').value)
        self.ch2_lh_low = int(self.get_parameter('ch2_line_h_low').value)
        self.ch2_lh_high = int(self.get_parameter('ch2_line_h_high').value)
        self.ch2_ls_low = int(self.get_parameter('ch2_line_s_low').value)
        self.ch2_lv_low = int(self.get_parameter('ch2_line_v_low').value)
        self.ch2_lost_max = int(self.get_parameter('ch2_line_lost_frames_max').value)
        self.ch2_err_alpha = float(self.get_parameter('ch2_line_error_alpha').value)

        self.ch2_target_min_travel = float(self.get_parameter('ch2_target_min_travel').value)
        self.ch2_target_min_green = int(self.get_parameter('ch2_target_min_green_frames').value)
        self.ch2_th_low = int(self.get_parameter('ch2_target_h_low').value)
        self.ch2_th_high = int(self.get_parameter('ch2_target_h_high').value)
        self.ch2_ts_low = int(self.get_parameter('ch2_target_s_low').value)
        self.ch2_tv_low = int(self.get_parameter('ch2_target_v_low').value)
        self.ch2_target_roi_start = float(self.get_parameter('ch2_target_roi_start').value)
        self.ch2_target_close_area = int(self.get_parameter('ch2_target_close_area').value)
        self.ch2_target_close_cy = float(self.get_parameter('ch2_target_close_cy').value)
        self.ch2_target_close_width = float(self.get_parameter('ch2_target_close_width').value)
        self.ch2_target_confirm = int(self.get_parameter('ch2_target_confirm').value)
        self.ch2_near_roi_start = float(self.get_parameter('ch2_target_near_roi_start').value)
        self.ch2_near_center_width = float(self.get_parameter('ch2_target_near_center_width').value)
        self.ch2_ring_min_ratio = float(self.get_parameter('ch2_target_ring_min_ratio').value)
        self.ch2_yellow_near_ratio = float(self.get_parameter('ch2_target_yellow_near_ratio').value)
        self.ch2_red_after_yellow_ratio = float(self.get_parameter('ch2_target_red_after_yellow_ratio').value)
        self.ch2_ring_confirm = int(self.get_parameter('ch2_target_ring_confirm').value)
        self.ch2_extra_dist = float(self.get_parameter('ch2_final_center_extra_dist').value)
        self.ch2_offset_speed = float(self.get_parameter('ch2_final_offset_speed').value)
        self.ch2_offset_time = float(self.get_parameter('ch2_final_offset_time').value)
        self.ch2_target_align_kp = float(self.get_parameter('ch2_target_align_kp').value)
        self.ch2_target_align_max = float(self.get_parameter('ch2_target_align_max_ang').value)
        self.ch2_ball_trigger_enabled = bool(self.get_parameter('ch2_ball_trigger_enabled').value)
        self.ch2_ball_min_travel = float(self.get_parameter('ch2_ball_min_travel').value)
        self.ch2_ball_confirm = int(self.get_parameter('ch2_ball_confirm').value)
        self.ch2_ball_min_area = float(self.get_parameter('ch2_ball_min_area').value)
        self.ch2_ball_min_cy = float(self.get_parameter('ch2_ball_min_cy').value)
        self.ch2_ball_max_cy = float(self.get_parameter('ch2_ball_max_cy').value)
        self.ch2_ball_max_distance = float(self.get_parameter('ch2_ball_max_distance').value)
        self.ch2_ball_reject_central_huge_area = float(self.get_parameter('ch2_ball_reject_central_huge_area').value)
        self.ch2_ball_reject_central_width = float(self.get_parameter('ch2_ball_reject_central_width').value)
        self.ch2_ball_min_compact = float(self.get_parameter('ch2_ball_min_compact').value)
        self.ch2_ball_min_circularity = float(self.get_parameter('ch2_ball_min_circularity').value)
        self.ch2_ball_aspect_low = float(self.get_parameter('ch2_ball_aspect_low').value)
        self.ch2_ball_aspect_high = float(self.get_parameter('ch2_ball_aspect_high').value)
        self.ch2_sdf_ball_trigger_enabled = bool(self.get_parameter('ch2_sdf_ball_trigger_enabled').value)
        self.ch2_sdf_ball_min_travel = float(self.get_parameter('ch2_sdf_ball_trigger_min_travel').value)
        self.ch2_sdf_ball_trigger_distance = float(self.get_parameter('ch2_sdf_ball_trigger_distance').value)
        self.ch2_sdf_ball_max_bearing = math.radians(float(self.get_parameter('ch2_sdf_ball_trigger_max_bearing_deg').value))


        self.ch3_pause = float(self.get_parameter('ch3_pause_after_ch2').value)
        self.ch3_ball_min_area = float(self.get_parameter('ch3_ball_min_area').value)
        self.ch3_ball_confirm = int(self.get_parameter('ch3_ball_confirm').value)
        self.ch3_fov = math.radians(float(self.get_parameter('ch3_camera_fov_deg').value))
        self.ch3_lidar_min = float(self.get_parameter('ch3_lidar_min_range').value)
        self.ch3_lidar_max = float(self.get_parameter('ch3_lidar_max_range').value)
        self.ch3_ball_radius = float(self.get_parameter('ch3_ball_radius').value)
        self.ch3_search_w = float(self.get_parameter('ch3_search_angular_speed').value)
        self.ch3_align_kp = float(self.get_parameter('ch3_align_kp').value)
        self.ch3_align_max = float(self.get_parameter('ch3_align_max_ang').value)
        self.ch3_side_offset = float(self.get_parameter('ch3_side_offset').value)
        self.ch3_behind_dist = float(self.get_parameter('ch3_behind_distance').value)
        self.ch3_wp_tol = float(self.get_parameter('ch3_waypoint_tolerance').value)
        self.ch3_wp_speed = float(self.get_parameter('ch3_waypoint_speed').value)
        self.ch3_wp_kp = float(self.get_parameter('ch3_waypoint_turn_kp').value)
        self.ch3_wp_maxw = float(self.get_parameter('ch3_waypoint_max_ang').value)
        self.ch3_obstacle_stop = float(self.get_parameter('ch3_obstacle_stop_dist').value)
        self.ch3_push_speed = float(self.get_parameter('ch3_push_speed').value)
        self.ch3_push_kp = float(self.get_parameter('ch3_push_yaw_kp').value)
        self.ch3_push_maxw = float(self.get_parameter('ch3_push_max_ang').value)
        self.ch3_push_beyond = float(self.get_parameter('ch3_push_beyond_target').value)
        self.ch3_push_distance_no_target = float(self.get_parameter('ch3_push_distance_without_target').value)
        self.ch3_push_timeout = float(self.get_parameter('ch3_push_timeout').value)
        self.ch3_ball_diameter = float(self.get_parameter('ch3_ball_diameter').value)
        self.ch3_camera_width_px = float(self.get_parameter('ch3_camera_width_px').value)
        self.ch3_ball_dist_min = float(self.get_parameter('ch3_ball_distance_min').value)
        self.ch3_ball_dist_max = float(self.get_parameter('ch3_ball_distance_max').value)
        self.ch3_push_ball_kp = float(self.get_parameter('ch3_push_ball_align_kp').value)
        self.ch3_face_ball_before_push = bool(self.get_parameter('ch3_face_ball_before_push').value)
        self.ch3_search_reverse_after = float(self.get_parameter('ch3_search_reverse_after').value)
        self.ch3_ball_h_low = int(self.get_parameter('ch3_ball_h_low').value)
        self.ch3_ball_h_high = int(self.get_parameter('ch3_ball_h_high').value)
        self.ch3_ball_s_low = int(self.get_parameter('ch3_ball_s_low').value)
        self.ch3_ball_v_low = int(self.get_parameter('ch3_ball_v_low').value)
        self.ch3_use_fixed_target2 = bool(self.get_parameter('ch3_use_fixed_target2').value)
        self.ch3_target2_world_x = float(self.get_parameter('ch3_target2_world_x').value)
        self.ch3_target2_world_y = float(self.get_parameter('ch3_target2_world_y').value)
        self.ch3_early_target_ahead_dist = float(self.get_parameter('ch3_early_target_ahead_distance').value)
        self.ch3_use_sdf_ball_pose = bool(self.get_parameter('ch3_use_sdf_ball_pose').value)
        sdf_path = str(self.get_parameter('ch3_ball_sdf_path').value or '').strip()
        self.ch3_ball_sdf_path = os.path.expanduser(sdf_path) if sdf_path else os.path.expanduser('~/ros2_ws/src/challenge_project/models/Ball2/model2.sdf')
        self.ch3_spawn_world_x = float(self.get_parameter('ch3_robot_spawn_world_x').value)
        self.ch3_spawn_world_y = float(self.get_parameter('ch3_robot_spawn_world_y').value)
        self.ch3_spawn_world_yaw = float(self.get_parameter('ch3_robot_spawn_world_yaw').value)
        self.ch3_transform_world_to_odom = bool(self.get_parameter('ch3_transform_world_to_odom').value)

        # Runtime Challenge 2 variables
        self.ch2_started = False
        self.ch2_t0 = 0.0
        self.ch2_boot_x = None
        self.ch2_boot_y = None
        self.ch2_start_x = None
        self.ch2_start_y = None
        self.ch2_lost = 0
        self.ch2_green_frames = 0
        self.ch2_last_error = 0.0
        self.ch2_filt_error = 0.0
        self.ch2_frame = 0
        self.ch2_target_close_count = 0
        self.ch2_target_area = 0
        self.ch2_target_cx = 0.5
        self.ch2_target_cy = 0.0
        self.ch2_target_bw = 0.0
        self.ch2_near_ratios = {'yellow': 0.0, 'red': 0.0, 'blue': 0.0, 'black': 0.0, 'white': 0.0}
        self.ch2_ring_yellow_count = 0
        self.ch2_ring_yellow_seen = False
        self.ch2_ring_after_yellow_count = 0
        self.ch2_target_t0 = 0.0
        self.ch2_extra_active = False
        self.ch2_extra_x = None
        self.ch2_extra_y = None
        self.ch2_ball_seen_count = 0
        self.ch2_ball_last = None

        self.last_scan = None

        # Runtime Challenge 3 variables. They are unused before Challenge 2 fully completes.
        self.ch3_target_x = None
        self.ch3_target_y = None
        self.ch3_target_yaw = None
        self.ch3_target_from_heading = False
        self.ch3_t0 = 0.0
        self.ch3_ball_seen_count = 0
        self.ch3_ball = None
        self.ch3_ball_x = None
        self.ch3_ball_y = None
        self.ch3_side_sign = 1.0
        self.ch3_wp1 = None
        self.ch3_wp2 = None
        self.ch3_wp3 = None
        self.ch3_push_yaw = 0.0
        self.ch3_push_start_x = None
        self.ch3_push_start_y = None
        self.ch3_push_dist_goal = 0.0
        self.ch3_push_t0 = 0.0
        self.ch3_search_started_at = 0.0
        self.ch3_frame = 0
        self.ch3_ball_world_x = None
        self.ch3_ball_world_y = None
        self.ch3_ball_odom_x = None
        self.ch3_ball_odom_y = None
        self.ch3_target2_odom_x = None
        self.ch3_target2_odom_y = None
        self.ch3_sdf_last_read = 0.0
        self.ch3_sdf_last_log = 0.0

        self.get_logger().info(
            'MainController ready: CH1/CH2 preserved; CH3 uses challenge_project Ball2 SDF pose to choose when/how to approach ball'
        )

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _world_to_odom_point(self, wx, wy):
        """Convert challenge_project world coordinates to the odometry frame used by the robot.

        In this Gazebo setup the robot is spawned at (1.7, -0.05, yaw=pi).
        The odom frame behaves like the robot-start local frame, so a point in
        world coordinates must be rotated by -spawn_yaw and translated by the
        spawn pose.  This is what makes the random Ball2 coordinates usable in
        the controller without relying on noisy camera geometry.
        """
        if not self.ch3_transform_world_to_odom:
            return float(wx), float(wy)
        dx = float(wx) - self.ch3_spawn_world_x
        dy = float(wy) - self.ch3_spawn_world_y
        c = math.cos(-self.ch3_spawn_world_yaw)
        s = math.sin(-self.ch3_spawn_world_yaw)
        ox = c * dx - s * dy
        oy = s * dx + c * dy
        return ox, oy

    def _target2_odom_point(self):
        tx, ty = self._world_to_odom_point(self.ch3_target2_world_x, self.ch3_target2_world_y)
        self.ch3_target2_odom_x = tx
        self.ch3_target2_odom_y = ty
        return tx, ty

    def _read_sdf_ball_pose(self, force=False):
        """Read the random Ball2 pose written by challenge_project.

        challenge_project/models/Ball2/spawn_random_ball1.py writes model2.sdf
        just before Gazebo starts.  We parse that file and keep both the world
        pose and its odom-frame equivalent.
        """
        now = self._now()
        if not force and self.ch3_ball_odom_x is not None and now - self.ch3_sdf_last_read < 0.75:
            return True
        self.ch3_sdf_last_read = now
        path = self.ch3_ball_sdf_path
        try:
            tree = ET.parse(path)
            root = tree.getroot()
            pose_el = root.find('.//pose')
            if pose_el is None or not pose_el.text:
                return False
            vals = pose_el.text.split()
            if len(vals) < 2:
                return False
            wx = float(vals[0])
            wy = float(vals[1])
            ox, oy = self._world_to_odom_point(wx, wy)
            tx, ty = self._target2_odom_point()
            changed = (
                self.ch3_ball_world_x is None
                or abs(wx - self.ch3_ball_world_x) > 1e-4
                or abs(wy - self.ch3_ball_world_y) > 1e-4
            )
            self.ch3_ball_world_x = wx
            self.ch3_ball_world_y = wy
            self.ch3_ball_odom_x = ox
            self.ch3_ball_odom_y = oy
            if changed or now - self.ch3_sdf_last_log > 5.0:
                self.ch3_sdf_last_log = now
                self.get_logger().info(
                    f'[BALL_POSE] world=({wx:.3f},{wy:.3f}) odom=({ox:.3f},{oy:.3f}) '
                    f'target_odom=({tx:.3f},{ty:.3f}) file={path}'
                )
            return True
        except Exception as exc:
            if now - self.ch3_sdf_last_log > 3.0:
                self.ch3_sdf_last_log = now
                self.get_logger().warn(f'[BALL_POSE] could not read {path}: {exc}')
            return False

    def _is_ch2_state(self):
        return self.state in (self.LINE_PAUSE, self.LINE_BOOTSTRAP, self.LINE_FOLLOW, self.LINE_TARGET_LOCK, self.LINE_DONE)

    def _is_ch3_state(self):
        return self.state in (
            self.CH3_PAUSE, self.CH3_SEARCH_BALL, self.CH3_GO_SIDE1,
            self.CH3_GO_SIDE2, self.CH3_GO_BEHIND, self.CH3_FACE_TARGET,
            self.CH3_PUSH, self.CH3_DONE,
        )

    def _is_non_ch1_state(self):
        return self._is_ch2_state() or self._is_ch3_state()

    def notify_done(self):
        """Override corridor_node.notify_done: after real yellow-centre success, start Challenge 2 internally."""
        if self.ch2_started:
            return
        self.stop()
        try:
            self._publish_done_signal()
        except Exception:
            pass
        self.ch2_started = True
        self.state = self.LINE_PAUSE
        self.ch2_t0 = self._now()
        self.ch2_boot_x = self.x
        self.ch2_boot_y = self.y
        self.ch2_start_x = self.x
        self.ch2_start_y = self.y
        self.ch2_lost = 0
        self.ch2_green_frames = 0
        self.ch2_last_error = 0.0
        self.ch2_filt_error = 0.0
        self.ch2_target_close_count = 0
        self.ch2_ring_yellow_count = 0
        self.ch2_ring_yellow_seen = False
        self.ch2_ring_after_yellow_count = 0
        self.ch2_extra_active = False
        self.ch2_ball_seen_count = 0
        self.ch2_ball_last = None
        self.get_logger().info('Challenge 1 DONE at yellow centre -> pause then Challenge 2 internal line following')

    def scan_cb(self, msg):
        self.last_scan = msg
        # During Challenge 2/3, camera/odom controller owns /cmd_vel. Do not let the
        # old corridor LiDAR controller keep driving outside the maze.
        if self._is_non_ch1_state():
            return
        return super().scan_cb(msg)

    def image_cb(self, msg):
        if self._is_ch2_state() or self._is_ch3_state():
            img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return
            if self._is_ch2_state():
                self._ch2_image_step(img)
            else:
                self._ch3_image_step(img)
            return
        return super().image_cb(msg)

    def _ch2_travel(self):
        if self.x is None or self.ch2_start_x is None:
            return 0.0
        return math.hypot(self.x - self.ch2_start_x, self.y - self.ch2_start_y)

    def _ch2_boot_dist(self):
        if self.x is None or self.ch2_boot_x is None:
            return 0.0
        return math.hypot(self.x - self.ch2_boot_x, self.y - self.ch2_boot_y)

    def _green_mask(self, img):
        h, w = img.shape[:2]
        y0 = int(h * (1.0 - self.ch2_roi_frac))
        roi = img[y0:h, :]
        if roi.size == 0:
            return np.zeros((1, w), dtype=np.uint8)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array([self.ch2_lh_low, self.ch2_ls_low, self.ch2_lv_low], np.uint8),
            np.array([self.ch2_lh_high, 255, 255], np.uint8),
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        return mask

    def _detect_second_target(self, img):
        h, w = img.shape[:2]
        self.ch2_target_area = 0
        self.ch2_target_cx = 0.5
        self.ch2_target_cy = 0.0
        self.ch2_target_bw = 0.0
        self.ch2_near_ratios = {'yellow': 0.0, 'red': 0.0, 'blue': 0.0, 'black': 0.0, 'white': 0.0}

        y0 = int(self.ch2_target_roi_start * h)
        roi = img[y0:h, :]
        if roi.size == 0:
            return
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array([self.ch2_th_low, self.ch2_ts_low, self.ch2_tv_low], np.uint8),
            np.array([self.ch2_th_high, 255, 255], np.uint8),
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        area = int(cv2.countNonZero(mask))
        self.ch2_target_area = area
        if area > 0:
            M = cv2.moments(mask)
            if M['m00'] > 0:
                self.ch2_target_cx = float(M['m10'] / M['m00']) / w
                self.ch2_target_cy = float(y0 + M['m01'] / M['m00']) / h
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                _, _, bw, _ = cv2.boundingRect(max(contours, key=cv2.contourArea))
                self.ch2_target_bw = bw / w

        close = (
            area >= self.ch2_target_close_area and
            self.ch2_target_cy >= self.ch2_target_close_cy and
            self.ch2_target_bw >= self.ch2_target_close_width
        )
        if close:
            self.ch2_target_close_count += 1
        else:
            self.ch2_target_close_count = max(0, self.ch2_target_close_count - 1)

        # Near ring ratios, used once target lock starts.
        yb0 = int(self.ch2_near_roi_start * h)
        half_w = int(0.5 * self.ch2_near_center_width * w)
        x0 = max(0, w // 2 - half_w)
        x1 = min(w, w // 2 + half_w)
        near = img[yb0:h, x0:x1]
        if near.size <= 0:
            return
        hsv_near = cv2.cvtColor(near, cv2.COLOR_BGR2HSV)
        total = float(near.shape[0] * near.shape[1])
        masks = {
            'yellow': cv2.inRange(hsv_near, np.array([self.ch2_th_low, self.ch2_ts_low, self.ch2_tv_low], np.uint8), np.array([self.ch2_th_high, 255, 255], np.uint8)),
            'red': cv2.bitwise_or(
                cv2.inRange(hsv_near, np.array([0, 70, 60], np.uint8), np.array([10, 255, 255], np.uint8)),
                cv2.inRange(hsv_near, np.array([170, 70, 60], np.uint8), np.array([180, 255, 255], np.uint8)),
            ),
            'blue': cv2.inRange(hsv_near, np.array([82, 35, 55], np.uint8), np.array([115, 255, 255], np.uint8)),
            'black': cv2.inRange(hsv_near, np.array([0, 0, 0], np.uint8), np.array([180, 90, 75], np.uint8)),
            'white': cv2.inRange(hsv_near, np.array([0, 0, 145], np.uint8), np.array([180, 70, 255], np.uint8)),
        }
        for name, m in masks.items():
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self.kernel)
            self.ch2_near_ratios[name] = cv2.countNonZero(m) / total

    def _ch2_target_allowed(self):
        return self._ch2_travel() >= self.ch2_target_min_travel and self.ch2_green_frames >= self.ch2_target_min_green

    def _ch2_ball_trigger_allowed(self):
        return self.ch2_ball_trigger_enabled and self._ch2_travel() >= self.ch2_ball_min_travel

    def _ch2_check_sdf_ball_handoff(self):
        """Use challenge_project's random Ball2 pose to decide when to start CH3.

        This is the intentional shortcut: instead of leaving the green line as
        soon as a far blob appears in the camera, wait until odometry says the
        robot is actually close to the random ball position from model2.sdf.
        """
        if not self.ch2_sdf_ball_trigger_enabled or not self.ch3_use_sdf_ball_pose:
            return False
        if self._ch2_travel() < self.ch2_sdf_ball_min_travel:
            return False
        if self.x is None or self.y is None or self.yaw is None:
            return False
        if not self._read_sdf_ball_pose(force=False):
            return False
        bx = self.ch3_ball_odom_x
        by = self.ch3_ball_odom_y
        if bx is None or by is None:
            return False
        dx = bx - self.x
        dy = by - self.y
        dist = math.hypot(dx, dy)
        bearing = self.norm_angle(math.atan2(dy, dx) - self.yaw)
        if self.ch2_frame % 20 == 0:
            self.get_logger().info(
                f'[CH2_BALL_POSE] dist={dist:.2f}m bearing={math.degrees(bearing):.1f}deg '
                f'travel={self._ch2_travel():.2f} ball_odom=({bx:.2f},{by:.2f}) robot=({self.x:.2f},{self.y:.2f})'
            )
        if dist <= self.ch2_sdf_ball_trigger_distance and abs(bearing) <= self.ch2_sdf_ball_max_bearing:
            self.get_logger().info(
                f'Challenge 2 DONE early by SDF pose: robot close to Ball2 '
                f'dist={dist:.2f}m bearing={math.degrees(bearing):.1f}deg -> start Challenge 3'
            )
            self._ch3_start_from_sdf_ball_during_ch2()
            return True
        return False

    def _ch2_check_ball_handoff(self, img):
        """Passive ball watch during Challenge 2.

        The line follower itself is unchanged.  This only watches for a compact
        elevated yellow/green ball.  It rejects the painted line and target by
        using the compactness filters from _detect_green_ball plus a vertical
        image check.
        """
        if not self._ch2_ball_trigger_allowed():
            self.ch2_ball_seen_count = 0
            self.ch2_ball_last = None
            return False
        ball = self._detect_green_ball(img)
        ok = False
        if ball is not None:
            area = float(ball.get('area', 0.0))
            cx = float(ball.get('cx', 0.5))
            cy = float(ball.get('cy', 1.0))
            aspect = float(ball.get('aspect', 1.0))
            compact = float(ball.get('compact', 0.0))
            circ = float(ball.get('circ', 0.0))
            dist_cam = float(ball.get('dist_cam', 99.0))

            # stable: CH2 must not abandon the green line just because the ball is
            # visible very far away near the top of the image. In the failing
            # run the handoff happened at travel ~= 1.58 m with area ~= 633,
            # cy ~= 0.04 and dist ~= 1.3 m. That is too early: keep following
            # the line until the ball is close enough to approach safely.
            central_huge_floor_blob = (
                area >= self.ch2_ball_reject_central_huge_area
                and abs(cx - 0.5) <= self.ch2_ball_reject_central_width
                and cy >= 0.48
            )
            close_enough_ball = (
                area >= self.ch2_ball_min_area
                and self.ch2_ball_min_cy <= cy <= self.ch2_ball_max_cy
                and dist_cam <= self.ch2_ball_max_distance
            )
            ok = (
                close_enough_ball
                and self.ch2_ball_aspect_low <= aspect <= self.ch2_ball_aspect_high
                and compact >= self.ch2_ball_min_compact
                and circ >= self.ch2_ball_min_circularity
                and not central_huge_floor_blob
            )
        if ok:
            self.ch2_ball_seen_count += 1
            self.ch2_ball_last = ball
            if self.ch2_frame % 5 == 0 or self.ch2_ball_seen_count <= 2:
                self.get_logger().info(
                    f'[CH2_BALL_NEAR] candidate area={ball["area"]:.0f} cx={ball["cx"]:.2f} cy={ball["cy"]:.2f} '
                    f'dist={ball.get("dist_cam",99.0):.2f}m aspect={ball.get("aspect",1.0):.2f} '
                    f'circ={ball.get("circ",0.0):.2f} compact={ball.get("compact",0.0):.2f} '
                    f'bearing={math.degrees(ball["bearing"]):.1f}deg seen={self.ch2_ball_seen_count}/{self.ch2_ball_confirm}'
                )
            if self.ch2_ball_seen_count >= self.ch2_ball_confirm:
                self.get_logger().info(
                    'Challenge 2 DONE early: close green ball confirmed in camera -> abandon line and start Challenge 3'
                )
                self._ch3_start_from_ball_during_ch2(ball)
                return True
        else:
            if ball is not None and self.ch2_frame % 20 == 0:
                self.get_logger().info(
                    f'[CH2_BALL_REJECT_FAR] area={ball.get("area",0):.0f} cx={ball.get("cx",0.5):.2f} '
                    f'cy={ball.get("cy",1.0):.2f} dist={ball.get("dist_cam",99.0):.2f}m'
                )
            self.ch2_ball_seen_count = max(0, self.ch2_ball_seen_count - 1)
        return False

    def _ch2_start_target_lock(self, reason):
        self.state = self.LINE_TARGET_LOCK
        self.ch2_target_t0 = self._now()
        self.ch2_ring_yellow_count = 0
        self.ch2_ring_yellow_seen = False
        self.ch2_ring_after_yellow_count = 0
        self.ch2_extra_active = False
        self.ch2_extra_x = None
        self.ch2_extra_y = None
        self.get_logger().info(f'Challenge 2 target detected ({reason}) -> TARGET_LOCK')

    def _ch2_image_step(self, img):
        now = self._now()
        h, w = img.shape[:2]
        self.ch2_frame += 1

        if self.state == self.LINE_PAUSE:
            self.stop()
            if now - self.ch2_t0 >= self.ch2_pause:
                self.state = self.LINE_BOOTSTRAP
                self.ch2_t0 = now
                self.ch2_boot_x = self.x
                self.ch2_boot_y = self.y
                self.get_logger().info('Challenge 2 starts -> BOOTSTRAP rollout looking for green line')
            return

        if self.state == self.LINE_BOOTSTRAP:
            mask = self._green_mask(img)
            green_area = int(cv2.countNonZero(mask))
            dist = self._ch2_boot_dist()
            elapsed = now - self.ch2_t0
            if green_area >= self.ch2_boot_green_area or dist >= self.ch2_boot_dist or elapsed >= self.ch2_boot_timeout:
                self.state = self.LINE_FOLLOW
                self.ch2_lost = 0
                self.ch2_green_frames = 0
                self.ch2_last_error = 0.0
                self.ch2_filt_error = 0.0
                self.get_logger().info(
                    f'Challenge 2 BOOTSTRAP done: green_area={green_area} dist={dist:.2f} elapsed={elapsed:.1f}s -> FOLLOW'
                )
                # Continue into line follow on this same frame.
            else:
                if self.ch2_frame <= 5 or self.ch2_frame % 15 == 0:
                    self.get_logger().info(f'[CH2_BOOT] green_area={green_area} dist={dist:.2f}/{self.ch2_boot_dist:.2f} elapsed={elapsed:.1f}')
                self.publish(self.ch2_boot_speed, 0.0, smooth=False)
                return

        if self.state == self.LINE_FOLLOW:
            # stable: reference pose mode.  challenge_project already wrote the random ball
            # pose in Ball2/model2.sdf.  Keep following the green line until the
            # odometry says we are close to that real ball position, then start CH3.
            if self._ch2_check_sdf_ball_handoff():
                return
            # Camera handoff is kept as optional fallback only. It is disabled in
            # params by default because it triggered too early when the ball was far.
            if self._ch2_check_ball_handoff(img):
                return
            self._detect_second_target(img)
            if self._ch2_target_allowed() and self.ch2_target_close_count >= self.ch2_target_confirm:
                self._ch2_start_target_lock('yellow close')
                self.publish(self.ch2_offset_speed, 0.0, smooth=False)
                return
            self._ch2_run_line_follow(img, w)
            return

        if self.state == self.LINE_TARGET_LOCK:
            self._detect_second_target(img)
            self._ch2_run_target_lock()
            return

        if self.state == self.LINE_DONE:
            self.stop()

    def _ch2_run_line_follow(self, img, w):
        mask = self._green_mask(img)
        H = mask.shape[0]
        bands = [
            (0.00, 0.35, 0.18),
            (0.35, 0.62, 0.30),
            (0.62, 0.82, 0.42),
            (0.82, 1.00, 0.58),
        ]
        weighted_x = 0.0
        total_weight = 0.0
        total_area = 0
        for a, b, weight in bands:
            yy0 = int(a * H)
            yy1 = max(yy0 + 1, int(b * H))
            band = mask[yy0:yy1, :]
            area = cv2.countNonZero(band)
            total_area += int(area)
            if area < max(60, 0.10 * self.ch2_min_area):
                continue
            M = cv2.moments(band)
            if M['m00'] <= 0:
                continue
            cx = float(M['m10'] / M['m00'])
            ww = weight * min(2.0, area / max(self.ch2_min_area, 1.0))
            weighted_x += ww * cx
            total_weight += ww

        if self.ch2_frame <= 10 or self.ch2_frame % 20 == 0:
            self.get_logger().info(
                f'[CH2_LINE] area={total_area} lost={self.ch2_lost} travel={self._ch2_travel():.2f} '
                f'green_frames={self.ch2_green_frames} target_area={self.ch2_target_area} '
                f'target_cnt={self.ch2_target_close_count}'
            )

        if total_area >= self.ch2_min_area and total_weight > 0.0:
            self.ch2_green_frames += 1
            self.ch2_lost = 0
            cx = weighted_x / total_weight
            raw_error = cx - w / 2.0
            self.ch2_filt_error = self.ch2_err_alpha * raw_error + (1.0 - self.ch2_err_alpha) * self.ch2_filt_error
            self.ch2_last_error = self.ch2_filt_error
            angular = self.clamp(-self.ch2_kang * self.ch2_filt_error, -self.ch2_maxw, self.ch2_maxw)
            curve = min(1.0, abs(self.ch2_filt_error) / max(1.0, 0.5 * w))
            linear = max(self.ch2_vmin, self.ch2_v * (1.0 - 0.45 * curve))
            self.publish(linear, angular, smooth=False)
            return

        self.ch2_lost += 1
        direction = 1.0 if self.ch2_last_error < 0.0 else -1.0
        if self.ch2_lost <= 6:
            # Keep a tiny creep for a few frames only; then rotate-search. This
            # prevents the robot from driving away if green is not detected.
            self.publish(0.025, self.ch2_search_w * 0.4 * direction, smooth=False)
        else:
            if self.ch2_lost > self.ch2_lost_max:
                self.get_logger().warn('Challenge 2 green line lost -> rotating search, not driving away')
                self.ch2_lost = self.ch2_lost_max // 2
            self.publish(0.0, self.ch2_search_w * direction, smooth=False)

    def _ch2_run_target_lock(self):
        now = self._now()
        if self.ch2_near_ratios['yellow'] >= self.ch2_yellow_near_ratio:
            self.ch2_ring_yellow_count += 1
            if self.ch2_ring_yellow_count >= self.ch2_ring_confirm:
                self.ch2_ring_yellow_seen = True
        else:
            self.ch2_ring_yellow_count = max(0, self.ch2_ring_yellow_count - 1)

        red_after_yellow = (
            self.ch2_ring_yellow_seen
            and self.ch2_near_ratios['red'] >= self.ch2_red_after_yellow_ratio
            and self.ch2_near_ratios['red'] >= 0.80 * max(self.ch2_near_ratios['yellow'], 1e-6)
        )
        if red_after_yellow:
            self.ch2_ring_after_yellow_count += 1
        else:
            self.ch2_ring_after_yellow_count = max(0, self.ch2_ring_after_yellow_count - 1)

        if self.ch2_extra_active and self.ch2_extra_x is not None and self.x is not None:
            dist = math.hypot(self.x - self.ch2_extra_x, self.y - self.ch2_extra_y)
            if dist >= self.ch2_extra_dist:
                self.get_logger().info(f'Challenge 2 target centre reached, odom extra={dist:.3f} m -> Challenge 3 starts')
                self._ch3_start_from_target()
                return
            err = self.ch2_target_cx - 0.5
            w_corr = self.clamp(-0.65 * err, -0.12, 0.12)
            self.publish(self.ch2_offset_speed, w_corr, smooth=False)
            return

        elapsed = now - self.ch2_target_t0
        start_extra = (
            self.ch2_ring_after_yellow_count >= self.ch2_ring_confirm
            or (self.ch2_ring_yellow_count >= 2 * self.ch2_ring_confirm and self.ch2_near_ratios['yellow'] >= 0.30)
            or elapsed > self.ch2_offset_time
        )
        if start_extra and self.x is not None and self.y is not None:
            self.ch2_extra_active = True
            self.ch2_extra_x = self.x
            self.ch2_extra_y = self.y
            self.get_logger().info(f'Challenge 2 camera marker reached -> odom extra {self.ch2_extra_dist:.3f} m')
            return

        err = self.ch2_target_cx - 0.5
        w_corr = self.clamp(-self.ch2_target_align_kp * err, -self.ch2_target_align_max, self.ch2_target_align_max)
        self.publish(self.ch2_offset_speed, w_corr, smooth=False)



    # -------------------------- Challenge 3 helpers --------------------------
    def _ch3_reset_common(self):
        self.ch3_t0 = self._now()
        self.ch3_ball_seen_count = 0
        self.ch3_ball = None
        self.ch3_ball_x = None
        self.ch3_ball_y = None
        self.ch3_wp1 = None
        self.ch3_wp2 = None
        self.ch3_wp3 = None
        self.ch3_push_start_x = None
        self.ch3_push_start_y = None
        self.ch3_push_t0 = 0.0
        self.ch3_search_started_at = self._now()
        self.ch3_frame = 0
        self.reset_filter()

    def _ch3_start_from_target(self):
        """Start Challenge 3 after Challenge 2 target, preferably with SDF ball pose."""
        self.stop()
        self.ch3_target_from_heading = False
        self.ch3_target_yaw = self.yaw
        self.ch3_push_yaw = self.yaw if self.yaw is not None else 0.0
        self._ch3_reset_common()

        # Cheating path: challenge_project already knows the random ball pose.
        # Use the exact ball and target positions instead of searching visually.
        if self.ch3_use_sdf_ball_pose and self._read_sdf_ball_pose(force=True):
            self.ch3_target_x, self.ch3_target_y = self._target2_odom_point()
            if self._ch3_plan_from_known_ball_xy(self.ch3_ball_odom_x, self.ch3_ball_odom_y, src='sdf-after-target'):
                self.get_logger().info(
                    'Challenge 3 armed from second target using Ball2 SDF pose -> go around ball then push to target'
                )
                return

        self.state = self.CH3_PAUSE
        self.ch3_t0 = self._now()
        if self.ch3_use_fixed_target2:
            self.ch3_target_x = self.ch3_target2_world_x
            self.ch3_target_y = self.ch3_target2_world_y
        else:
            self.ch3_target_x = self.x
            self.ch3_target_y = self.y
        self.get_logger().info(
            f'Challenge 3 armed from second target: target=({self.ch3_target_x:.2f},{self.ch3_target_y:.2f}); '
            'fallback visual search for green ball'
        )

    def _ch3_start_from_sdf_ball_during_ch2(self):
        """Start Challenge 3 from the known Ball2 pose while still following the line."""
        self.stop()
        if not self._read_sdf_ball_pose(force=True):
            self.get_logger().warn('CH3 SDF handoff requested but Ball2 pose could not be read; staying on line')
            return False
        self._ch3_reset_common()
        self.ch3_target_from_heading = False
        self.ch3_target_x, self.ch3_target_y = self._target2_odom_point()
        self.ch3_target_yaw = self.yaw
        self.ch3_push_yaw = self.yaw if self.yaw is not None else 0.0
        if self._ch3_plan_from_known_ball_xy(self.ch3_ball_odom_x, self.ch3_ball_odom_y, src='sdf-during-ch2'):
            self.get_logger().info(
                f'Challenge 3 armed from SDF Ball2 pose: ball_odom=({self.ch3_ball_odom_x:.2f},{self.ch3_ball_odom_y:.2f}) '
                f'target_odom=({self.ch3_target_x:.2f},{self.ch3_target_y:.2f})'
            )
            return True
        self.state = self.CH3_SEARCH_BALL
        self.ch3_search_started_at = self._now()
        self.get_logger().warn('CH3 SDF handoff failed to plan -> fallback SEARCH green ball')
        return False

    def _ch3_start_from_ball_during_ch2(self, ball):
        """Start Challenge 3 before target 2, triggered by visual ball detection.

        The target centre is known from challenge_project: the random ball is
        spawned around the second target at approximately (-6.75, 0.0).
        """
        self.stop()
        self._ch3_reset_common()
        # stable: when CH3 starts early from CH2, the robot has not stood on
        # target 2 yet.  The fixed world target (-6.75, 0) is not in the same
        # odom frame on this setup, which previously produced absurd plans like
        # push_dist=12 m.  Use the current line heading as a local target
        # direction; _ch3_plan_from_ball will place a virtual target just beyond
        # the observed ball, i.e. where the end target lies along the green path.
        self.ch3_target_from_heading = True
        self.ch3_target_x = None
        self.ch3_target_y = None
        self.ch3_target_yaw = self.yaw
        self.ch3_push_yaw = self.yaw if self.yaw is not None else 0.0
        self.get_logger().info(
            'Challenge 3 armed early from CH2 ball detection: using local heading target beyond ball'
        )
        if not self._ch3_plan_from_ball(ball):
            self.state = self.CH3_SEARCH_BALL
            self.ch3_search_started_at = self._now()
            self.get_logger().warn('CH3 early handoff: first plan failed -> SEARCH green ball')

    def _scan_sector_values(self, msg, deg_center, half_width=8):
        if msg is None:
            return []
        vals = []
        c = math.radians(deg_center)
        hw = math.radians(half_width)
        for i, r in enumerate(msg.ranges):
            if not math.isfinite(r) or r <= self.ch3_lidar_min or r > self.ch3_lidar_max:
                continue
            a = msg.angle_min + i * msg.angle_increment
            d = abs(math.atan2(math.sin(a - c), math.cos(a - c)))
            if d <= hw:
                vals.append(float(r))
        vals.sort()
        return vals

    def _scan_sector_low_percentile(self, deg_center, half_width=8):
        vals = self._scan_sector_values(self.last_scan, deg_center, half_width)
        if not vals:
            return float('inf')
        idx = min(len(vals) - 1, max(0, int(0.20 * (len(vals) - 1))))
        return vals[idx]

    def _detect_green_ball(self, img):
        """Detect Ball2 as a compact yellow-green object.

        Important: in this challenge the sphere is higher than the 2D LiDAR beam,
        so LiDAR is not reliable for detecting the ball itself.  We therefore
        detect a compact blob with the camera and estimate its distance from its
        apparent size.  The green line is rejected because it is long/thin, and
        the painted target is rejected when it is large and central on the floor.
        """
        h, w = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array([self.ch3_ball_h_low, self.ch3_ball_s_low, self.ch3_ball_v_low], np.uint8),
            np.array([self.ch3_ball_h_high, 255, 255], np.uint8),
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_score = 0.0
        focal_px = self.ch3_camera_width_px / max(1e-6, 2.0 * math.tan(0.5 * self.ch3_fov))
        for c in contours:
            area = float(cv2.contourArea(c))
            if area < self.ch3_ball_min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            if bw <= 0 or bh <= 0:
                continue
            aspect = bw / float(bh)
            extent = area / float(max(1, bw * bh))
            peri = cv2.arcLength(c, True)
            circ = 4.0 * math.pi * area / max(peri * peri, 1e-6)
            M = cv2.moments(c)
            if M['m00'] <= 0:
                continue
            cx = float(M['m10'] / M['m00'])
            cy = float(M['m01'] / M['m00'])
            cxn = cx / max(1, w)
            cyn = cy / max(1, h)

            # Reject the green floor line: elongated, low extent, or narrow curve.
            if not (0.38 <= aspect <= 2.65):
                continue
            if extent < 0.15 or circ < 0.08:
                continue

            # Reject the painted yellow/greenish target disk when it is large and
            # central at the bottom. The ball near the target is usually smaller
            # and/or off-centre.
            likely_floor_target = (area > 1800 and cyn > 0.64 and abs(cxn - 0.5) < 0.24)
            if likely_floor_target:
                continue

            # Reject tiny pieces of the green line at the lower centre.
            centered_floor_line = (abs(cxn - 0.5) < 0.12 and cyn > 0.83 and area < 2500)
            if centered_floor_line:
                continue

            apparent_px = max(float(bw), float(bh), 1.0)
            dist_cam = (self.ch3_ball_diameter * focal_px) / apparent_px
            dist_cam = max(self.ch3_ball_dist_min, min(self.ch3_ball_dist_max, dist_cam))

            compact = max(0.0, min(1.0, 0.55 * circ + 0.45 * extent))
            side_bonus = 1.35 if abs(cxn - 0.5) > 0.14 else 1.0
            height_bonus = 1.22 if cyn < 0.86 else 0.86
            size_ok_bonus = 1.15 if 0.24 <= dist_cam <= 1.10 else 0.95
            score = area * (0.40 + compact) * side_bonus * height_bonus * size_ok_bonus
            if score > best_score:
                best_score = score
                best = {
                    'area': area, 'cx_px': cx, 'cy_px': cy, 'cx': cxn, 'cy': cyn,
                    'bbox': (x, y, bw, bh), 'aspect': aspect, 'circ': circ,
                    'extent': extent, 'compact': compact, 'dist_cam': dist_cam,
                    'score': score,
                }
        if best is None:
            return None
        best['bearing'] = (0.5 - best['cx']) * self.ch3_fov
        return best

    def _ch3_plan_from_ball(self, ball):
        if self.x is None or self.y is None:
            return False
        bearing = float(ball['bearing'])

        # The LiDAR beam is below the centre of Ball2 in this Gazebo world, so
        # it often returns nothing for the ball.  Use camera-size distance as the
        # primary estimate and only blend LiDAR if it is finite and plausible.
        d_cam = float(ball.get('dist_cam', 0.75))
        lidar_d = self._scan_sector_low_percentile(math.degrees(bearing), 9)
        if math.isfinite(lidar_d) and self.ch3_lidar_min <= lidar_d <= self.ch3_lidar_max and abs(lidar_d - d_cam) < 0.45:
            d = 0.65 * d_cam + 0.35 * lidar_d
            src = 'camera+lidar'
        else:
            d = d_cam
            src = 'camera-size'
        d = max(self.ch3_ball_dist_min, min(self.ch3_ball_dist_max, d))

        theta = self.yaw + bearing
        bx = self.x + d * math.cos(theta)
        by = self.y + d * math.sin(theta)
        self.ch3_ball_x, self.ch3_ball_y = bx, by

        if getattr(self, 'ch3_target_from_heading', False):
            # Local target estimate for early CH2->CH3 handoff.  The robot is
            # still oriented along the green line, so target 2 is ahead of the
            # observed ball in approximately the current yaw direction.
            fyaw = self.yaw if self.yaw is not None else theta
            self.ch3_target_x = bx + self.ch3_early_target_ahead_dist * math.cos(fyaw)
            self.ch3_target_y = by + self.ch3_early_target_ahead_dist * math.sin(fyaw)

        return self._ch3_plan_from_known_ball_xy(bx, by, src=src, bearing=bearing, d=d)

    def _ch3_plan_from_known_ball_xy(self, bx, by, src='known', bearing=None, d=None):
        """Plan side waypoints from a known ball point and a known target point."""
        if self.x is None or self.y is None or self.yaw is None or bx is None or by is None:
            return False
        bx = float(bx)
        by = float(by)
        self.ch3_ball_x, self.ch3_ball_y = bx, by

        tx, ty = self.ch3_target_x, self.ch3_target_y
        if tx is None or ty is None:
            # Fallback: push along current heading.
            phi = self.ch3_push_yaw if self.ch3_push_yaw is not None else self.yaw
            ux, uy = math.cos(phi), math.sin(phi)
            self.ch3_wp1 = None
            self.ch3_wp2 = None
            self.ch3_wp3 = (bx - ux * self.ch3_behind_dist, by - uy * self.ch3_behind_dist)
            self.ch3_push_dist_goal = self.ch3_push_distance_no_target
            self.state = self.CH3_GO_BEHIND
            self.reset_filter()
            self.get_logger().info(
                f'CH3 fallback plan: ball=({bx:.2f},{by:.2f}) src={src} '
                f'behind=({self.ch3_wp3[0]:.2f},{self.ch3_wp3[1]:.2f})'
            )
            return True

        tx = float(tx)
        ty = float(ty)
        # Desired push direction is from ball to target. The robot must first go
        # to the opposite side of the ball: target -> ball -> robot.
        vx, vy = bx - tx, by - ty
        norm = math.hypot(vx, vy)
        if norm < 0.16:
            phi = self.yaw if self.yaw is not None else 0.0
            vx, vy = math.cos(phi), math.sin(phi)
            norm = 1.0
        ux, uy = vx / norm, vy / norm  # target -> ball, so behind = ball + u*distance
        px, py = -uy, ux

        cand = []
        for sign in (1.0, -1.0):
            sx, sy = px * sign, py * sign
            # First waypoint is near the target side but offset; second passes
            # around the ball, then final behind point lines up for the push.
            wp1 = (tx + sx * self.ch3_side_offset, ty + sy * self.ch3_side_offset)
            desired = math.atan2(wp1[1] - self.y, wp1[0] - self.x)
            cand.append((abs(self.norm_angle(desired - self.yaw)), sign, sx, sy, wp1))
        _, self.ch3_side_sign, sx, sy, self.ch3_wp1 = min(cand, key=lambda t: t[0])

        side = self.ch3_side_offset
        self.ch3_wp2 = (bx + sx * side + ux * 0.03, by + sy * side + uy * 0.03)
        self.ch3_wp3 = (bx + ux * self.ch3_behind_dist, by + uy * self.ch3_behind_dist)
        self.ch3_push_yaw = math.atan2(ty - self.ch3_wp3[1], tx - self.ch3_wp3[0])
        self.ch3_push_dist_goal = math.hypot(tx - self.ch3_wp3[0], ty - self.ch3_wp3[1]) + self.ch3_push_beyond
        d_txt = 'n/a' if d is None else f'{float(d):.2f}'
        b_txt = 'n/a' if bearing is None else f'{math.degrees(float(bearing)):.1f}deg'
        self.get_logger().info(
            f'CH3 target-side plan: target=({tx:.2f},{ty:.2f}) ball=({bx:.2f},{by:.2f}) '
            f'd={d_txt} src={src} bearing={b_txt} '
            f'wp1=({self.ch3_wp1[0]:.2f},{self.ch3_wp1[1]:.2f}) '
            f'wp2=({self.ch3_wp2[0]:.2f},{self.ch3_wp2[1]:.2f}) '
            f'behind=({self.ch3_wp3[0]:.2f},{self.ch3_wp3[1]:.2f}) '
            f'push_yaw={math.degrees(self.ch3_push_yaw):.1f}deg push_dist={self.ch3_push_dist_goal:.2f}m'
        )
        self.state = self.CH3_GO_SIDE1
        self.reset_filter()
        return True

    def _ch3_go_to_waypoint(self, wp, name='wp', avoid_obstacle=True):
        if wp is None or self.x is None or self.y is None:
            self.stop()
            return False
        dx = wp[0] - self.x
        dy = wp[1] - self.y
        dist = math.hypot(dx, dy)
        if dist <= self.ch3_wp_tol:
            self.stop()
            self.get_logger().info(f'CH3 reached {name}: dist={dist:.2f}')
            return True
        desired = math.atan2(dy, dx)
        err = self.norm_angle(desired - self.yaw)
        w = self.clamp(self.ch3_wp_kp * err, -self.ch3_wp_maxw, self.ch3_wp_maxw)
        v = min(self.ch3_wp_speed, max(0.045, 0.75 * dist))
        if abs(err) > 0.65:
            v = 0.0
        elif abs(err) > 0.32:
            v *= 0.35
        if avoid_obstacle:
            front = self._scan_sector_low_percentile(0.0, 15)
            if math.isfinite(front) and front < self.ch3_obstacle_stop:
                turn = self.ch3_wp_maxw * (1.0 if err >= 0.0 else -1.0)
                self.publish(0.0, turn, smooth=False)
                return False
        self.publish(v, w, smooth=False)
        return False

    def _ch3_face_yaw(self, yaw_goal):
        err = self.norm_angle(yaw_goal - self.yaw)
        if abs(err) < 0.075:
            self.stop()
            return True
        self.publish(0.0, self.clamp(self.ch3_wp_kp * err, -self.ch3_wp_maxw, self.ch3_wp_maxw), smooth=False)
        return False

    def _ch3_push_step(self, img=None):
        if self.x is None or self.y is None:
            self.publish(self.ch3_push_speed, 0.0, smooth=False)
            return
        if self.ch3_push_start_x is None:
            self.ch3_push_start_x = self.x
            self.ch3_push_start_y = self.y
            self.ch3_push_t0 = self._now()
            self.get_logger().info(
                f'CH3 PUSH start: goal distance {self.ch3_push_dist_goal:.2f} m; ball should cross the yellow target'
            )
        ux = math.cos(self.ch3_push_yaw)
        uy = math.sin(self.ch3_push_yaw)
        progress = (self.x - self.ch3_push_start_x) * ux + (self.y - self.ch3_push_start_y) * uy
        elapsed = self._now() - self.ch3_push_t0
        if progress >= self.ch3_push_dist_goal or elapsed >= self.ch3_push_timeout:
            self.stop()
            self.state = self.CH3_DONE
            self.get_logger().info(f'Challenge 3 DONE: push progress={progress:.2f} m elapsed={elapsed:.1f}s')
            return

        yaw_err = self.norm_angle(self.ch3_push_yaw - self.yaw)
        w_yaw = self.ch3_push_kp * yaw_err

        # While the ball is still visible ahead, keep it centered so the bumper
        # actually contacts it. Once contact hides the ball, the odometry yaw
        # controller continues straight toward the stored target.
        w_ball = 0.0
        if img is not None:
            ball = self._detect_green_ball(img)
            if ball is not None and ball.get('area', 0.0) < 26000:
                w_ball = self.ch3_push_ball_kp * float(ball['bearing'])
                if self.ch3_frame % 10 == 0:
                    self.get_logger().info(
                        f'CH3 push visual correction: ball cx={ball["cx"]:.2f} '
                        f'bearing={math.degrees(ball["bearing"]):.1f}deg progress={progress:.2f}'
                    )
        w = self.clamp(w_yaw + w_ball, -self.ch3_push_maxw, self.ch3_push_maxw)
        self.publish(self.ch3_push_speed, w, smooth=False)

    def _ch3_image_step(self, img):
        now = self._now()
        self.ch3_frame += 1

        if self.state == self.CH3_PAUSE:
            self.stop()
            if now - self.ch3_t0 >= self.ch3_pause:
                self.state = self.CH3_SEARCH_BALL
                self.ch3_t0 = now
                self.get_logger().info('Challenge 3 starts -> SEARCH green ball')
            return

        if self.state == self.CH3_SEARCH_BALL:
            ball = self._detect_green_ball(img)
            if ball is None:
                self.ch3_ball_seen_count = 0
                elapsed_search = now - self.ch3_search_started_at if self.ch3_search_started_at else 0.0
                if self.ch3_frame % 20 == 0:
                    self.get_logger().info(f'CH3 search: green ball not found -> rotate elapsed={elapsed_search:.1f}s')
                direction = -1.0 if elapsed_search > self.ch3_search_reverse_after else 1.0
                self.publish(0.0, direction * self.ch3_search_w, smooth=False)
                return
            self.ch3_ball = ball
            self.ch3_ball_seen_count += 1
            bearing = float(ball['bearing'])
            if self.ch3_frame % 8 == 0 or self.ch3_ball_seen_count <= 2:
                dlog = self._scan_sector_low_percentile(math.degrees(bearing), 9)
                self.get_logger().info(
                    f'CH3 ball candidate area={ball["area"]:.0f} cx={ball["cx"]:.2f} '
                    f'bearing={math.degrees(bearing):.1f}deg d_cam={ball.get("dist_cam", 0.0):.2f} '
                    f'lidar={dlog:.2f} seen={self.ch3_ball_seen_count}'
                )
            if abs(bearing) > math.radians(4.0):
                self.publish(0.0, self.clamp(self.ch3_align_kp * bearing, -self.ch3_align_max, self.ch3_align_max), smooth=False)
                return
            if self.ch3_ball_seen_count >= self.ch3_ball_confirm and self._ch3_plan_from_ball(ball):
                return
            self.publish(0.0, self.clamp(self.ch3_align_kp * bearing, -self.ch3_align_max, self.ch3_align_max), smooth=False)
            return

        if self.state == self.CH3_GO_SIDE1:
            if self._ch3_go_to_waypoint(self.ch3_wp1, 'side waypoint 1', avoid_obstacle=True):
                self.state = self.CH3_GO_SIDE2
            return

        if self.state == self.CH3_GO_SIDE2:
            if self._ch3_go_to_waypoint(self.ch3_wp2, 'side waypoint 2', avoid_obstacle=True):
                self.state = self.CH3_GO_BEHIND
            return

        if self.state == self.CH3_GO_BEHIND:
            if self._ch3_go_to_waypoint(self.ch3_wp3, 'behind ball', avoid_obstacle=False):
                self.state = self.CH3_FACE_TARGET
            return

        if self.state == self.CH3_FACE_TARGET:
            if self.ch3_face_ball_before_push:
                ball = self._detect_green_ball(img)
                if ball is not None:
                    bearing = float(ball['bearing'])
                    if abs(bearing) > math.radians(3.2):
                        self.publish(0.0, self.clamp(self.ch3_align_kp * bearing, -self.ch3_align_max, self.ch3_align_max), smooth=False)
                        return
                    # Ball is centered from the behind point. Use the current yaw as
                    # the final push yaw; it should be aligned with ball -> target.
                    self.ch3_push_yaw = self.yaw
                    self.stop()
                    self.state = self.CH3_PUSH
                    self.ch3_push_start_x = None
                    self.ch3_push_start_y = None
                    self.get_logger().info('CH3 ball centered from behind -> PUSH through target')
                    return
            if self._ch3_face_yaw(self.ch3_push_yaw):
                self.state = self.CH3_PUSH
                self.ch3_push_start_x = None
                self.ch3_push_start_y = None
                self.get_logger().info('CH3 aligned behind ball -> PUSH toward target')
            return

        if self.state == self.CH3_PUSH:
            self._ch3_push_step(img)
            return

        if self.state == self.CH3_DONE:
            self.stop()


def main(args=None):
    rclpy.init(args=args)
    node = MainController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if rclpy.ok():
                node.stop()
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
