#!/usr/bin/env python3
import math
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage, LaserScan
from nav_msgs.msg import Odometry
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


class FullRunController(Node):

    # global states
    CORRIDOR    = 'CORRIDOR'
    LINE_FOLLOW = 'LINE_FOLLOW'
    BALL_PUSH   = 'BALL_PUSH'
    DONE        = 'DONE'

    # ball sub-states
    BALL_SEARCHING = 'BALL_SEARCHING'
    BALL_ALIGNING  = 'BALL_ALIGNING'
    BALL_PUSHING   = 'BALL_PUSHING'

    # corridor sub-states
    CORR_CENTERING     = 'CENTERING'
    CORR_TURNING       = 'TURNING'
    CORR_RECOVERY      = 'RECOVERY'
    CORR_EXIT_APPROACH = 'EXIT_APPROACH'
    CORR_EXIT_LOCK     = 'EXIT_LOCK'

    def __init__(self) -> None:
        super().__init__('full_run_controller')

        # --- corridor ---
        self.declare_parameter('corridor_fast_speed',        0.28)
        self.declare_parameter('corridor_slow_speed',        0.08)
        self.declare_parameter('corridor_turn_lin',          0.05)
        self.declare_parameter('corridor_turn_speed',        0.48)
        self.declare_parameter('corridor_turn_min_time',     0.55)
        self.declare_parameter('corridor_turn_target_angle', 1.35)
        self.declare_parameter('corridor_turn_max_time',     2.4)
        self.declare_parameter('corridor_reverse_speed',     0.05)
        self.declare_parameter('corridor_max_angular',       1.4)
        self.declare_parameter('corridor_alpha',             0.35)
        self.declare_parameter('corridor_min_front_fast',    1.20)
        self.declare_parameter('corridor_front_stop',        0.35)
        self.declare_parameter('corridor_turn_detect',       0.68)
        self.declare_parameter('corridor_slow_dist',         0.85)
        self.declare_parameter('corridor_clear_dist',        0.55)
        self.declare_parameter('corridor_kp',                1.8)
        self.declare_parameter('corridor_enter_thresh',      1.50)
        self.declare_parameter('corridor_enter_count',       5)
        self.declare_parameter('corridor_stuck_time',         2.5)
        self.declare_parameter('corridor_stuck_dist',         0.02)
        # single-wall following
        self.declare_parameter('corridor_wall_detect_max',    0.85)
        self.declare_parameter('corridor_wall_desired',       0.35)
        self.declare_parameter('corridor_kp_single_wall',     0.8)
        self.declare_parameter('corridor_single_wall_max',    0.25)
        # exit detection
        self.declare_parameter('corridor_exit_trigger_front', 0.90)
        self.declare_parameter('corridor_exit_frames',        4)
        self.declare_parameter('corridor_exit_speed',         0.16)
        self.declare_parameter('corridor_exit_max_ang',       0.15)
        self.declare_parameter('corridor_exit_kp_yaw',        0.5)
        self.declare_parameter('corridor_exit_timeout',       15.0)
        self.declare_parameter('corridor_exit_emerg',         0.20)
        self.declare_parameter('corridor_tgt_h_low',          18)
        self.declare_parameter('corridor_tgt_h_high',         38)
        self.declare_parameter('corridor_tgt_s_low',          80)
        self.declare_parameter('corridor_tgt_v_low',          80)
        self.declare_parameter('corridor_tgt_roi',            0.40)
        self.declare_parameter('corridor_tgt_far_area',       1500)
        self.declare_parameter('corridor_tgt_area',           12000)
        self.declare_parameter('corridor_tgt_cy',             0.70)
        self.declare_parameter('corridor_tgt_width',          0.35)
        self.declare_parameter('corridor_tgt_confirm',        5)
        self.declare_parameter('corridor_final_offset_time',   5.0)  # safety timeout only
        self.declare_parameter('corridor_final_offset_speed',  0.060)
        self.declare_parameter('corridor_final_center_extra_dist', 0.115)
        self.declare_parameter('corridor_target_near_roi_start', 0.72)
        self.declare_parameter('corridor_target_near_center_width', 0.32)
        self.declare_parameter('corridor_target_ring_min_ratio', 0.030)
        self.declare_parameter('corridor_target_yellow_near_ratio', 0.110)
        self.declare_parameter('corridor_target_red_after_yellow_ratio', 0.045)
        self.declare_parameter('corridor_target_ring_confirm', 4)

        # --- line follow ---
        self.declare_parameter('line_linear_speed',    0.18)
        self.declare_parameter('line_angular_gain',    0.004)
        self.declare_parameter('line_roi_fraction',    0.50)
        self.declare_parameter('line_min_area',        500)
        self.declare_parameter('obstacle_dist',        0.40)
        self.declare_parameter('obstacle_angle_deg',   25)
        self.declare_parameter('line_h_low',  40)
        self.declare_parameter('line_h_high', 80)
        self.declare_parameter('line_s_low',  50)
        self.declare_parameter('line_v_low',  50)
        self.declare_parameter('line_lost_frames_max', 60)

        # --- ball push ---
        self.declare_parameter('ball_linear_speed',    0.18)
        self.declare_parameter('ball_push_speed',      0.12)
        self.declare_parameter('ball_angular_gain',    0.005)
        self.declare_parameter('ball_min_area',        300)
        self.declare_parameter('ball_push_area',       12000)
        self.declare_parameter('ball_done_area',       70000)
        self.declare_parameter('ball_h_low',   18)
        self.declare_parameter('ball_h_high',  35)
        self.declare_parameter('ball_s_low',   100)
        self.declare_parameter('ball_v_low',   100)
        self.declare_parameter('ball_search_angular', 0.40)

        # corridor params
        self._c_vf     = self.get_parameter('corridor_fast_speed').value
        self._c_vs     = self.get_parameter('corridor_slow_speed').value
        self._c_vtl    = self.get_parameter('corridor_turn_lin').value
        self._c_wt     = self.get_parameter('corridor_turn_speed').value
        self._c_tmin   = self.get_parameter('corridor_turn_min_time').value
        self._c_tang   = self.get_parameter('corridor_turn_target_angle').value
        self._c_tmax   = self.get_parameter('corridor_turn_max_time').value
        self._c_vr     = self.get_parameter('corridor_reverse_speed').value
        self._c_mw     = self.get_parameter('corridor_max_angular').value
        self._c_al     = self.get_parameter('corridor_alpha').value
        self._c_dff    = self.get_parameter('corridor_min_front_fast').value
        self._c_stop   = self.get_parameter('corridor_front_stop').value
        self._c_turn_d = self.get_parameter('corridor_turn_detect').value
        self._c_slow_d = self.get_parameter('corridor_slow_dist').value
        self._c_clear  = self.get_parameter('corridor_clear_dist').value
        self._c_kp     = self.get_parameter('corridor_kp').value
        self._c_dent   = self.get_parameter('corridor_enter_thresh').value
        self._c_nent   = int(self.get_parameter('corridor_enter_count').value)
        self._c_tstk   = self.get_parameter('corridor_stuck_time').value
        self._c_dstk   = self.get_parameter('corridor_stuck_dist').value
        self._c_dwmax  = self.get_parameter('corridor_wall_detect_max').value
        self._c_dwdes  = self.get_parameter('corridor_wall_desired').value
        self._c_kpsw   = self.get_parameter('corridor_kp_single_wall').value
        self._c_swmax  = self.get_parameter('corridor_single_wall_max').value
        self._c_etf    = self.get_parameter('corridor_exit_trigger_front').value
        self._c_nexf   = int(self.get_parameter('corridor_exit_frames').value)
        self._c_vex    = self.get_parameter('corridor_exit_speed').value
        self._c_wex    = self.get_parameter('corridor_exit_max_ang').value
        self._c_kpy    = self.get_parameter('corridor_exit_kp_yaw').value
        self._c_tex    = self.get_parameter('corridor_exit_timeout').value
        self._c_emerg  = self.get_parameter('corridor_exit_emerg').value
        self._c_thlo   = self.get_parameter('corridor_tgt_h_low').value
        self._c_thhi   = self.get_parameter('corridor_tgt_h_high').value
        self._c_tslo   = self.get_parameter('corridor_tgt_s_low').value
        self._c_tvlo   = self.get_parameter('corridor_tgt_v_low').value
        self._c_troi   = self.get_parameter('corridor_tgt_roi').value
        self._c_tfar   = self.get_parameter('corridor_tgt_far_area').value
        self._c_tarea  = self.get_parameter('corridor_tgt_area').value
        self._c_tcy    = self.get_parameter('corridor_tgt_cy').value
        self._c_tbw    = self.get_parameter('corridor_tgt_width').value
        self._c_tconf  = int(self.get_parameter('corridor_tgt_confirm').value)
        self._c_offset_time = float(self.get_parameter('corridor_final_offset_time').value)
        self._c_offset_speed = float(self.get_parameter('corridor_final_offset_speed').value)
        self._c_center_extra_dist = float(self.get_parameter('corridor_final_center_extra_dist').value)
        self._c_near_roi_start = float(self.get_parameter('corridor_target_near_roi_start').value)
        self._c_near_center_width = float(self.get_parameter('corridor_target_near_center_width').value)
        self._c_ring_min_ratio = float(self.get_parameter('corridor_target_ring_min_ratio').value)
        self._c_yellow_near_ratio = float(self.get_parameter('corridor_target_yellow_near_ratio').value)
        self._c_red_after_yellow_ratio = float(self.get_parameter('corridor_target_red_after_yellow_ratio').value)
        self._c_ring_confirm = int(self.get_parameter('corridor_target_ring_confirm').value)

        # line follow params
        self._l_v      = self.get_parameter('line_linear_speed').value
        self._l_k      = self.get_parameter('line_angular_gain').value
        self._l_roi    = self.get_parameter('line_roi_fraction').value
        self._l_mina   = self.get_parameter('line_min_area').value
        self._obs_dist = self.get_parameter('obstacle_dist').value
        self._obs_ang  = self.get_parameter('obstacle_angle_deg').value
        self._lh_lo    = self.get_parameter('line_h_low').value
        self._lh_hi    = self.get_parameter('line_h_high').value
        self._ls_lo    = self.get_parameter('line_s_low').value
        self._lv_lo    = self.get_parameter('line_v_low').value
        self._lost_max = self.get_parameter('line_lost_frames_max').value

        # ball push params
        self._b_v      = self.get_parameter('ball_linear_speed').value
        self._b_push   = self.get_parameter('ball_push_speed').value
        self._b_k      = self.get_parameter('ball_angular_gain').value
        self._b_mina   = self.get_parameter('ball_min_area').value
        self._b_pusha  = self.get_parameter('ball_push_area').value
        self._b_donea  = self.get_parameter('ball_done_area').value
        self._bh_lo    = self.get_parameter('ball_h_low').value
        self._bh_hi    = self.get_parameter('ball_h_high').value
        self._bs_lo    = self.get_parameter('ball_s_low').value
        self._bv_lo    = self.get_parameter('ball_v_low').value
        self._b_wsrch  = self.get_parameter('ball_search_angular').value

        # global state
        self._state             = self.CORRIDOR
        self._obstacle          = False
        self._line_lost_counter = 0
        self._line_seen_once    = False
        self._ball_state        = self.BALL_SEARCHING
        self._kernel            = np.ones((5, 5), np.uint8)

        # corridor runtime
        self._corr_sub      = self.CORR_CENTERING
        self._corr_tdir     = 1
        self._corr_turn_t   = 0.0
        self._corr_turn_yaw = 0.0
        self._corr_ent_cnt  = 0
        self._corr_exc_cnt  = 0
        self._corr_entered  = False
        self._corr_rec_t    = 0.0
        self._corr_exit_t   = 0.0
        self._corr_exit_yaw = 0.0
        self._corr_log_t    = 0.0
        self._corr_yarea    = 0
        self._corr_yfar     = False
        self._corr_ycy      = 0.0
        self._corr_ycx      = 0.5
        self._corr_ybw      = 0.0
        self._corr_ycnt     = 0
        self._corr_final_offset_active = False
        self._corr_final_offset_t0 = 0.0
        self._corr_near_color = 'none'
        self._corr_near_ratios = {'yellow': 0.0, 'red': 0.0, 'blue': 0.0, 'black': 0.0, 'white': 0.0}
        self._corr_ring_yellow_seen = False
        self._corr_ring_yellow_seen_count = 0
        self._corr_ring_after_yellow_count = 0
        self._corr_ring_done = False
        self._corr_center_extra_active = False
        self._corr_center_extra_x = None
        self._corr_center_extra_y = None
        self._corr_center_extra_reason = 'none'
        self._c_prev_lin    = 0.0
        self._c_prev_ang    = 0.0

        # odom
        self._odom_x   = None
        self._odom_y   = None
        self._odom_yaw = 0.0
        self._ref_x    = None
        self._ref_y    = None
        self._stuck_t  = None

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self._pub      = self.create_publisher(Twist, 'cmd_vel', 10)
        self._scan_sub = self.create_subscription(
            LaserScan, 'scan', self._scan_cb, sensor_qos)
        self._img_sub  = self.create_subscription(
            CompressedImage, '/image_raw/compressed', self._image_cb, sensor_qos)
        self._odom_sub = self.create_subscription(
            Odometry, 'odom', self._odom_cb, 10)

        self.get_logger().info('Full Run Controller started – CORRIDOR')

    # ------------------------------------------------------------------ helpers

    def _rng(self, ranges, amin, ainc, center_deg, half_deg=12):
        cr   = math.radians(center_deg % 360)
        hr   = math.radians(half_deg)
        n    = len(ranges)
        vals = []
        for i in range(n):
            a    = amin + i * ainc
            diff = abs(math.atan2(math.sin(a - cr), math.cos(a - cr)))
            if diff <= hr:
                v = ranges[i]
                if math.isfinite(v) and v > 0.05:
                    vals.append(v)
        if not vals:
            return float('inf')
        vals.sort()
        return vals[len(vals) // 2]

    @staticmethod
    def _yaw_from_quat(q):
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny, cosy)

    @staticmethod
    def _norm_angle(a):
        return math.atan2(math.sin(a), math.cos(a))

    def _publish(self, v: float, w: float) -> None:
        t = Twist()
        t.linear.x  = float(v)
        t.angular.z = float(w)
        self._pub.publish(t)

    def _stop(self) -> None:
        self._publish(0.0, 0.0)

    def _transition(self, new_state: str) -> None:
        self.get_logger().info(f'State: {self._state} → {new_state}')
        self._state = new_state

    def _odom_cb(self, msg: Odometry) -> None:
        self._odom_x   = msg.pose.pose.position.x
        self._odom_y   = msg.pose.pose.position.y
        self._odom_yaw = self._yaw_from_quat(msg.pose.pose.orientation)

    def _reset_stuck(self) -> None:
        self._ref_x   = self._odom_x
        self._ref_y   = self._odom_y
        self._stuck_t = self.get_clock().now().nanoseconds * 1e-9

    def _is_stuck(self) -> bool:
        if self._odom_x is None or self._ref_x is None:
            return False
        dist = math.hypot(self._odom_x - self._ref_x, self._odom_y - self._ref_y)
        if dist > self._c_dstk:
            self._reset_stuck()
            return False
        dt = self.get_clock().now().nanoseconds * 1e-9 - (self._stuck_t or 0.0)
        return dt > self._c_tstk

    def _c_smooth(self, new_lin, new_ang):
        a = self._c_al
        lin = a * new_lin + (1.0 - a) * self._c_prev_lin
        ang = a * new_ang + (1.0 - a) * self._c_prev_ang
        self._c_prev_lin = lin
        self._c_prev_ang = ang
        return lin, ang

    def _c_dyn_lin(self, front, w_raw):
        front_f = max(0.0, min(1.0,
            (front - self._c_stop) / max(0.01, self._c_dff - self._c_stop)))
        turn_f  = max(0.2, min(1.0, 1.0 - abs(w_raw) / self._c_mw))
        return max(self._c_vs, min(self._c_vf, self._c_vf * front_f * turn_f))

    # ---------------------------------------------------------------- callbacks

    def _scan_cb(self, msg: LaserScan) -> None:
        if len(msg.ranges) == 0:
            return
        if self._state == self.CORRIDOR:
            self._corridor_scan(msg)
        elif self._state == self.LINE_FOLLOW:
            self._line_scan(msg)

    def _line_scan(self, msg: LaserScan) -> None:
        n    = len(msg.ranges)
        half = int(self._obs_ang)
        vals = [msg.ranges[i % n]
                for i in range(-half, half + 1)
                if math.isfinite(msg.ranges[i % n]) and msg.ranges[i % n] > 0.02]
        if vals:
            prev           = self._obstacle
            self._obstacle = min(vals) < self._obs_dist
            if self._obstacle and not prev:
                self.get_logger().warn('[OBSTACLE] stopped.')

    def _corridor_scan(self, msg: LaserScan) -> None:
        amin = msg.angle_min
        ainc = msg.angle_increment
        R    = lambda deg, w=12: self._rng(msg.ranges, amin, ainc, deg, w)

        front       = R(0,   18)
        front_left  = R(35,  15)
        front_right = R(325, 15)
        left        = R(90,  20)
        right       = R(270, 20)
        wide_left   = R(75,  45)
        wide_right  = R(285, 45)

        left_vis       = math.isfinite(left)  and left  < self._c_dwmax
        right_vis      = math.isfinite(right) and right < self._c_dwmax
        front_open     = not math.isfinite(front) or front > self._c_etf
        side_open      = not left_vis or not right_vis
        yellow_partial = self._corr_yfar
        now = self.get_clock().now().nanoseconds * 1e-9

        if now - self._corr_log_t > 0.5:
            self._corr_log_t = now
            wall_mode = ('dual'  if left_vis and right_vis
                         else 'left'  if left_vis
                         else 'right' if right_vis
                         else 'open')
            self.get_logger().info(
                f'[CORR/{self._corr_sub}|{wall_mode}] F={front:.2f} L={left:.2f} R={right:.2f} '
                f'WL={wide_left:.2f} WR={wide_right:.2f} '
                f'yel={self._corr_yarea} far={self._corr_yfar} cy={self._corr_ycy:.2f} bw={self._corr_ybw:.2f} ycnt={self._corr_ycnt} '
                f'ent={self._corr_entered} exc={self._corr_exc_cnt}')

        # ---- EXIT_LOCK: camera ring gate before LINE_FOLLOW ----
        # stable: the front camera only tells us what is under the nose. Once the
        # camera sees the yellow-centre marker/ring transition, continue by a
        # calibrated odometry distance so the robot base reaches the yellow disk.
        if self._corr_sub == self.CORR_EXIT_LOCK:
            if not self._corr_final_offset_active:
                self._corr_final_offset_active = True
                self._corr_final_offset_t0 = now
                self._corr_ring_yellow_seen = False
                self._corr_ring_yellow_seen_count = 0
                self._corr_ring_after_yellow_count = 0
                self._corr_ring_done = False
                self._corr_center_extra_active = False
                self._corr_center_extra_x = None
                self._corr_center_extra_y = None
                self._corr_center_extra_reason = 'none'
                self._c_prev_lin = 0.0
                self._c_prev_ang = 0.0
                self.get_logger().info('[CORR] Yellow close -> stable target-centre final approach')

            if self._corr_ring_done:
                self.get_logger().info('[CORR] Target-centre odom extra completed -> LINE_FOLLOW')
                self._corr_sub = self.CORR_CENTERING
                self._corr_final_offset_active = False
                self._corr_center_extra_active = False
                self._c_prev_lin = 0.0
                self._c_prev_ang = 0.0
                self._transition(self.LINE_FOLLOW)
                self._stop()
                return

            elapsed = now - self._corr_final_offset_t0

            if (self._corr_center_extra_active and self._corr_center_extra_x is not None
                    and self._odom_x is not None and self._odom_y is not None):
                dist = math.hypot(self._odom_x - self._corr_center_extra_x,
                                  self._odom_y - self._corr_center_extra_y)
                if dist >= self._c_center_extra_dist:
                    self.get_logger().info(f'[CORR] Final odom extra done ({dist:.3f} m)')
                    self._corr_ring_done = True
                    self._stop()
                    return
                err_x = self._corr_ycx - 0.5 if self._corr_yfar else 0.0
                w_corr = max(-0.12, min(0.12, -0.65 * err_x))
                self._publish(self._c_offset_speed, w_corr)
                return

            if elapsed > self._c_offset_time:
                if self._odom_x is not None and self._odom_y is not None:
                    self.get_logger().warn('[CORR] Ring gate timeout -> start odom extra fallback')
                    self._corr_center_extra_active = True
                    self._corr_center_extra_x = self._odom_x
                    self._corr_center_extra_y = self._odom_y
                    self._corr_center_extra_reason = 'timeout'
                    return
                self.get_logger().warn('[CORR] Ring gate timeout without odom -> LINE_FOLLOW')
                self._corr_sub = self.CORR_CENTERING
                self._corr_final_offset_active = False
                self._transition(self.LINE_FOLLOW)
                self._stop()
                return

            err_x = self._corr_ycx - 0.5 if self._corr_yfar else 0.0
            w_corr = max(-0.20, min(0.20, -1.15 * err_x))
            self._publish(self._c_offset_speed, w_corr)
            return

        # ---- EXIT_APPROACH: open straight section ----
        if self._corr_sub == self.CORR_EXIT_APPROACH:
            elapsed = now - self._corr_exit_t
            if self._corr_ycnt >= self._c_tconf:
                self.get_logger().info('[CORR] Yellow close in EXIT_APPROACH → EXIT_LOCK offset')
                self._corr_sub = self.CORR_EXIT_LOCK
                self._corr_final_offset_active = False
                self._corr_final_offset_active = False
                self._c_prev_lin = 0.0
                self._c_prev_ang = 0.0
                return
            if elapsed > self._c_tex:
                self.get_logger().info('[CORR] EXIT_APPROACH timeout → LINE_FOLLOW')
                self._corr_sub = self.CORR_CENTERING
                self._transition(self.LINE_FOLLOW)
                self._stop()
                return

            if math.isfinite(front) and front < self._c_emerg:
                self._publish(0.0, 0.0)
                return

            yaw_err = self._norm_angle(self._odom_yaw - self._corr_exit_yaw)
            if abs(yaw_err) > math.pi / 2:
                w_corr = max(-self._c_wex, min(self._c_wex, -self._c_kpy * yaw_err))
                lin, ang = self._c_smooth(0.0, w_corr)
            else:
                w_corr = max(-self._c_wex, min(self._c_wex, -self._c_kpy * yaw_err))
                lin, ang = self._c_smooth(self._c_vex, w_corr)
            self._publish(lin, ang)
            return

        # ---- entry bookkeeping (CENTERING only) ----
        if self._corr_sub == self.CORR_CENTERING:
            if left_vis and right_vis:
                self._corr_ent_cnt += 1
                if self._corr_ent_cnt >= self._c_nent:
                    self._corr_entered = True
            else:
                self._corr_ent_cnt = max(0, self._corr_ent_cnt - 1)

        # ---- open-area exit detection (CENTERING only) ----
        # Do not trigger exit while TURNING.  A corner can look open before the
        # robot is aligned with the next straight section.
        no_side_walls = (not left_vis and not right_vis)
        exit_candidate = (
            self._corr_entered
            and self._corr_sub == self.CORR_CENTERING
            and front_open
            and (self._corr_yfar or no_side_walls)
        )
        if exit_candidate:
            self._corr_exc_cnt += 1
            if self._corr_exc_cnt >= self._c_nexf:
                self.get_logger().info(
                    f'[CORR] Open/yellow area → EXIT_APPROACH '
                    f'(F={front:.2f} L={left:.2f} R={right:.2f} yellow={self._corr_yfar})')
                self._corr_exit_yaw = self._odom_yaw
                self._corr_exit_t   = now
                self._corr_sub      = self.CORR_EXIT_APPROACH
                self._c_prev_lin    = 0.0
                self._c_prev_ang    = 0.0
                return
        elif self._corr_sub == self.CORR_CENTERING:
            self._corr_exc_cnt = 0

        # ---- yellow confirmed ----
        if self._corr_ycnt >= self._c_tconf and self._corr_sub not in (
                self.CORR_EXIT_LOCK, self.CORR_EXIT_APPROACH):
            self.get_logger().info(
                f'[CORR] Yellow confirmed area={self._corr_yarea} → EXIT_LOCK')
            self._corr_sub   = self.CORR_EXIT_LOCK
            self._c_prev_lin = 0.0
            self._c_prev_ang = 0.0
            return

        # ---- anti-stuck ----
        if self._stuck_t is None:
            self._reset_stuck()
        if self._corr_sub != self.CORR_RECOVERY and self._is_stuck():
            self.get_logger().warn('[CORR] Stuck → RECOVERY')
            self._corr_sub   = self.CORR_RECOVERY
            self._corr_rec_t = now
            self._c_prev_lin = 0.0
            self._c_prev_ang = 0.0

        # ---- FSM ----
        if self._corr_sub == self.CORR_CENTERING:
            if math.isfinite(front) and front < self._c_turn_d and not yellow_partial:
                wl = wide_left  if math.isfinite(wide_left)  else 5.0
                wr = wide_right if math.isfinite(wide_right) else 5.0
                self._corr_tdir = 1 if wl >= wr else -1
                self.get_logger().info(
                    f'[CORR] anticipated bend → {"L" if self._corr_tdir > 0 else "R"} '
                    f'(F={front:.2f} FL={front_left:.2f} FR={front_right:.2f} '
                    f'wl={wl:.2f} wr={wr:.2f})')
                self._corr_turn_t = now
                self._corr_turn_yaw = self._odom_yaw
                self._corr_sub   = self.CORR_TURNING
                self._c_prev_lin = 0.0
                self._c_prev_ang = 0.0
                self._publish(0.0, self._c_wt * self._corr_tdir)
            elif yellow_partial:
                lin, ang = self._c_smooth(self._c_vs, 0.0)
                self._publish(lin, ang)
            elif left_vis and right_vis:
                error  = right - left
                w_raw  = max(-self._c_mw, min(self._c_mw, -self._c_kp * error))
                v_raw  = self._c_dyn_lin(front, w_raw)
                if math.isfinite(front) and front < self._c_slow_d:
                    v_raw = min(v_raw, self._c_vs)
                lin, ang = self._c_smooth(v_raw, w_raw)
                self._publish(lin, ang)
            elif left_vis and not right_vis:
                err   = self._c_dwdes - left
                w_raw = max(-self._c_swmax, min(self._c_swmax, -self._c_kpsw * err))
                v_raw = self._c_dyn_lin(front, w_raw)
                if math.isfinite(front) and front < self._c_slow_d:
                    v_raw = min(v_raw, self._c_vs)
                lin, ang = self._c_smooth(v_raw, w_raw)
                self._publish(lin, ang)
            elif right_vis and not left_vis:
                err   = right - self._c_dwdes
                w_raw = max(-self._c_swmax, min(self._c_swmax, -self._c_kpsw * err))
                v_raw = self._c_dyn_lin(front, w_raw)
                if math.isfinite(front) and front < self._c_slow_d:
                    v_raw = min(v_raw, self._c_vs)
                lin, ang = self._c_smooth(v_raw, w_raw)
                self._publish(lin, ang)
            else:
                lin, ang = self._c_smooth(self._c_vs, 0.0)
                self._publish(lin, ang)

        elif self._corr_sub == self.CORR_TURNING:
            turn_elapsed = now - self._corr_turn_t
            yaw_delta = abs(self._norm_angle(self._odom_yaw - self._corr_turn_yaw))
            # Do not leave a turn just because front is briefly free.
            stable_corridor = False
            yaw_turn_done = (turn_elapsed >= self._c_tmin and yaw_delta >= self._c_tang)
            timeout_done = (turn_elapsed > self._c_tmax and front > self._c_emerg)
            if yaw_turn_done or stable_corridor or timeout_done:
                reason = 'YAW_DONE' if yaw_turn_done else 'STABLE_CORRIDOR' if stable_corridor else 'TURN_TIMEOUT'
                self.get_logger().info(
                    f'[CORR] turn done → CENTERING reason={reason} F={front:.2f} '
                    f'yaw_delta={yaw_delta:.2f} elapsed={turn_elapsed:.1f}s')
                self._corr_sub   = self.CORR_CENTERING
                self._c_prev_lin = 0.0
                self._c_prev_ang = 0.0
                self._reset_stuck()
            else:
                lin, ang = self._c_smooth(self._c_vtl, self._c_wt * self._corr_tdir)
                self._publish(lin, ang)

        elif self._corr_sub == self.CORR_RECOVERY:
            elapsed = now - self._corr_rec_t
            if elapsed < 0.6:
                lin, ang = self._c_smooth(-self._c_vr, 0.0)
                self._publish(lin, ang)
            elif elapsed < 1.4:
                wl = wide_left  if math.isfinite(wide_left)  else 5.0
                wr = wide_right if math.isfinite(wide_right) else 5.0
                d  = 1 if wl >= wr else -1
                lin, ang = self._c_smooth(0.0, self._c_wt * d)
                self._publish(lin, ang)
            else:
                self.get_logger().info('[CORR] recovery → CENTERING')
                self._corr_sub   = self.CORR_CENTERING
                self._c_prev_lin = 0.0
                self._c_prev_ang = 0.0
                self._reset_stuck()

    def _image_cb(self, msg: CompressedImage) -> None:
        if self._state == self.DONE:
            return

        buf   = np.frombuffer(msg.data, dtype=np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if image is None:
            return
        h, w = image.shape[:2]

        if self._state == self.CORRIDOR:
            roi = image[int(self._c_troi * h):, :]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            lo  = np.array([self._c_thlo, self._c_tslo, self._c_tvlo], np.uint8)
            hi  = np.array([self._c_thhi, 255,           255         ], np.uint8)
            mask = cv2.inRange(hsv, lo, hi)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
            self._corr_yarea = int(cv2.countNonZero(mask))
            self._corr_yfar = self._corr_yarea >= self._c_tfar
            self._corr_ycy = 0.0
            self._corr_ycx = 0.5
            self._corr_ybw = 0.0
            if self._corr_yarea > 0:
                M = cv2.moments(mask)
                if M['m00'] > 0:
                    cx_roi = M['m10'] / M['m00']
                    cy_roi = M['m01'] / M['m00']
                    self._corr_ycx = cx_roi / w
                    self._corr_ycy = (int(self._c_troi * h) + cy_roi) / h
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    _, _, bw, _ = cv2.boundingRect(max(contours, key=cv2.contourArea))
                    self._corr_ybw = bw / w

            close = (self._corr_yarea >= self._c_tarea
                     and self._corr_ycy >= self._c_tcy
                     and self._corr_ybw >= self._c_tbw)
            if close:
                self._corr_ycnt += 1
            else:
                self._corr_ycnt = max(0, self._corr_ycnt - 1)

            # Near-field ring gate for final target centering.
            yb0 = int(self._c_near_roi_start * h)
            half_w = int(0.5 * self._c_near_center_width * w)
            x0 = max(0, w // 2 - half_w)
            x1 = min(w, w // 2 + half_w)
            near = image[yb0:h, x0:x1]
            self._corr_near_color = 'none'
            self._corr_near_ratios = {'yellow': 0.0, 'red': 0.0, 'blue': 0.0, 'black': 0.0, 'white': 0.0}
            if near.size > 0:
                hsv_near = cv2.cvtColor(near, cv2.COLOR_BGR2HSV)
                total = float(near.shape[0] * near.shape[1])
                yellow_mask = cv2.inRange(
                    hsv_near,
                    np.array([self._c_thlo, self._c_tslo, self._c_tvlo], np.uint8),
                    np.array([self._c_thhi, 255, 255], np.uint8),
                )
                red_mask1 = cv2.inRange(hsv_near, np.array([0, 70, 60], np.uint8), np.array([10, 255, 255], np.uint8))
                red_mask2 = cv2.inRange(hsv_near, np.array([170, 70, 60], np.uint8), np.array([180, 255, 255], np.uint8))
                red_mask = cv2.bitwise_or(red_mask1, red_mask2)
                blue_mask = cv2.inRange(hsv_near, np.array([82, 35, 55], np.uint8), np.array([115, 255, 255], np.uint8))
                black_mask = cv2.inRange(hsv_near, np.array([0, 0, 0], np.uint8), np.array([180, 90, 75], np.uint8))
                white_mask = cv2.inRange(hsv_near, np.array([0, 0, 145], np.uint8), np.array([180, 70, 255], np.uint8))
                masks = {'yellow': yellow_mask, 'red': red_mask, 'blue': blue_mask, 'black': black_mask, 'white': white_mask}
                counts = {}
                for name, m in masks.items():
                    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self._kernel)
                    counts[name] = int(cv2.countNonZero(m))
                    self._corr_near_ratios[name] = counts[name] / total
                best = max(counts, key=counts.get)
                if self._corr_near_ratios[best] >= self._c_ring_min_ratio:
                    self._corr_near_color = best

            if self._corr_final_offset_active:
                # Use the narrow bottom-centre ROI. Do not stop on ring colour;
                # ring colour only starts the odometry correction for the front camera.
                if self._corr_near_ratios['yellow'] >= self._c_yellow_near_ratio:
                    self._corr_ring_yellow_seen_count += 1
                    if self._corr_ring_yellow_seen_count >= self._c_ring_confirm:
                        self._corr_ring_yellow_seen = True
                else:
                    self._corr_ring_yellow_seen_count = max(0, self._corr_ring_yellow_seen_count - 1)
                red_after_yellow = (
                    self._corr_ring_yellow_seen
                    and self._corr_near_ratios['red'] >= self._c_red_after_yellow_ratio
                    and self._corr_near_ratios['red'] >= 0.80 * max(self._corr_near_ratios['yellow'], 1e-6)
                )
                if red_after_yellow:
                    self._corr_ring_after_yellow_count += 1
                else:
                    self._corr_ring_after_yellow_count = max(0, self._corr_ring_after_yellow_count - 1)

                # Main marker: yellow then red again. Fallback: very strong stable
                # yellow in the centre for longer than the normal confirmation.
                start_extra = (
                    self._corr_ring_after_yellow_count >= self._c_ring_confirm
                    or (self._corr_ring_yellow_seen_count >= 2 * self._c_ring_confirm
                        and self._corr_near_ratios['yellow'] >= 0.30)
                )
                if start_extra and not self._corr_center_extra_active and self._odom_x is not None and self._odom_y is not None:
                    self._corr_center_extra_active = True
                    self._corr_center_extra_x = self._odom_x
                    self._corr_center_extra_y = self._odom_y
                    self._corr_center_extra_reason = 'strong_yellow' if self._corr_ring_after_yellow_count < self._c_ring_confirm else 'yellow_then_red'
                    self.get_logger().info(
                        f'[CORR] Camera marker reached ({self._corr_center_extra_reason}) -> odom extra '
                        f'{self._c_center_extra_dist:.3f} m')

            # transition to EXIT_LOCK only on CLOSE target, not far yellow paint
            if (self._corr_ycnt >= self._c_tconf
                    and self._corr_sub not in (self.CORR_EXIT_LOCK,)):
                self.get_logger().info(
                    f'[CORR] Yellow CLOSE confirmed area={self._corr_yarea} '
                    f'cy={self._corr_ycy:.2f} bw={self._corr_ybw:.2f} → EXIT_LOCK')
                self._corr_sub = self.CORR_EXIT_LOCK

        elif self._state == self.LINE_FOLLOW:
            self._run_line_follow(image, h, w)
        elif self._state == self.BALL_PUSH:
            self._run_ball_push(image, h, w)

    def _run_line_follow(self, image, h: int, w: int) -> None:
        if self._obstacle:
            self._publish(0.0, 0.0)
            return
        roi_y = int(h * (1.0 - self._l_roi))
        roi   = image[roi_y:, :]
        hsv   = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lo    = np.array([self._lh_lo, self._ls_lo, self._lv_lo], dtype=np.uint8)
        hi    = np.array([self._lh_hi, 255,         255         ], dtype=np.uint8)
        mask  = cv2.inRange(hsv, lo, hi)
        mask  = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self._kernel)
        mask  = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)
        M    = cv2.moments(mask)
        area = M['m00']
        if area > self._l_mina:
            self._line_seen_once    = True
            self._line_lost_counter = 0
            cx      = int(M['m10'] / area)
            error   = cx - w // 2
            angular = -self._l_k * float(error)
            self._publish(self._l_v, angular)
        else:
            if self._line_seen_once:
                self._line_lost_counter += 1
                if self._line_lost_counter >= self._lost_max:
                    self._transition(self.BALL_PUSH)
                    self._stop()
                    return
            self._publish(0.0, 0.25)

    def _run_ball_push(self, image, h: int, w: int) -> None:
        hsv  = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lo   = np.array([self._bh_lo, self._bs_lo, self._bv_lo], dtype=np.uint8)
        hi   = np.array([self._bh_hi, 255,         255         ], dtype=np.uint8)
        mask = cv2.inRange(hsv, lo, hi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area = cv2.contourArea(max(contours, key=cv2.contourArea)) \
               if contours else 0.0
        if area < self._b_mina:
            if self._ball_state != self.BALL_SEARCHING:
                self.get_logger().info('Ball lost → searching')
                self._ball_state = self.BALL_SEARCHING
            self._publish(0.0, self._b_wsrch)
            return
        M   = cv2.moments(mask)
        cx  = int(M['m10'] / M['m00']) if M['m00'] > 0 else w // 2
        err = cx - w // 2
        ang = -self._b_k * float(err)
        if area >= self._b_donea:
            self.get_logger().info('Ball in target – DONE!')
            self._transition(self.DONE)
            self._stop()
            return
        if area >= self._b_pusha:
            if self._ball_state != self.BALL_PUSHING:
                self.get_logger().info(f'PUSHING area={int(area)}')
                self._ball_state = self.BALL_PUSHING
            self._publish(self._b_push, ang)
        else:
            if self._ball_state != self.BALL_ALIGNING:
                self.get_logger().info(f'ALIGNING area={int(area)} err={err}')
                self._ball_state = self.BALL_ALIGNING
            lin = self._b_v if abs(err) < 80 else 0.05
            self._publish(lin, ang)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FullRunController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
