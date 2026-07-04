#!/usr/bin/env python3
"""
Challenge 1 corridor navigator - reactive controller.

Key design choice:
- No fixed 90° TURNING state for the S-shaped maze. The previous version got
  confused in successive bends because it treated a curve like a discrete turn.
- This version continuously steers toward the most open forward direction while
  using side-wall repulsion and corridor centering.
- Camera is used for final target approach: align on yellow, detect the target ring colours in a narrow bottom-centre ROI, then continue by odometry for the calibrated camera-to-base offset so the robot body reaches the yellow centre.
"""
import math
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, CompressedImage
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy


class CorridorNavigator(Node):
    RUNNING = 'RUNNING'
    EXIT_APPROACH = 'EXIT_APPROACH'
    EXIT_LOCK = 'EXIT_LOCK'
    DONE = 'DONE'
    RECOVERY = 'RECOVERY'

    def __init__(self):
        super().__init__('corridor_navigator')

        # Handoff control for main.launch.py. Default keeps challenge1.launch.py identical.
        self.declare_parameter('shutdown_on_done', False)

        # speeds
        self.declare_parameter('fast_speed', 0.24)
        self.declare_parameter('medium_speed', 0.16)
        self.declare_parameter('slow_speed', 0.075)
        self.declare_parameter('max_angular_speed', 1.10)
        self.declare_parameter('smoothing_alpha', 0.45)

        # lidar/navigation
        self.declare_parameter('wall_detect_max', 0.95)
        self.declare_parameter('desired_wall_dist', 0.34)
        self.declare_parameter('side_min_dist', 0.23)
        self.declare_parameter('emergency_stop_dist', 0.18)
        self.declare_parameter('front_slow_dist', 0.80)
        self.declare_parameter('front_curve_dist', 0.62)
        self.declare_parameter('exit_front_threshold', 1.10)
        self.declare_parameter('exit_side_threshold', 0.95)
        self.declare_parameter('exit_confirm_frames', 8)

        # gains
        self.declare_parameter('kp_center', 1.00)
        self.declare_parameter('kp_gap', 1.55)
        self.declare_parameter('kp_diag', 0.75)
        self.declare_parameter('kp_side_repulse', 1.90)
        self.declare_parameter('angular_deadband', 0.025)

        # recovery
        self.declare_parameter('stuck_time', 3.0)
        self.declare_parameter('stuck_dist', 0.018)
        self.declare_parameter('recovery_reverse_speed', 0.045)
        self.declare_parameter('recovery_turn_speed', 0.42)

        # exit/yellow target
        self.declare_parameter('exit_speed', 0.17)
        self.declare_parameter('exit_max_angular', 0.10)
        self.declare_parameter('exit_kp_yaw', 0.35)
        self.declare_parameter('yellow_align_kp', 1.15)
        self.declare_parameter('yellow_align_max_ang', 0.38)
        self.declare_parameter('yellow_approach_speed', 0.13)
        self.declare_parameter('yellow_capture_area', 8000)
        self.declare_parameter('yellow_capture_cy', 0.66)
        self.declare_parameter('yellow_capture_width', 0.22)
        self.declare_parameter('target_h_low', 18)
        self.declare_parameter('target_h_high', 38)
        self.declare_parameter('target_s_low', 80)
        self.declare_parameter('target_v_low', 80)
        self.declare_parameter('target_roi_start', 0.40)
        self.declare_parameter('yellow_far_area', 1500)
        self.declare_parameter('yellow_close_area', 12000)
        self.declare_parameter('yellow_close_cy', 0.70)
        self.declare_parameter('yellow_close_width', 0.35)
        self.declare_parameter('yellow_confirm', 6)
        # Camera is mounted at the front: do NOT stop as soon as yellow is close.
        # stable: continue slowly until the lower-centre camera ROI has seen yellow,
        # then sees the red ring again. At that moment the camera has passed the
        # yellow centre and the robot base is inside the yellow disk.
        self.declare_parameter('final_offset_time', 5.0)       # safety timeout only
        self.declare_parameter('final_offset_speed', 0.060)
        # stable: the camera is on the robot nose. After the camera sees the yellow
        # centre/ring transition, the base must still advance this measured
        # distance to put the robot centre in the yellow disk.
        self.declare_parameter('final_center_extra_dist', 0.115)
        self.declare_parameter('target_near_roi_start', 0.72)  # bottom part of image
        self.declare_parameter('target_near_center_width', 0.32)
        self.declare_parameter('target_ring_min_ratio', 0.030)
        self.declare_parameter('target_yellow_near_ratio', 0.110)
        self.declare_parameter('target_red_after_yellow_ratio', 0.045)
        self.declare_parameter('target_ring_confirm', 4)

        self.v_fast = float(self.get_parameter('fast_speed').value)
        self.v_med = float(self.get_parameter('medium_speed').value)
        self.v_slow = float(self.get_parameter('slow_speed').value)
        self.max_w = float(self.get_parameter('max_angular_speed').value)
        self.alpha = float(self.get_parameter('smoothing_alpha').value)
        self.wall_max = float(self.get_parameter('wall_detect_max').value)
        self.wall_des = float(self.get_parameter('desired_wall_dist').value)
        self.side_min = float(self.get_parameter('side_min_dist').value)
        self.d_emerg = float(self.get_parameter('emergency_stop_dist').value)
        self.d_slow = float(self.get_parameter('front_slow_dist').value)
        self.d_curve = float(self.get_parameter('front_curve_dist').value)
        self.d_exit_front = float(self.get_parameter('exit_front_threshold').value)
        self.d_exit_side = float(self.get_parameter('exit_side_threshold').value)
        self.exit_confirm_frames = int(self.get_parameter('exit_confirm_frames').value)
        self.k_center = float(self.get_parameter('kp_center').value)
        self.k_gap = float(self.get_parameter('kp_gap').value)
        self.k_diag = float(self.get_parameter('kp_diag').value)
        self.k_repulse = float(self.get_parameter('kp_side_repulse').value)
        self.deadband = float(self.get_parameter('angular_deadband').value)
        self.stuck_time = float(self.get_parameter('stuck_time').value)
        self.stuck_dist = float(self.get_parameter('stuck_dist').value)
        self.v_rev = float(self.get_parameter('recovery_reverse_speed').value)
        self.w_rec = float(self.get_parameter('recovery_turn_speed').value)
        self.v_exit = float(self.get_parameter('exit_speed').value)
        self.w_exit = float(self.get_parameter('exit_max_angular').value)
        self.k_exit_yaw = float(self.get_parameter('exit_kp_yaw').value)
        self.k_yellow_align = float(self.get_parameter('yellow_align_kp').value)
        self.w_yellow_max = float(self.get_parameter('yellow_align_max_ang').value)
        self.v_yellow = float(self.get_parameter('yellow_approach_speed').value)
        self.y_capture_area = int(self.get_parameter('yellow_capture_area').value)
        self.y_capture_cy = float(self.get_parameter('yellow_capture_cy').value)
        self.y_capture_bw = float(self.get_parameter('yellow_capture_width').value)
        self.h_low = int(self.get_parameter('target_h_low').value)
        self.h_high = int(self.get_parameter('target_h_high').value)
        self.s_low = int(self.get_parameter('target_s_low').value)
        self.v_low = int(self.get_parameter('target_v_low').value)
        self.roi_start = float(self.get_parameter('target_roi_start').value)
        self.y_far = int(self.get_parameter('yellow_far_area').value)
        self.y_close = int(self.get_parameter('yellow_close_area').value)
        self.y_cy = float(self.get_parameter('yellow_close_cy').value)
        self.y_bw = float(self.get_parameter('yellow_close_width').value)
        self.y_confirm = int(self.get_parameter('yellow_confirm').value)
        self.final_offset_time = float(self.get_parameter('final_offset_time').value)
        self.final_offset_speed = float(self.get_parameter('final_offset_speed').value)
        self.final_center_extra_dist = float(self.get_parameter('final_center_extra_dist').value)
        self.near_roi_start = float(self.get_parameter('target_near_roi_start').value)
        self.near_center_width = float(self.get_parameter('target_near_center_width').value)
        self.ring_min_ratio = float(self.get_parameter('target_ring_min_ratio').value)
        self.yellow_near_ratio_min = float(self.get_parameter('target_yellow_near_ratio').value)
        self.red_after_yellow_ratio_min = float(self.get_parameter('target_red_after_yellow_ratio').value)
        self.ring_confirm = int(self.get_parameter('target_ring_confirm').value)
        self.shutdown_on_done = bool(self.get_parameter('shutdown_on_done').value)
        self.done_published = False
        self.shutdown_requested = False
        self.done_timer = None

        self.state = self.RUNNING
        self.prev_v = 0.0
        self.prev_w = 0.0
        self.exit_count = 0
        self.exit_yaw = 0.0
        self.recovery_t0 = 0.0
        self.recovery_dir = 1
        self.log_t = 0.0

        # odom/stuck
        self.x = None
        self.y = None
        self.yaw = 0.0
        self.ref_x = None
        self.ref_y = None
        self.ref_t = None

        # yellow detection
        self.img_count = 0
        self.yellow_area = 0
        self.yellow_cy_ratio = 0.0
        self.yellow_cx_ratio = 0.5
        self.yellow_bw_ratio = 0.0
        self.yellow_seen_far = False
        self.yellow_close = False
        self.yellow_close_count = 0
        self.final_offset_active = False
        self.final_offset_start = 0.0
        # Target ring tracking in the lower-centre camera ROI.
        self.near_color = 'none'
        self.near_ratios = {'yellow': 0.0, 'red': 0.0, 'blue': 0.0, 'black': 0.0, 'white': 0.0}
        self.ring_yellow_seen = False
        self.ring_yellow_seen_count = 0
        self.ring_after_yellow_count = 0
        self.center_extra_active = False
        self.center_extra_x = None
        self.center_extra_y = None
        self.center_extra_reason = 'none'
        self.kernel = np.ones((5, 5), np.uint8)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)
        done_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.done_pub = self.create_publisher(Bool, '/challenge1_done', done_qos)
        self.create_subscription(LaserScan, 'scan', self.scan_cb, qos)
        self.create_subscription(CompressedImage, '/image_raw/compressed', self.image_cb, qos)
        self.create_subscription(Odometry, 'odom', self.odom_cb, 10)
        self.get_logger().info('CorridorNavigator stable: stable challenge1 preserved + reliable /challenge1_done handoff')

    @staticmethod
    def norm_angle(a):
        return math.atan2(math.sin(a), math.cos(a))

    @staticmethod
    def yaw_from_quat(q):
        return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                          1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    @staticmethod
    def finite(v):
        return math.isfinite(v) and v > 0.04

    @staticmethod
    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    def odom_cb(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.yaw = self.yaw_from_quat(msg.pose.pose.orientation)

    def reset_filter(self):
        self.prev_v = 0.0
        self.prev_w = 0.0

    def publish(self, v, w, smooth=True):
        w = self.clamp(w, -self.max_w, self.max_w)
        if abs(w) < self.deadband:
            w = 0.0
        if smooth:
            v = self.alpha * v + (1.0 - self.alpha) * self.prev_v
            w = self.alpha * w + (1.0 - self.alpha) * self.prev_w
            self.prev_v = v
            self.prev_w = w
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.pub.publish(msg)

    def stop(self):
        self.reset_filter()
        self.publish(0.0, 0.0, smooth=False)

    def _publish_done_signal(self):
        msg = Bool()
        msg.data = True
        self.done_pub.publish(msg)

    def notify_done(self):
        # Called only when the stable target-centre logic has actually finished.
        # It repeatedly publishes a latched /challenge1_done signal so the line
        # follower cannot miss the handoff, then this process exits in main.launch.py.
        if self.done_published:
            return
        self.state = self.DONE
        self.stop()
        self._publish_done_signal()
        self.done_published = True
        self.done_timer = self.create_timer(0.10, self._publish_done_signal)
        self.get_logger().info('Challenge 1 DONE -> published /challenge1_done repeatedly for handoff')
        if self.shutdown_on_done and not self.shutdown_requested:
            self.shutdown_requested = True
            self.create_timer(0.90, self._shutdown_after_done)

    def _shutdown_after_done(self):
        self.get_logger().info('Challenge 1 node exits; challenge 2 is already armed and should now drive')
        if rclpy.ok():
            rclpy.shutdown()

    def range_sector(self, msg, deg_center, half_width=10):
        vals = []
        c = math.radians(deg_center)
        hw = math.radians(half_width)
        for i, r in enumerate(msg.ranges):
            if not math.isfinite(r) or r <= 0.05:
                continue
            a = msg.angle_min + i * msg.angle_increment
            d = abs(math.atan2(math.sin(a - c), math.cos(a - c)))
            if d <= hw:
                vals.append(float(r))
        if not vals:
            return float('inf')
        vals.sort()
        return vals[len(vals) // 2]

    def gap_angle(self, msg, amin_deg=-95, amax_deg=95, step_deg=5):
        """Pick a forward angle with high clearance but a bias toward straight.
        This handles successive curves better than fixed TURN_LEFT/TURN_RIGHT.
        """
        best_score = -1e9
        best_deg = 0.0
        for deg in range(amin_deg, amax_deg + 1, step_deg):
            r = self.range_sector(msg, deg, 5)
            rr = 2.2 if not math.isfinite(r) else min(r, 2.2)
            # forward preference prevents unnecessary 180°/side-looking choices
            forward = math.cos(math.radians(deg))
            # reject very close directions
            close_penalty = 2.5 * max(0.0, 0.30 - rr)
            score = rr + 0.75 * forward - close_penalty
            if score > best_score:
                best_score = score
                best_deg = deg
        return math.radians(best_deg), best_score

    def reset_stuck_ref(self):
        self.ref_x = self.x
        self.ref_y = self.y
        self.ref_t = self.get_clock().now().nanoseconds * 1e-9

    def is_stuck(self, now, front):
        if self.x is None or self.ref_x is None or self.ref_t is None:
            return False
        if self.state != self.RUNNING:
            return False
        if self.finite(front) and front < self.d_emerg + 0.04:
            return False
        dist = math.hypot(self.x - self.ref_x, self.y - self.ref_y)
        if dist > self.stuck_dist:
            self.reset_stuck_ref()
            return False
        return (now - self.ref_t) > self.stuck_time

    def image_cb(self, msg):
        if self.state == self.DONE:
            return
        img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return

        self.img_count += 1
        h, w = img.shape[:2]

        # ------------------------------------------------------------------
        # 1) Yellow target detection over the lower image: used for long-range
        #    target acquisition and centering.
        # ------------------------------------------------------------------
        y0 = int(self.roi_start * h)
        roi = img[y0:, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array([self.h_low, self.s_low, self.v_low], np.uint8),
            np.array([self.h_high, 255, 255], np.uint8),
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        area = int(cv2.countNonZero(mask))

        self.yellow_area = area
        self.yellow_cy_ratio = 0.0
        self.yellow_cx_ratio = 0.5
        self.yellow_bw_ratio = 0.0
        if area > 0:
            M = cv2.moments(mask)
            if M['m00'] > 0:
                cx_roi = M['m10'] / M['m00']
                cy_roi = M['m01'] / M['m00']
                self.yellow_cx_ratio = cx_roi / w
                self.yellow_cy_ratio = (y0 + cy_roi) / h
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                _, _, bw, _ = cv2.boundingRect(max(contours, key=cv2.contourArea))
                self.yellow_bw_ratio = bw / w

        self.yellow_seen_far = area >= self.y_far
        self.yellow_close = (
            ((area >= self.y_close and self.yellow_cy_ratio >= self.y_cy and self.yellow_bw_ratio >= self.y_bw)
             or (area >= self.y_capture_area and self.yellow_cy_ratio >= self.y_capture_cy and self.yellow_bw_ratio >= self.y_capture_bw))
        )
        if self.yellow_close:
            self.yellow_close_count += 1
        else:
            self.yellow_close_count = max(0, self.yellow_close_count - 1)

        # ------------------------------------------------------------------
        # 2) Near-field ring detection in the bottom-centre image.
        #    The target sequence while driving forward is:
        #       white -> black -> blue -> red -> yellow -> red -> blue ...
        #    Because the camera is in front of the robot, stopping at the first
        #    yellow frame is too early. We wait until yellow has been in the
        #    near-field ROI, then the red ring appears again. That means the
        #    camera is slightly past the centre and the robot base is inside
        #    the yellow disk.
        # ------------------------------------------------------------------
        yb0 = int(self.near_roi_start * h)
        half_w = int(0.5 * self.near_center_width * w)
        x0 = max(0, w // 2 - half_w)
        x1 = min(w, w // 2 + half_w)
        near = img[yb0:h, x0:x1]
        self.near_color = 'none'
        self.near_ratios = {'yellow': 0.0, 'red': 0.0, 'blue': 0.0, 'black': 0.0, 'white': 0.0}

        if near.size > 0:
            hsv_near = cv2.cvtColor(near, cv2.COLOR_BGR2HSV)
            total = float(near.shape[0] * near.shape[1])

            yellow_mask = cv2.inRange(
                hsv_near,
                np.array([self.h_low, self.s_low, self.v_low], np.uint8),
                np.array([self.h_high, 255, 255], np.uint8),
            )
            red_mask1 = cv2.inRange(hsv_near, np.array([0, 70, 60], np.uint8), np.array([10, 255, 255], np.uint8))
            red_mask2 = cv2.inRange(hsv_near, np.array([170, 70, 60], np.uint8), np.array([180, 255, 255], np.uint8))
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            blue_mask = cv2.inRange(hsv_near, np.array([82, 35, 55], np.uint8), np.array([115, 255, 255], np.uint8))
            black_mask = cv2.inRange(hsv_near, np.array([0, 0, 0], np.uint8), np.array([180, 90, 75], np.uint8))
            white_mask = cv2.inRange(hsv_near, np.array([0, 0, 145], np.uint8), np.array([180, 70, 255], np.uint8))

            masks = {
                'yellow': yellow_mask,
                'red': red_mask,
                'blue': blue_mask,
                'black': black_mask,
                'white': white_mask,
            }
            counts = {}
            for name, m in masks.items():
                m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self.kernel)
                counts[name] = int(cv2.countNonZero(m))
                self.near_ratios[name] = counts[name] / total

            best = max(counts, key=counts.get)
            if self.near_ratios[best] >= self.ring_min_ratio:
                self.near_color = best

        # Start the final colour-sequence gate only when the yellow centre is
        # already close. Before that, red/blue/white rings are just approach info.
        if self.yellow_close_count >= self.y_confirm and not self.final_offset_active:
            self.get_logger().info('Yellow close confirmed -> RING_GATE final approach')
            self.final_offset_active = True
            self.final_offset_start = self.get_clock().now().nanoseconds * 1e-9
            self.ring_yellow_seen = False
            self.ring_yellow_seen_count = 0
            self.ring_after_yellow_count = 0
            self.center_extra_active = False
            self.center_extra_x = None
            self.center_extra_y = None
            self.center_extra_reason = 'none'
            self.reset_filter()

        if self.final_offset_active:
            # Narrow bottom-centre ROI only: this avoids false red detections on the
            # side of the target. First wait until yellow is really under the camera.
            if self.near_ratios['yellow'] >= self.yellow_near_ratio_min:
                self.ring_yellow_seen_count += 1
                if self.ring_yellow_seen_count >= self.ring_confirm:
                    self.ring_yellow_seen = True
            else:
                self.ring_yellow_seen_count = max(0, self.ring_yellow_seen_count - 1)

            # Red after yellow is still useful as a crossing marker, but in stable it
            # DOES NOT stop the robot. It only starts the calibrated odometry push.
            red_after_yellow = (
                self.ring_yellow_seen
                and self.near_ratios['red'] >= self.red_after_yellow_ratio_min
                and self.near_ratios['red'] >= 0.80 * max(self.near_ratios['yellow'], 1e-6)
            )
            if red_after_yellow:
                self.ring_after_yellow_count += 1
            else:
                self.ring_after_yellow_count = max(0, self.ring_after_yellow_count - 1)

            # Start the extra odometry distance either when yellow is dominant in
            # the narrow near ROI, or when the camera has crossed yellow then red.
            # This compensates the front-mounted camera: the robot base is still
            # behind the point seen by the camera.
            # Main marker: yellow then red again. Fallback: very strong stable
            # yellow in the centre for longer than the normal confirmation.
            start_extra = (
                self.ring_after_yellow_count >= self.ring_confirm
                or (self.ring_yellow_seen_count >= 2 * self.ring_confirm
                    and self.near_ratios['yellow'] >= 0.30)
            )
            if start_extra and not self.center_extra_active and self.x is not None and self.y is not None:
                self.center_extra_active = True
                self.center_extra_x = self.x
                self.center_extra_y = self.y
                self.center_extra_reason = 'strong_yellow' if self.ring_after_yellow_count < self.ring_confirm else 'yellow_then_red'
                self.get_logger().info(
                    f'Camera target marker reached ({self.center_extra_reason}) -> odom extra '
                    f'{self.final_center_extra_dist:.3f} m before stop'
                )

        if self.img_count <= 3 or self.img_count % 15 == 0 or self.final_offset_active:
            ratios = self.near_ratios
            self.get_logger().info(
                f'[CAM] #{self.img_count} area={area} cx={self.yellow_cx_ratio:.2f} '
                f'cy={self.yellow_cy_ratio:.2f} bw={self.yellow_bw_ratio:.2f} '
                f'far={self.yellow_seen_far} close={self.yellow_close} cnt={self.yellow_close_count} '
                f'near={self.near_color} Y={ratios["yellow"]:.2f} R={ratios["red"]:.2f} '
                f'B={ratios["blue"]:.2f} K={ratios["black"]:.2f} W={ratios["white"]:.2f} '
                f'sawY={self.ring_yellow_seen} afterR={self.ring_after_yellow_count} extra={self.center_extra_active}'
            )

    def scan_cb(self, msg):
        if self.state == self.DONE:
            return
        now = self.get_clock().now().nanoseconds * 1e-9

        # Final target gate after close yellow confirmation.
        # Drive very slowly while the image callback watches the colour sequence.
        # Stop is decided by camera rings, not by a blind timed offset.
        if self.final_offset_active:
            elapsed = now - self.final_offset_start

            # Phase 2: after the camera has found the centre marker, use odometry
            # to move the robot body into the centre of the yellow disk.
            if self.center_extra_active and self.center_extra_x is not None and self.x is not None:
                dist = math.hypot(self.x - self.center_extra_x, self.y - self.center_extra_y)
                if dist >= self.final_center_extra_dist:
                    self.get_logger().info(
                        f'Final odom extra done ({dist:.3f} m) -> DONE at yellow centre'
                    )
                    self.notify_done()
                    return
                err_x = self.yellow_cx_ratio - 0.5 if self.yellow_seen_far else 0.0
                w = self.clamp(-0.55 * self.k_yellow_align * err_x, -0.12, 0.12)
                self.publish(self.final_offset_speed, w, smooth=False)
                return

            # Timeout now starts the odometry push instead of stopping early.
            if elapsed > self.final_offset_time:
                if self.x is not None and self.y is not None:
                    self.get_logger().warn('RING_GATE timeout -> start odom extra fallback')
                    self.center_extra_active = True
                    self.center_extra_x = self.x
                    self.center_extra_y = self.y
                    self.center_extra_reason = 'timeout'
                    return
                self.get_logger().warn('RING_GATE timeout without odom -> DONE')
                self.notify_done()
                return

            # Phase 1: approach slowly while the camera detects the yellow centre.
            err_x = self.yellow_cx_ratio - 0.5 if self.yellow_seen_far else 0.0
            w = self.clamp(-self.k_yellow_align * err_x, -0.20, 0.20)
            self.publish(self.final_offset_speed, w, smooth=False)
            return

        front = self.range_sector(msg, 0, 16)
        fl = self.range_sector(msg, 35, 14)
        fr = self.range_sector(msg, -35, 14)
        left = self.range_sector(msg, 90, 18)
        right = self.range_sector(msg, -90, 18)
        wl = self.range_sector(msg, 70, 45)
        wr = self.range_sector(msg, -70, 45)

        left_vis = self.finite(left) and left < self.wall_max
        right_vis = self.finite(right) and right < self.wall_max
        front_open = (not self.finite(front)) or front > self.d_exit_front
        side_open = ((not left_vis) or left > self.d_exit_side or
                     (not right_vis) or right > self.d_exit_side)

        if self.ref_t is None:
            self.reset_stuck_ref()

        if now - self.log_t > 0.5:
            self.log_t = now
            self.get_logger().info(
                f'[{self.state}] F={front:.2f} FL={fl:.2f} FR={fr:.2f} '
                f'L={left:.2f} R={right:.2f} WL={wl:.2f} WR={wr:.2f} '
                f'yaw={self.yaw:.2f} exit_cnt={self.exit_count} '
                f'yel={self.yellow_area} far={self.yellow_seen_far} '
                f'close={self.yellow_close} yc={self.yellow_close_count}'
            )

        if self.state == self.EXIT_APPROACH:
            if self.finite(front) and front < self.d_emerg:
                self.get_logger().warn('Obstacle in EXIT_APPROACH -> stop')
                self.stop()
                return
            # Final approach: once yellow is visible, steer toward its centroid.
            # The LIDAR has already done the maze; the camera now recenters the robot
            # so it does not pass beside the painted target.
            if self.yellow_seen_far:
                err_x = self.yellow_cx_ratio - 0.5
                w = self.clamp(-self.k_yellow_align * err_x, -self.w_yellow_max, self.w_yellow_max)
                v = self.v_yellow if abs(err_x) < 0.32 else min(self.v_yellow, 0.075)
                self.publish(v, w, smooth=False)
            else:
                yaw_err = self.norm_angle(self.yaw - self.exit_yaw)
                w = self.clamp(-self.k_exit_yaw * yaw_err, -self.w_exit, self.w_exit)
                self.publish(self.v_exit, w, smooth=False)
            return

        if self.state == self.RECOVERY:
            dt = now - self.recovery_t0
            if dt < 0.55:
                self.publish(-self.v_rev, 0.0)
            elif dt < 1.25:
                self.publish(0.0, self.recovery_dir * self.w_rec)
            else:
                self.get_logger().info('Recovery done -> RUNNING')
                self.state = self.RUNNING
                self.reset_filter()
                self.reset_stuck_ref()
            return

        # If the final painted target is visible, switch immediately to camera-guided approach.
        # Do not wait for side walls to disappear, otherwise the robot can pass beside it.
        if self.yellow_seen_far and (not self.finite(front) or front > self.d_emerg + 0.12):
            self.exit_count += 1
            if self.exit_count >= 2:
                self.get_logger().info('Yellow seen ahead -> EXIT_APPROACH camera-guided')
                self.state = self.EXIT_APPROACH
                self.exit_yaw = self.yaw
                self.reset_filter()
                return

        # exit: only trust open-area exit if yellow is seen or both side walls are mostly gone
        no_side_walls = (not left_vis and not right_vis)
        if front_open and (self.yellow_seen_far or no_side_walls) and side_open:
            self.exit_count += 1
        else:
            self.exit_count = max(0, self.exit_count - 1)
        if self.exit_count >= self.exit_confirm_frames:
            self.get_logger().info('Open/yellow final area -> EXIT_APPROACH')
            self.state = self.EXIT_APPROACH
            self.exit_yaw = self.yaw
            self.reset_filter()
            return

        if self.is_stuck(now, front):
            self.get_logger().warn('Stuck -> RECOVERY')
            # turn toward the larger side sector
            wr_eff = 2.2 if not self.finite(wr) else wr
            wl_eff = 2.2 if not self.finite(wl) else wl
            self.recovery_dir = 1 if wl_eff >= wr_eff else -1
            self.recovery_t0 = now
            self.state = self.RECOVERY
            self.reset_filter()
            return

        # Pure reactive corridor control.
        # 1) steer toward largest forward gap, strongly when the front is near.
        gap_ang, _ = self.gap_angle(msg)
        front_eff = 2.2 if not self.finite(front) else front
        curve_factor = self.clamp((self.d_slow - front_eff) / max(0.01, self.d_slow - self.d_emerg), 0.0, 1.0)
        w_gap = self.k_gap * gap_ang * (0.30 + 0.85 * curve_factor)

        # 2) diagonal correction: if obstacle is closer front-left, turn right, and vice versa.
        fl_eff = 2.2 if not self.finite(fl) else min(fl, 2.2)
        fr_eff = 2.2 if not self.finite(fr) else min(fr, 2.2)
        w_diag = self.k_diag * (fl_eff - fr_eff)

        # 3) side centering and hard repulsion.
        w_center = 0.0
        if left_vis and right_vis:
            # right too close => left-right positive => turn left; left too close => turn right.
            w_center = self.k_center * (left - right)
        elif left_vis:
            # keep left wall at desired distance; too close -> right.
            w_center = self.k_center * 0.55 * (left - self.wall_des)
        elif right_vis:
            # keep right wall at desired distance; too close -> left.
            w_center = self.k_center * 0.55 * (self.wall_des - right)

        w_rep = 0.0
        if left_vis and left < self.side_min:
            w_rep -= self.k_repulse * (self.side_min - left)
        if right_vis and right < self.side_min:
            w_rep += self.k_repulse * (self.side_min - right)
        if self.finite(fl) and fl < self.side_min + 0.04:
            w_rep -= 1.25 * self.k_repulse * (self.side_min + 0.04 - fl)
        if self.finite(fr) and fr < self.side_min + 0.04:
            w_rep += 1.25 * self.k_repulse * (self.side_min + 0.04 - fr)

        w = w_gap + w_diag + w_center + w_rep
        w = self.clamp(w, -self.max_w, self.max_w)

        # Linear speed adapts to front distance and turning strength.
        if self.finite(front) and front < self.d_emerg:
            v = 0.0
            # rotate toward larger side; do not drive into the wall.
            wl_eff = 2.2 if not self.finite(wl) else wl
            wr_eff = 2.2 if not self.finite(wr) else wr
            w = self.w_rec * (1 if wl_eff >= wr_eff else -1)
        else:
            front_factor = self.clamp((front_eff - self.d_emerg) / max(0.01, self.d_slow - self.d_emerg), 0.0, 1.0)
            turn_factor = self.clamp(1.0 - abs(w) / self.max_w, 0.28, 1.0)
            base = self.v_slow + (self.v_fast - self.v_slow) * front_factor
            v = self.clamp(base * turn_factor, 0.055, self.v_fast)
            if abs(w) > 0.65:
                v = min(v, self.v_slow)

        self.publish(v, w)


def main(args=None):
    rclpy.init(args=args)
    node = CorridorNavigator()
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
