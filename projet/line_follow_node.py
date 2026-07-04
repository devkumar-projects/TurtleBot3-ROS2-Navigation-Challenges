#!/usr/bin/env python3

import math
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage, LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy


class LineFollower(Node):
    BOOTSTRAP = 'BOOTSTRAP'
    FOLLOW = 'FOLLOW'
    TARGET_LOCK = 'TARGET_LOCK'
    DONE = 'DONE'

    def __init__(self) -> None:
        super().__init__('line_follower')

        # Start control
        # False = standalone challenge2 launch starts immediately.
        # True  = main.launch waits for /challenge1_done from corridor_node.
        # Default is True on purpose: in main.launch the line follower must never
        # publish on /cmd_vel before challenge 1 is really finished.
        self.declare_parameter('wait_for_challenge1', True)

        # Small rollout after challenge 1: the robot is stopped inside the first
        # target, and the green line may not yet be in the lower camera ROI.
        # Drive straight a little bit until green is visible, then switch to FOLLOW.
        self.declare_parameter('bootstrap_distance', 0.32)
        self.declare_parameter('bootstrap_speed', 0.075)
        self.declare_parameter('bootstrap_green_area', 180)
        self.declare_parameter('bootstrap_timeout', 5.0)

        # Green line tracking
        self.declare_parameter('linear_speed', 0.17)
        self.declare_parameter('min_linear_speed', 0.07)
        self.declare_parameter('angular_gain', 0.0045)
        self.declare_parameter('max_angular_speed', 0.65)
        self.declare_parameter('search_angular_speed', 0.28)
        self.declare_parameter('roi_fraction', 0.68)
        self.declare_parameter('min_line_area', 420)
        self.declare_parameter('line_h_low', 35)
        self.declare_parameter('line_h_high', 88)
        self.declare_parameter('line_s_low', 45)
        self.declare_parameter('line_v_low', 35)
        self.declare_parameter('line_lost_frames_max', 80)
        self.declare_parameter('line_error_alpha', 0.40)

        # Lidar safety
        self.declare_parameter('obstacle_dist', 0.25)
        self.declare_parameter('obstacle_angle_deg', 18)

        # Ignore target for the first centimetres after challenge 1 so the first
        # target does not trigger a false stop when challenge 2 starts.
        self.declare_parameter('target_min_travel', 0.55)
        self.declare_parameter('target_min_green_frames', 10)

        # Target colour detection
        self.declare_parameter('target_h_low', 18)
        self.declare_parameter('target_h_high', 38)
        self.declare_parameter('target_s_low', 80)
        self.declare_parameter('target_v_low', 80)
        self.declare_parameter('target_roi_start', 0.40)
        self.declare_parameter('target_close_area', 10500)
        self.declare_parameter('target_close_cy', 0.66)
        self.declare_parameter('target_close_width', 0.26)
        self.declare_parameter('target_confirm', 5)

        # Near-field ring gate + odom extra for front camera offset
        self.declare_parameter('target_near_roi_start', 0.72)
        self.declare_parameter('target_near_center_width', 0.34)
        self.declare_parameter('target_ring_min_ratio', 0.030)
        self.declare_parameter('target_yellow_near_ratio', 0.100)
        self.declare_parameter('target_red_after_yellow_ratio', 0.040)
        self.declare_parameter('target_ring_confirm', 4)
        self.declare_parameter('final_center_extra_dist', 0.115)
        self.declare_parameter('final_offset_speed', 0.060)
        self.declare_parameter('final_offset_time', 6.0)
        self.declare_parameter('target_align_kp', 0.95)
        self.declare_parameter('target_align_max_ang', 0.22)

        self._wait_for_challenge1 = bool(self.get_parameter('wait_for_challenge1').value)
        self._active = not self._wait_for_challenge1
        self._bootstrap_distance = float(self.get_parameter('bootstrap_distance').value)
        self._bootstrap_speed = float(self.get_parameter('bootstrap_speed').value)
        self._bootstrap_green_area = int(self.get_parameter('bootstrap_green_area').value)
        self._bootstrap_timeout = float(self.get_parameter('bootstrap_timeout').value)

        self._v = float(self.get_parameter('linear_speed').value)
        self._v_min = float(self.get_parameter('min_linear_speed').value)
        self._k_ang = float(self.get_parameter('angular_gain').value)
        self._max_w = float(self.get_parameter('max_angular_speed').value)
        self._search_w = float(self.get_parameter('search_angular_speed').value)
        self._roi_frac = float(self.get_parameter('roi_fraction').value)
        self._min_area = float(self.get_parameter('min_line_area').value)
        self._lh_low = int(self.get_parameter('line_h_low').value)
        self._lh_high = int(self.get_parameter('line_h_high').value)
        self._ls_low = int(self.get_parameter('line_s_low').value)
        self._lv_low = int(self.get_parameter('line_v_low').value)
        self._lost_max = int(self.get_parameter('line_lost_frames_max').value)
        self._err_alpha = float(self.get_parameter('line_error_alpha').value)
        self._obs_dist = float(self.get_parameter('obstacle_dist').value)
        self._obs_ang = int(self.get_parameter('obstacle_angle_deg').value)
        self._target_min_travel = float(self.get_parameter('target_min_travel').value)
        self._target_min_green = int(self.get_parameter('target_min_green_frames').value)
        self._th_low = int(self.get_parameter('target_h_low').value)
        self._th_high = int(self.get_parameter('target_h_high').value)
        self._ts_low = int(self.get_parameter('target_s_low').value)
        self._tv_low = int(self.get_parameter('target_v_low').value)
        self._target_roi_start = float(self.get_parameter('target_roi_start').value)
        self._target_close_area = int(self.get_parameter('target_close_area').value)
        self._target_close_cy = float(self.get_parameter('target_close_cy').value)
        self._target_close_width = float(self.get_parameter('target_close_width').value)
        self._target_confirm = int(self.get_parameter('target_confirm').value)
        self._near_roi_start = float(self.get_parameter('target_near_roi_start').value)
        self._near_center_width = float(self.get_parameter('target_near_center_width').value)
        self._ring_min_ratio = float(self.get_parameter('target_ring_min_ratio').value)
        self._yellow_near_ratio = float(self.get_parameter('target_yellow_near_ratio').value)
        self._red_after_yellow_ratio = float(self.get_parameter('target_red_after_yellow_ratio').value)
        self._ring_confirm = int(self.get_parameter('target_ring_confirm').value)
        self._extra_dist = float(self.get_parameter('final_center_extra_dist').value)
        self._offset_speed = float(self.get_parameter('final_offset_speed').value)
        self._offset_time = float(self.get_parameter('final_offset_time').value)
        self._target_align_kp = float(self.get_parameter('target_align_kp').value)
        self._target_align_max = float(self.get_parameter('target_align_max_ang').value)

        self._state = self.FOLLOW
        self._obstacle = False
        self._lost = 0
        self._green_frames = 0
        self._last_error = 0.0
        self._filt_error = 0.0
        self._target_close_count = 0
        self._target_ycx = 0.5
        self._target_ycy = 0.0
        self._target_ybw = 0.0
        self._target_area = 0
        self._frame_count = 0
        self._near_ratios = {'yellow': 0.0, 'red': 0.0, 'blue': 0.0, 'black': 0.0, 'white': 0.0}
        self._ring_yellow_seen_count = 0
        self._ring_yellow_seen = False
        self._ring_after_yellow_count = 0
        self._extra_active = False
        self._extra_x = None
        self._extra_y = None
        self._target_t0 = 0.0
        self._bootstrap_t0 = 0.0
        self._bootstrap_x = None
        self._bootstrap_y = None
        self._odom_x = None
        self._odom_y = None
        self._start_x = None
        self._start_y = None
        self._kernel = np.ones((5, 5), np.uint8)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self._pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.create_subscription(CompressedImage, '/image_raw/compressed', self._image_cb, qos)
        self.create_subscription(LaserScan, 'scan', self._scan_cb, qos)
        self.create_subscription(Odometry, 'odom', self._odom_cb, 10)
        done_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(Bool, '/challenge1_done', self._challenge1_done_cb, done_qos)
        if self._active:
            self.get_logger().info('Challenge 2 line follower started immediately')
        else:
            self.get_logger().info('Challenge 2 line follower armed: waiting for reliable /challenge1_done')

    def _challenge1_done_cb(self, msg: Bool) -> None:
        if not msg.data or self._active:
            return
        self._active = True
        self._state = self.BOOTSTRAP
        self._lost = 0
        self._green_frames = 0
        self._last_error = 0.0
        self._filt_error = 0.0
        self._target_close_count = 0
        self._ring_yellow_seen_count = 0
        self._ring_yellow_seen = False
        self._ring_after_yellow_count = 0
        self._extra_active = False
        self._start_x = self._odom_x
        self._start_y = self._odom_y
        self._bootstrap_x = self._odom_x
        self._bootstrap_y = self._odom_y
        self._bootstrap_t0 = self.get_clock().now().nanoseconds * 1e-9
        self.get_logger().info('Challenge 1 done received -> Challenge 2 bootstrap rollout, then green line following')
        self._publish(self._bootstrap_speed, 0.0)

    def _odom_cb(self, msg: Odometry) -> None:
        self._odom_x = msg.pose.pose.position.x
        self._odom_y = msg.pose.pose.position.y
        if self._active and self._start_x is None:
            self._start_x = self._odom_x
            self._start_y = self._odom_y

    def _travel_from_start(self) -> float:
        if self._odom_x is None or self._start_x is None:
            return 0.0
        return math.hypot(self._odom_x - self._start_x, self._odom_y - self._start_y)

    def _scan_cb(self, msg: LaserScan) -> None:
        if not self._active:
            return
        n = len(msg.ranges)
        if n == 0:
            return
        half = min(self._obs_ang, n // 2)
        vals = []
        for i in range(-half, half + 1):
            v = msg.ranges[i % n]
            if math.isfinite(v) and v > 0.02:
                vals.append(v)
        self._obstacle = bool(vals and min(vals) < self._obs_dist)

    def _publish(self, v: float, w: float) -> None:
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self._pub.publish(msg)

    def _stop(self) -> None:
        self._publish(0.0, 0.0)

    def _image_cb(self, msg: CompressedImage) -> None:
        if not self._active or self._state == self.DONE:
            return
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if image is None:
            return
        h, w = image.shape[:2]
        self._detect_target(image, h, w)
        if self._state == self.TARGET_LOCK:
            self._run_target_lock()
        elif self._state == self.BOOTSTRAP:
            self._run_bootstrap(image, h, w)
        else:
            self._run_line_follow(image, h, w)

    def _detect_target(self, image, h: int, w: int) -> None:
        roi = image[int(self._target_roi_start * h):, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        yellow = cv2.inRange(
            hsv,
            np.array([self._th_low, self._ts_low, self._tv_low], np.uint8),
            np.array([self._th_high, 255, 255], np.uint8),
        )
        yellow = cv2.morphologyEx(yellow, cv2.MORPH_OPEN, self._kernel)
        self._target_area = int(cv2.countNonZero(yellow))
        self._target_ycx = 0.5
        self._target_ycy = 0.0
        self._target_ybw = 0.0
        if self._target_area > 0:
            M = cv2.moments(yellow)
            if M['m00'] > 0:
                self._target_ycx = float(M['m10'] / M['m00']) / float(w)
                cy_roi = float(M['m01'] / M['m00'])
                self._target_ycy = (int(self._target_roi_start * h) + cy_roi) / float(h)
            contours, _ = cv2.findContours(yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                _, _, bw, _ = cv2.boundingRect(max(contours, key=cv2.contourArea))
                self._target_ybw = bw / float(w)

        close = (
            self._target_area >= self._target_close_area
            and self._target_ycy >= self._target_close_cy
            and self._target_ybw >= self._target_close_width
        )
        if close:
            self._target_close_count += 1
        else:
            self._target_close_count = max(0, self._target_close_count - 1)

        # Bottom-centre ROI for the ring sequence: yellow then red again.
        y0 = int(self._near_roi_start * h)
        half_w = int(0.5 * self._near_center_width * w)
        x0 = max(0, w // 2 - half_w)
        x1 = min(w, w // 2 + half_w)
        near = image[y0:h, x0:x1]
        self._near_ratios = {'yellow': 0.0, 'red': 0.0, 'blue': 0.0, 'black': 0.0, 'white': 0.0}
        if near.size == 0:
            return
        hsvn = cv2.cvtColor(near, cv2.COLOR_BGR2HSV)
        total = float(near.shape[0] * near.shape[1])
        masks = {
            'yellow': cv2.inRange(hsvn, np.array([self._th_low, self._ts_low, self._tv_low], np.uint8), np.array([self._th_high, 255, 255], np.uint8)),
            'red': cv2.bitwise_or(
                cv2.inRange(hsvn, np.array([0, 70, 60], np.uint8), np.array([10, 255, 255], np.uint8)),
                cv2.inRange(hsvn, np.array([170, 70, 60], np.uint8), np.array([180, 255, 255], np.uint8)),
            ),
            'blue': cv2.inRange(hsvn, np.array([82, 35, 55], np.uint8), np.array([115, 255, 255], np.uint8)),
            'black': cv2.inRange(hsvn, np.array([0, 0, 0], np.uint8), np.array([180, 90, 75], np.uint8)),
            'white': cv2.inRange(hsvn, np.array([0, 0, 145], np.uint8), np.array([180, 70, 255], np.uint8)),
        }
        for name, mask in masks.items():
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
            self._near_ratios[name] = cv2.countNonZero(mask) / total

        if self._state == self.TARGET_LOCK:
            if self._near_ratios['yellow'] >= self._yellow_near_ratio:
                self._ring_yellow_seen_count += 1
                if self._ring_yellow_seen_count >= self._ring_confirm:
                    self._ring_yellow_seen = True
            else:
                self._ring_yellow_seen_count = max(0, self._ring_yellow_seen_count - 1)

            red_after_yellow = (
                self._ring_yellow_seen
                and self._near_ratios['red'] >= self._red_after_yellow_ratio
                and self._near_ratios['red'] >= 0.80 * max(self._near_ratios['yellow'], 1e-6)
            )
            if red_after_yellow:
                self._ring_after_yellow_count += 1
            else:
                self._ring_after_yellow_count = max(0, self._ring_after_yellow_count - 1)

    def _target_allowed(self) -> bool:
        return self._travel_from_start() >= self._target_min_travel and self._green_frames >= self._target_min_green

    def _start_target_lock(self, reason: str) -> None:
        self._state = self.TARGET_LOCK
        self._target_t0 = self.get_clock().now().nanoseconds * 1e-9
        self._ring_yellow_seen_count = 0
        self._ring_yellow_seen = False
        self._ring_after_yellow_count = 0
        self._extra_active = False
        self._extra_x = None
        self._extra_y = None
        self.get_logger().info(f'Target 2 detected ({reason}) -> TARGET_LOCK')


    def _green_mask(self, image, h: int, w: int):
        """Return a cleaned binary mask of the green line in the camera ROI.

        stable accidentally called this same function from inside itself, which
        caused a RecursionError as soon as challenge 2 received an image.
        """
        y0 = int(h * (1.0 - self._roi_frac))
        roi = image[y0:h, :]
        if roi.size == 0:
            return np.zeros((1, w), dtype=np.uint8)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array([self._lh_low, self._ls_low, self._lv_low], np.uint8),
            np.array([self._lh_high, 255, 255], np.uint8),
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)
        return mask

    def _run_bootstrap(self, image, h: int, w: int) -> None:
        """Leave the first target safely before normal line following.

        At the end of challenge 1 the robot is centred on the yellow disk. If the
        challenge 2 node immediately searches for green, it can rotate on the target
        and lose the intended heading. This short odometry rollout keeps the same
        heading until the green line is actually visible in the camera ROI.
        """
        if self._obstacle:
            self._publish(0.0, 0.0)
            return
        mask = self._green_mask(image, h, w)
        green_area = int(cv2.countNonZero(mask))
        now = self.get_clock().now().nanoseconds * 1e-9
        dist = 0.0
        if self._odom_x is not None and self._bootstrap_x is not None:
            dist = math.hypot(self._odom_x - self._bootstrap_x, self._odom_y - self._bootstrap_y)
        elapsed = now - self._bootstrap_t0
        if green_area >= self._bootstrap_green_area or dist >= self._bootstrap_distance or elapsed >= self._bootstrap_timeout:
            self._state = self.FOLLOW
            self._lost = 0
            self._green_frames = 0
            self._last_error = 0.0
            self._filt_error = 0.0
            self.get_logger().info(
                f'Challenge 2 bootstrap done: green_area={green_area} dist={dist:.2f} elapsed={elapsed:.1f}s -> FOLLOW'
            )
            self._run_line_follow(image, h, w)
            return
        if self._frame_count <= 5 or self._frame_count % 15 == 0:
            self.get_logger().info(
                f'[BOOTSTRAP] green_area={green_area} dist={dist:.2f}/{self._bootstrap_distance:.2f} elapsed={elapsed:.1f}s'
            )
        self._publish(self._bootstrap_speed, 0.0)

    def _run_line_follow(self, image, h: int, w: int) -> None:
        if self._obstacle:
            self._publish(0.0, 0.0)
            return

        if self._target_allowed() and self._target_close_count >= self._target_confirm:
            self._start_target_lock('yellow close')
            self._publish(self._offset_speed, 0.0)
            return

        mask = self._green_mask(image, h, w)

        # Multi-band centroid: bottom bands keep the robot on the line, upper
        # bands give look-ahead for the S curve.
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
            if area < max(60, 0.10 * self._min_area):
                continue
            M = cv2.moments(band)
            if M['m00'] <= 0:
                continue
            cx = float(M['m10'] / M['m00'])
            # Stronger weight when the band contains a lot of green pixels.
            ww = weight * min(2.0, area / max(self._min_area, 1.0))
            weighted_x += ww * cx
            total_weight += ww

        self._frame_count += 1
        if self._frame_count <= 5 or self._frame_count % 20 == 0:
            self.get_logger().info(
                f'[LINE] frame={self._frame_count} area={total_area} lost={self._lost} '
                f'travel={self._travel_from_start():.2f} green_frames={self._green_frames} '
                f'obstacle={self._obstacle} target_area={self._target_area}'
            )

        if total_area >= self._min_area and total_weight > 0.0:
            self._green_frames += 1
            self._lost = 0
            cx = weighted_x / total_weight
            raw_error = cx - w / 2.0
            self._filt_error = self._err_alpha * raw_error + (1.0 - self._err_alpha) * self._filt_error
            self._last_error = self._filt_error
            angular = max(-self._max_w, min(self._max_w, -self._k_ang * self._filt_error))
            # Slow down when the curve is strong.
            curve = min(1.0, abs(self._filt_error) / max(1.0, 0.5 * w))
            linear = max(self._v_min, self._v * (1.0 - 0.45 * curve))
            self._publish(linear, angular)
        else:
            self._lost += 1
            direction = 1.0 if self._last_error < 0.0 else -1.0
            if self._lost > self._lost_max:
                # Do not jump to challenge 3 here; challenge 2 expects the second
                # target. Keep searching instead of declaring success too early.
                self.get_logger().warn('Green line lost for a long time -> searching')
                self._lost = self._lost_max // 2
            self._publish(0.0, self._search_w * direction)

    def _run_target_lock(self) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        if self._extra_active and self._extra_x is not None and self._odom_x is not None:
            dist = math.hypot(self._odom_x - self._extra_x, self._odom_y - self._extra_y)
            if dist >= self._extra_dist:
                self.get_logger().info(f'Target 2 centre reached, odom extra={dist:.3f} m -> DONE')
                self._state = self.DONE
                self._stop()
                return
            err = self._target_ycx - 0.5
            w_corr = max(-0.12, min(0.12, -0.65 * err))
            self._publish(self._offset_speed, w_corr)
            return

        elapsed = now - self._target_t0
        start_extra = (
            self._ring_after_yellow_count >= self._ring_confirm
            or (self._ring_yellow_seen_count >= 2 * self._ring_confirm and self._near_ratios['yellow'] >= 0.30)
            or elapsed > self._offset_time
        )
        if start_extra and self._odom_x is not None and self._odom_y is not None:
            self._extra_active = True
            self._extra_x = self._odom_x
            self._extra_y = self._odom_y
            self.get_logger().info(f'Target 2 marker reached -> odom extra {self._extra_dist:.3f} m')
            return

        err = self._target_ycx - 0.5
        w_corr = max(-self._target_align_max, min(self._target_align_max, -self._target_align_kp * err))
        self._publish(self._offset_speed, w_corr)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LineFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node._stop()
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
