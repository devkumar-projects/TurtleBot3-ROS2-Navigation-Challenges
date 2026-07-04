#!/usr/bin/env python3
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


class BallPusher(Node):

    SEARCHING = 'SEARCHING'
    ALIGNING  = 'ALIGNING'
    PUSHING   = 'PUSHING'
    DONE      = 'DONE'

    def __init__(self) -> None:
        super().__init__('ball_pusher')

        self.declare_parameter('search_angular_speed', 0.40)
        self.declare_parameter('linear_speed',         0.18)
        self.declare_parameter('push_speed',           0.12)
        self.declare_parameter('angular_gain',         0.005)
        self.declare_parameter('min_ball_area',        300)
        self.declare_parameter('push_area_threshold',  12000)
        self.declare_parameter('done_area_threshold',  70000)
        self.declare_parameter('align_threshold_px',   30)
        self.declare_parameter('ball_h_low',   18)
        self.declare_parameter('ball_h_high',  35)
        self.declare_parameter('ball_s_low',   100)
        self.declare_parameter('ball_v_low',   100)

        self._w_search  = self.get_parameter('search_angular_speed').value
        self._v_lin     = self.get_parameter('linear_speed').value
        self._v_push    = self.get_parameter('push_speed').value
        self._k_ang     = self.get_parameter('angular_gain').value
        self._min_area  = self.get_parameter('min_ball_area').value
        self._push_thr  = self.get_parameter('push_area_threshold').value
        self._done_thr  = self.get_parameter('done_area_threshold').value
        self._align_thr = self.get_parameter('align_threshold_px').value
        self._h_lo      = self.get_parameter('ball_h_low').value
        self._h_hi      = self.get_parameter('ball_h_high').value
        self._s_lo      = self.get_parameter('ball_s_low').value
        self._v_lo      = self.get_parameter('ball_v_low').value

        self._state  = self.SEARCHING
        self._kernel = np.ones((7, 7), np.uint8)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self._pub     = self.create_publisher(Twist, 'cmd_vel', 10)
        self._img_sub = self.create_subscription(
            CompressedImage, '/image_raw/compressed',
            self._image_cb, sensor_qos)

    def _image_cb(self, msg: CompressedImage) -> None:
        if self._state == self.DONE:
            self._stop()
            return

        buf   = np.frombuffer(msg.data, dtype=np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if image is None:
            return

        h, w = image.shape[:2]

        hsv  = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lo   = np.array([self._h_lo, self._s_lo, self._v_lo], dtype=np.uint8)
        hi   = np.array([self._h_hi, 255,        255        ], dtype=np.uint8)
        mask = cv2.inRange(hsv, lo, hi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            biggest = max(contours, key=cv2.contourArea)
            area    = cv2.contourArea(biggest)
        else:
            area = 0.0

        if area < self._min_area:
            if self._state != self.SEARCHING:
                self.get_logger().info('Ball lost → SEARCHING')
                self._state = self.SEARCHING
            self._publish(0.0, self._w_search)
            return

        M    = cv2.moments(mask)
        cx   = int(M['m10'] / M['m00']) if M['m00'] > 0 else w // 2
        err  = cx - w // 2
        angular = -self._k_ang * float(err)

        if area >= self._done_thr:
            self.get_logger().info(f'Ball in target area (area={int(area)}px²). DONE!')
            self._state = self.DONE
            self._stop()
            return

        if area >= self._push_thr:
            if self._state != self.PUSHING:
                self.get_logger().info(f'PUSHING  area={int(area)}')
                self._state = self.PUSHING
            self._publish(self._v_push, angular)
        else:
            if self._state != self.ALIGNING:
                self.get_logger().info(f'ALIGNING  area={int(area)}  err={err}px')
                self._state = self.ALIGNING
            linear = self._v_lin if abs(err) < 80 else 0.05
            self._publish(linear, angular)

    def _publish(self, v: float, w: float) -> None:
        msg = Twist()
        msg.linear.x  = float(v)
        msg.angular.z = float(w)
        self._pub.publish(msg)

    def _stop(self) -> None:
        self._publish(0.0, 0.0)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BallPusher()
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
