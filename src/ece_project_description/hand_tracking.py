#!/usr/bin/env python3
"""
gesture_direct_mapping.py

Implements a direct mapping teleoperation node using MediaPipe Hand Tracking.
Control Scheme:
  - Palm Y      -> Shoulder Joint (Up/Down)
  - Palm X      -> Elbow Joint (Retract/Extend)
  - Wrist Tilt  -> Wrist Joint (Roll)
  - Pinch Mode  -> Base Rotation (Thumb+Index+Middle)
  - Fist        -> Unlock/Open Gripper
  - Open Palm   -> Lock/Close Gripper
  - Peace Sign  -> Reset to Neutral
"""

import os
import math
import cv2
import numpy as np
import mediapipe as mp
from typing import List, Tuple, Optional

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from mediapipe.tasks.python import vision

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TIMER_PERIOD = 0.033  # 30 Hz

# Control Parameters
SMOOTH_FACTOR = 0.22
MAX_STEP_DEG = 6.0
DEADZONE_DEG = 0.08
PINCH_THRESHOLD = 0.05
WRIST_STABILITY_BAND = 8.0

# Joint Limits (Degrees)
LIMIT_SHOULDER = 60.0
LIMIT_ELBOW = 45.0
LIMIT_WRIST = 90.0
LIMIT_BASE = 180.0

# Sensitivity Scales
SCALE_SHOULDER = 1.0
SCALE_ELBOW = 1.0
SCALE_BASE = 120.0

# Joint Names
JOINT_NAMES = [
    "base_link_part1_joint",
    "part1_part2_joint",
    "part2_part3_joint",
    "part3_component_joint",
    "component_arm_joint"
]

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def map_value(v: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    """Maps a value from one range to another with clamping."""
    if in_max == in_min:
        return (out_min + out_max) / 2.0
    v_clamped = max(min(v, in_max), in_min)
    return (v_clamped - in_min) / (in_max - in_min) * (out_max - out_min) + out_min

def smooth_value(prev: float, target: float, factor: float = SMOOTH_FACTOR) -> float:
    return prev * (1 - factor) + target * factor

def limit_step(prev: float, new_val: float, max_step: float = MAX_STEP_DEG) -> float:
    diff = new_val - prev
    if abs(diff) > max_step:
        return prev + max_step * (1 if diff > 0 else -1)
    return new_val

def clamp(val: float, low: float, high: float) -> float:
    return max(low, min(high, val))

def get_dist(lm_a, lm_b) -> float:
    return math.hypot(lm_a.x - lm_b.x, lm_a.y - lm_b.y)

# -----------------------------------------------------------------------------
# Gesture Primitives
# -----------------------------------------------------------------------------
def is_finger_extended(lm, tip_idx: int, pip_idx: int) -> bool:
    return lm[tip_idx].y < lm[pip_idx].y

def is_finger_folded(lm, tip_idx: int, pip_idx: int) -> bool:
    return lm[tip_idx].y > lm[pip_idx].y

def is_fist(lm) -> bool:
    return (is_finger_folded(lm, 8, 6) and is_finger_folded(lm, 12, 10) and
            is_finger_folded(lm, 16, 14) and is_finger_folded(lm, 20, 18))

def is_open_palm(lm) -> bool:
    return (is_finger_extended(lm, 8, 6) and is_finger_extended(lm, 12, 10) and
            is_finger_extended(lm, 16, 14) and is_finger_extended(lm, 20, 18))

def is_peace_sign(lm) -> bool:
    return (is_finger_extended(lm, 8, 6) and is_finger_extended(lm, 12, 10) and
            is_finger_folded(lm, 16, 14) and is_finger_folded(lm, 20, 18))

def is_pinching(lm) -> bool:
    # Requires Thumb to be close to both Index and Middle fingertips
    d_index = get_dist(lm[4], lm[8])
    d_middle = get_dist(lm[4], lm[12])
    return (d_index < PINCH_THRESHOLD) and (d_middle < PINCH_THRESHOLD)

# -----------------------------------------------------------------------------
# ROS 2 Node
# -----------------------------------------------------------------------------
class GestureControl(Node):
    def __init__(self):
        super().__init__("gesture_direct_mapping_node")
        self.get_logger().info("Initializing Gesture Control Node...")

        # ROS Publishers
        self.traj_pub = self.create_publisher(
            JointTrajectory,
            "/joint_trajectory_controller/joint_trajectory",
            10
        )

        # MediaPipe Setup
        try:
            pkg_share = get_package_share_directory('ece_project_description')
            model_path = os.path.join(pkg_share, "models", "hand_landmarker.task")
        except LookupError:
            self.get_logger().error("Package 'ece_project_description' not found. Check environment.")
            raise

        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        # Camera Setup
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open Camera (Index 0)")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        # State Variables
        self.joint_state = [0.0] * 5  # [Base, Shoulder, Elbow, Wrist, Gripper]
        self.prev_sent = [0.0] * 5
        
        # Pinch/Base Control State
        self.is_pinching_mode = False
        self.pinch_base_hold = 0.0

        # Gripper State
        self.gripper_locked = False

        # Timer
        self.timer = self.create_timer(TIMER_PERIOD, self.control_loop)
        self.get_logger().info("Gesture Control Ready.")

    def control_loop(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Frame read failed.")
            return

        # Pre-process image
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        detection_result = self.detector.detect(mp_image)

        if not detection_result.hand_landmarks:
            cv2.imshow("Gesture Control", frame)
            cv2.waitKey(1)
            return

        landmarks = detection_result.hand_landmarks[0]
        self._draw_landmarks(frame, landmarks)
        
        # Calculate targets
        targets = self._process_gestures(landmarks)
        
        # Apply smoothing and limits
        final_cmd = self._apply_kinematics(targets)
        
        # Publish
        self._publish_trajectory(final_cmd)
        
        # Visualization
        self._draw_hud(frame, final_cmd)
        cv2.imshow("Gesture Control", frame)
        cv2.waitKey(1)

    def _process_gestures(self, lm) -> List[float]:
        """Analyzes landmarks to determine raw target angles."""
        
        # 1. Reset Gesture (Peace Sign)
        if is_peace_sign(lm):
            self.get_logger().info("Peace Sign: Resetting System.")
            self.gripper_locked = False
            self.is_pinching_mode = False
            self.pinch_base_hold = 0.0
            return [0.0, 0.0, 0.0, 0.0, 0.0]

        # 2. Pinch Logic (Base Control)
        current_pinching = is_pinching(lm)
        if current_pinching and not self.is_pinching_mode:
            self.is_pinching_mode = True
            self.pinch_base_hold = self.joint_state[0]
        elif not current_pinching and self.is_pinching_mode:
            self.is_pinching_mode = False
            self.pinch_base_hold = self.joint_state[0]

        # 3. Gripper Logic
        gripper_target = -45.0 if self.gripper_locked else 45.0
        if is_fist(lm):
            self.gripper_locked = False
            gripper_target = 45.0
        elif is_open_palm(lm):
            self.gripper_locked = True
            gripper_target = -45.0

        # 4. Wrist Tilt Calculation
        # Use vector from Wrist(0) to MCP_Middle(9)
        h, w, _ = (480, 640, 3) # Approx dimensions
        p0 = (int(lm[0].x * w), int(lm[0].y * h))
        p9 = (int(lm[9].x * w), int(lm[9].y * h))
        tilt_rad = -math.atan2(p9[1] - p0[1], p9[0] - p0[0])
        wrist_mapped = map_value(tilt_rad, -1.2, 1.2, -LIMIT_WRIST, LIMIT_WRIST)
        
        # Deadband for wrist stability
        wrist_target = wrist_mapped if abs(wrist_mapped) >= WRIST_STABILITY_BAND else self.joint_state[3]

        # 5. Base Calculation
        if self.is_pinching_mode:
            # Map wrist X position to base rotation
            base_target = map_value(1.0 - lm[0].x, 0.0, 1.0, -SCALE_BASE, SCALE_BASE)
        else:
            base_target = self.pinch_base_hold

        # 6. Shoulder & Elbow Calculation
        # Palm Y -> Shoulder (Inverted: Hand Up = Shoulder Up)
        shoulder_target = map_value(1.0 - lm[0].y, 0.0, 1.0, 
                                    -LIMIT_SHOULDER * SCALE_SHOULDER, 
                                    LIMIT_SHOULDER * SCALE_SHOULDER)
        
        # Palm X -> Elbow
        elbow_target = map_value(lm[0].x, 0.0, 1.0, 
                                 -LIMIT_ELBOW * SCALE_ELBOW, 
                                 LIMIT_ELBOW * SCALE_ELBOW)

        return [base_target, shoulder_target, elbow_target, wrist_target, gripper_target]

    def _apply_kinematics(self, targets: List[float]) -> List[float]:
        """Applies smoothing, step limits, and hard limits."""
        smoothed_joints = []
        limits = [LIMIT_BASE, LIMIT_SHOULDER, LIMIT_ELBOW, LIMIT_WRIST, 180.0]

        for i, (prev, target) in enumerate(zip(self.prev_sent, targets)):
            # Apply Deadzone
            if abs(target - prev) < DEADZONE_DEG:
                target = prev
            
            # Smooth
            val = smooth_value(prev, target)
            
            # Limit Step Velocity
            val = limit_step(prev, val)
            
            # Hard Limits
            limit = limits[i]
            val = clamp(val, -limit, limit)
            
            smoothed_joints.append(val)

        # Update internal state
        self.joint_state = smoothed_joints
        self.prev_sent = list(smoothed_joints)
        return smoothed_joints

    def _publish_trajectory(self, joints_deg: List[float]):
        msg = JointTrajectory()
        msg.joint_names = JOINT_NAMES
        
        point = JointTrajectoryPoint()
        point.positions = [math.radians(j) for j in joints_deg]
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = 50 * 1_000_000  # 50ms duration
        
        msg.points.append(point)
        self.traj_pub.publish(msg)

    def _draw_landmarks(self, frame, landmarks):
        h, w, _ = frame.shape
        for lm in landmarks:
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (0, 255, 0), -1)

    def _draw_hud(self, frame, joints):
        status_color = (0, 255, 255) if self.is_pinching_mode else (200, 200, 200)
        mode_text = "MODE: PINCH (Base)" if self.is_pinching_mode else "MODE: DIRECT"
        
        cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(frame, f"Base: {joints[0]:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, f"Grip: {'LOCKED' if self.gripper_locked else 'OPEN'}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    def cleanup(self):
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = GestureControl()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down node...")
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
