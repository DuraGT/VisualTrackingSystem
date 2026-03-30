"""
╔════════════════════════════════════════════════════════════════════════════╗
║                  COMBINED 6DOF POSITION & ORIENTATION TRACKER               ║
║                                                                             ║
║  Hybrid approach: Camera (primary) + BNO055 IMU (backup)                   ║
║                                                                             ║
║  PURPOSE                                                                    ║
║  ──────────────────────────────────────────────────────────────────────── ║
║  Tracks the 6-degree-of-freedom pose (3D position + 3D rotation) of an     ║
║  object marked with 4 ArUco markers. When the camera can see all markers,  ║
║  it's the primary data source. When markers are obscured, an inertial      ║
║  measurement unit (IMU) takes over, allowing brief camera-loss tolerance.  ║
║                                                                             ║
║  HARDWARE REQUIRED                                                          ║
║  ──────────────────────────────────────────────────────────────────────── ║
║  • Webcam (USB, supporting MJPG codec)                                     ║
║  • BNO055 IMU (9-DOF: accel + gyro + magnetometer) on serial port COM4     ║
║  • Exactly 4 ArUco markers (IDs: 0, 1, 2, 3) attached to target object     ║
║    - Marker 0: reference frame origin                                      ║
║    - Markers 1-3: define the tracked plane/surface                         ║
║                                                                             ║
║  COORDINATE SYSTEM                                                          ║
║  ──────────────────────────────────────────────────────────────────────── ║
║  • Position: X, Y, Z in millimetres (origin at marker 0)                   ║
║  • Rotation: Roll (X-axis), Pitch (Y-axis), Yaw (Z-axis) in degrees        ║
║  • ZYX Euler convention (aerospace standard)                               ║
║                                                                             ║
║  KEYBOARD CONTROLS                                                          ║
║  ──────────────────────────────────────────────────────────────────────── ║
║  • Q = Quit the program                                                    ║
║  • A = Toggle 3-D axis overlay on detected markers                         ║
║                                                                             ║
║  OUTPUT                                                                     ║
║  ──────────────────────────────────────────────────────────────────────── ║
║  • Console: real-time pose + FPS + data source (CAMERA/IMU/HOLD)           ║
║  • UDP: CSV line to TARGET_IP:TARGET_PORT (6 float values per frame)       ║
║  • Display: OpenCV window with HUD overlay showing live pose data           ║
║                                                                             ║
║  FALLBACK BEHAVIOR WHEN CAMERA LOSES MARKERS                               ║
║  ──────────────────────────────────────────────────────────────────────── ║
║  Phase 1 (0–5 seconds):  Use last known position + live IMU orientation    ║
║  Phase 2 (>5 seconds):   Freeze position, continue updating rotation only  ║
║  No IMU:                 Hold last known complete pose until recovery       ║
║                                                                             ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import time
import threading
import socket
import serial
import sys
import os
from collections import deque
from pynput import keyboard


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                         CONFIGURATION SECTION                              ║
# ╚════════════════════════════════════════════════════════════════════════════╝

# Network: Where to send UDP pose updates
TARGET_IP   = "10.98.109.221"
TARGET_PORT = 5006

# Hardware: Serial port and camera device
IMU_PORT    = "COM4"
IMU_BAUD    = 115200
CAM_INDEX   = 0

# Tracking: When to switch IMU strategy if markers are lost
IMU_TAKEOVER_FULL_SECS = 5.0  # Freeze XYZ position after this many seconds


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                    QUATERNION MATHEMATICS HELPERS                          ║
# ║                                                                             ║
# ║  Quaternions are a compact, numerically stable way to represent 3-D        ║
# ║  rotations as a single [w, x, y, z] unit vector. This module provides:     ║
# ║    • Multiplication (chaining rotations)                                   ║
# ║    • Conjugation (rotation inverse)                                        ║
# ║    • Conversion to/from Euler angles and rotation matrices                 ║
# ╚════════════════════════════════════════════════════════════════════════════╝

def qmul(q1, q2):
    """
    Quaternion multiplication (Hamilton product).

    Chains two rotations: q_relative = q_origin_inv * q_current.
    This tells us "how much did the IMU rotate since its calibration point?"

    Args:
        q1, q2: Unit quaternions [w, x, y, z]

    Returns:
        q_result: The combined rotation
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2])


def qconj(q):
    """
    Quaternion conjugate (equivalent to rotation inverse).

    For a unit quaternion, negating x, y, z reverses the rotation direction.
    We pre-compute and store the conjugate of the IMU's origin quaternion,
    allowing fast "undo" of the calibration rotation.

    Args:
        q: Unit quaternion [w, x, y, z]

    Returns:
        q_inverse: The quaternion that undoes this rotation
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])


def q2euler(q):
    """
    Convert a unit quaternion to Euler angles.

    Uses ZYX convention (aerospace standard):
      • Roll  = rotation around X-axis (left/right tilt)
      • Pitch = rotation around Y-axis (forward/back tilt)
      • Yaw   = rotation around Z-axis (left/right turn)

    Args:
        q: Unit quaternion [w, x, y, z]

    Returns:
        np.array([roll, pitch, yaw]) in degrees
    """
    w, x, y, z = q

    roll  = np.degrees(np.arctan2(2*(w*x + y*z),  1 - 2*(x*x + y*y)))
    pitch = np.degrees(np.arcsin(np.clip(2*(w*y - z*x), -1, 1)))
    yaw   = np.degrees(np.arctan2(2*(w*z + x*y),  1 - 2*(y*y + z*z)))

    return np.array([roll, pitch, yaw])


def rotation_matrix_to_quaternion(R):
    """
    Convert a 3×3 rotation matrix to a unit quaternion.

    Uses Shepperd's method: selects one of four numerically stable branches
    based on which diagonal element of R is largest.

    Args:
        R: 3×3 rotation matrix

    Returns:
        q: Unit quaternion [w, x, y, z]
    """
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        return np.array([0.25/s,
                         (R[2,1]-R[1,2])*s,
                         (R[0,2]-R[2,0])*s,
                         (R[1,0]-R[0,1])*s])
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        return np.array([(R[2,1]-R[1,2])/s, 0.25*s,
                         (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s])
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        return np.array([(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s,
                         0.25*s,             (R[1,2]+R[2,1])/s])
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        return np.array([(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s,
                         (R[1,2]+R[2,1])/s, 0.25*s])


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                       IMU READER (Background Thread)                       ║
# ║                                                                             ║
# ║  The BNO055 sensor reads orientation and calibration data continuously.    ║
# ║  A background daemon thread manages the serial connection, allowing the    ║
# ║  main loop to fetch the latest quaternion without blocking.                ║
# ╚════════════════════════════════════════════════════════════════════════════╝

class IMUReader:
    """
    Manages BNO055 IMU connection and reads quaternion + calibration data.

    The IMU is optional; the tracker works camera-only if not available.

    Expected serial format from IMU:
        QUAT:w,x,y,z|CAL:sys,gyro,accel,mag

    Both tokens optional; order doesn't matter.
    """

    def __init__(self, port, baud=115200):
        """
        Open serial connection and start the read loop in a background thread.

        Args:
            port: Serial port name (e.g., "COM4" on Windows, "/dev/ttyUSB0" on Linux)
            baud: Baud rate (BNO055 default is 115200)
        """
        self._lock   = threading.Lock()
        self.cur_q   = np.array([1., 0., 0., 0.])  # Identity quaternion
        self.cal     = np.zeros(4, dtype=int)       # Calibration scores [0-3]
        self.running = True

        try:
            self.ser = serial.Serial(port, baud, timeout=1)
            time.sleep(2)  # Wait for Arduino bootloader to finish
            self.ser.reset_input_buffer()
            self.available = True
            print(f"✓ IMU connected on {port}")
        except serial.SerialException as e:
            print(f"⚠  IMU not available ({e}). Backup disabled.")
            self.available = False
            return

        # Launch the background thread (daemon = dies with main process)
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        """
        Background thread loop: reads lines from serial and parses them.
        Silently recovers from errors (timeouts, decode issues) to stay alive.
        """
        while self.running:
            try:
                line = self.ser.readline().decode(errors='ignore').strip()
                if 'QUAT:' in line or 'EULER:' in line:
                    self._parse(line)
            except Exception:
                pass  # Timeout or decode error — continue

    def _parse(self, line):
        """
        Parse a '|'-separated line and update quaternion + calibration scores.
        Thread-safe: runs under self._lock.

        Args:
            line: Serial line like "QUAT:0.5,0.1,0.2,0.3|CAL:3,3,2,1"
        """
        with self._lock:
            for part in line.split('|'):
                try:
                    if part.startswith('QUAT:'):
                        q = np.array([float(v) for v in part[5:].split(',')])
                        n = np.linalg.norm(q)
                        if n > 0:
                            self.cur_q = q / n  # Normalize to unit quaternion
                    elif part.startswith('CAL:'):
                        self.cal = np.array([int(float(v)) for v in part[4:].split(',')])
                except Exception:
                    pass  # Malformed token — skip it

    def get_quaternion(self):
        """Thread-safe snapshot of the latest quaternion."""
        with self._lock:
            return self.cur_q.copy()

    def get_cal(self):
        """Thread-safe snapshot of calibration scores [sys, gyro, accel, mag]."""
        with self._lock:
            return self.cal.copy()

    def stop(self):
        """Signal the thread to stop and close the serial port."""
        self.running = False
        if self.available and self.ser.is_open:
            self.ser.close()


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                         MAIN TRACKER CLASS                                 ║
# ║                                                                             ║
# ║  CombinedTracker is the core engine. It:                                   ║
# ║    1. Manages the camera, IMU, and UDP socket                              ║
# ║    2. Detects ArUco markers and solves for their 3-D pose                  ║
# ║    3. Computes relative pose (plane relative to reference marker)          ║
# ║    4. Smooths the pose output via median + velocity clamp + EMA            ║
# ║    5. Falls back to IMU when camera loses tracking                         ║
# ║    6. Displays real-time HUD and sends UDP packets                         ║
# ╚════════════════════════════════════════════════════════════════════════════╝

class CombinedTracker:
    """
    Main tracker integrating camera, IMU, and pose filtering.

    Call run() to enter the main loop.
    """

    def __init__(self):
        """Initialize all hardware, filters, and state variables."""

        # ─────────────────────────────────────────────────────────────────
        # UDP SOCKET
        # ─────────────────────────────────────────────────────────────────
        # Opens a non-blocking UDP socket. Pre-connect to avoid per-packet
        # DNS lookups. Silently drops packets if the OS buffer is full.
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.sock.connect((TARGET_IP, TARGET_PORT))
        print(f"✓ UDP → {TARGET_IP}:{TARGET_PORT}")

        # ─────────────────────────────────────────────────────────────────
        # IMU READER
        # ─────────────────────────────────────────────────────────────────
        self.imu = IMUReader(IMU_PORT)

        # imu_origin_q_inv: The conjugate of the IMU's quaternion at the
        # moment we calibrate it to the camera's reference frame. When we
        # multiply this against a future quaternion, the result tells us
        # "how much did the IMU rotate since calibration?"
        self.imu_origin_q_inv = np.array([1., 0., 0., 0.])
        self.imu_calibrated   = False

        # Short history for smoothing IMU Euler output (gyro noise reduction)
        self._imu_euler_hist  = deque(maxlen=5)

        # ─────────────────────────────────────────────────────────────────
        # CAMERA GEOMETRY
        # ─────────────────────────────────────────────────────────────────
        # Physical size of one ArUco marker in metres (15 mm square)
        self.marker_size = 0.020

        # Marker layout:
        #   - ID 0: origin/reference marker (defines coordinate system)
        #   - IDs 1-3: form a plane we track relative to ID 0
        self.ref_id    = 0
        self.plane_ids = [1, 2, 3]

        # 3-D corner positions of a marker in its own local frame (Z=0 = flat).
        # OpenCV corner order: top-left, top-right, bottom-right, bottom-left.
        s = self.marker_size / 2
        self.obj_pts = np.array([[-s,  s, 0],
                                  [ s,  s, 0],
                                  [ s, -s, 0],
                                  [-s, -s, 0]], dtype=np.float32)

        # ─────────────────────────────────────────────────────────────────
        # CAMERA CAPTURE
        # ─────────────────────────────────────────────────────────────────
        # DirectShow (CAP_DSHOW) initializes much faster than MSMF on Windows.
        self.cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)

        # MJPG codec: compresses on-camera so 60 FPS fits over USB 2.0.
        # Without it, uncompressed YUY2 saturates bandwidth and caps ~25 FPS.
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS,          60)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)  # Minimal buffer = lowest latency
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS,    0)  # Let camera autofocus continuously

        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        actual_w   = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h   = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"✓ Camera: {actual_w}×{actual_h} @ {actual_fps:.0f} FPS")

        # ─────────────────────────────────────────────────────────────────
        # CAMERA CALIBRATION
        # ─────────────────────────────────────────────────────────────────
        # Camera calibration corrects for lens distortion, making depth
        # measurements (Z) much more accurate. Without it, poses are skewed.
        try:
            import json
            with open('camera_calibration_1280x720.json') as f:
                calib = json.load(f)
            self.cam_mat = np.array(calib['camera_matrix'],      dtype=np.float32)
            self.dist    = np.array(calib['dist_coeffs'], dtype=np.float32).flatten()
            print(f"✓ Camera calibration loaded (err: {calib['rms_error']:.4f}px)")
        except FileNotFoundError:
            # Fallback: pinhole model (less accurate but workable)
            self.cam_mat = np.array([[800,0,320],[0,800,240],[0,0,1]], dtype=np.float32)
            self.dist    = np.zeros(5, dtype=np.float32)
            print("⚠  Default camera matrix in use — run calibration for accuracy")

        # ─────────────────────────────────────────────────────────────────
        # ARUCORE DETECTOR
        # ─────────────────────────────────────────────────────────────────
        # DICT_4X4_50: 50 unique marker IDs, 4×4 bit pattern.
        # Smaller dictionary = faster detection, fewer false positives.
        self.dict   = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.params = aruco.DetectorParameters()

        # Sub-pixel corner refinement improves pose accuracy
        self.params.cornerRefinementMethod        = aruco.CORNER_REFINE_SUBPIX
        self.params.cornerRefinementMaxIterations = 15
        self.params.cornerRefinementMinAccuracy   = 0.05

        # Fewer adaptive threshold windows = fewer image passes = faster
        self.params.adaptiveThreshWinSizeMin  = 5
        self.params.adaptiveThreshWinSizeMax  = 21
        self.params.adaptiveThreshWinSizeStep = 8

        # Filter out unreasonably sized marker candidates
        self.params.minMarkerPerimeterRate = 0.05
        self.params.maxMarkerPerimeterRate = 4.0

        self.detector = aruco.ArucoDetector(self.dict, self.params)

        # ─────────────────────────────────────────────────────────────────
        # POSE SMOOTHING PIPELINE
        # ─────────────────────────────────────────────────────────────────
        # Three sequential stages kill noise while preserving fast motion:
        #
        # 1. MEDIAN FILTER
        #    Rolls raw pose through a 5-frame window, takes median.
        #    Robust to single-frame outlier spikes (e.g., failed marker detection).
        #
        # 2. VELOCITY CLAMP
        #    Blocks any DOF from jumping more than MAX_*_STEP per frame.
        #    Protects against corrupt detections while allowing real fast motion.
        #
        # 3. EXPONENTIAL MOVING AVERAGE (EMA)
        #    Smooth residual jitter via weighted average of current + filtered.
        #    Higher α = faster response but noisier. Lower α = smoother but lags.

        self.MEDIAN_WIN      = 5      # Rolling window size
        self.raw_history     = deque(maxlen=self.MEDIAN_WIN)

        self.alpha           = 0.20   # EMA weight (0.0–1.0)
        self.MAX_POS_STEP    = 25.0   # mm per frame
        self.MAX_ANG_STEP    = 10.0   # degrees per frame

        # ─────────────────────────────────────────────────────────────────
        # TRACKER STATE
        # ─────────────────────────────────────────────────────────────────
        self.filtered_pose    = None  # Pose after all filter stages
        self.last_valid_pose  = np.zeros(6, dtype=np.float32)  # Held during IMU fallback
        self.pose_confidence  = 0     # Rises when markers seen, decays when lost
        self.max_confidence   = 20    # Ceiling (scales velocity clamp limits)

        # ─────────────────────────────────────────────────────────────────
        # CAMERA-LOSS STATE
        # ─────────────────────────────────────────────────────────────────
        # When all 4 markers disappear, we record when that happened (start
        # of a timer) and handle the IMU takeover in two phases.
        self.camera_lost_time = None  # None = camera is tracking
        self.imu_frozen_xyz   = None  # Snapshot of position at 5-second mark

        # ─────────────────────────────────────────────────────────────────
        # DISPLAY & TIMING
        # ─────────────────────────────────────────────────────────────────
        self.draw_axes   = True                    # Toggle with 'A'
        self.fps         = 0.0                     # Computed once per 30 frames
        self.frame_count = 0
        self.last_time   = time.perf_counter()

        # ─────────────────────────────────────────────────────────────────
        # THREADED FRAME GRABBER
        # ─────────────────────────────────────────────────────────────────
        # A background thread calls cap.read() in a tight loop, storing the
        # latest frame. The main loop picks up waiting frames rather than
        # blocking on camera I/O, reducing capture latency (~16 ms at 60 FPS).
        self._frame         = None
        self._frame_id      = 0
        self._last_frame_id = 0
        self._frame_lock    = threading.Lock()
        self._grab_running  = True
        self._grab_thread   = threading.Thread(target=self._grabber, daemon=True)
        self._grab_thread.start()

        # Block until first frame arrives (poll every 50 ms, up to 1.5 s)
        print("Starting camera...", end='', flush=True)
        for _ in range(30):
            if self._frame is not None:
                break
            time.sleep(0.05)
        print(" ready.")

        # ─────────────────────────────────────────────────────────────────
        # KEYBOARD LISTENER
        # ─────────────────────────────────────────────────────────────────
        # pynput captures key presses globally, even inside IDE consoles
        # where cv2.waitKey() may not have focus.
        self._key_queue   = deque()
        self._kb_listener = keyboard.Listener(on_press=self._on_key)
        self._kb_listener.start()

        print("=" * 60)
        print("COMBINED 6DOF  |  Q=quit  A=toggle axes")
        print("=" * 60)

    # ════════════════════════════════════════════════════════════════════════
    # BACKGROUND THREAD: FRAME GRABBER
    # ════════════════════════════════════════════════════════════════════════

    def _grabber(self):
        """
        Daemon thread: continuously reads frames from the camera.

        Each new frame gets a unique ID. The main loop checks if the current
        frame ID has changed; if not, it's a duplicate and we skip processing.
        This avoids re-processing the same image multiple times per frame period.
        """
        while self._grab_running:
            ret, frame = self.cap.read()
            if ret:
                with self._frame_lock:
                    self._frame    = frame
                    self._frame_id += 1

    # ════════════════════════════════════════════════════════════════════════
    # BACKGROUND THREAD: KEYBOARD LISTENER
    # ════════════════════════════════════════════════════════════════════════

    def _on_key(self, key):
        """pynput callback: queue any printable key press."""
        try:
            self._key_queue.append(key.char.lower())
        except AttributeError:
            pass  # Ignore special keys (Shift, Ctrl, arrows)

    # ════════════════════════════════════════════════════════════════════════
    # POSE MATHEMATICS
    # ════════════════════════════════════════════════════════════════════════

    def _avg_quaternions(self, quats):
        """
        Average multiple unit quaternions using eigendecomposition.

        Marks et al. (2007) method: build a 4×4 accumulator matrix from
        outer products, then return the eigenvector of the largest eigenvalue.
        This eigenvector IS the mean quaternion and is always a valid rotation.

        Args:
            quats: List of unit quaternions [w, x, y, z]

        Returns:
            q_mean: Unit quaternion representing the average rotation
        """
        M = np.zeros((4, 4))
        for q in quats:
            if q[0] < 0:      # Force same hemisphere (q and -q are identical)
                q = -q
            M += np.outer(q, q)
        M /= len(quats)

        eigvals, eigvecs = np.linalg.eigh(M)
        return eigvecs[:, np.argmax(eigvals)]

    def _quat_to_rot(self, q):
        """
        Convert a unit quaternion to a 3×3 rotation matrix.

        Args:
            q: Unit quaternion [w, x, y, z]

        Returns:
            R: 3×3 rotation matrix
        """
        w, x, y, z = q / np.linalg.norm(q)
        return np.array([
            [1-2*(y*y+z*z),  2*(x*y-z*w),   2*(x*z+y*w)],
            [2*(x*y+z*w),    1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w),    2*(y*z+x*w),   1-2*(x*x+y*y)]])

    def _euler_from_mat(self, R):
        """
        Extract Euler angles [Roll, Pitch, Yaw] from a rotation matrix.

        Uses ZYX convention. Handles gimbal lock (pitch ≈ ±90°) by setting
        yaw to 0 in degenerate cases.

        Args:
            R: 3×3 rotation matrix

        Returns:
            np.array([roll, pitch, yaw]) in degrees
        """
        sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
        if sy > 1e-6:  # Normal case
            roll  = np.arctan2(R[2,1], R[2,2])
            pitch = np.arctan2(-R[2,0], sy)
            yaw   = np.arctan2(R[1,0], R[0,0])
        else:          # Gimbal lock
            roll  = np.arctan2(-R[1,2], R[1,1])
            pitch = np.arctan2(-R[2,0], sy)
            yaw   = 0.0
        return np.degrees([roll, pitch, yaw])

    def _compute_relative_pose(self, rvecs, tvecs):
        """
        Compute the 6-DOF pose of the plane (markers 1-3) relative to
        the reference marker (marker 0).

        The math:
          R_0, t_0  = camera-space pose of reference marker
          R_avg     = average rotation of plane markers (via quaternion method)
          R_rel     = R_0^T * R_avg  (relative rotation in marker-0 frame)
          t_avg     = mean camera-space position of plane markers
          t_rel     = R_0^T * (t_avg - t_0)  (relative position in marker-0 frame)

        Args:
            rvecs: Rotation vectors (axis-angle) keyed by marker ID
            tvecs: Translation vectors keyed by marker ID

        Returns:
            (pos_mm, euler_deg): position in mm, Euler angles in degrees
                                 or (None, None) on error
        """
        try:
            # Reference marker (ID 0)
            R_0, _ = cv2.Rodrigues(rvecs[self.ref_id])
            t_0    = tvecs[self.ref_id].flatten()

            # Average the three plane-marker rotations
            quats = []
            for pid in self.plane_ids:
                R_p, _ = cv2.Rodrigues(rvecs[pid])
                quats.append(rotation_matrix_to_quaternion(R_p))
            q_avg = self._avg_quaternions(quats)
            R_avg = self._quat_to_rot(q_avg)

            # Relative rotation: R_0.T is the inverse
            R_rel = R_0.T @ R_avg

            # Relative position in marker-0 frame
            t_avg = np.mean([tvecs[pid].flatten() for pid in self.plane_ids], axis=0)
            t_rel = R_0.T @ (t_avg - t_0)

            return t_rel * 1000.0, self._euler_from_mat(R_rel)  # m→mm, rad→deg

        except Exception:
            return None, None

    def _clamp_step(self, new_pose, prev_pose):
        """
        Velocity clamp: prevent any single DOF from changing too much in one frame.

        Protects against corrupt marker detections (single-frame spikes) while
        allowing genuine fast motion. Limits scale with confidence so a locked
        track can move faster than a freshly-acquired one.

        Args:
            new_pose: Raw pose from current frame [X, Y, Z, Roll, Pitch, Yaw]
            prev_pose: Filtered pose from previous frame

        Returns:
            clamped: Velocity-clamped pose
        """
        clamped = new_pose.copy()
        delta   = new_pose - prev_pose

        # Confidence scale: 1.0 (none) → 4.0 (full)
        scale     = 1.0 + (self.pose_confidence / self.max_confidence) * 3.0
        pos_limit = self.MAX_POS_STEP * scale
        ang_limit = self.MAX_ANG_STEP * scale

        for i in range(3):     # X, Y, Z
            if abs(delta[i]) > pos_limit:
                clamped[i] = prev_pose[i] + np.sign(delta[i]) * pos_limit

        for i in range(3, 6):  # Roll, Pitch, Yaw
            if abs(delta[i]) > ang_limit:
                clamped[i] = prev_pose[i] + np.sign(delta[i]) * ang_limit

        return clamped

    def _apply_deadzone(self, pose):
        """
        Zero out values smaller than the noise floor.

        Prevents the output from jittering when stationary.
        Position noise floor: 0.4 mm. Angle noise floor: 0.25°.

        Args:
            pose: 6-DOF pose [X, Y, Z, Roll, Pitch, Yaw]

        Returns:
            out: Pose with small values zeroed out
        """
        out = pose.copy()
        out[:3][np.abs(out[:3]) < 0.4]  = 0.0
        out[3:][np.abs(out[3:]) < 0.25] = 0.0
        return out

    # ════════════════════════════════════════════════════════════════════════
    # IMU CALIBRATION & ORIENTATION
    # ════════════════════════════════════════════════════════════════════════

    def _calibrate_imu_to_camera(self, camera_euler_deg):
        """
        Snapshot the IMU's current quaternion as the calibration origin.

        From this point on, _imu_relative_euler() reports angles as deltas
        from this snapshot, not absolute magnetic headings. This keeps the
        IMU continuously synchronized with the camera's reference frame.

        Called every frame the camera is successfully tracking.

        Args:
            camera_euler_deg: Ignored; present for symmetry
        """
        if not self.imu.available:
            return
        imu_q = self.imu.get_quaternion()
        self.imu_origin_q_inv = qconj(imu_q)
        self.imu_calibrated   = True

    def _imu_relative_euler(self):
        """
        Return the IMU's current orientation relative to its calibration origin.

        Smooths output with a 5-frame moving average to reduce gyro noise.

        Returns:
            np.array([roll, pitch, yaw]) in degrees, relative to calibration
        """
        q  = self.imu.get_quaternion()
        qr = qmul(self.imu_origin_q_inv, q)
        if qr[0] < 0:
            qr = -qr

        e = q2euler(qr)

        self._imu_euler_hist.append(e)
        if len(self._imu_euler_hist) >= 3:
            return np.mean(self._imu_euler_hist, axis=0)
        return e

    # ════════════════════════════════════════════════════════════════════════
    # NETWORK OUTPUT
    # ════════════════════════════════════════════════════════════════════════

    def _send_udp(self, pose):
        """
        Send pose as CSV line over UDP (non-blocking).

        Silently drops the packet if the OS socket buffer is full
        (acceptable for real-time data where old packets are stale).

        Format: "X,Y,Z,Roll,Pitch,Yaw,"

        Args:
            pose: 6-element numpy array [X, Y, Z, Roll, Pitch, Yaw]
        """
        try:
            self.sock.send(
              f"{float(pose[0]):.2f},"
              f"{float(pose[1]):.2f},"
              f"{float(pose[2]):.2f},"
              f"{float(pose[3]):.2f},"
              f"{float(pose[4]):.2f},"
              f"{float(pose[5]):.2f},".encode())
        except (BlockingIOError, OSError):
            pass  # Buffer full or target unreachable — drop silently

    # ════════════════════════════════════════════════════════════════════════
    # DISPLAY / HUD
    # ════════════════════════════════════════════════════════════════════════

    def _draw_hud(self, frame, pose, color, mode_label):
        """
        Render 6 pose values and a mode label onto the video frame.

        Color encodes the data source:
          • Green:       Camera (high confidence)
          • Cyan-yellow: Camera (building confidence)
          • Orange:      IMU Phase 1 (full 6-DOF)
          • Red-orange:  IMU Phase 2 (XYZ frozen)
          • Grey:        Holding last pose (no IMU)

        Args:
            frame: OpenCV frame (modified in-place)
            pose: 6-element array [X, Y, Z, Roll, Pitch, Yaw]
            color: BGR tuple (B, G, R)
            mode_label: Text label showing data source
        """
        y = 80
        for label, val in zip(("X mm:", "Y mm:", "Z mm:", "Roll:", "Pitch:", "Yaw:"), pose):
            cv2.putText(frame, f"{label}{val:8.1f}", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y += 35
        cv2.putText(frame, mode_label, (20, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # ════════════════════════════════════════════════════════════════════════
    # MAIN LOOP
    # ════════════════════════════════════════════════════════════════════════

    def run(self):
        """
        Main tracking loop. Each iteration:

        1. Drain the keyboard queue (Q to quit, A to toggle axes)
        2. Fetch latest frame; skip if it's a duplicate
        3. Detect ArUco markers; solve for each marker's 3-D pose
        4. If all 4 markers visible:
           - Compute relative pose (plane relative to reference marker)
           - Apply three-stage smoothing pipeline (median → clamp → EMA)
           - Send UDP, draw HUD
           - Calibrate IMU to camera's reference frame
        5. If markers lost:
           - Phase 1 (0–5s):   Use last camera position + live IMU rotation
           - Phase 2 (>5s):    Freeze position, continue updating rotation
           - No IMU:           Hold last known pose
        6. Display frame; update FPS counter
        """
        while True:

            # ─────────────────────────────────────────────────────────────
            # KEYBOARD HANDLING
            # ─────────────────────────────────────────────────────────────
            while self._key_queue:
                k = self._key_queue.popleft()
                if k == 'q':
                    self._shutdown()
                    return
                elif k == 'a':
                    self.draw_axes = not self.draw_axes

            # ─────────────────────────────────────────────────────────────
            # FRAME ACQUISITION
            # ─────────────────────────────────────────────────────────────
            with self._frame_lock:
                if self._frame is None:
                    continue
                current_id = self._frame_id
                frame      = self._frame.copy()

            if current_id == self._last_frame_id:
                continue  # Duplicate frame — skip processing
            self._last_frame_id = current_id

            # ─────────────────────────────────────────────────────────────
            # FPS COUNTER (TRUE CAMERA DELIVERY RATE)
            # ─────────────────────────────────────────────────────────────
            self.frame_count += 1
            if self.frame_count >= 30:
                now = time.perf_counter()
                self.fps = 30 / max(now - self.last_time, 1e-6)
                self.last_time   = now
                self.frame_count = 0

            # ─────────────────────────────────────────────────────────────
            # MARKER DETECTION
            # ─────────────────────────────────────────────────────────────
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = self.detector.detectMarkers(gray)

            rvecs = {}  # Rotation vectors by marker ID
            tvecs = {}  # Translation vectors by marker ID

            if ids is not None:
                aruco.drawDetectedMarkers(frame, corners, ids)
                flat_ids = ids.flatten()

                for i, mid in enumerate(flat_ids):
                    if mid not in (0, 1, 2, 3):
                        continue

                    # solvePnP: find the camera-space pose that maps
                    # known 3-D corner points to detected 2-D pixel corners.
                    # Uses the Levenberg-Marquardt optimizer.
                    ok, rvec, tvec = cv2.solvePnP(
                        self.obj_pts,
                        corners[i][0],
                        self.cam_mat,
                        self.dist,
                        flags=cv2.SOLVEPNP_ITERATIVE)

                    if ok:
                        rvecs[mid] = rvec
                        tvecs[mid] = tvec
                        if self.draw_axes:
                            # Draw 3-D axes (2 cm long) on marker
                            cv2.drawFrameAxes(frame, self.cam_mat, self.dist,
                                              rvec, tvec, 0.02, 2)

            # True when all 4 required markers have valid poses
            all_detected = all(m in rvecs for m in (0, 1, 2, 3))
            now_t = time.perf_counter()

            # ═════════════════════════════════════════════════════════════
            # PRIMARY PATH: CAMERA HAS ALL 4 MARKERS
            # ═════════════════════════════════════════════════════════════
            if all_detected:
                self.camera_lost_time = None  # Reset loss timer
                self.imu_frozen_xyz   = None  # Clear XYZ freeze

                pos, euler = self._compute_relative_pose(rvecs, tvecs)

                if pos is not None:
                    raw = np.array([pos[0], pos[1], pos[2],
                                    euler[0], euler[1], euler[2]])

                    # SMOOTHING STAGE 1: MEDIAN FILTER
                    self.raw_history.append(raw)
                    median_pose = (np.median(self.raw_history, axis=0)
                                   if len(self.raw_history) >= 3 else raw)

                    # SMOOTHING STAGE 2: VELOCITY CLAMP
                    if self.filtered_pose is not None:
                        median_pose = self._clamp_step(median_pose, self.filtered_pose)

                    # SMOOTHING STAGE 3: EXPONENTIAL MOVING AVERAGE
                    if self.filtered_pose is None:
                        self.filtered_pose = median_pose.copy()
                    else:
                        self.filtered_pose = (self.alpha * median_pose
                                              + (1 - self.alpha) * self.filtered_pose)

                    # SMOOTHING STAGE 4: DEADZONE
                    output_pose = self._apply_deadzone(self.filtered_pose)

                    self.last_valid_pose = output_pose.copy()
                    self.pose_confidence = min(self.pose_confidence + 1, self.max_confidence)

                    # Keep IMU continuously synchronized with camera
                    self._calibrate_imu_to_camera(output_pose[3:])

                    self._send_udp(output_pose)

                    # Draw magenta triangle connecting the three plane markers
                    if ids is not None:
                        flat = ids.flatten().tolist()
                        pts  = []
                        for pid in self.plane_ids:
                            if pid in flat:
                                j   = flat.index(pid)
                                ctr = tuple(np.mean(corners[j][0], axis=0).astype(int))
                                pts.append(ctr)
                        if len(pts) == 3:
                            cv2.line(frame, pts[0], pts[1], (255, 0, 255), 2)
                            cv2.line(frame, pts[1], pts[2], (255, 0, 255), 2)
                            cv2.line(frame, pts[2], pts[0], (255, 0, 255), 2)

                    # HUD color: green if confident, cyan-yellow if acquiring
                    color = (0, 255, 0) if self.pose_confidence > 10 else (0, 220, 220)
                    self._draw_hud(frame, output_pose, color, "SOURCE: CAMERA")

                    print(f"\r[CAM] X:{output_pose[0]:7.1f} Y:{output_pose[1]:7.1f} "
                          f"Z:{output_pose[2]:7.1f} | R:{output_pose[3]:6.1f} "
                          f"P:{output_pose[4]:6.1f} Yw:{output_pose[5]:6.1f} "
                          f"| {self.fps:.0f}fps   ", end='', flush=True)

            # ═════════════════════════════════════════════════════════════
            # FALLBACK PATH: CAMERA LOST MARKERS
            # ═════════════════════════════════════════════════════════════
            else:
                if self.camera_lost_time is None:
                    self.camera_lost_time = now_t

                lost_for = now_t - self.camera_lost_time
                self.pose_confidence = max(0, self.pose_confidence - 1)

                if self.imu.available and self.imu_calibrated:
                    imu_rpy = self._imu_relative_euler()

                    if lost_for <= IMU_TAKEOVER_FULL_SECS:
                        # PHASE 1 (0–5s): Full 6-DOF
                        # Position from last camera, rotation from live IMU
                        output_pose = np.array([
                            self.last_valid_pose[0],
                            self.last_valid_pose[1],
                            self.last_valid_pose[2],
                            imu_rpy[0],
                            imu_rpy[1],
                            imu_rpy[2]],
                            dtype=np.float32)
                        mode  = f"IMU BACKUP (full 6DOF) {lost_for:.1f}s"
                        color = (0, 200, 255)  # Orange

                    else:
                        # PHASE 2 (>5s): XYZ frozen, only rotation updates
                        # Accelerometer drift makes position estimates unreliable
                        # beyond ~5 seconds, so we freeze it.
                        if self.imu_frozen_xyz is None:
                            self.imu_frozen_xyz = self.last_valid_pose[:3].copy()

                        output_pose = np.array([
                            self.imu_frozen_xyz[0],
                            self.imu_frozen_xyz[1],
                            self.imu_frozen_xyz[2],
                            imu_rpy[0],
                            imu_rpy[1],
                            imu_rpy[2]],
                            dtype=np.float32)
                        mode  = f"IMU BACKUP (XYZ frozen) {lost_for:.1f}s"
                        color = (0, 100, 255)  # Red-orange

                    cal = self.imu.get_cal()
                    self._send_udp(output_pose)
                    self._draw_hud(frame, output_pose, color, mode)
                    print(f"\r[IMU] X:{output_pose[0]:7.1f} Y:{output_pose[1]:7.1f} "
                          f"Z:{output_pose[2]:7.1f} | R:{output_pose[3]:6.1f} "
                          f"P:{output_pose[4]:6.1f} Yw:{output_pose[5]:6.1f} "
                          f"| lost:{lost_for:.1f}s cal:{cal} {self.fps:.0f}fps   ",
                          end='', flush=True)

                else:
                    # NO IMU: Hold last camera pose
                    output_pose = self.last_valid_pose.copy()
                    self._send_udp(output_pose)
                    self._draw_hud(frame, output_pose, (80, 80, 80),
                                   "MARKERS LOST — HOLDING LAST POSE")
                    cv2.putText(frame, "NO IMU BACKUP", (20, 310),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # ─────────────────────────────────────────────────────────────
            # STATUS BAR (TOP OF FRAME)
            # ─────────────────────────────────────────────────────────────
            src = ("CAM"  if all_detected else
                   "IMU"  if (self.imu.available and self.imu_calibrated) else
                   "HOLD")
            cv2.putText(frame,
                        f"{self.fps:.0f}FPS | {src} | UDP:{TARGET_IP} | Q=quit A=axes",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

            cv2.imshow("6DOF Tracker (Camera + IMU)", frame)

            # waitKey drives the OpenCV GUI event loop
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self._shutdown()
                return
            elif key == ord('a'):
                self.draw_axes = not self.draw_axes

    # ════════════════════════════════════════════════════════════════════════
    # SHUTDOWN
    # ════════════════════════════════════════════════════════════════════════

    def _shutdown(self):
        """Gracefully stop all threads, release hardware, close windows."""
        self._grab_running = False
        self._grab_thread.join(timeout=1)
        self._kb_listener.stop()
        self.cap.release()
        cv2.destroyAllWindows()
        self.imu.stop()
        self.sock.close()
        print("\nDone.")


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                           PROGRAM ENTRY POINT                              ║
# ╚════════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    tracker = CombinedTracker()
    tracker.run()