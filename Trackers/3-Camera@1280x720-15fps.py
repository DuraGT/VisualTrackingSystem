"""
╔════════════════════════════════════════════════════════════════════════════╗
║            3-CAMERA 6DOF POSITION & ORIENTATION TRACKER                    ║
║          With Kalman Filtering & Best-View-Wins Camera Selection           ║
║                                                                             ║
║  PURPOSE                                                                    ║
║  ──────────────────────────────────────────────────────────────────────── ║
║  Tracks full 6-DOF pose using THREE redundant cameras. Each camera runs    ║
║  marker detection independently; the "best" camera (most markers + lowest  ║
║  reprojection error) is selected each frame. Kalman filtering smooths      ║
║  position + rotation velocity. Falls back to IMU when all cameras lose     ║
║  markers.                                                                   ║
║                                                                             ║
║  KEY DIFFERENCES FROM SINGLE-CAMERA VERSION                                ║
║  ──────────────────────────────────────────────────────────────────────── ║
║  • 3 independent camera streams, grabbed in parallel                       ║
║  • "Best view" selection (most markers, lowest reprojection error)         ║
║  • Kalman filter as final smoothing stage (12-state constant-velocity)     ║
║  • No deadzone — all 6 values reported, never zeroed out                   ║
║  • Multi-view display with live thumbnails + centralized HUD panel         ║
║                                                                             ║
║  HARDWARE REQUIRED                                                          ║
║  ──────────────────────────────────────────────────────────────────────── ║
║  • 3 × Webcams (USB, MJPG codec, minimum 640×480 each)                    ║
║  • BNO055 IMU (optional, for fallback rotation)                            ║
║  • 4 × ArUco markers (IDs: 0, 1, 2, 3) on target object                   ║
║                                                                             ║
║  OUTPUT                                                                     ║
║  ──────────────────────────────────────────────────────────────────────── ║
║  • Display: 3 live camera tiles (top) + unified HUD panel (bottom right)   ║
║  • Active camera highlighted with green border + ★ACTIVE label             ║
║  • HUD shows all 6 values, FPS, data source, IMU calibration status        ║
║  • UDP: Kalman-filtered pose to TARGET_IP:TARGET_PORT, 60 Hz               ║
║  • Console: Real-time status per frame                                     ║
║                                                                             ║
║  KEYBOARD CONTROLS                                                          ║
║  ──────────────────────────────────────────────────────────────────────── ║
║  • Q = Quit                                                                ║
║  • A = Toggle 3-D axis overlay on all detected markers                     ║
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
from collections import deque
from pynput import keyboard


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                         CONFIGURATION SECTION                              ║
# ╚════════════════════════════════════════════════════════════════════════════╝

# Network: Where to send UDP pose updates
TARGET_IP   = "10.98.109.221"
TARGET_PORT = 5006

# Hardware: Serial port for IMU, camera indices to use
IMU_PORT    = "COM4"
IMU_BAUD    = 115200
CAM_INDICES = [2, 1, 0]  # Device indices for 3 cameras

# Camera capture settings (applied to all 3)
CAM_WIDTH  = 1280
CAM_HEIGHT = 720
CAM_FPS    = 60

# Tracking: When to switch IMU strategy if all cameras lose markers
IMU_TAKEOVER_FULL_SECS = 5.0

# Kalman filter tuning
# Raise *_NOISE values to increase smoothing (less responsive to noise).
# Lower *_NOISE values to increase responsiveness (less smoothing).
KF_PROCESS_NOISE_POS = 1e-2   # mm² per step (position)
KF_PROCESS_NOISE_ANG = 1e-2   # deg² per step (rotation)
KF_MEAS_NOISE_POS    = 5.0    # mm² (measurement noise — how much to trust raw pose)
KF_MEAS_NOISE_ANG    = 1.0    # deg²


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                    QUATERNION MATHEMATICS HELPERS                          ║
# ╚════════════════════════════════════════════════════════════════════════════╝

def qmul(q1, q2):
    """
    Quaternion multiplication (Hamilton product).
    Chains two rotations: q_relative = q_origin_inv * q_current.
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
    Quaternion conjugate (rotation inverse).
    For a unit quaternion, negating x,y,z reverses the rotation.
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])


def q2euler(q):
    """
    Convert unit quaternion [w, x, y, z] to Euler angles [Roll, Pitch, Yaw].
    Uses ZYX convention (aerospace standard).
    Returns angles in degrees.
    """
    w, x, y, z = q
    roll  = np.degrees(np.arctan2(2*(w*x + y*z),  1 - 2*(x*x + y*y)))
    pitch = np.degrees(np.arcsin(np.clip(2*(w*y - z*x), -1, 1)))
    yaw   = np.degrees(np.arctan2(2*(w*z + x*y),  1 - 2*(y*y + z*z)))
    return np.array([roll, pitch, yaw])


def rotation_matrix_to_quaternion(R):
    """
    Convert a 3×3 rotation matrix to a unit quaternion [w, x, y, z].
    Uses Shepperd's method for numerical stability.
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
# ║                         KALMAN FILTER (Final Stage)                        ║
# ║                                                                             ║
# ║  12-state constant-velocity model for smooth, lag-free pose tracking.      ║
# ║  State: [X, Y, Z, Roll, Pitch, Yaw, Vx, Vy, Vz, Vroll, Vpitch, Vyaw]     ║
# ║                                                                             ║
# ║  Why final stage? After median + clamp + EMA, the pose is already clean.  ║
# ║  The Kalman filter adds one more layer of smoothing while estimating      ║
# ║  velocity, allowing us to better predict future states.                   ║
# ╚════════════════════════════════════════════════════════════════════════════╝

class PoseKalmanFilter:
    """
    12-state constant-velocity Kalman filter for 6-DOF pose.

    State vector:
        [X, Y, Z, Roll, Pitch, Yaw, Vx, Vy, Vz, Vroll, Vpitch, Vyaw]

    Measurement:
        [X, Y, Z, Roll, Pitch, Yaw]

    The filter is seeded directly from the first measurement, so there's
    no warm-up lag — it's immediately responsive.
    """

    def __init__(self):
        """Initialize the Kalman filter with constant-velocity dynamics."""
        kf = cv2.KalmanFilter(12, 6, 0, cv2.CV_32F)

        # Transition matrix: position += velocity each step
        # This is the "constant velocity" model assumption.
        kf.transitionMatrix = np.eye(12, dtype=np.float32)
        for i in range(6):
            kf.transitionMatrix[i, i + 6] = 1.0

        # Measurement matrix: we observe position/angle directly
        # (not velocity — that's estimated from the pose difference)
        kf.measurementMatrix = np.zeros((6, 12), dtype=np.float32)
        for i in range(6):
            kf.measurementMatrix[i, i] = 1.0

        # Process noise: how much we expect the system to deviate from
        # the constant-velocity model. Higher = trusts measurements more.
        # Lower = trusts the motion model more (smoother but may lag).
        pn = np.eye(12, dtype=np.float32)
        for i in range(3):
            pn[i,   i]   = KF_PROCESS_NOISE_POS           # Position process noise
            pn[i+3, i+3] = KF_PROCESS_NOISE_ANG           # Rotation process noise
            pn[i+6, i+6] = KF_PROCESS_NOISE_POS * 10      # Velocity process noise (higher)
            pn[i+9, i+9] = KF_PROCESS_NOISE_ANG * 10
        kf.processNoiseCov = pn

        # Measurement noise: how much we expect measurements to be wrong.
        # Higher = trusts measurements less (more smoothing).
        # Lower = trusts measurements more (less smoothing, noisier).
        mn = np.eye(6, dtype=np.float32)
        for i in range(3):
            mn[i,   i]   = KF_MEAS_NOISE_POS   # Position measurement noise
            mn[i+3, i+3] = KF_MEAS_NOISE_ANG   # Rotation measurement noise
        kf.measurementNoiseCov = mn

        kf.errorCovPost = np.eye(12, dtype=np.float32)
        self._kf   = kf
        self._init = False

    def update(self, meas: np.ndarray) -> np.ndarray:
        """
        Feed a new measurement and get the smoothed pose.

        Args:
            meas: 6-element array [X, Y, Z, Roll, Pitch, Yaw]

        Returns:
            smoothed_pose: 6-element array after Kalman filtering
        """
        m = meas.astype(np.float32).reshape(6, 1)

        # On first call, seed the state directly from the measurement.
        # This avoids the "ramp-up" lag that some Kalman filters have.
        if not self._init:
            self._kf.statePre  = np.zeros((12, 1), dtype=np.float32)
            self._kf.statePost = np.zeros((12, 1), dtype=np.float32)
            self._kf.statePre[:6]  = m
            self._kf.statePost[:6] = m
            self._init = True

        # Predict the next state based on motion model
        self._kf.predict()

        # Correct the prediction based on the new measurement
        return self._kf.correct(m)[:6].flatten()

    def reset(self):
        """Reset the filter (e.g., when camera recovers after long loss)."""
        self._init = False


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                       IMU READER (Background Thread)                       ║
# ║                                                                             ║
# ║  Manages BNO055 IMU connection. Reads quaternion + calibration data        ║
# ║  continuously in a daemon thread. Optional; tracker works without it.      ║
# ╚════════════════════════════════════════════════════════════════════════════╝

class IMUReader:
    """
    Manages BNO055 IMU connection and reads quaternion + calibration data.

    Expected serial format from IMU:
        QUAT:w,x,y,z|CAL:sys,gyro,accel,mag

    Both tokens optional; order doesn't matter.
    """

    def __init__(self, port, baud=115200):
        """
        Open serial connection and start the read loop in a background thread.

        Args:
            port: Serial port name (e.g., "COM4" on Windows)
            baud: Baud rate (BNO055 default is 115200)
        """
        self._lock   = threading.Lock()
        self.cur_q   = np.array([1., 0., 0., 0.])  # Identity quaternion
        self.cal     = np.zeros(4, dtype=int)       # Calibration scores [0-3]
        self.running = True

        try:
            self.ser = serial.Serial(port, baud, timeout=1)
            time.sleep(2)  # Wait for Arduino bootloader
            self.ser.reset_input_buffer()
            self.available = True
            print(f"✓ IMU connected on {port}")
        except serial.SerialException as e:
            print(f"⚠  IMU not available ({e}). Backup disabled.")
            self.available = False
            return

        # Start background read thread (daemon = dies with main process)
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        """Background thread: read and parse serial lines continuously."""
        while self.running:
            try:
                line = self.ser.readline().decode(errors='ignore').strip()
                if 'QUAT:' in line or 'EULER:' in line:
                    self._parse(line)
            except Exception:
                pass  # Timeout or decode error — continue

    def _parse(self, line):
        """Parse a line and update quaternion + calibration. Thread-safe."""
        with self._lock:
            for part in line.split('|'):
                try:
                    if part.startswith('QUAT:'):
                        q = np.array([float(v) for v in part[5:].split(',')])
                        n = np.linalg.norm(q)
                        if n > 0:
                            self.cur_q = q / n  # Normalize to unit quaternion
                    elif part.startswith('CAL:'):
                        self.cal = np.array([int(float(v))
                                             for v in part[4:].split(',')])
                except Exception:
                    pass

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
# ║                      CAMERA GRABBER (Per-Camera Thread)                    ║
# ║                                                                             ║
# ║  One background thread per camera. Opens staggered 2 seconds apart to      ║
# ║  avoid overwhelming the USB hub with simultaneous device negotiations.     ║
# ╚════════════════════════════════════════════════════════════════════════════╝

class CameraGrabber:
    """
    Manages one USB camera with a dedicated background frame-grab thread.

    Staggered initialization (2 s apart) prevents USB hub saturation.
    Each thread calls cap.read() in a tight loop, storing the latest frame.
    Main loop picks up waiting frames without blocking on camera I/O.
    """

    def __init__(self, cam_index, label):
        """
        Initialize a camera grabber.

        Args:
            cam_index: OpenCV device index (0, 1, 2, etc.)
            label: Human-readable label (e.g., "CAM-0")
        """
        self.label     = label
        self.index     = cam_index
        self._frame    = None
        self._frame_id = 0
        self._lock     = threading.Lock()
        self._running  = True
        self.available = False
        self.cap       = None

        # Stagger opening by 2 seconds per camera to reduce USB contention
        time.sleep(cam_index * 2.0)

        # Try to open the camera with DirectShow (fast initialization on Windows)
        cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print(f"⚠  {label}: could not open device {cam_index}")
            return

        # MJPG codec: compresses on-camera, fitting 60 FPS over USB 2.0
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS,          CAM_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)         # Minimal latency
        cap.set(cv2.CAP_PROP_AUTOFOCUS,    0)         # Continuous autofocus

        # Test first read to verify the camera works
        ret, _ = cap.read()
        if not ret:
            print(f"⚠  {label}: opened but first read failed")
            cap.release()
            return

        # Log what the driver actually agreed to
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"✓ {label} (dev {cam_index}): {w}×{h} @ {fps:.0f} FPS")

        self.cap = cap

        # Start the frame grabber thread (daemon = dies with main process)
        threading.Thread(target=self._grabber, daemon=True).start()

        # Block until first frame arrives (poll every 50 ms, up to 1.5 s)
        for _ in range(30):
            if self._frame is not None:
                self.available = True
                break
            time.sleep(0.05)

        if not self.available:
            print(f"⚠  {label}: grabber thread produced no frames")

    def _grabber(self):
        """Background thread: grab frames continuously from the camera."""
        while self._running:
            ret, frame = self.cap.read()
            if ret:
                with self._lock:
                    self._frame    = frame
                    self._frame_id += 1

    def get_frame(self):
        """
        Get the latest frame without blocking.

        Returns:
            (frame, frame_id): Latest frame + its ID, or (None, -1) if no frame ready
        """
        with self._lock:
            if self._frame is None:
                return None, -1
            return self._frame.copy(), self._frame_id

    def release(self):
        """Stop the grabber thread and release the camera device."""
        self._running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                         MAIN TRACKER CLASS                                 ║
# ║                                                                             ║
# ║  Orchestrates 3 cameras, detects markers on each, selects the best view,   ║
# ║  computes 6-DOF pose, applies Kalman filtering, and outputs via UDP.       ║
# ╚════════════════════════════════════════════════════════════════════════════╝

class CombinedTracker:
    """
    Main 3-camera tracker. Manages cameras, marker detection, best-view
    selection, Kalman filtering, and pose output.
    """

    def __init__(self):
        """Initialize all hardware, filters, and state variables."""

        # ─────────────────────────────────────────────────────────────────
        # UDP SOCKET
        # ─────────────────────────────────────────────────────────────────
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.sock.connect((TARGET_IP, TARGET_PORT))
        print(f"✓ UDP → {TARGET_IP}:{TARGET_PORT}")

        # ─────────────────────────────────────────────────────────────────
        # IMU READER
        # ─────────────────────────────────────────────────────────────────
        self.imu = IMUReader(IMU_PORT)

        # Quaternion that "undoes" the IMU's calibrated origin orientation
        self.imu_origin_q_inv = np.array([1., 0., 0., 0.])
        self.imu_calibrated   = False

        # Short history for smoothing IMU Euler output (gyro noise reduction)
        self._imu_euler_hist  = deque(maxlen=5)

        # ─────────────────────────────────────────────────────────────────
        # MARKER GEOMETRY
        # ─────────────────────────────────────────────────────────────────
        self.marker_size = 0.020  # 15 mm square
        self.plane_ids   = [1, 2, 3]  # Markers forming the tracked plane
        self.ref_id      = 0           # Reference marker (origin)

        # 3-D corner positions of a marker in its local frame
        s = self.marker_size / 2
        self.obj_pts = np.array([[-s,  s, 0],
                                  [ s,  s, 0],
                                  [ s, -s, 0],
                                  [-s, -s, 0]], dtype=np.float32)

        # ─────────────────────────────────────────────────────────────────
        # CAMERA CALIBRATION
        # ─────────────────────────────────────────────────────────────────
        # Load intrinsics for accurate depth/pose estimation
        try:
            import json
            with open('camera_calibration_1280x720.json') as f:
                calib = json.load(f)
            self.cam_mat = np.array(calib['camera_matrix'], dtype=np.float32)
            self.dist    = np.array(calib['dist_coeffs'],   dtype=np.float32).flatten()
            print(f"✓ Calibration loaded (err: {calib['rms_error']:.4f}px)")
        except FileNotFoundError:
            # Fallback: pinhole model (less accurate)
            self.cam_mat = np.array([[800, 0, 320],
                                     [0, 800, 240],
                                     [0,   0,   1]], dtype=np.float32)
            self.dist = np.zeros(5, dtype=np.float32)
            print("⚠  Default camera matrix — run calibration for accuracy")

        # ─────────────────────────────────────────────────────────────────
        # ARUCORE DETECTOR
        # ─────────────────────────────────────────────────────────────────
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        params = aruco.DetectorParameters()
        params.cornerRefinementMethod        = aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementMaxIterations = 15
        params.cornerRefinementMinAccuracy   = 0.05
        params.adaptiveThreshWinSizeMin  = 5
        params.adaptiveThreshWinSizeMax  = 21
        params.adaptiveThreshWinSizeStep = 8
        params.minMarkerPerimeterRate    = 0.05
        params.maxMarkerPerimeterRate    = 4.0
        self.detector = aruco.ArucoDetector(self.aruco_dict, params)

        # ─────────────────────────────────────────────────────────────────
        # CAMERAS
        # ─────────────────────────────────────────────────────────────────
        # Open all 3 cameras (staggered 2 s apart for USB stability)
        print("Opening cameras (staggered 2 s apart) ...")
        self.cameras = []
        for i, idx in enumerate(CAM_INDICES):
            self.cameras.append(CameraGrabber(idx, f"CAM-{i}"))

        n_ok = sum(1 for c in self.cameras if c.available)
        if n_ok == 0:
            print("✗ No cameras available — exiting.")
            sys.exit(1)
        print(f"✓ {n_ok}/{len(self.cameras)} cameras ready")

        # ─────────────────────────────────────────────────────────────────
        # SMOOTHING PIPELINE (Median + Clamp + EMA before Kalman)
        # ─────────────────────────────────────────────────────────────────
        self.MEDIAN_WIN   = 5
        self.raw_history  = deque(maxlen=self.MEDIAN_WIN)
        self.alpha        = 0.20   # EMA weight
        self.MAX_POS_STEP = 25.0   # mm per frame
        self.MAX_ANG_STEP = 10.0   # degrees per frame
        self.filtered_pose = None

        # ─────────────────────────────────────────────────────────────────
        # KALMAN FILTER
        # ─────────────────────────────────────────────────────────────────
        # Final smoothing stage: estimates velocity, smooths output further
        self.kalman = PoseKalmanFilter()

        # ─────────────────────────────────────────────────────────────────
        # TRACKER STATE
        # ─────────────────────────────────────────────────────────────────
        self.last_valid_pose  = np.zeros(6, dtype=np.float32)
        self.pose_confidence  = 0      # Rises when markers visible
        self.max_confidence   = 20     # Ceiling
        self.camera_lost_time = None   # When all cameras lost markers
        self.imu_frozen_xyz   = None   # Position snapshot at 5-second mark

        # ─────────────────────────────────────────────────────────────────
        # DISPLAY & TIMING
        # ─────────────────────────────────────────────────────────────────
        self.draw_axes   = True    # Toggle axes overlay with 'A'
        self.fps         = 0.0
        self.frame_count = 0
        self.last_time   = time.perf_counter()

        # Display layout: 3 camera tiles (top) + info panel (bottom-right)
        self.TILE_W      = 426     # Width of each camera tile
        self.TILE_H      = 320     # Height of each camera tile

        # ─────────────────────────────────────────────────────────────────
        # KEYBOARD LISTENER
        # ─────────────────────────────────────────────────────────────────
        self._key_queue   = deque()
        self._kb_listener = keyboard.Listener(on_press=self._on_key)
        self._kb_listener.start()

        print("=" * 60)
        print("3-CAM 6DOF TRACKER  |  Q=quit  A=toggle axes")
        print("=" * 60)

    # ════════════════════════════════════════════════════════════════════════
    # KEYBOARD INPUT
    # ════════════════════════════════════════════════════════════════════════

    def _on_key(self, key):
        """pynput callback: queue any printable key press."""
        try:
            self._key_queue.append(key.char.lower())
        except AttributeError:
            pass  # Special keys (Shift, Ctrl, etc.)

    # ════════════════════════════════════════════════════════════════════════
    # POSE MATHEMATICS
    # ════════════════════════════════════════════════════════════════════════

    def _avg_quaternions(self, quats):
        """
        Average multiple unit quaternions via eigendecomposition.
        Returns the eigenvector of the largest eigenvalue of the
        accumulator matrix built from outer products.
        """
        M = np.zeros((4, 4))
        for q in quats:
            if q[0] < 0:  # Force same hemisphere
                q = -q
            M += np.outer(q, q)
        M /= len(quats)
        _, eigvecs = np.linalg.eigh(M)
        return eigvecs[:, -1]

    def _quat_to_rot(self, q):
        """Convert a unit quaternion to a 3×3 rotation matrix."""
        w, x, y, z = q / np.linalg.norm(q)
        return np.array([
            [1-2*(y*y+z*z),  2*(x*y-z*w),   2*(x*z+y*w)],
            [2*(x*y+z*w),    1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w),    2*(y*z+x*w),   1-2*(x*x+y*y)]])

    def _euler_from_mat(self, R):
        """
        Extract Euler angles [Roll, Pitch, Yaw] from a rotation matrix.
        Uses ZYX convention. Handles gimbal lock by setting yaw=0.
        Returns angles in degrees.
        """
        sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
        if sy > 1e-6:
            roll  = np.arctan2(R[2,1], R[2,2])
            pitch = np.arctan2(-R[2,0], sy)
            yaw   = np.arctan2(R[1,0], R[0,0])
        else:
            roll  = np.arctan2(-R[1,2], R[1,1])
            pitch = np.arctan2(-R[2,0], sy)
            yaw   = 0.0
        return np.degrees([roll, pitch, yaw])

    def _compute_relative_pose(self, rvecs, tvecs):
        """
        Compute the 6-DOF pose of the plane (markers 1-3) relative to
        the reference marker (marker 0).

        Returns:
            (pos_mm, euler_deg): Position in mm, rotation in degrees
                                 or (None, None) on error
        """
        try:
            R_0, _ = cv2.Rodrigues(rvecs[self.ref_id])
            t_0    = tvecs[self.ref_id].flatten()
            quats  = []
            for pid in self.plane_ids:
                R_p, _ = cv2.Rodrigues(rvecs[pid])
                quats.append(rotation_matrix_to_quaternion(R_p))
            R_avg = self._quat_to_rot(self._avg_quaternions(quats))
            R_rel = R_0.T @ R_avg
            t_avg = np.mean([tvecs[pid].flatten() for pid in self.plane_ids], axis=0)
            t_rel = R_0.T @ (t_avg - t_0)
            return t_rel * 1000.0, self._euler_from_mat(R_rel)
        except Exception:
            return None, None

    def _clamp_step(self, new_pose, prev_pose):
        """
        Velocity clamp: prevent any DOF from jumping more than the limit.
        Limits scale with confidence so locked tracks can move faster.
        """
        clamped   = new_pose.copy()
        delta     = new_pose - prev_pose
        scale     = 1.0 + (self.pose_confidence / self.max_confidence) * 3.0
        pos_limit = self.MAX_POS_STEP * scale
        ang_limit = self.MAX_ANG_STEP * scale
        for i in range(3):
            if abs(delta[i]) > pos_limit:
                clamped[i] = prev_pose[i] + np.sign(delta[i]) * pos_limit
        for i in range(3, 6):
            if abs(delta[i]) > ang_limit:
                clamped[i] = prev_pose[i] + np.sign(delta[i]) * ang_limit
        return clamped

    # ════════════════════════════════════════════════════════════════════════
    # IMU CALIBRATION & ORIENTATION
    # ════════════════════════════════════════════════════════════════════════

    def _calibrate_imu_to_camera(self, _euler):
        """
        Snapshot the IMU's current quaternion as the calibration origin.
        From this point, _imu_relative_euler() reports angles as deltas.
        Called every frame the camera is successfully tracking.
        """
        if not self.imu.available:
            return
        self.imu_origin_q_inv = qconj(self.imu.get_quaternion())
        self.imu_calibrated   = True

    def _imu_relative_euler(self):
        """
        Return IMU's current orientation relative to calibration origin.
        Smoothed with a 5-frame moving average to reduce gyro noise.
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
        Silently drops if socket buffer is full.

        Format: X,Y,Z,Roll,Pitch,Yaw (no trailing comma, no newline)
        """
        try:
            self.sock.send(
                f"{float(pose[0]):.2f},"
                f"{float(pose[1]):.2f},"
                f"{float(pose[2]):.2f},"
                f"{float(pose[3]):.2f},"
                f"{float(pose[4]):.2f},"
                f"{float(pose[5]):.2f}".encode())
        except (BlockingIOError, OSError):
            pass
    # ════════════════════════════════════════════════════════════════════════
    # MARKER DETECTION
    # ════════════════════════════════════════════════════════════════════════

    def _detect_on_frame(self, frame):
        """
        Run ArUco marker detection on a single frame.

        Returns:
            (rvecs, tvecs, corners): Detected markers' rotation vectors,
                                     translation vectors, and 2-D corners
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        raw_corners, ids, _ = self.detector.detectMarkers(gray)
        rvecs = {}
        tvecs = {}
        corners = {}

        if ids is not None:
            for i, mid in enumerate(ids.flatten()):
                if mid not in (0, 1, 2, 3):
                    continue
                ok, rvec, tvec = cv2.solvePnP(
                    self.obj_pts, raw_corners[i][0],
                    self.cam_mat, self.dist,
                    flags=cv2.SOLVEPNP_ITERATIVE)
                if ok:
                    rvecs[mid]   = rvec
                    tvecs[mid]   = tvec
                    corners[mid] = raw_corners[i][0]

        return rvecs, tvecs, corners

    def _select_best(self, all_rvecs, all_tvecs, all_corners):
        """
        Select the "best" camera view based on:
          1. Most markers detected
          2. Lowest reprojection error (if tied on marker count)

        Returns:
            best_idx: Index of the best camera (0, 1, or 2), or -1 if none available
        """
        best_idx = -1
        best_count = -1
        best_err = float('inf')

        for ci, cam in enumerate(self.cameras):
            if not cam.available:
                continue

            rv, tv, co = all_rvecs[ci], all_tvecs[ci], all_corners[ci]

            # Count how many of the 4 required markers this camera has
            count = sum(1 for m in (0, 1, 2, 3) if m in rv)

            # Compute reprojection error: how well do the solved poses
            # predict the detected 2-D corners?
            errs  = []
            for mid in rv:
                proj, _ = cv2.projectPoints(self.obj_pts, rv[mid], tv[mid],
                                            self.cam_mat, self.dist)
                errs.append(float(np.mean(np.linalg.norm(
                    proj.reshape(-1, 2) - co[mid], axis=1))))
            err = float(np.mean(errs)) if errs else float('inf')

            # Update best if this camera is better
            if count > best_count or (count == best_count and err < best_err):
                best_idx, best_count, best_err = ci, count, err

        return best_idx

    # ════════════════════════════════════════════════════════════════════════
    # DISPLAY RENDERING
    # ════════════════════════════════════════════════════════════════════════

    def _draw_on_tile(self, tile, rvecs, tvecs, corners, is_active, cam_idx):
        """
        Draw detected markers, axes, and plane triangle on a camera tile.

        Highlights the active camera with a green border and ★ACTIVE label.
        """
        th, tw = tile.shape[:2]
        sx, sy = tw / CAM_WIDTH, th / CAM_HEIGHT

        # Scale the camera matrix to match the tile resolution
        cm = self.cam_mat.copy()
        cm[0,0]*=sx
        cm[1,1]*=sy
        cm[0,2]*=sx
        cm[1,2]*=sy

        # Draw detected markers as green polygons
        for mid, c2d in corners.items():
            sc = c2d.copy()
            sc[:,0] *= sx
            sc[:,1] *= sy
            pts = sc.astype(int)
            cv2.polylines(tile, [pts], True, (0, 255, 0), 1)
            cv2.putText(tile, str(mid), tuple(pts[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            # Draw 3-D axes if requested
            if self.draw_axes:
                cv2.drawFrameAxes(tile, cm, self.dist,
                                  rvecs[mid], tvecs[mid], 0.015, 1)

        # Draw magenta triangle connecting the three plane markers
        tri = []
        for pid in self.plane_ids:
            if pid in corners:
                tri.append((int(np.mean(corners[pid][:,0]) * sx),
                             int(np.mean(corners[pid][:,1]) * sy)))
        if len(tri) == 3:
            cv2.line(tile, tri[0], tri[1], (255, 0, 255), 1)
            cv2.line(tile, tri[1], tri[2], (255, 0, 255), 1)
            cv2.line(tile, tri[2], tri[0], (255, 0, 255), 1)

        # Header bar: camera label, active status, marker count
        n    = sum(1 for m in (0,1,2,3) if m in rvecs)
        col  = (0, 255, 0) if is_active else (100, 100, 100)
        star = " ★ACTIVE" if is_active else ""
        cv2.rectangle(tile, (0, 0), (tw, 22), (0, 0, 0), -1)
        cv2.putText(tile, f"CAM-{cam_idx}{star}  [{n}/4]",
                    (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

        # Border: green if active, grey if not
        cv2.rectangle(tile, (0,0), (tw-1,th-1),
                      (0, 220, 0) if is_active else (50, 50, 50), 2)

    def _build_info_panel(self, pose, src_lbl, hud_col, mode_lbl):
        """
        Build the unified info panel showing all 6 pose values, FPS, and status.
        This panel is bright and always visible (no greying).
        """
        PW, PH = self.TILE_W, self.TILE_H
        panel = np.full((PH, PW, 3), (18, 18, 28), dtype=np.uint8)

        # Title bar
        cv2.rectangle(panel, (0, 0), (PW, 28), (30, 30, 50), -1)
        cv2.putText(panel, "6-DOF OUTPUT", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)

        # Source badge (top-right)
        cv2.rectangle(panel, (PW-112, 4), (PW-4, 24), hud_col, -1)
        cv2.putText(panel, src_lbl, (PW-108, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)

        # Pose values (always bright green — never greyed)
        for i, (lbl, val) in enumerate(
                zip(["X mm", "Y mm", "Z mm", "Roll ", "Pitch", "Yaw  "], pose)):
            y = 52 + i * 38
            cv2.line(panel, (8, y-12), (PW-8, y-12), (35, 35, 50), 1)
            cv2.putText(panel, lbl, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (130, 130, 150), 1)
            cv2.putText(panel, f"{val:+9.2f}", (72, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (80, 255, 120), 1)

        # Footer section: FPS, UDP target, mode, IMU calibration
        sy = 52 + 6*38 + 4
        cv2.line(panel, (8, sy-6), (PW-8, sy-6), (50, 50, 70), 1)
        for line in [f"FPS  : {self.fps:.0f}",
                     f"UDP  : {TARGET_IP}:{TARGET_PORT}",
                     f"Mode : {mode_lbl}"]:
            cv2.putText(panel, line, (10, sy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 170, 190), 1)
            sy += 22

        # IMU calibration status (if available)
        if self.imu.available:
            cal = self.imu.get_cal()
            cv2.putText(panel,
                        f"IMU  : sys={cal[0]} gyr={cal[1]} acc={cal[2]} mag={cal[3]}",
                        (10, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (150, 150, 175), 1)

        # Border
        cv2.rectangle(panel, (0, 0), (PW-1, PH-1), (55, 55, 75), 1)
        return panel

    def _build_display(self, frames, all_rvecs, all_tvecs, all_corners,
                       best_idx, pose, src_lbl, hud_col, mode_lbl):
        """
        Build the full composite display: 3 camera tiles (top) + info panel (bottom-right).
        """
        TW, TH = self.TILE_W, self.TILE_H
        tiles = []

        # Process each camera
        for ci in range(3):
            if frames[ci] is None:
                # Camera not available: grey placeholder
                tile = np.zeros((TH, TW, 3), dtype=np.uint8)
                cv2.putText(tile, f"CAM-{ci}  N/A", (TW//2-55, TH//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (55, 55, 55), 1)
                cv2.rectangle(tile, (0,0), (TW-1,TH-1), (40,40,40), 1)
            else:
                # Resize frame to tile size and draw markers
                tile = cv2.resize(frames[ci], (TW, TH))
                self._draw_on_tile(tile, all_rvecs[ci], all_tvecs[ci],
                                   all_corners[ci], ci == best_idx, ci)
            tiles.append(tile)

        # Top row: 3 camera tiles
        top    = np.hstack(tiles)
        # Bottom-left: empty spacer
        spacer = np.zeros((TH, TW*2, 3), dtype=np.uint8)
        # Bottom-right: info panel
        info   = self._build_info_panel(pose, src_lbl, hud_col, mode_lbl)

        return np.vstack([top, np.hstack([spacer, info])])

    # ════════════════════════════════════════════════════════════════════════
    # MAIN LOOP
    # ════════════════════════════════════════════════════════════════════════

    def run(self):
        """
        Main tracking loop. Each iteration:
          1. Grab frames from all 3 cameras
          2. Detect markers on each
          3. Select best camera view (most markers, lowest error)
          4. Compute 6-DOF pose via median+clamp+EMA+Kalman pipeline
          5. Fall back to IMU if all cameras lose markers
          6. Send UDP, render display, update console
        """
        time.sleep(0.2)  # Let everything settle
        self._key_queue.clear()

        # Initialize pose display variables
        output_pose = np.zeros(6, dtype=np.float32)
        hud_col     = (80, 80, 80)
        mode_lbl    = "WAITING..."
        src_lbl     = "---"

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
            # GRAB FRAMES & DETECT ON ALL CAMERAS
            # ─────────────────────────────────────────────────────────────
            frames      = []
            all_rvecs   = []
            all_tvecs   = []
            all_corners = []

            for cam in self.cameras:
                if not cam.available:
                    frames.append(None)
                    all_rvecs.append({})
                    all_tvecs.append({})
                    all_corners.append({})
                    continue

                frame, _ = cam.get_frame()
                if frame is None:
                    frames.append(None)
                    all_rvecs.append({})
                    all_tvecs.append({})
                    all_corners.append({})
                    continue

                frames.append(frame)
                rv, tv, co = self._detect_on_frame(frame)
                all_rvecs.append(rv)
                all_tvecs.append(tv)
                all_corners.append(co)

            # Skip this iteration if no cameras produced frames
            if all(f is None for f in frames):
                time.sleep(0.005)
                continue

            # ─────────────────────────────────────────────────────────────
            # SELECT BEST CAMERA
            # ─────────────────────────────────────────────────────────────
            best_idx   = self._select_best(all_rvecs, all_tvecs, all_corners)
            best_rvecs = all_rvecs[best_idx] if best_idx >= 0 else {}
            best_tvecs = all_tvecs[best_idx] if best_idx >= 0 else {}

            # ─────────────────────────────────────────────────────────────
            # FPS COUNTER
            # ─────────────────────────────────────────────────────────────
            self.frame_count += 1
            if self.frame_count >= 30:
                now = time.perf_counter()
                self.fps       = 30 / max(now - self.last_time, 1e-6)
                self.last_time = now
                self.frame_count = 0

            all_detected = all(m in best_rvecs for m in (0, 1, 2, 3))
            now_t = time.perf_counter()

            # ═════════════════════════════════════════════════════════════
            # PRIMARY PATH: CAMERA HAS ALL 4 MARKERS
            # ═════════════════════════════════════════════════════════════
            if all_detected:
                self.camera_lost_time = None
                self.imu_frozen_xyz   = None

                pos, euler = self._compute_relative_pose(best_rvecs, best_tvecs)

                if pos is not None:
                    raw = np.array([pos[0], pos[1], pos[2],
                                    euler[0], euler[1], euler[2]])

                    # STAGE 1: MEDIAN FILTER
                    self.raw_history.append(raw)
                    median_pose = (np.median(self.raw_history, axis=0)
                                   if len(self.raw_history) >= 3 else raw)

                    # STAGE 2: VELOCITY CLAMP
                    if self.filtered_pose is not None:
                        median_pose = self._clamp_step(median_pose, self.filtered_pose)

                    # STAGE 3: EXPONENTIAL MOVING AVERAGE
                    if self.filtered_pose is None:
                        self.filtered_pose = median_pose.copy()
                    else:
                        self.filtered_pose = (self.alpha * median_pose
                                              + (1 - self.alpha) * self.filtered_pose)

                    # STAGE 4: KALMAN FILTER
                    output_pose = self.kalman.update(self.filtered_pose).astype(np.float32)

                    self.last_valid_pose = output_pose.copy()
                    self.pose_confidence = min(self.pose_confidence + 1,
                                              self.max_confidence)

                    self._calibrate_imu_to_camera(output_pose[3:])
                    self._send_udp(output_pose)

                    cam_lbl  = self.cameras[best_idx].label
                    hud_col  = (0, 255, 0) if self.pose_confidence > 10 else (0, 220, 220)
                    mode_lbl = "CAMERA TRACKING"
                    src_lbl  = cam_lbl

                    print(f"\r[{cam_lbl}] "
                          f"X:{output_pose[0]:7.1f} Y:{output_pose[1]:7.1f} "
                          f"Z:{output_pose[2]:7.1f} | "
                          f"R:{output_pose[3]:6.1f} P:{output_pose[4]:6.1f} "
                          f"Yw:{output_pose[5]:6.1f} | {self.fps:.0f}fps   ",
                          end='', flush=True)

            # ═════════════════════════════════════════════════════════════
            # FALLBACK PATH: ALL CAMERAS LOST MARKERS
            # ═════════════════════════════════════════════════════════════
            else:
                if self.camera_lost_time is None:
                    self.camera_lost_time = now_t

                lost_for = now_t - self.camera_lost_time
                self.pose_confidence = max(0, self.pose_confidence - 1)

                if self.imu.available and self.imu_calibrated:
                    imu_rpy = self._imu_relative_euler()

                    if lost_for <= IMU_TAKEOVER_FULL_SECS:
                        # Phase 1: Full 6-DOF
                        output_pose = np.array([
                            self.last_valid_pose[0], self.last_valid_pose[1],
                            self.last_valid_pose[2],
                            imu_rpy[0], imu_rpy[1], imu_rpy[2]],
                            dtype=np.float32)
                        hud_col  = (0, 200, 255)
                        mode_lbl = f"IMU BACKUP (full 6DOF) {lost_for:.1f}s"
                        src_lbl  = "IMU"
                    else:
                        # Phase 2: XYZ frozen
                        if self.imu_frozen_xyz is None:
                            self.imu_frozen_xyz = self.last_valid_pose[:3].copy()
                        output_pose = np.array([
                            self.imu_frozen_xyz[0], self.imu_frozen_xyz[1],
                            self.imu_frozen_xyz[2],
                            imu_rpy[0], imu_rpy[1], imu_rpy[2]],
                            dtype=np.float32)
                        hud_col  = (0, 100, 255)
                        mode_lbl = f"IMU BACKUP (XYZ frozen) {lost_for:.1f}s"
                        src_lbl  = "IMU(frz)"

                    self._send_udp(output_pose)
                    cal = self.imu.get_cal()
                    print(f"\r[IMU] "
                          f"X:{output_pose[0]:7.1f} Y:{output_pose[1]:7.1f} "
                          f"Z:{output_pose[2]:7.1f} | "
                          f"R:{output_pose[3]:6.1f} P:{output_pose[4]:6.1f} "
                          f"Yw:{output_pose[5]:6.1f} | "
                          f"lost:{lost_for:.1f}s cal:{cal} {self.fps:.0f}fps   ",
                          end='', flush=True)

                else:
                    # No IMU: hold last camera pose
                    output_pose = self.last_valid_pose.copy()
                    hud_col     = (80, 80, 80)
                    mode_lbl    = "MARKERS LOST — HOLDING"
                    src_lbl     = "HOLD"
                    self._send_udp(output_pose)

            # ─────────────────────────────────────────────────────────────
            # RENDER DISPLAY
            # ─────────────────────────────────────────────────────────────
            display = self._build_display(
                frames, all_rvecs, all_tvecs, all_corners,
                best_idx, output_pose, src_lbl, hud_col, mode_lbl)

            cv2.imshow("3-Cam 6DOF Tracker", display)

            # Handle window keyboard input
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
        self._kb_listener.stop()
        for cam in self.cameras:
            cam.release()
        cv2.destroyAllWindows()
        self.imu.stop()
        self.sock.close()
        print("\nDone.")


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                           PROGRAM ENTRY POINT                              ║
# ╚════════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    CombinedTracker().run()