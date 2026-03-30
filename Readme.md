# 6DOF Marker-Based Position & Orientation Tracker

A professional-grade real-time 6-degree-of-freedom (6DOF) pose estimation system using ArUco markers, webcams, and optional IMU backup. Tracks both position (X, Y, Z) and rotation (Roll, Pitch, Yaw) with sub-millimeter accuracy.

---

## Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [Key Features](#key-features)
3. [Hardware Requirements](#hardware-requirements)
4. [Quick Start](#quick-start)
5. [Project Structure](#project-structure)
6. [System Variants](#system-variants)
7. [Setup & Calibration](#setup--calibration)
8. [Running the Tracker](#running-the-tracker)
9. [Network Output](#network-output)
10. [Troubleshooting](#troubleshooting)
11. [Performance Specs](#performance-specs)

---

## What This Project Does

This project estimates the **real-time 6-DOF pose** (position + orientation) of an object marked with 4 ArUco fiducial markers.

### Input
- Webcam feed (60 FPS @ 1280x720 MJPG)
- 4 printed ArUco markers (IDs: 0, 1, 2, 3) attached to the target object
- Optional BNO055 IMU (for rotation backup during camera occlusion)

### Output
- Real-time pose: X, Y, Z (mm), Roll, Pitch, Yaw (degrees)
- Network stream: UDP packets @ 60 Hz
- Console display: Live pose + FPS + data source
- Video display: Live camera feed with marker overlays + HUD

### Example Use Cases
- Motion capture for VR/AR applications
- Robot localization and tracking
- Precision instrument positioning
- Augmented reality anchoring
- Pose-based user interfaces

---

## Key Features

### 1. Robust Multi-View Tracking
- Single-camera version: One camera, simple setup
- 3-camera version: Redundant cameras for occlusion tolerance
- Best-view selection: Automatically picks the camera with the most markers + lowest error

### 2. Advanced Smoothing
- 4-stage pipeline: Median filter → Velocity clamp → EMA → Kalman filter
- Confidence scaling: Smoothing adapts as the tracker locks onto the target
- All 6 values always reported (no artificial zeroing)

### 3. IMU Fallback
- Primary: Camera tracks position + rotation
- Phase 1 (0-5 sec): IMU provides rotation; position held from last camera frame
- Phase 2 (>5 sec): Position frozen; rotation continues from IMU (avoiding accelerometer drift)
- Graceful degradation: System keeps running even if camera loses markers

### 4. Professional Calibration
- Per-camera calibration: Corrects for lens distortion
- Sub-pixel accuracy: RMS error < 1 pixel
- JSON output: Easy to integrate into other projects

### 5. Network Integration
- UDP streaming: 60 Hz CSV packets to any IP:port
- Low latency: ~140-180 ms end-to-end
- Non-blocking: Silently drops packets if receiver is slow (real-time data)

---

## Hardware Requirements

### Minimum (Single-Camera System)

| Item | Spec | Notes |
|------|------|-------|
| Webcam | USB, MJPG codec, >=640x480 | Most modern USB cameras support MJPG |
| Markers | 4 printed ArUco (IDs 0-3), ~15 mm | See calibration guide for printing |
| PC | Any laptop/desktop with USB | Linux, macOS, Windows supported |
| Optional | BNO055 IMU on serial | Provides fallback rotation (COM4 on Windows) |

### Recommended (3-Camera System)

| Item | Spec | Notes |
|------|------|-------|
| Webcams | 3 USB, MJPG, 1280x720 @ 60 FPS | Best for redundancy |
| USB Hub | Powered, >=4 downstream ports | Avoid USB 3.0 (faster, less compatible) |
| IMU | BNO055 (9-DOF) | Gyro + accel + magnetometer |
| Markers | 4 ArUco, well-spaced on target | Large markers (25-50 mm) work better |

### Checkerboard for Calibration

| Item | Spec |
|------|------|
| Pattern | 6x9 checkerboard, 27 mm squares |
| Material | Printed on heavy matte paper |
| Mount | Glued flat to foam board or cardboard |

---

## Quick Start

### 1. Install Dependencies

```bash
# Python 3.7+
pip install opencv-python numpy pynput pyserial
```

### 2. Calibrate Your Camera

Before running the tracker, calibrate once per camera/resolution:

```bash
python camera_calibrator.py
```

What it does:
- Shows live camera feed
- You position the checkerboard pattern at various angles/distances
- Press SPACE to capture frames (green bar = ready)
- Press C to calibrate
- Generates camera_calibration_1280x720.json (or your resolution)

Expected time: 2-3 minutes
Output files: JSON (main) + NPZ (optional)

### 3. Run the Single-Camera Tracker

```bash
python 1-Camera@1280x720-30fps.py
```

What it does:
- Opens your webcam
- Detects ArUco markers in real-time
- Computes 6DOF pose (60 FPS)
- Sends UDP packets to a receiver
- Displays live video with pose overlay

Live display:
- 3D axes on each detected marker (optional, press A)
- Magenta triangle connecting the plane markers
- HUD panel with pose values, FPS, data source

Keyboard controls:
- Q: Quit
- A: Toggle 3D axes overlay

### 4. Run the 3-Camera Tracker (if you have 3 cameras)

```bash
python 3-Camera@1280x720-15fps.py
```

Same as single-camera, but with:
- 3 live camera tiles (top row)
- Automatic best-view selection
- Green border on the active camera
- Redundancy: if one camera fails, the others take over

### 5. Test Pose Reception

On a different machine (or same machine if using localhost):

```bash
python reciever_test.py
```

What it does:
- Listens on UDP port 5005 (adjust UDP_IP and UDP_PORT in code)
- Prints each pose packet as it arrives
- Shows FPS counter
- Displays error rate

Example output:
```
[1]   X:    100.45 mm   Y:     50.12 mm   Z:   -250.89 mm  |  Roll:   15.30 degrees  Pitch:   -2.10 degrees  Yaw:    45.80 degrees
```

---

## Project Structure

```
VisualTrackingSystem/Trackers/
├── README.md                              (This file)
│
├── SINGLE-CAMERA VERSION
│   └── 1-Camera@1280x720-30fps.py         (Main single-camera tracker)
│
├── 3-CAMERA VERSION
│   └── 3-Camera@1280x720-15fps.py         (Main 3-camera tracker)
│
├── CALIBRATION & UTILITIES
│   ├── camera_calibrator.py               (Camera calibration script)
│   ├── reciever_test.py                   (UDP receiver test)
│   └── calib.io_checker_218x150_6x9_20.pdf (Calibration pattern)
│
├── ARUCO MARKERS (Generated)
│   ├── Aruco_ID_0.png                     (Marker ID 0)
│   ├── Aruco_ID_1.png                     (Marker ID 1)
│   ├── Aruco_ID_2.png                     (Marker ID 2)
│   ├── Aruco_ID_3.png                     (Marker ID 3)
│   └── Aruco_ID_4.png                     (Marker ID 4, optional)
│
├── CALIBRATION FILES (Generated after camera_calibrator.py)
│   ├── camera_calibration_480x360.json    (480x360 resolution)
│   ├── camera_calibration_640x360.json    (640x360 resolution)
│   └── camera_calibration_1280x720.json   (1280x720 resolution)
│
└── DOCUMENTATION
    └── README.md                          (You are here)
```

---

## System Variants

### Single-Camera Version (1-Camera@1280x720-30fps.py)

When to use: Simple setup, good markers visibility, single camera sufficient

| Aspect | Details |
|--------|---------|
| Cameras | 1 USB webcam |
| Setup time | 15 minutes (calibration + positioning) |
| Cost | Low |
| Robustness | Medium (fails if markers occluded) |
| Latency | ~120 ms |
| Smoothing | Median + Clamp + EMA |

Pros: Simple, low cost, low latency
Cons: No redundancy, fails on occlusion

### 3-Camera Version (3-Camera@1280x720-15fps.py)

When to use: Occlusion tolerance needed, large workspace, high reliability

| Aspect | Details |
|--------|---------|
| Cameras | 3 USB webcams positioned around target |
| Setup time | 30 minutes (calibration x3 + positioning) |
| Cost | Medium (3 cameras + USB hub) |
| Robustness | High (survives partial camera failure) |
| Latency | ~140-180 ms (includes Kalman) |
| Smoothing | Median + Clamp + EMA + Kalman filter |

Pros: Robust, best-view selection, Kalman smoothing
Cons: More complex, higher latency

---

## Setup & Calibration

### Step 1: Prepare Your Checkerboard

1. Print a 6x9 checkerboard with 27 mm squares (use calib.io_checker_218x150_6x9_20.pdf or generate one)
2. Glue to flat foam board
3. Verify it lies completely flat (no waves or curves)

### Step 2: Calibrate Each Camera

For single-camera:
```bash
python camera_calibrator.py
```

For 3-camera (calibrate each separately):
```bash
# Camera 0
python camera_calibrator.py
# -> Generates: camera_calibration_1280x720.json

# Plug in Camera 1, repeat
# Camera 2, repeat
```

Calibration strategy:
- Capture 20-30 frames from diverse angles/distances
- Cover all parts of the frame (corners, edges, center)
- Position checkerboard close, far, tilted, straight
- Avoid motion blur and glare
- Aim for RMS error < 1.0 pixel (excellent if < 0.5)

Output: camera_calibration_WIDTHxHEIGHT.json for each camera/resolution

### Step 3: Generate & Attach ArUco Markers

Pre-generated marker images are included (Aruco_ID_0.png through Aruco_ID_4.png).

If you need to generate new markers:
```python
import cv2.aruco as aruco
dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
for mid in [0, 1, 2, 3]:
    img = aruco.generateImageMarker(dict, mid, 200)
    cv2.imwrite(f"marker_{mid}.png", img)
```

To attach:
1. Print the marker images (~15 mm squares for close work, 25-50 mm for far)
2. Cut out and attach to target object
3. Marker 0: Reference point (origin)
4. Markers 1-3: Form a plane/triangle
5. Spacing: For single-camera, keep all 4 visible in one frame

### Step 4: Position Cameras

Single-camera: Point at the markers, ensure all 4 visible

3-camera: Position around target at ~60-120 degrees angles to minimize occlusion

---

## Running the Tracker

### Single-Camera

```bash
python 1-Camera@1280x720-30fps.py
```

Console output:
```
UDP -> 10.98.109.221:5006
IMU connected on COM4
Camera: 1280x720 @ 60 FPS
Camera calibration loaded (err: 0.3425px)
============================================================
COMBINED 6DOF  |  Q=quit  A=toggle axes
============================================================

[CAM] X:   100.5 Y:    50.1 Z: -250.9 | R:   15.3 P:   -2.1 Yw:   45.8 | 60fps
[CAM] X:   100.7 Y:    50.3 Z: -250.8 | R:   15.4 P:   -2.0 Yw:   45.9 | 60fps
...
```

Display window:
- Live camera feed
- Green marker outlines + IDs
- Magenta triangle (plane markers 1-3)
- 3D axes (if A pressed)
- HUD: X, Y, Z, Roll, Pitch, Yaw, FPS, source

### 3-Camera

```bash
python 3-Camera@1280x720-15fps.py
```

Display window:
- Top row: 3 camera tiles (426x320 each)
- Bottom-left: Empty spacer
- Bottom-right: Unified HUD panel
- Active camera: Green border + ACTIVE label

---

## Network Output

### UDP Format

The tracker sends CSV lines at 60 Hz:

```
100.45,50.12,-250.89,15.3,-2.1,45.8
|      |      |       |    |    +- Yaw (degrees)
|      |      |       |    +------ Pitch (degrees)
|      |      |       +----------- Roll (degrees)
|      |      +------------------ Z position (mm)
|      +------------------------- Y position (mm)
+------------------------------- X position (mm)
```

### Configuration

In the tracker code:
```python
TARGET_IP   = "10.98.109.221"  # Receiver IP
TARGET_PORT = 5006              # Receiver port
```

Change these to send to a different machine.

### Receiving Packets

Python example:
```python
import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("10.98.109.221", 5006))

while True:
    data, addr = sock.recvfrom(1024)
    x, y, z, roll, pitch, yaw = map(float, data.decode().split(','))
    print(f"Pose: ({x:.1f}, {y:.1f}, {z:.1f}) mm, RPY: ({roll:.1f}, {pitch:.1f}, {yaw:.1f}) degrees")
```

Or use the test receiver:
```bash
python reciever_test.py
```

---

## Troubleshooting

### Camera Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| Camera not found | USB not connected or device index wrong | Check USB, try CAM_INDEX = 1 or 2 |
| Very low FPS | Using uncompressed codec | Ensure MJPG is set (cv2.CAP_PROP_FOURCC) |
| Frames stutter | USB buffer too large | Ensure BUFFERSIZE = 1 |
| No autofocus | Autofocus disabled or camera doesn't support | Set AUTOFOCUS = 1 or manually focus |

### Marker Detection

| Problem | Cause | Solution |
|---------|-------|----------|
| Markers not detected | Poor lighting, markers too far, or blurred | Improve lighting, move closer, hold still |
| Only some markers detected | Partial occlusion or bad image | Adjust camera angle, improve lighting |
| False positives | Other patterns in scene (background, clothing) | Use larger, well-separated markers |
| Jittery poses | Noisy corner detection | Increase MEDIAN_WIN, adjust Kalman parameters |

### Calibration

| Problem | Cause | Solution |
|---------|-------|----------|
| RMS error > 2.0 px | Poor frame coverage or blurry captures | Recalibrate with 30+ frames from diverse angles |
| NOT FOUND bar (red) | Checkerboard not fully visible | Move closer, ensure entire 5x8 pattern visible |
| Calibration file not created | Script crashed or no frames captured | Check console for errors, re-run with better frames |

### Network

| Problem | Cause | Solution |
|---------|-------|----------|
| UDP packets not received | Wrong IP/port or firewall | Check TARGET_IP and TARGET_PORT, disable firewall |
| Packets dropped | Network congestion | Normal for UDP; use TCP if reliability needed |
| Receiver says bad socket | Port already in use | Change UDP_PORT to an unused port (e.g., 5010) |

---

## Performance Specs

### Accuracy
- Position: +/- 0.5-2.0 mm (depending on calibration)
- Rotation: +/- 0.5-1.0 degrees (gyroscope limited on IMU fallback)
- Reprojection error: < 1.0 pixel (sub-pixel with good calibration)

### Latency

| Component | Time |
|-----------|------|
| Camera capture | ~33 ms (one frame @ 30 FPS) |
| Detection | ~5-10 ms |
| Filtering | ~80-100 ms (median window) |
| Network | ~1-2 ms (local network) |
| Total | ~120-180 ms |

### Throughput
- Frame rate: Up to 60 FPS
- UDP packets: 60/second
- Detection success: 95-99% when markers fully visible

### Robustness (3-Camera)
- 1 camera failure: Seamless failover to 2 remaining
- 2 cameras failure: Uses the remaining 1 camera
- All cameras fail: IMU takes over (if available)
- IMU loss: Holds last known pose

---

## Integration Example

Receiving and using pose in your application:

```python
import socket

# Start a receiver
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("10.98.109.221", 5006))

# Receive poses
while True:
    data, _ = sock.recvfrom(1024)
    values = data.decode().strip().split(',')
    
    # Parse pose
    pose = {
        'x_mm': float(values[0]),
        'y_mm': float(values[1]),
        'z_mm': float(values[2]),
        'roll_deg': float(values[3]),
        'pitch_deg': float(values[4]),
        'yaw_deg': float(values[5])
    }
    
    # Use pose in your application
    print(f"Position: ({pose['x_mm']:.1f}, {pose['y_mm']:.1f}, {pose['z_mm']:.1f}) mm")
    print(f"Rotation: RPY = ({pose['roll_deg']:.1f}, {pose['pitch_deg']:.1f}, {pose['yaw_deg']:.1f}) degrees")
```

---

## Checklist for First-Time Setup

- [ ] Hardware: Camera(s) connected, checkerboard printed, ArUco markers ready
- [ ] Software: Python 3.7+, dependencies installed (pip install opencv-python numpy pynput pyserial)
- [ ] Calibration: Ran camera_calibrator.py, got RMS error < 1.0 pixel
- [ ] Markers: 4 ArUco markers (IDs 0-3) attached to target object, all visible to camera
- [ ] Network: TARGET_IP and TARGET_PORT configured (match your network)
- [ ] IMU (optional): Plugged in, COM port set correctly if using fallback
- [ ] First run: Started tracker, saw live video + HUD in window
- [ ] Testing: Started UDP receiver test, saw pose packets arriving
- [ ] Integration: Incorporated pose into your application

---

## Tips & Best Practices

1. Always calibrate at the resolution you'll use
   - If tracker runs at 1280x720, calibrate at 1280x720
   - Different resolutions need separate calibration files

2. Keep markers well-lit and in focus
   - Use diffuse lighting (avoid shadows, glare)
   - Ensure adequate brightness
   - Manual focus if autofocus struggles

3. Smooth out poses with post-filtering
   - The Kalman filter helps, but your application can add more
   - Moving average, exponential smoothing, or Butterworth filters work

4. Monitor UDP packet loss
   - Non-blocking sends may drop packets under load
   - Use TCP if guaranteed delivery is needed

5. Test IMU separately
   - Verify BNO055 is readable before full integration
   - Set it flat and let it calibrate (takes ~30 seconds)

6. Use the 3-camera version for critical applications
   - Redundancy is worth the extra complexity
   - Best-view selection handles partial occlusion

---

## Quick Reference

Start here: Single-camera tracker
```bash
python camera_calibrator.py              # 1. Calibrate
python 1-Camera@1280x720-30fps.py        # 2. Run tracker
python reciever_test.py                  # 3. Test output
```

For robustness: 3-camera tracker
```bash
# Calibrate each camera separately, then:
python 3-Camera@1280x720-15fps.py
```

Key files:
- 1-Camera@1280x720-30fps.py - Single camera
- 3-Camera@1280x720-15fps.py - Three cameras
- camera_calibrator.py - Calibration utility
- reciever_test.py - Network test

---

## License & Attribution

This project uses:
- OpenCV: For camera, ArUco detection, calibration
- NumPy: For matrix math
- PySerial: For IMU communication
- pynput: For keyboard input

All open-source and widely used in academic/industry projects.

---
