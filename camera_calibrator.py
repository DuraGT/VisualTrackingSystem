"""
╔════════════════════════════════════════════════════════════════════════════╗
║                      CAMERA CALIBRATION UTILITY                            ║
║                                                                             ║
║  Generates intrinsic camera parameters (focal length, principal point,     ║
║  lens distortion) using a checkerboard pattern. Output is a JSON file      ║
║  for the ArUco tracker.                                                     ║
║                                                                             ║
║  PURPOSE                                                                    ║
║  ──────────────────────────────────────────────────────────────────────── ║
║  Raw camera output has lens distortion and unknown focal length. This      ║
║  script captures images of a known checkerboard pattern from various       ║
║  angles and distances, then solves for the camera matrix and distortion    ║
║  coefficients via non-linear optimization.                                 ║
║                                                                             ║
║  HARDWARE REQUIRED                                                          ║
║  ──────────────────────────────────────────────────────────────────────── ║
║  • Webcam (USB, same one the tracker will use)                             ║
║  • Printed 6×9 checkerboard pattern (27 mm squares)                        ║
║  • Flat surface to mount the checkerboard (e.g., cardboard)                ║
║  • Good lighting (avoid shadows, glare, motion blur)                       ║
║                                                                             ║
║  OUTPUT                                                                     ║
║  ──────────────────────────────────────────────────────────────────────── ║
║  • camera_calibration_WIDTHxHEIGHT.json — JSON intrinsics (load in tracker)║
║  • camera_calibration.npz — NumPy format (optional, for cv2 convenience)   ║
║  • Live preview of undistortion effect                                     ║
║                                                                             ║
║  QUALITY METRICS                                                            ║
║  ──────────────────────────────────────────────────────────────────────── ║
║  • RMS error < 0.5 px   → Excellent                                        ║
║  • RMS error < 1.0 px   → Good                                             ║
║  • RMS error < 2.0 px   → Acceptable                                       ║
║  • RMS error > 2.0 px   → Recalibrate (poor coverage or checkerboard fit) ║
║                                                                             ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

import cv2
import numpy as np
import os
import json
import time


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                         CONFIGURATION SECTION                              ║
# ║                                                                             ║
# ║  Adjust these values to match your checkerboard pattern and camera.       ║
# ╚════════════════════════════════════════════════════════════════════════════╝

# Checkerboard pattern: inner corners count
# A 6×9 printed board has 5×8 inner corners (corners where lines meet)
CHECKERBOARD_SIZE = (5, 8)

# Physical size of each square on the printed checkerboard (in mm)
SQUARE_SIZE_MM = 27

# Resolution at which to capture frames for calibration
# Must match the resolution your tracker will use!
# Common: 480x360 (fast), 640x480 (standard), 1280x720 (high detail)
CAPTURE_WIDTH  = 480
CAPTURE_HEIGHT = 360

# Minimum number of good frames needed for reliable calibration
# Recommended: 20–30 frames from different angles and distances
MIN_CAPTURES = 20

# Output filename (automatically includes resolution)
OUTPUT_FILE = f"camera_calibration_{CAPTURE_WIDTH}x{CAPTURE_HEIGHT}.json"


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                      OPENCV CALIBRATION SETUP                              ║
# ╚════════════════════════════════════════════════════════════════════════════╝

# Criteria for sub-pixel corner refinement
# Higher precision in corner detection → better calibration
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 3-D object points (world coordinates) for the checkerboard
# All points lie on Z=0 plane; X and Y scale with SQUARE_SIZE_MM
objp = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE_MM

# Arrays to store 3-D and 2-D points from all captured frames
obj_points = []  # 3-D points in real-world space (mm)
img_points = []  # 2-D points in image plane (pixels)


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                        CAMERA INITIALIZATION                               ║
# ╚════════════════════════════════════════════════════════════════════════════╝

def open_camera(index=0):
    """
    Open a USB camera with MJPG codec and requested resolution.

    Args:
        index: Camera device index (0 = first camera, 1 = second, etc.)

    Returns:
        cap: OpenCV VideoCapture object, or None if initialization failed

    Notes:
        - Uses DirectShow (CAP_DSHOW) on Windows for speed
        - Uses V4L2 (CAP_V4L2) on Linux
        - Requests MJPG codec (much faster than uncompressed YUY2)
        - Sets frame size and FPS; actual values may differ from request
    """
    # DirectShow on Windows, V4L2 on Linux — faster initialization
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_V4L2
    cap = cv2.VideoCapture(index, backend)

    if not cap.isOpened():
        print(f"[Error] Cannot open camera device {index}")
        return None

    # MJPG codec: compresses on-camera, much faster capture rate
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Read what the driver actually agreed to
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"[Camera] Opened at {actual_w}×{actual_h} @ {actual_fps:.0f} FPS")

    # Warn if the driver didn't accept the requested resolution
    if (actual_w, actual_h) != (CAPTURE_WIDTH, CAPTURE_HEIGHT):
        print(f"[Warning] Requested {CAPTURE_WIDTH}×{CAPTURE_HEIGHT} but got {actual_w}×{actual_h}")
        print(f"[Warning] Output file will reflect actual resolution: camera_calibration_{actual_w}x{actual_h}.json")

    return cap


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                           DISPLAY OVERLAY                                  ║
# ╚════════════════════════════════════════════════════════════════════════════╝

def draw_overlay(frame, captured, corners_found, corners):
    """
    Draw status bar, progress bar, and checkerboard corners on the frame.

    Args:
        frame: Input frame (BGR)
        captured: Number of frames successfully captured so far
        corners_found: Boolean, whether checkerboard corners were detected in this frame
        corners: 2-D array of corner positions if found, else None

    Returns:
        frame: Frame with overlays drawn

    Visual elements:
        • Status bar (top): Shows if corners are detected, capture count
        • Progress bar: Fills as you approach MIN_CAPTURES
        • Checkerboard overlay: Green lines connecting corners (if found)
        • Instructions (bottom): Keyboard shortcuts
    """
    h, w = frame.shape[:2]

    # Draw checkerboard corners as green lines (if found)
    if corners_found and corners is not None:
        cv2.drawChessboardCorners(frame, CHECKERBOARD_SIZE, corners, corners_found)

    # Status bar (top): green if corners found, red if not
    bar_color = (0, 180, 0) if corners_found else (0, 0, 200)
    cv2.rectangle(frame, (0, 0), (w, 40), bar_color, -1)

    status = f"Corners: {'FOUND' if corners_found else 'NOT FOUND'}   Captured: {captured}/{MIN_CAPTURES}"
    cv2.putText(frame, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Instructions bar (bottom): keyboard shortcuts
    cv2.rectangle(frame, (0, h - 40), (w, h), (30, 30, 30), -1)
    cv2.putText(frame, "SPACE: capture    R: reset    C: calibrate & save    Q: quit",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # Progress bar: fills as captured frames approach MIN_CAPTURES
    if captured > 0:
        progress = min(captured / MIN_CAPTURES, 1.0)
        cv2.rectangle(frame, (0, 40), (int(w * progress), 46), (0, 255, 100), -1)

    return frame


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                      FRAME CAPTURE LOOP                                    ║
# ║                                                                             ║
# ║  Main interactive loop: detect checkerboard, capture frames on demand,     ║
# ║  reset, or calibrate when ready.                                          ║
# ╚════════════════════════════════════════════════════════════════════════════╝

def run_calibration(cap):
    """
    Main interactive calibration loop.

    Detects the checkerboard corners in each frame and allows the user to:
      • Capture frames (SPACE) — only when corners are detected
      • Reset all captures (R) — start over
      • Calibrate (C) — solve for intrinsics (needs ≥6 frames, ideally ≥20)
      • Quit (Q) — exit without saving

    Args:
        cap: OpenCV VideoCapture object

    Strategy:
        • Position the checkerboard at various distances and angles
        • Cover different parts of the frame (corners, edges, center)
        • Avoid shadows, glare, and motion blur
        • Aim for ~20–30 frames from diverse views
    """
    print("\n" + "="*60)
    print("CAMERA CALIBRATION")
    print("="*60)
    print(f"Checkerboard inner corners : {CHECKERBOARD_SIZE[0]} × {CHECKERBOARD_SIZE[1]}")
    print(f"Square size                : {SQUARE_SIZE_MM} mm")
    print(f"Target resolution          : {CAPTURE_WIDTH}×{CAPTURE_HEIGHT} MJPG")
    print(f"Min captures recommended   : {MIN_CAPTURES}")
    print("\nInstructions:")
    print("  1. Hold the checkerboard at various angles and distances")
    print("  2. Press SPACE to capture when corners are detected (green bar)")
    print("  3. Cover different parts of the frame:")
    print("     • Top-left, top-right, bottom-left, bottom-right corners")
    print("     • Close-up and far away")
    print("     • Rotated at different angles")
    print("  4. Avoid shadows, glare, and motion blur")
    print("  5. Press C to calibrate once you have enough captures")
    print("  6. Press R to reset if you want to start over")
    print("="*60 + "\n")

    last_capture_time = 0
    cooldown = 1.0  # Seconds between captures to avoid duplicates/blur

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Error] Cannot read frame from camera.")
            break

        # Convert to grayscale for corner detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find the checkerboard corners
        # CALIB_CB_FAST_CHECK: quick check, returns immediately if corners unlikely
        found, corners = cv2.findChessboardCorners(
            gray, CHECKERBOARD_SIZE,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        )

        # If corners found, refine them to sub-pixel accuracy
        if found:
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        else:
            corners_refined = None

        # Draw overlay and display
        display = draw_overlay(frame.copy(), len(obj_points), found, corners_refined)
        cv2.imshow("Camera Calibration", display)

        key = cv2.waitKey(1) & 0xFF

        # ───────────────────────────────────────────────────────────────
        # Quit
        # ───────────────────────────────────────────────────────────────
        if key == ord("q"):
            print("[Quit] Exiting without calibrating.")
            break

        # ───────────────────────────────────────────────────────────────
        # Capture (SPACE)
        # ───────────────────────────────────────────────────────────────
        elif key == ord(" "):
            now = time.time()
            # Only capture if corners found AND enough time has passed
            if found and corners_refined is not None and (now - last_capture_time) > cooldown:
                obj_points.append(objp)
                img_points.append(corners_refined)
                last_capture_time = now
                count = len(obj_points)
                print(f"[Captured] Frame {count}/{MIN_CAPTURES}")

                # Visual feedback: flash the frame with green border
                flash = frame.copy()
                cv2.rectangle(flash, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 8)
                cv2.imshow("Camera Calibration", flash)
                cv2.waitKey(200)
            elif not found:
                print("[Skip] No corners detected in this frame.")
            elif (now - last_capture_time) <= cooldown:
                print(f"[Skip] Cooldown active ({cooldown:.1f}s) — wait before next capture.")

        # ───────────────────────────────────────────────────────────────
        # Reset (R)
        # ───────────────────────────────────────────────────────────────
        elif key == ord("r"):
            obj_points.clear()
            img_points.clear()
            print("[Reset] Cleared all captured frames.")

        # ───────────────────────────────────────────────────────────────
        # Calibrate (C)
        # ───────────────────────────────────────────────────────────────
        elif key == ord("c"):
            if len(obj_points) < 6:
                print(f"[Error] Need at least 6 captures (have {len(obj_points)}).")
            else:
                break

    cv2.destroyAllWindows()


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                    CALIBRATION & ERROR ANALYSIS                            ║
# ╚════════════════════════════════════════════════════════════════════════════╝

def calibrate_and_save():
    """
    Solve for camera intrinsics using cv2.calibrateCamera, compute reprojection
    error, and save the result in JSON and NumPy formats.

    The calibration process:
      1. Non-linear least-squares fitting of the pinhole camera model
      2. Solves for: focal length (fx, fy), principal point (cx, cy),
         and distortion coefficients (k1, k2, p1, p2, k3)
      3. Computes reprojection error: projects 3-D points back to image
         and measures how well they match the detected 2-D corners

    Output files:
      • JSON: Easy to load in Python (what the tracker uses)
      • NPZ: NumPy format, convenient for cv2 functions

    Quality interpretation:
      • RMS < 0.5 px: Excellent, sub-pixel precision
      • RMS < 1.0 px: Good, acceptable for most tracking
      • RMS < 2.0 px: Acceptable, but consider recalibrating
      • RMS > 2.0 px: Poor, likely needs better frame coverage
    """
    if len(obj_points) == 0:
        print("[Error] No frames captured. Calibration aborted.")
        return

    print(f"\n[Calibrating] Using {len(obj_points)} frames...")
    image_size = (CAPTURE_WIDTH, CAPTURE_HEIGHT)

    # Main calibration: non-linear optimization of the pinhole camera model
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )

    # ─────────────────────────────────────────────────────────────────
    # Reprojection Error Analysis
    # ─────────────────────────────────────────────────────────────────
    # For each frame, project the 3-D corners back to the image using
    # the solved camera matrix and distortion. Measure error in pixels.

    total_error = 0
    for i in range(len(obj_points)):
        projected, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i],
                                         camera_matrix, dist_coeffs)
        # L2 distance between detected and projected corners
        error = cv2.norm(img_points[i], projected, cv2.NORM_L2) / len(projected)
        total_error += error

    mean_error = total_error / len(obj_points)

    # ─────────────────────────────────────────────────────────────────
    # Print Results
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("CALIBRATION RESULTS")
    print("="*60)
    print(f"RMS reprojection error  : {ret:.4f} px")
    print(f"Mean reprojection error : {mean_error:.4f} px")

    if ret < 0.5:
        quality = "[Quality] EXCELLENT — sub-pixel precision (RMS < 0.5)"
    elif ret < 1.0:
        quality = "[Quality] GOOD — acceptable tracking accuracy (RMS < 1.0)"
    elif ret < 2.0:
        quality = "[Quality] ACCEPTABLE — usable but consider refinement (RMS < 2.0)"
    else:
        quality = "[Quality] POOR — consider recalibrating with better coverage (RMS > 2.0)"

    print(quality)
    print("\n" + "="*60)
    print("INTRINSIC PARAMETERS")
    print("="*60)
    print(f"\nCamera Matrix (3×3):")
    print(camera_matrix)
    print(f"\nDistortion Coefficients [k1, k2, p1, p2, k3]:")
    print(dist_coeffs.ravel())
    print("="*60 + "\n")

    # ─────────────────────────────────────────────────────────────────
    # Save as JSON (easy to load in tracker)
    # ─────────────────────────────────────────────────────────────────
    calibration_data = {
        "resolution": [CAPTURE_WIDTH, CAPTURE_HEIGHT],
        "checkerboard_size": list(CHECKERBOARD_SIZE),
        "checkerboard_size_description": f"{CHECKERBOARD_SIZE[0]+1}×{CHECKERBOARD_SIZE[1]+1} (with border)",
        "square_size_mm": SQUARE_SIZE_MM,
        "rms_error": float(ret),
        "mean_error": float(mean_error),
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.flatten().tolist(),
        "frames_used": len(obj_points),
    }

    json_file = f"camera_calibration_{CAPTURE_WIDTH}x{CAPTURE_HEIGHT}.json"
    with open(json_file, "w") as f:
        json.dump(calibration_data, f, indent=2)
    print(f"[Saved] JSON: '{json_file}'")

    # ─────────────────────────────────────────────────────────────────
    # Save as NumPy (convenient for cv2 functions)
    # ─────────────────────────────────────────────────────────────────
    np.savez("camera_calibration.npz",
             camera_matrix=camera_matrix,
             dist_coeffs=dist_coeffs,
             rms_error=ret)
    print(f"[Saved] NumPy: 'camera_calibration.npz'")

    # ─────────────────────────────────────────────────────────────────
    # Preview: Show original vs. undistorted side-by-side
    # ─────────────────────────────────────────────────────────────────
    print("\n[Preview] Showing undistorted feed (press Q to close)...")
    cap2 = open_camera(0)
    if cap2 is None:
        print("[Warning] Cannot reopen camera for preview.")
        return

    # Compute the optimal new camera matrix (can crop black borders)
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, image_size, 1, image_size
    )

    while True:
        ret2, frame2 = cap2.read()
        if not ret2:
            break

        # Apply undistortion
        undistorted = cv2.undistort(frame2, camera_matrix, dist_coeffs,
                                     None, new_camera_matrix)

        # Crop to ROI (removes black borders)
        x, y, w, h = roi
        undistorted_cropped = (undistorted[y:y+h, x:x+w]
                               if all([x, y, w, h]) else undistorted)

        # Side-by-side comparison
        combined = np.hstack([
            cv2.resize(frame2, (640, 360)),
            cv2.resize(undistorted_cropped, (640, 360))
        ])

        # Labels
        cv2.putText(combined, "Original", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)
        cv2.putText(combined, "Undistorted", (650, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)

        cv2.imshow("Undistortion Preview (Q to quit)", combined)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap2.release()
    cv2.destroyAllWindows()
    print("[Done] Calibration complete!")


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                           PROGRAM ENTRY POINT                              ║
# ╚════════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    print("\n" + "="*60)
    print("CAMERA CALIBRATION UTILITY")
    print("="*60)

    cap = open_camera(0)
    if cap is None:
        print("[Error] Failed to open camera. Exiting.")
        exit(1)

    run_calibration(cap)
    cap.release()

    if len(obj_points) > 0:
        calibrate_and_save()
    else:
        print("[Info] No frames captured. Skipping calibration.")