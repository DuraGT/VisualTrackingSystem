"""
╔════════════════════════════════════════════════════════════════════════════╗
║                      UDP 6DOF POSE RECEIVER TEST                           ║
║                                                                             ║
║  Simple UDP listener for testing 6DOF tracker output before integration    ║
║  into a larger system.                                                     ║
║                                                                             ║
║  PURPOSE                                                                    ║
║  ──────────────────────────────────────────────────────────────────────── ║
║  Receives 6-DOF pose packets from the tracker at TARGET_IP:TARGET_PORT     ║
║  and displays them in real-time. Useful for:                              ║
║    • Verifying tracker is sending data                                     ║
║    • Testing network connectivity                                          ║
║    • Parsing CSV pose format before building a full consumer               ║
║    • Debugging packet loss or latency                                      ║
║                                                                             ║
║  EXPECTED INPUT                                                             ║
║  ──────────────────────────────────────────────────────────────────────── ║
║  CSV line from tracker: X,Y,Z,Roll,Pitch,Yaw                              ║
║  Example: "100.45,50.12,-250.89,15.3,-2.1,45.8"                           ║
║                                                                             ║
║  OUTPUT                                                                     ║
║  ──────────────────────────────────────────────────────────────────────── ║
║  Console: Raw packets + parsed pose values                                 ║
║           Optionally: FPS counter, error checking                          ║
║                                                                             ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

import socket
import time
from collections import deque

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                         CONFIGURATION SECTION                              ║
# ╚════════════════════════════════════════════════════════════════════════════╝

# Network: IP and port to listen on
# Must match TARGET_IP and TARGET_PORT in the tracker code
UDP_IP = "10.0.0.146"  # IP address of this machine (receiver)
UDP_PORT = 5005  # Port to listen on (tracker sends to 5006, adjust as needed)

# Optional: FPS counter and statistics
ENABLE_FPS_COUNTER = True
FPS_WINDOW = 30  # Calculate FPS every N packets


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                         PACKET PARSING                                     ║
# ╚════════════════════════════════════════════════════════════════════════════╝

def parse_pose_packet(data_str):
    """
    Parse a 6-DOF pose CSV string into individual values.

    Input format: "X,Y,Z,Roll,Pitch,Yaw"
    Example: "100.45,50.12,-250.89,15.3,-2.1,45.8"

    Args:
        data_str: CSV string (already decoded from bytes)

    Returns:
        (success, pose_dict): Tuple of (bool, dict) where:
            - success: True if parsing succeeded, False if format error
            - pose_dict: Dictionary with keys 'X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw'
                        or None if parsing failed

    Example:
        >>> parse_pose_packet("100.45,50.12,-250.89,15.3,-2.1,45.8")
        (True, {'X': 100.45, 'Y': 50.12, 'Z': -250.89, 'Roll': 15.3, 'Pitch': -2.1, 'Yaw': 45.8})
    """
    try:
        # Split by comma and convert to floats
        values = [float(v) for v in data_str.strip().split(',')]

        # Verify we have exactly 6 values
        if len(values) != 6:
            return False, None

        # Create dictionary with named keys
        pose_dict = {
            'X': values[0],  # Position X (mm)
            'Y': values[1],  # Position Y (mm)
            'Z': values[2],  # Position Z (mm)
            'Roll': values[3],  # Rotation Roll (degrees)
            'Pitch': values[4],  # Rotation Pitch (degrees)
            'Yaw': values[5]  # Rotation Yaw (degrees)
        }

        return True, pose_dict

    except (ValueError, IndexError) as e:
        # Failed to parse: malformed CSV or non-numeric values
        return False, None


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                         UDP RECEIVER                                       ║
# ╚════════════════════════════════════════════════════════════════════════════╝

def create_socket(ip, port):
    """
    Create and bind a UDP socket.

    Args:
        ip: IP address to bind to (e.g., "10.0.0.146")
        port: Port number to listen on (e.g., 5005)

    Returns:
        sock: Bound socket object, or None if binding failed
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((ip, port))
        return sock
    except OSError as e:
        print(f"[Error] Cannot bind to {ip}:{port} — {e}")
        print(f"        Possible causes:")
        print(f"        • Port {port} already in use (another process listening)")
        print(f"        • IP {ip} not available on this machine")
        print(f"        • Permission denied (port < 1024 may need admin)")
        return None


def format_pose(pose_dict):
    """
    Format a pose dictionary into a readable string.

    Args:
        pose_dict: Dictionary with X, Y, Z, Roll, Pitch, Yaw

    Returns:
        Formatted string with aligned columns
    """
    return (f"  X: {pose_dict['X']:9.2f} mm   "
            f"Y: {pose_dict['Y']:9.2f} mm   "
            f"Z: {pose_dict['Z']:9.2f} mm  |  "
            f"Roll: {pose_dict['Roll']:7.2f}°  "
            f"Pitch: {pose_dict['Pitch']:7.2f}°  "
            f"Yaw: {pose_dict['Yaw']:7.2f}°")


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                         MAIN RECEIVER LOOP                                 ║
# ╚════════════════════════════════════════════════════════════════════════════╝

def run_receiver():
    """
    Main UDP receiver loop.

    Listens for 6-DOF pose packets, parses them, and displays in real-time.
    Press Ctrl+C to stop.
    """

    # Create socket
    sock = create_socket(UDP_IP, UDP_PORT)
    if sock is None:
        return

    print("=" * 80)
    print("UDP 6DOF POSE RECEIVER TEST")
    print("=" * 80)
    print(f"Listening on {UDP_IP}:{UDP_PORT}")
    print(f"Waiting for packets from tracker...")
    print("(Press Ctrl+C to stop)\n")

    # Optional: FPS counter
    packet_times = deque(maxlen=FPS_WINDOW) if ENABLE_FPS_COUNTER else None
    packet_count = 0
    parse_errors = 0

    try:
        while True:
            # Receive packet (blocks until data arrives)
            data, addr = sock.recvfrom(1024)
            packet_count += 1

            # Record timestamp for FPS calculation
            if ENABLE_FPS_COUNTER:
                packet_times.append(time.time())

            # Decode bytes to string
            try:
                data_str = data.decode('utf-8')
            except UnicodeDecodeError:
                print(f"[Error] Packet {packet_count}: Cannot decode as UTF-8")
                parse_errors += 1
                continue

            # Parse CSV
            success, pose = parse_pose_packet(data_str)

            if success and pose is not None:
                # Successful parse: display formatted output
                formatted_pose = format_pose(pose)

                # FPS counter (if enabled)
                fps_str = ""
                if ENABLE_FPS_COUNTER and len(packet_times) >= 2:
                    time_delta = packet_times[-1] - packet_times[0]
                    if time_delta > 0:
                        fps = (len(packet_times) - 1) / time_delta
                        fps_str = f" | FPS: {fps:.1f}"

                print(f"[{packet_count}]{formatted_pose}{fps_str}")

            else:
                # Parse error: malformed packet
                print(f"[Error] Packet {packet_count}: Invalid format — '{data_str[:50]}...'")
                parse_errors += 1
                if parse_errors <= 5:
                    print(f"        Expected: X,Y,Z,Roll,Pitch,Yaw (6 comma-separated floats)")

    except KeyboardInterrupt:
        print(f"\n{'=' * 80}")
        print("RECEIVER STOPPED")
        print("=" * 80)
        print(f"Total packets received: {packet_count}")
        print(f"Parse errors: {parse_errors}")
        if parse_errors == 0:
            print("[Status] All packets parsed successfully!")
        else:
            success_rate = (packet_count - parse_errors) / packet_count * 100
            print(f"[Status] Success rate: {success_rate:.1f}%")

    finally:
        # Clean up
        sock.close()
        print("Socket closed.")


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                           PROGRAM ENTRY POINT                              ║
# ╚════════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    print("\n")
    run_receiver()