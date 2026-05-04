"""
main_pi.py — SmartBin for Raspberry Pi 5
FINAL VERSION: Optimized for low-latency catching with Prediction Lock.
"""

import argparse
import time
import sys
import os
import cv2
import numpy as np

try:
    from src.camera_real import RealCamera
    from src.motors_real import RealMotors
    from src.detect_real import RealDetector
    from src.predict import Predictor
    from src.ml_predictor import MLPredictor, ThrowDataCollector
    from src.logger import Logger
    from config.sim_settings import SimSettings
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    print("[ERROR] Make sure you're in the smartbin_v4 directory")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description='SmartBin on Raspberry Pi 5')
    parser.add_argument('--ml', action='store_true',
                        help='Use ML predictor instead of Kalman physics')
    parser.add_argument('--collect-data', action='store_true',
                        help='Collect throw data for ML training')
    parser.add_argument('--debug', action='store_true',
                        help='Save debug depth images to captures/')
    return parser.parse_args()


def build_debug_view(frame, position):
    """Convert depth frame to a viewable BGR image with detection overlay."""
    h, w = frame.shape
    center_z = frame[h // 2, w // 2]
    # Clean the frame before visualizing
    clean = cv2.medianBlur(frame.astype(np.float32), 5)  # 5x5 median
    depth_gray = np.clip((clean / 4.0) * 255, 0, 255).astype(np.uint8)


    # Map depth to 0-255 grayscale (0-4m range)

    debug_view = cv2.cvtColor(depth_gray, cv2.COLOR_GRAY2BGR)

    # Stamp center depth
    cv2.putText(debug_view, f"Z: {center_z:.3f}m", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Draw crosshair on detected position
    if position is not None:
        cx, cy, pz = position
        cv2.drawMarker(debug_view, (int(cx), int(cy)), (0, 0, 255),
                       cv2.MARKER_CROSS, 15, 2)
        cv2.putText(debug_view, f"BALL: ({cx:.0f}, {cy:.0f}) {pz:.2f}m",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return debug_view


def main():
    args = parse_args()
    settings = SimSettings()
    log = Logger()

    log.header("SmartBin — Raspberry Pi 5 (FINAL DEMO MODE)")
    log.info(f"Predictor : {'ML MODEL' if args.ml else 'Kalman + Physics'}")
    log.info(f"Hardware  : Raspberry Pi 5 | Camera | L298N | Mecanum")

    if args.debug:
        os.makedirs("captures", exist_ok=True)
        log.info("Debug mode ON — saving frames to captures/")

    camera = None
    motors = None
    detector = None

    try:
        # Initialize hardware
        log.info("Initializing hardware...")
        camera = RealCamera(settings=settings)
        motors = RealMotors(settings=settings)
        detector = RealDetector(settings=settings)

        # Initialize predictor
        predictor = MLPredictor(settings=settings) if args.ml else Predictor(settings=settings)

        # Data collector
        collector = ThrowDataCollector() if args.collect_data else None

        log.success("All modules loaded. Waiting for calibration...")
        log.info("Press Ctrl+C to stop")

        frame_count = 0
        detection_count = 0
        last_detection_time = time.time()

        # --- STATE MACHINE VARIABLES ---
        active_throw = False
        prediction_locked = False
        final_target_x = None
        lock_start_time = 0
        still_frames = 0
        last_cx = 0
        LOCK_THRESHOLD = 2

        # Main loop
        while True:
            loop_start = time.time()
            frame_count += 1

            # 1. SENSE
            frame = camera.get_frame()
            if frame is None:
                continue

            # Edge mask — black out noisy left/right 10 pixels
            depth_map = frame.copy()
            depth_map[:, 0:10] = 0
            depth_map[:, -10:] = 0

            # 2. DETECT
            position = detector.find_object(depth_map)
            current_time = time.time()

            # 3. DEBUG VIEW — only save if --debug flag is set
            if args.debug:
                debug_view = build_debug_view(depth_map, position)
                cv2.imwrite(f"captures/raw_depth_{frame_count:04d}.jpg", debug_view)

            if position is not None:
                cx, cy, depth = position

                # --- SETTLED OBJECT LOGIC ---
                if prediction_locked:
                    if abs(cx - last_cx) < 2:
                        still_frames += 1
                    else:
                        still_frames = 0
                    last_cx = cx

                    if still_frames > 20:
                        log.success("OBJECT SETTLED: Forcing Reset/Recalibration")
                        motors.stop()
                        motors.reset()
                        predictor.reset()
                        detector.reset_calibration()
                        active_throw = False
                        prediction_locked = False
                        still_frames = 0
                        continue

                # --- START OF NEW THROW ---
                if not active_throw:
                    log.info("NEW THROW DETECTED")
                    active_throw = True
                    prediction_locked = False
                    final_target_x = None
                    still_frames = 0
                    motors.reset()

                detection_count += 1
                last_detection_time = current_time


                # 4. PREDICT & ACT
                if not prediction_locked:
                    predictor.add_point(position)
                    point_count = predictor._updates
                    predicted_x = predictor.get_predicted_landing_x()

                    if predicted_x is not None:
                        if point_count >= LOCK_THRESHOLD:
                            log.success(f"› [LOCK] Target Locked at {predicted_x:.1f}px. Committing!")
                            prediction_locked = True
                            final_target_x = predicted_x
                            lock_start_time = time.time()
                        motors.move_to_x(predicted_x)
                else:
                    motors.move_to_x(final_target_x)

            else:
                # --- NO BALL IN VIEW ---
                if active_throw:
                    time_since_last_seen = current_time - last_detection_time

                    if time_since_last_seen > 0.8:
                        log.info("THROW COMPLETED: Resetting and Recalibrating for next object...")
                        motors.stop()
                        motors.reset()
                        predictor.reset()
                        detector.reset_calibration()
                        active_throw = False
                        prediction_locked = False
                        final_target_x = None
                        still_frames = 0

                    elif prediction_locked:
                        if time.time() - lock_start_time > 2.0:
                            log.info("LOCK TIMEOUT: Forcing Reset...")
                            motors.stop()
                            motors.reset()
                            predictor.reset()
                            detector.reset_calibration()
                            active_throw = False
                            prediction_locked = False
                            continue
                        motors.move_to_x(final_target_x)

            # Periodic status
            if frame_count % 30 == 0:
                status = detector.get_calibration_status()
                log.info(f"[RUNNING] Frames: {frame_count}, Detections: {detection_count} | Detector: {status}")

            # Timing
            elapsed = time.time() - loop_start
            target_delay = 1.0 / 20
            if elapsed < target_delay:
                time.sleep(target_delay - elapsed)

    except KeyboardInterrupt:
        log.info("Interrupted by user")
    except Exception as e:
        log.error(f"Runtime error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        log.info("Cleaning up hardware...")
        if motors: motors.stop(); motors.cleanup()
        if camera: camera.cleanup()
        log.header(f"FINAL: {detection_count} detections, {frame_count} frames processed")


if __name__ == '__main__':
    main()
