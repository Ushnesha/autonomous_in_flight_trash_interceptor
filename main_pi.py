
"""
main_pi.py — SmartBin for Raspberry Pi 5
FINAL VERSION: Optimized for low-latency catching with Prediction Lock.
"""

import argparse
import time
import sys

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
    return parser.parse_args()


def main():
    args = parse_args()
    settings = SimSettings()
    log = Logger()

    log.header("SmartBin — Raspberry Pi 5 (FINAL DEMO MODE)")
    log.info(f"Predictor : {'ML MODEL' if args.ml else 'Kalman + Physics'}")
    log.info(f"Hardware  : Raspberry Pi 5 | Camera | L298N | Mecanum")

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
        
        # How many frames to see before we lock the motors (Using 2 for speed)
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

            # 2. DETECT
            position = detector.find_object(frame)
            current_time = time.time()

            if position is not None:
                cx, cy, depth = position

                # --- SETTLED OBJECT LOGIC ---
                if prediction_locked:
                    if abs(cx - last_cx) < 2: # If X position changed by less than 2 pixels
                        still_frames += 1
                    else:
                        still_frames = 0
                    last_cx = cx

                    # If the object is still for ~1 second (20 frames), consider it caught
                    if still_frames > 20:
                        log.success("OBJECT SETTLED: Forcing Reset/Recalibration")
                        motors.stop()
                        motors.reset()
                        predictor.reset()
                        detector.reset_calibration() # Triggers the 5s recalibration
                        active_throw = False
                        prediction_locked = False
                        still_frames = 0
                        continue # Skip to next frame to allow recalibration to start

                # --- START OF NEW THROW ---
                if not active_throw:
                    log.info("NEW THROW DETECTED")
                    active_throw = True
                    prediction_locked = False
                    final_target_x = None
                    still_frames = 0

                detection_count += 1
                last_detection_time = current_time

                # 3. PREDICT & ACT
                if not prediction_locked:
                    predictor.add_point(position)
                    
                    # Use the internal _updates counter from your Predictor class
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
                    # LOCKED STATE: Ignore ghosts/noise and drive to the target
                    motors.move_to_x(final_target_x)

            else:
                # --- NO BALL IN VIEW ---
                if active_throw:
                    time_since_last_seen = current_time - last_detection_time

                    # THROW COMPLETED (800ms second wait after ball is lost)
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

                    # MID-AIR PERSISTENCE: If ball flies high/out of frame, keep driving
                    elif prediction_locked:
                        if time.time() - lock_start_time > 2.0:
                            log.info("LOCK TIMEOUT: Forcing Reset...")
                            motors.stop()
                            motors.reset()
                            predictor.reset()
                            detector.reset_calibration()
                            active_throw = False
                            prediction_locked = False
                            continue # Skip to next frame
                        motors.move_to_x(final_target_x)
                    # elif predictor._last_x is not None:
                    #     motors.move_to_x(predictor._last_x)

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
