import cv2
import numpy as np
import ArducamDepthCamera
import os
import time

def main():
    ac_lib = ArducamDepthCamera.ArducamDepthCamera
    cam = ac_lib.ArducamCamera()
    
    output_dir = "captures"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Attempting to open camera...")
    if cam.open(ac_lib.Connection.CSI, 0).value != 0:
        print("Error: Could not open ToF camera.")
        return

    cam.start(ac_lib.FrameType.DEPTH)
    print(f"Capture started. Saving images to ./{output_dir}")
    print("Press Ctrl+C to stop.")

    frame_count = 0
    try:
        while True:
            frame = cam.requestFrame(200)
            if frame is None:
                continue

            # --- UPDATED EXTRACTION ---
            # The error showed 'frame' is already a DepthData object.
            # We access the internal buffers directly:
            try:
                depth_buf = frame.depth_data
                amp_buf = frame.amplitude_data
            except AttributeError:
                # If the above fails, it might be a simple byte buffer
                depth_buf = np.asarray(frame)
                amp_buf = None

            # 1. Process Depth Map (The most important for your robot)
            depth_img = np.array(depth_buf, dtype=np.float32)
            # Scale: 0 to 4 meters mapped to 0-255
            depth_viz = (depth_img / 4.0) * 255 
            depth_viz = np.clip(depth_viz, 0, 255).astype(np.uint8)
            depth_colored = cv2.applyColorMap(depth_viz, cv2.COLORMAP_JET)

            # 2. Save the files
            timestamp = int(time.time() * 100)
            cv2.imwrite(f"{output_dir}/depth_{timestamp}.jpg", depth_colored)

            # 3. Process and save Amplitude (IR) if it exists
            if amp_buf is not None:
                amp_img = np.array(amp_buf, dtype=np.float32)
                amp_img = cv2.normalize(amp_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                cv2.imwrite(f"{output_dir}/ir_{timestamp}.jpg", amp_img)

            frame_count += 1
            if frame_count % 10 == 0:
                print(f"Captured {frame_count} frames...")

            cam.releaseFrame(frame)

    except KeyboardInterrupt:
        print("\nStopping capture...")
    finally:
        cam.stop()
        cam.close()
        print(f"Done. Saved {frame_count} images to '{output_dir}'.")

if __name__ == "__main__":
    main()
