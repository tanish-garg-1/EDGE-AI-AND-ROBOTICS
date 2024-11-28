import cv2
import numpy as np


def adjust_brightness(frame, brightness_factor=1.2, threshold=50):
    # Convert to grayscale to calculate brightness
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)

    # If the average brightness is below the threshold, brighten the frame
    if avg_brightness < threshold:
        frame = cv2.convertScaleAbs(frame, alpha=brightness_factor, beta=0)

    return frame


def analyze_crowd_movement(input_video, output_video, decay=0.9, frame_skip=1, save_heatmap=False):
    # Load video
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Error: Cannot open video {input_video}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Initialize previous frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Unable to read the first frame.")
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Parameters for visualization
    motion_heatmap = np.zeros((height, width), dtype=np.float32)

    frame_count = 0
    while True:
        # Read the next frame
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Skip frames for faster processing
        if frame_count % frame_skip != 0:
            continue

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Compute magnitude and angle of flow
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Update heatmap with motion intensity
        motion_heatmap = cv2.addWeighted(motion_heatmap, decay, magnitude, 1 - decay, 0)

        # Normalize heatmap for visualization
        normalized_heatmap = cv2.normalize(motion_heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Convert heatmap to color
        heatmap_color = cv2.applyColorMap(normalized_heatmap, cv2.COLORMAP_JET)

        # Adjust brightness of the original frame if it's too dark
        frame = adjust_brightness(frame)

        # Overlay heatmap on the original frame
        overlay = cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)

        # Write the output frame
        out.write(overlay)

        # Update the previous frame
        prev_gray = gray

        # Optional: Show the frame (comment out if running on headless servers)
        cv2.imshow("Crowd Movement Analysis", overlay)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processed video saved as {output_video}")

    # Save the final heatmap as an image if needed
    if save_heatmap:
        final_heatmap = cv2.normalize(motion_heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        final_heatmap_color = cv2.applyColorMap(final_heatmap, cv2.COLORMAP_JET)
        cv2.imwrite("final_heatmap.png", final_heatmap_color)
        print("Final heatmap saved as 'final_heatmap.png'.")


# Run the analysis with enhanced features
analyze_crowd_movement(
    input_video="crowd_video1.mp4",
    output_video="output_crowd_analysis.mp4",
    decay=0.8,
    frame_skip=1,
    save_heatmap=True
)