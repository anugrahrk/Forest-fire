import streamlit as st
import cv2
import numpy as np
import time

# Function to detect fire and smoke in a frame
def detect_fire_and_smoke(frame):
    # Convert the frame from BGR to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # --- Fire Detection ---
    # Define a tight range for fire (bright, vivid flames)
    fire_lower = np.array([0, 150, 200])  # Red-orange, high saturation, bright
    fire_upper = np.array([30, 255, 255])  # Yellow, full saturation and value
    fire_mask = cv2.inRange(hsv, fire_lower, fire_upper)

    # Morphological operations for fire
    kernel = np.ones((5, 5), np.uint8)
    fire_mask = cv2.dilate(fire_mask, kernel, iterations=2)
    fire_mask = cv2.erode(fire_mask, kernel, iterations=1)

    # Brightness check for fire
    value_channel = hsv[:, :, 2]
    bright_mask = cv2.threshold(value_channel, 200, 255, cv2.THRESH_BINARY)[1]
    fire_mask = cv2.bitwise_and(fire_mask, bright_mask)

    # --- Smoke Detection ---
    # Define a tighter range for smoke (grayish, low saturation, moderate value)
    smoke_lower = np.array([0, 0, 100])    # Wide Hue, very low Saturation, higher Value
    smoke_upper = np.array([180, 50, 200])  # Wide Hue, low Saturation, moderate-to-high Value
    smoke_mask = cv2.inRange(hsv, smoke_lower, smoke_upper)

    # Blur the frame and detect edges for smoke (smoke is diffused, low edges)
    blurred = cv2.GaussianBlur(frame, (21, 21), 0)
    edges = cv2.Canny(blurred, 50, 150)  # Adjusted edge thresholds
    edges_mask = cv2.bitwise_not(edges)  # Invert: high values where edges are weak
    smoke_mask = cv2.bitwise_and(smoke_mask, edges_mask)  # Combine with color mask

    # Morphological operations for smoke
    smoke_mask = cv2.dilate(smoke_mask, kernel, iterations=2)  # Reduced dilation
    smoke_mask = cv2.erode(smoke_mask, kernel, iterations=2)   # Increased erosion

    # Additional check for texture to differentiate smoke from uniform areas like sky
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 100:  # Adjust threshold based on your environment
        smoke_mask = np.zeros_like(smoke_mask)  # Discard detection in uniform areas

    # Combine masks for visualization (optional)
    combined_mask = cv2.bitwise_or(fire_mask, smoke_mask)

    # Find contours for fire
    fire_contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fire_detected = False
    for contour in fire_contours:
        area = cv2.contourArea(contour)
        if area > 500:
            mean_val = cv2.mean(value_channel, mask=cv2.drawContours(np.zeros_like(fire_mask), [contour], -1, 255, thickness=cv2.FILLED))[0]
            if mean_val > 200:
                fire_detected = True
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green for fire
                cv2.putText(frame, "Fire", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Find contours for smoke
    smoke_contours, _ = cv2.findContours(smoke_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    smoke_detected = False
    for contour in smoke_contours:
        area = cv2.contourArea(contour)
        if area > 1500:  # Increased threshold for smoke due to its spread
            smoke_detected = True
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue for smoke
            cv2.putText(frame, "Smoke", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return frame, combined_mask, fire_detected, smoke_detected

# Streamlit app
def main():
    st.title("Fire and Smoke Detection App")
    st.write("Upload a video file or use your webcam to detect fire and smoke.")

    # Sidebar for input selection
    input_type = st.sidebar.selectbox("Choose input type", ["Video File", "Webcam"])

    # Placeholder for input and output
    input_container = st.container()
    output_container = st.container()

    # Session state to control webcam loop
    if "running" not in st.session_state:
        st.session_state.running = False

    with input_container:
        if input_type == "Video File":
            # Video file upload
            uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
            if uploaded_file is not None:
                # Save the uploaded video to a temporary file
                temp_file = "temp_video.mp4"
                with open(temp_file, "wb") as f:
                    f.write(uploaded_file.read())
                
                # Open the video and grab the first frame
                cap = cv2.VideoCapture(temp_file)
                ret, frame = cap.read()
                if ret:
                    frame_to_process = cv2.resize(frame, (640, 480))
                    cap.release()

                    # Detect button for video file
                    if st.button("Detect"):
                        # Process the frame for fire and smoke detection
                        result_frame, combined_mask, fire_detected, smoke_detected = detect_fire_and_smoke(frame_to_process)

                        # Convert to RGB for Streamlit
                        result_frame_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
                        combined_mask_colored = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2RGB)

                        # Display results in output container
                        with output_container:
                            st.subheader("Detection Result")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.image(result_frame_rgb, caption="Processed Frame", use_column_width=True)
                            with col2:
                                st.image(combined_mask_colored, caption="Fire & Smoke Mask", use_column_width=True)

                            # Display detection status
                            if fire_detected and smoke_detected:
                                st.success("Fire and smoke detected!")
                            elif fire_detected:
                                st.success("Fire detected!")
                            elif smoke_detected:
                                st.success("Smoke detected!")
                            else:
                                st.warning("No fire or smoke detected.")

        elif input_type == "Webcam":
            st.write("Click 'Start Detection' to begin webcam feed and detect fire and smoke.")
            
            # Buttons to start and stop webcam feed
            start_button = st.button("Start Detection")
            stop_button = st.button("Stop Detection")

            if start_button:
                st.session_state.running = True
            if stop_button:
                st.session_state.running = False

            # Placeholder for webcam feed
            webcam_frame = st.empty()
            status_placeholder = st.empty()

            # Open webcam
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Error: Could not access webcam.")
                return

            # Webcam loop
            while st.session_state.running:
                ret, frame = cap.read()
                if not ret:
                    st.error("Error: Could not read webcam frame.")
                    break

                # Resize frame for faster processing
                frame = cv2.resize(frame, (640, 480))

                # Detect fire and smoke
                result_frame, combined_mask, fire_detected, smoke_detected = detect_fire_and_smoke(frame)

                # Convert to RGB for Streamlit
                result_frame_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)

                # Update the webcam feed display
                webcam_frame.image(result_frame_rgb, caption="Webcam Feed", use_column_width=True)

                # Update detection status
                if fire_detected and smoke_detected:
                    status_placeholder.success("Fire and smoke detected!")
                elif fire_detected:
                    status_placeholder.success("Fire detected!")
                elif smoke_detected:
                    status_placeholder.success("Smoke detected!")
                else:
                    status_placeholder.warning("No fire or smoke detected.")

                # Small delay to avoid overwhelming the UI
                time.sleep(0.1)

            # Release webcam when stopped
            cap.release()

if __name__ == "__main__":
    main()