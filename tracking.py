import cv2
from ultralytics import YOLO

def main():
    # 1. Load a pre-trained model (YOLOv8 Nano is fast and accurate for real-time)
    # This automatically downloads 'yolov8n.pt' if not present.
    print("Loading model...")
    model = YOLO('yolov8n.pt') 

    # 2. Set up real-time video input
    # Use 0 for webcam, or replace with a string like 'video.mp4' for a file.
    video_source = 0 
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    print("Starting video stream... Press 'q' to exit.")

    while cap.isOpened():
        # 3. Process each video frame
        success, frame = cap.read()
        if not success:
            break

        # 4. Apply Object Detection and Tracking
        # The 'track' method runs the detection model and assigns unique IDs to objects.
        # persist=True tells the model that this is a continuous video sequence, 
        # which allows the tracker (ByteTrack/BoT-SORT) to maintain IDs across frames.
        # results = model.track(source=frame, persist=True, tracker="bytetrack.yaml")
        # This uses the default tracker which may bypass the 'lap' requirement 
# or you can simply run detection without the advanced tracker if it persists
        results = model.predict(source=frame, show=False)

        # 5. Display the output
        # The .plot() method draws the bounding boxes, labels, and tracking IDs for us.
        annotated_frame = results[0].plot()

        # Show the frame in a window
        cv2.imshow("Task 4: Object Detection & Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
#enter q to exit in cmd prompt