from ultralytics import YOLO
import cv2

# Load more accurate model
model = YOLO("yolov8m.pt")

# Objects to detect
allowed_classes = [
    "person",
    "bottle",
    "chair",
    "laptop",
    "cell phone"
]

# Open webcam
cap = cv2.VideoCapture(0)

# Camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Create resizable window
cv2.namedWindow("Improved Object Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Improved Object Detection", 1280, 720)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera not found!")
        break

    # Run YOLO
    results = model(
        frame,
        conf=0.75,
        verbose=False
    )

    annotated_frame = frame.copy()

    object_counts = {}

    for box in results[0].boxes:

        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        if class_name not in allowed_classes:
            continue

        confidence = float(box.conf[0])

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Draw rectangle
        cv2.rectangle(
            annotated_frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Label
        label = f"{class_name} {confidence:.2f}"

        cv2.putText(
            annotated_frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        object_counts[class_name] = object_counts.get(class_name, 0) + 1

    # Display counts
    y_pos = 30

    for obj, count in object_counts.items():

        cv2.putText(
            annotated_frame,
            f"{obj}: {count}",
            (10, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        y_pos += 35

    # Resize output to fill window
    annotated_frame = cv2.resize(
        annotated_frame,
        (1280, 720)
    )

    cv2.imshow(
        "Improved Object Detection",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()