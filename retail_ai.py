import cv2
from ultralytics import YOLO
import winsound
import time

# -------------------------------------------------
# 1. LOAD YOLO MODEL AND CAMERA
# -------------------------------------------------
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera not found")
    exit()

# -------------------------------------------------
# 2. QUEUE SETTINGS
# -------------------------------------------------
QUEUE_X, QUEUE_Y = 300, 180
QUEUE_WIDTH, QUEUE_HEIGHT = 280, 280

# Alert when queue reaches this number
QUEUE_LIMIT = 3

# -------------------------------------------------
# 3. SHELF SETTINGS
# -------------------------------------------------
SHELF_X, SHELF_Y = 30, 180
SHELF_WIDTH, SHELF_HEIGHT = 220, 220

# -------------------------------------------------
# 4. OTHER SETTINGS
# -------------------------------------------------
PERSON = 0

last_alarm = 0
ALARM_COOLDOWN = 5

# -------------------------------------------------
# 5. MAIN LOOP
# -------------------------------------------------
while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Resize camera frame
    frame = cv2.resize(frame, (800, 600))

    # YOLO detection
    result = model(frame, verbose=False)[0]

    people = 0
    queue = 0
    shelf_objects = 0

    # -------------------------------------------------
    # 6. PROCESS DETECTED OBJECTS
    # -------------------------------------------------
    for box in result.boxes:

        # Confidence filtering
        if float(box.conf[0]) < 0.40:
            continue

        cls = int(box.cls[0])

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Center of detected object
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # -------------------------------------------------
        # PEOPLE + QUEUE DETECTION
        # -------------------------------------------------
        if cls == PERSON:

            people += 1

            # Check whether person is inside queue area
            if (
                QUEUE_X <= cx <= QUEUE_X + QUEUE_WIDTH
                and
                QUEUE_Y <= cy <= QUEUE_Y + QUEUE_HEIGHT
            ):

                queue += 1
                color = (0, 255, 0)

            else:

                color = (255, 0, 0)

            # Draw person bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

        # -------------------------------------------------
        # SHELF OBJECT DETECTION
        # -------------------------------------------------
        else:

            if (
                SHELF_X <= cx <= SHELF_X + SHELF_WIDTH
                and
                SHELF_Y <= cy <= SHELF_Y + SHELF_HEIGHT
            ):

                shelf_objects += 1

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

    # -------------------------------------------------
    # 7. SHELF STATUS
    # -------------------------------------------------
    if shelf_objects > 0:

        shelf_status = "AVAILABLE"
        shelf_color = (0, 255, 0)

    else:

        shelf_status = "EMPTY - RESTOCK"
        shelf_color = (0, 0, 255)

    # -------------------------------------------------
    # 8. DRAW QUEUE AREA
    # -------------------------------------------------
    cv2.rectangle(
        frame,
        (QUEUE_X, QUEUE_Y),
        (QUEUE_X + QUEUE_WIDTH,
         QUEUE_Y + QUEUE_HEIGHT),
        (0, 0, 255),
        2
    )

    cv2.putText(
        frame,
        "QUEUE AREA",
        (QUEUE_X, QUEUE_Y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2
    )

    # -------------------------------------------------
    # 9. DRAW SHELF AREA
    # -------------------------------------------------
    cv2.rectangle(
        frame,
        (SHELF_X, SHELF_Y),
        (SHELF_X + SHELF_WIDTH,
         SHELF_Y + SHELF_HEIGHT),
        shelf_color,
        2
    )

    cv2.putText(
        frame,
        "SHELF AREA",
        (SHELF_X, SHELF_Y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        shelf_color,
        2
    )

    # -------------------------------------------------
    # 10. DASHBOARD
    # -------------------------------------------------
    cv2.rectangle(
        frame,
        (0, 0),
        (800, 120),
        (25, 25, 25),
        -1
    )

    cv2.putText(
        frame,
        "SMART RETAIL AI - SIH PS 179",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"People: {people}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Queue: {queue}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Shelf: {shelf_status}",
        (200, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        shelf_color,
        2
    )

    # -------------------------------------------------
    # 11. ALERT SYSTEM
    # -------------------------------------------------

    # Queue has priority over shelf alert
    if queue >= QUEUE_LIMIT:

        alert = "LONG QUEUE - OPEN ANOTHER COUNTER"
        alert_color = (0, 0, 255)

    elif shelf_objects == 0:

        alert = "SHELF EMPTY - RESTOCK"
        alert_color = (0, 0, 255)

    else:

        alert = "SYSTEM NORMAL"
        alert_color = (0, 255, 0)

    # Display alert
    cv2.putText(
        frame,
        alert,
        (20, 580),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        alert_color,
        2
    )

    # -------------------------------------------------
    # 12. BUZZER
    # -------------------------------------------------
    if alert != "SYSTEM NORMAL":

        current_time = time.time()

        if current_time - last_alarm >= ALARM_COOLDOWN:

            winsound.Beep(1200, 2000)

            last_alarm = current_time

    # -------------------------------------------------
    # 13. SHOW WINDOW
    # -------------------------------------------------
    cv2.imshow(
        "Smart Retail AI - PS 179",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# -------------------------------------------------
# 14. RELEASE CAMERA
# -------------------------------------------------
cap.release()
cv2.destroyAllWindows()