import cv2
import time
import threading
from playsound import playsound
from ultralytics import YOLO


class BirdDetector:

    def __init__(self):

        # Load YOLO model
        self.model = YOLO("yolov8n.pt")

        # Open webcam directly
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            raise Exception("Could not open webcam!")

        self.prev_time = 0
        self.alarm_playing = False

    def play_alarm(self):
        if not self.alarm_playing:
            self.alarm_playing = True

            try:
                playsound("alarm.mp3")
            except Exception as e:
                print("Alarm Error:", e)

            self.alarm_playing = False

    def run(self):

        cv2.namedWindow("Bird Detection", cv2.WINDOW_NORMAL)

        while True:

            ret, frame = self.cap.read()

            if not ret:
                break

            bird_count = 0

            results = self.model(
                frame,
                conf=0.5,
                verbose=False
            )

            for result in results:

                for box in result.boxes:

                    cls_id = int(box.cls[0])
                    class_name = self.model.names[cls_id]

                    if class_name == "bird":

                        bird_count += 1

                        x1, y1, x2, y2 = map(
                            int,
                            box.xyxy[0]
                        )

                        confidence = float(box.conf[0])

                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (0, 255, 0),
                            3
                        )

                        cv2.putText(
                            frame,
                            f"Bird {confidence:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2
                        )

            # Alarm
            if bird_count > 0 and not self.alarm_playing:

                threading.Thread(
                    target=self.play_alarm,
                    daemon=True
                ).start()

            # FPS
            current_time = time.time()

            fps = 0
            if self.prev_time:
                fps = 1 / (current_time - self.prev_time)

            self.prev_time = current_time

            cv2.putText(
                frame,
                f"Bird Count: {bird_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

            cv2.putText(
                frame,
                f"FPS: {int(fps)}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2
            )

            cv2.putText(
                frame,
                "ESC = Exit",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            cv2.imshow(
                "Bird Detection",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":

    detector = BirdDetector()
    detector.run()