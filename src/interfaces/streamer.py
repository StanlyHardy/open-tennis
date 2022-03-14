import cv2
from src import AppContext


class Streamer(AppContext):

    def __init__(self):
        self.video = cv2.VideoCapture(self.streamer_profile["video_path"])
        print("Input Video Path : ", self.streamer_profile["video_path"])

        self.fw = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.fh = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def update_frames(self):
        _, frame = self.video.read()

        if not _:
            print("Completed!...")
            exit()

        self.set_overlay_frame(frame)
        self.set_detection_frame(frame)

    def set_overlay_frame(self, frame):
        self.overlayframe = frame.copy()

    def set_detection_frame(self, frame):
        self.detection_frame = frame.copy()

    def get_detection_frame(self):
        return self.detection_frame

    def get_overlay_frame(self):
        return self.overlayframe

    def is_interrupted(self):
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return True
        return False

    def __del__(self):
        self.switch_off()

    def switch_off(self):
        self.video.release()
        cv2.destroyAllWindows()
