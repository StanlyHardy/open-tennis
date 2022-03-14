import cv2
from src.session.session_context import SessionContext


class VideoSession(SessionContext):

    def __init__(self):
        super().__init__()
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
        self.frame_count += 1

    def __del__(self):
        self._switch_off()

    def _switch_off(self):
        self.video.release()
        cv2.destroyAllWindows()
