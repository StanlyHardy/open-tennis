import cv2
from src.session.session_context import SessionContext


class VideoStreamer(SessionContext):

    def __init__(self):
        """
        Video Playback session handler
        """
        super().__init__()
        self.video = cv2.VideoCapture(self.streamer_profile["video_path"])
        print("Input Video Path : ", self.streamer_profile["video_path"])

        self.fw = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.fh = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def update(self):
        """
        Retrieve the frame from the video and update the session
        """
        _, frame = self.video.read()

        if not _:
            print("Completed!...")
            exit()

        self._set_detection_frame(frame)
        self.frame_count += 1

    def get_frame_count(self):
        """
        :return: return the frame count
        """
        return self.frame_count

    def __del__(self):
        self._switch_off()

    def _switch_off(self):
        """
        gracefully exit the session
        :return:
        """
        self.video.release()
        cv2.destroyAllWindows()

