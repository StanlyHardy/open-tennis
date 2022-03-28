import os.path

import cv2

from src.session.session_context import SessionContext


class VideoStreamer(SessionContext):

    def __init__(self):
        """
        Video Playback session handler
        """
        super().__init__()
        self.video_path = os.path.expanduser(self.app_profile["paths"]["video_path"])
        if not os.path.exists(self.video_path):
            print("Please check the video path".format(self.video_path))
            exit()
        self.video = cv2.VideoCapture(self.video_path)
        print("Input Video Path :{}".format(self.video_path))

        self.fw = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.fh = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def width(self) -> int:
        return self.fw

    @property
    def height(self) -> int:
        return self.fh

    def update(self):
        """
        Retrieve the frame from the video and update the session
        """
        _, frame = self.video.read()

        if not _:
            print("Completed!...")
            exit()

        self._set_detection_frame(frame)
        self._frame_count += 1

    @property
    def frame_count(self) -> int:
        """
        :return: return the frame count
        """
        return self._frame_count

    def __del__(self):
        self._switch_off()

    def _switch_off(self):
        """
        gracefully exit the session
        :return:
        """
        if hasattr(self, 'video'):
            self.video.release()
            cv2.destroyAllWindows()
