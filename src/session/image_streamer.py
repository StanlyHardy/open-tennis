import os.path
import time
import cv2
from glob import glob
from tqdm.auto import tqdm

from src.session.session_context import SessionContext


class ImageStreamer(SessionContext):

    def __init__(self):
        """
        Video Playback session handler for evaluation
        """
        super().__init__()
        self.img_paths = glob(self.app_profile["streamer"]["img_path"] + "/*.jpg")
        print("Input Video Path : ", self.app_profile["streamer"]["img_path"])

        self.img_count = 0
        self.img_paths = sorted(self.img_paths, key=lambda x: str(os.path.splitext(x)[0]))
        self.total_frame_count = len(self.img_paths)
        self.p_bar = tqdm(range(len(self.img_paths)), desc="Streaming Images...")

    def update(self):
        """
        Retrieve the frame from the disk and update the session
        """
        if self.img_count == self.total_frame_count:
            self.csv_logger.persist(self.gt_ann, self.total_frame_count)

            print("Completed!...")
            time.sleep(0.5)
            exit()

        self.p_bar.update()
        file_path = self.img_paths[self.img_count]
        self.frame_count = os.path.basename(file_path).split(".")[0]
        frame = cv2.imread(file_path)
        self._set_detection_frame(frame)
        self.img_count += 1

    def get_frame_count(self):
        """
        Get the current framecount
        :return:
        """
        return self.frame_count

    def __del__(self):
        self._switch_off()

    def _switch_off(self):
        cv2.destroyAllWindows()
