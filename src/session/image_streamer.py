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
        self.img_dir = os.path.expanduser(self.app_profile["paths"]["img_path"])
        if not os.path.isdir(self.img_dir):
            print("Please check if the image directory is vald {}".format(self.img_dir))
            exit()
        self.img_paths = glob(self.img_dir + "*.jpg")
        if len(self.img_paths) == 0:
            print("The folder {} has got zero images.".format(os.path.expanduser(self.img_dir)))
            exit()
        else:
            self.fh, self.fw, _ = cv2.imread(self.img_paths[0]).shape

        print("Input Image Path : ", self.img_dir)

        self.img_count = 0
        self.img_paths = sorted(self.img_paths, key=lambda x: str(os.path.splitext(x)[0]))[:100]
        self.total_frame_count = len(self.img_paths)
        self.p_bar = tqdm(range(len(self.img_paths)), desc="Streaming Images...")

    @property
    def width(self):
        return self.fw

    @property
    def height(self):
        return self.fh

    def update(self):
        """
        Retrieve the frame from the disk and update the session
        """
        if self.img_count == self.total_frame_count:
            self.result_coordinator.persist(os.path.expanduser(self.app_profile["paths"]["logs_path"]), self.gt_ann,
                                            self.total_frame_count)

            print("Completed!...")
            time.sleep(0.5)
            exit()

        self.p_bar.update()
        file_path = self.img_paths[self.img_count]
        self._frame_count = os.path.basename(file_path).split(".")[0]
        frame = cv2.imread(file_path)
        self._set_detection_frame(frame)
        self.img_count += 1

    @property
    def frame_count(self):
        """
        Get the current framecount
        :return:
        """
        return self._frame_count

    def __del__(self):
        self._switch_off()

    def _switch_off(self):
        cv2.destroyAllWindows()
