import os.path
import time

import cv2

from glob import glob

from tqdm.auto import tqdm

from src.session.session_context import SessionContext


class ImageStreamer(SessionContext):

    def __init__(self):
        super().__init__()
        PATH = '/home/hardy/Workspace/Python/SportRadar/AnnotationCreator/assets/dataset/split/test/images'
        self.img_paths = glob(PATH + "/*.jpg")
        self.img_count = 0
        self.img_paths = sorted(self.img_paths, key=lambda x: str(os.path.splitext(x)[0]))
        self.total_frame_count = len(self.img_paths)
        self.p_bar = tqdm(range(len(self.img_paths)),desc="Streaming Images...")

    def update_frames(self):
        if self.img_count == self.total_frame_count:
            self.csv_logger.persist(self.gt_ann, self.total_frame_count)

            print("Completed!...")
            time.sleep(0.5)
            exit()

        self.p_bar.update()
        file_path = self.img_paths[self.img_count]
        self.frame_count = os.path.basename(file_path).split(".")[0]
        frame = cv2.imread(file_path)
        self.set_overlay_frame(frame)
        self.set_detection_frame(frame)
        self.img_count += 1

    def get_frame_count(self):
        return self.frame_count

    def __del__(self):
        self._switch_off()

    def _switch_off(self):
        cv2.destroyAllWindows()
