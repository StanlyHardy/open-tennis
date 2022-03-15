from src import AppContext
import onnxruntime as rt

from src.controllers.ocr.dl_txt_recognizer import DLTextRecognizer
from src.controllers.ocr.tesseract_ocr import TesserTextRecognizer


class ModelManager(AppContext):
    def __init__(self):
        """
        Manages the models for detection and recognition
        """
        self.detector_session = self.load_detector()
        self.text_recognizer = self.load_text_recognizer()

    def load_detector(self):
        """
        Load the detector Model
        :return:
        """
        sess_options = rt.SessionOptions()
        session = rt.InferenceSession(self.app_profile["models"]["score_det_model"],
                                      providers=["CUDAExecutionProvider"],
                                      sess_options=sess_options)

        return session

    def load_text_recognizer(self):
        """
        Load the recognizer model
        :return:
        """
        if self.app_profile["models"]["ocr_engine"] == "PyTesseract":
            print("Initializing OCR Backend : PyTesseract")
            text_recognizer = TesserTextRecognizer()
        else:
            print("Initializing OCR Backend : CRNN")
            text_recognizer = DLTextRecognizer()
        return text_recognizer
