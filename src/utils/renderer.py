import cv2
import numpy as np
from PIL import Image, ImageDraw

from src.utils.daos import Result


class Renderer(object):
    """
    Renderer is responsible to respond for every draw calls that it receives about the player data.
    """

    def __init__(self, app_profile):
        self.should_draw = app_profile["streamer"]["should_draw"]
        self.tl, self.h, self.w = 0, 0, 0

    def draw_canvas(
        self, drawable_frame: np.ndarray, output_vis: np.ndarray, alpha=0.4
    ):
        """
        Creates a drawable to write player data.
        :param drawable_frame: drawable that will be used as the canvas
        :param output_vis: the overlapped image
        :param alpha:  alpha to control the transparency.
        :return: rendered image output
        """
        if self.should_draw:
            drawable_frame = cv2.rectangle(
                drawable_frame, (1350, 900), (1800, 1000), (0, 0, 0), -1
            )
            drawable_frame = cv2.rectangle(
                drawable_frame, (850, 900), (1300, 1000), (0, 0, 0), -1
            )
            drawable_frame = cv2.rectangle(
                drawable_frame, (850, 830), (1800, 890), (0, 0, 0), -1
            )

            cv2.addWeighted(drawable_frame, alpha, output_vis, 1 - alpha, 0, output_vis)
        return output_vis

    def text(
        self,
        image: np.ndarray,
        data: str,
        coordinate: tuple,
        font_face=cv2.LINE_AA,
        fontscale=1,
        font_color=(125, 246, 55),
        thickness=2,
    ):
        """
        Input text data.
        :param image: image on which the data needs to be written
        :param data: player information that needs to be written
        :param coordinate: coordinate of the text
        :param font_face: type of font
        :param fontscale: font size
        :param font_color: color of the font
        :param thickness: thickness of the written text
        :return: rendered image
        """
        if self.should_draw:
            cv2.putText(
                img=image,
                text=data,
                org=coordinate,
                fontFace=font_face,
                fontScale=fontscale,
                color=font_color,
                thickness=thickness,
            )
        return image

    def rect(
        self,
        image: np.ndarray,
        pt1: tuple,
        pt2: tuple,
        color=(0, 255, 0),
        thickness=2,
        lineType=cv2.LINE_AA,
    ):
        """
        Draws a rectangle at specified position
        :param image: image in which the rectangle needs to be drawn
        :param pt1: top coordinate
        :param pt2: bottom coordinate
        :param color: color of the rectangle
        :param thickness: thickness of the border
        :param lineType: line type in which the borders will be drawn
        :return:
        """
        if self.should_draw:
            cv2.rectangle(
                image, pt1, pt2, color, thickness=thickness, lineType=lineType
            )
        return image

    def render_result(self, det_frame, result: Result):
        """
        :param det_frame: Scoreboard object with its metadata
        :param result: Processed result
        """
        if self.tl == 0:
            self.h, self.w = det_frame.shape[:2]
        det_frame = self.draw_canvas(det_frame.copy(), det_frame)

        det_frame = self.draw_canvas(det_frame.copy(), det_frame)

        x1, y1, x2, y2 = map(int, result.score_board.bbox)
        self.text(det_frame, "scoreboard", (x1 + 3, y1 - 4), 0, self.tl / 3)
        self.rect(
            det_frame,
            (x1, y1),
            (x2, y2),
            thickness=max(int((self.w + self.h) / 600), 1),
        )
        self.text(
            det_frame,
            "Player 1: {}".format(result.name_1.title()),
            coordinate=(870, 940),
        )
        if len(result.score_1) > 0:
            self.text(
                det_frame, "Score:    {}".format(result.score_1), coordinate=(870, 980)
            )
        else:
            self.text(
                det_frame, "Score:    {}".format("Recognizing"), coordinate=(870, 980)
            )

        self.text(
            det_frame,
            "Player 2: {}".format(result.name_2.title()),
            coordinate=(1370, 940),
        )
        if len(result.score_2) > 0:
            self.text(
                det_frame, "Score:    {}".format(result.score_2), coordinate=(1370, 990)
            )
        else:
            self.text(
                det_frame, "Score:    {}".format("Recognizing"), coordinate=(1370, 990)
            )

        if result.serving_player == "unknown":
            draw_text = "Recognizing..."
        elif result.serving_player == "name_1":
            draw_text = result.name_1.title()
        else:
            draw_text = result.name_2.title()

        self.text(
            det_frame, "Serving Player: {}".format(draw_text), coordinate=(880, 870)
        )

    def convert_from_cv2_to_image(self, img: np.ndarray) -> Image:
        """

        @param img: input OpenCv image
        @return: PIL Image
        """

        return Image.fromarray(img)

    def render_court_points(self, original_img, final_pts) -> np.ndarray:
        """

        @param original_img: input image that shall be used as a canvas
        @param final_pts: final points that is obtained via homography
        @return:
        """

        pil_img = self.convert_from_cv2_to_image(original_img)
        image_draw = ImageDraw.ImageDraw(pil_img)
        image_draw.line([final_pts[6], final_pts[9]], fill=(120, 0, 255, 255), width=20)

        image_draw.line(
            (final_pts[1], final_pts[5], final_pts[8], final_pts[12]),
            fill=(0, 120, 255, 255),
            width=20,
        )

        image_draw.line(
            (final_pts[3], final_pts[7], final_pts[10], final_pts[13]),
            fill=(0, 120, 255, 255),
            width=20,
        )

        image_draw.line(
            (final_pts[5], final_pts[6], final_pts[7]),
            fill=(0, 120, 255, 255),
            width=20,
        )
        image_draw.line(
            (final_pts[8], final_pts[9], final_pts[10]),
            fill=(0, 120, 255, 255),
            width=20,
        )

        image_draw.line(
            (
                (final_pts[0][0] - 10, final_pts[0][1]),
                final_pts[1],
                final_pts[2],
                final_pts[3],
                (final_pts[4][0] + 10, final_pts[4][1]),
                final_pts[0],
                final_pts[11],
                (final_pts[12][0] - 10, final_pts[12][1]),
                final_pts[13],
                final_pts[14],
                final_pts[4],
            ),
            fill=(255, 102, 168, 255),
            width=20,
        )

        np_image = np.array(pil_img)

        return np_image
