import cv2


class Renderer(object):
    """
    Renderer is responsible to respond for every draw calls that it receives about the player data.
    """

    def __init__(self, should_draw):
        self.should_draw = should_draw

    def draw_canvas(self, drawable_frame, output_vis, alpha=0.4):
        """
        Creates a drawable to write player data.
        :param drawable_frame: drawable that will be used as the canvas
        :param output_vis: the overlapped image
        :param alpha:  alpha to control the transparency.
        :return: rendered image output
        """
        if self.should_draw:
            drawable_frame = cv2.rectangle(drawable_frame, (1350, 900), (1800, 1000), (0, 0, 0), -1)
            drawable_frame = cv2.rectangle(drawable_frame, (850, 900), (1300, 1000), (0, 0, 0), -1)
            drawable_frame = cv2.rectangle(drawable_frame, (850, 830), (1800, 890), (0, 0, 0), -1)

            cv2.addWeighted(drawable_frame, alpha, output_vis, 1 - alpha,
                            0, output_vis)
        return output_vis

    def text(self, image, data, coordinate, font_face=cv2.LINE_AA, fontscale=1, font_color=(125, 246, 55), thickness=2):
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
                thickness=thickness
            )
        return image

    def rect(self, image, pt1, pt2, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA):
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
            cv2.rectangle(image, pt1, pt2, color, thickness=thickness, lineType=lineType)
        return image
