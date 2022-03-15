import cv2


class Renderer():

    def __init__(self, should_draw):
        self.should_draw = should_draw

    def text(self, image , data, coordinate, font_face = cv2.FONT_HERSHEY_DUPLEX, fontscale=3.0, font_color=(125,246,55),thickness=3):
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