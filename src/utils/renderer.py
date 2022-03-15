import cv2


class Renderer(object):

    def __init__(self, should_draw):
        self.should_draw = should_draw

    def text(self, image, data, coordinate, font_face=cv2.LINE_AA, fontscale=1, font_color=(125, 246, 55), thickness=2):
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

    def draw_boundary(self, drawable_frame, output_vis, alpha=0.4):
        if self.should_draw:
            drawable_frame = cv2.rectangle(drawable_frame, (1350, 900), (1800, 1000), (0, 0, 0), -1)
            drawable_frame = cv2.rectangle(drawable_frame, (850, 900), (1300, 1000), (0, 0, 0), -1)
            drawable_frame = cv2.rectangle(drawable_frame, (850, 830), (1800, 890), (0, 0, 0), -1)

            cv2.addWeighted(drawable_frame, alpha, output_vis, 1 - alpha,
                            0, output_vis)
        return output_vis

    def rect(self, image, pt1, pt2, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA):
        if self.should_draw:
            cv2.rectangle(image, pt1, pt2, color, thickness=thickness, lineType=lineType)
        return image
