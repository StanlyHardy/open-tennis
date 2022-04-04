import cv2
import numpy as np


class MathUtils(object):
    """
    General math resolvers.
    """
    @classmethod
    def group_pts(cls, input_coords, size=2):
        """
        Group points into tuples
        @param input_coords: input polygons
        @param size: size of the individual pairs.
        @return:
        """
        return [tuple(input_coords[i:i + size]) for i in range(0, len(input_coords), size)]

    @classmethod
    def apply_tx(cls, position: np.ndarray, transformation_matrix: np.ndarray):
        """
        Apply transformation matrix to the reference vector
        @param position: Vector on which the transformation matrix needs to be applied.
        @param transformation_matrix: The transformation matrix between src and dst
        @return:
        """
        transformed = cv2.perspectiveTransform(np.float32([position]).reshape(1, 1, 2), transformation_matrix)
        trans_2d_pos = transformed[0][0]
        return trans_2d_pos
