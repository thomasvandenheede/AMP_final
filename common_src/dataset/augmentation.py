import random
import numpy as np

class Augment:
    def __init__(self, rot_range=(-0.1, 0.1), scale_range=(0.95, 1.05),
                 flip_prob=0.5, trans_std=0.2):
        self.rot_range = rot_range
        self.scale_range = scale_range
        self.flip_prob = flip_prob
        self.trans_std = trans_std

    def apply_randaugment(self, points, boxes):
        # Random rotation
        rot_angle = random.uniform(*self.rot_range)
        cos_val, sin_val = np.cos(rot_angle), np.sin(rot_angle)
        rot_mat = np.array([[cos_val, -sin_val], [sin_val, cos_val]])
        points[:, :2] = points[:, :2] @ rot_mat.T
        boxes.tensor[:, :2] = boxes.tensor[:, :2] @ rot_mat.T
        boxes.tensor[:, 6] += rot_angle

        # Random scaling
        scale = random.uniform(*self.scale_range)
        points[:, :3] *= scale
        boxes.tensor[:, :6] *= scale

        # Random flip
        if random.random() < self.flip_prob:
            points[:, 1] *= -1
            boxes.tensor[:, 1] *= -1
            boxes.tensor[:, 6] *= -1

        # Random translation
        trans = np.random.normal(scale=self.trans_std, size=2)
        points[:, 0] += trans[0]
        points[:, 1] += trans[1]
        boxes.tensor[:, 0] += trans[0]
        boxes.tensor[:, 1] += trans[1]

        return points, boxes

    def __call__(self, points, boxes, labels):
        points, boxes = self.apply_randaugment(points, boxes)
        return points, boxes, labels
