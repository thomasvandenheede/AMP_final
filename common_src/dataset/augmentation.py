import random
import numpy as np
import cv2

class Augment:
    def __init__(self, rot_range=(-0.1, 0.1), scale_range=(0.95, 1.05),
                 flip_prob=0.5, image_size=(1936, 1216)):
        """

        The image size used for the point painting is 520x520


        Augmentation for LiDAR and image data (no translation).

        Parameters:
        - rot_range: range of random rotation angles (radians)
        - scale_range: range for random scaling factors
        - flip_prob: probability of applying a Y-axis flip
        - image_size: (width, height) for image transformations
        """
        self.rot_range = rot_range
        self.scale_range = scale_range
        self.flip_prob = flip_prob
        self.image_size = image_size

    def apply_randaugment(self, points, boxes, image=None, boxes_2d=None):
        # Random rotation
        rot_angle = random.uniform(*self.rot_range)
        cos_val, sin_val = np.cos(rot_angle), np.sin(rot_angle)
        rot_mat = np.array([[cos_val, -sin_val], [sin_val, cos_val]])

        points[:, :2] = points[:, :2] @ rot_mat.T
        boxes.tensor[:, :2] = boxes.tensor[:, :2] @ rot_mat.T
        boxes.tensor[:, 6] += rot_angle

        if image is not None:
            center = (self.image_size[0] / 2, self.image_size[1] / 2)
            rot_deg = np.degrees(rot_angle)
            M = cv2.getRotationMatrix2D(center, rot_deg, 1.0)
            image = cv2.warpAffine(image, M, self.image_size)

            if boxes_2d is not None:
                for i in range(boxes_2d.shape[0]):
                    x1, y1, x2, y2 = boxes_2d[i]
                    corners = np.array([
                        [x1, y1],
                        [x2, y1],
                        [x2, y2],
                        [x1, y2]
                    ])
                    ones = np.ones((4, 1))
                    corners_h = np.hstack([corners, ones])
                    transformed = (M @ corners_h.T).T
                    x_min, y_min = transformed.min(axis=0)
                    x_max, y_max = transformed.max(axis=0)
                    boxes_2d[i] = [x_min, y_min, x_max, y_max]

        # Random scaling
        scale = random.uniform(*self.scale_range)
        points[:, :3] *= scale
        boxes.tensor[:, :6] *= scale

        if boxes_2d is not None:
            boxes_2d *= scale
        if image is not None:
            new_size = (int(self.image_size[0] * scale), int(self.image_size[1] * scale))
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)

        # Random flip along Y (horizontal flip in image)
        if random.random() < self.flip_prob:
            points[:, 1] *= -1
            boxes.tensor[:, 1] *= -1
            boxes.tensor[:, 6] *= -1

            if image is not None:
                image = cv2.flip(image, 1)
                if boxes_2d is not None:
                    w = image.shape[1]
                    x_min = w - boxes_2d[:, 2]
                    x_max = w - boxes_2d[:, 0]
                    boxes_2d[:, 0] = x_min
                    boxes_2d[:, 2] = x_max

        if image is not None:
            image = image.astype(np.float32) / 255.0
            image = torch.tensor(image).permute(2, 0, 1)  # HWC -> CHW
            image = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])(image)

        return points, boxes, image, boxes_2d

    def __call__(self, points, boxes, labels, image=None, boxes_2d=None):
        points, boxes, image, boxes_2d = self.apply_randaugment(points, boxes, image, boxes_2d)
        return points, boxes, labels, image, boxes_2d
