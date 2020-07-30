#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math

from importlib import resources
import torch
import torch.nn as nn
from .boxes import (batched_nms, boxes_to_square, adjust_boxes, compute_boxes,
                    crop_boxes)

from .modules import PNet, RNet, ONet


class MTCNN(nn.Module):
    """MTCNN completely in PyTorch - no more numpy.

    Many of these methods pass around a "bounding_boxes" object.

    To avoid repeatedly describing it, here is what that object is like:

    //
    bounding_boxes : torch.Tensor
        size [num_boxes, 10]

        Slightly hair data structure, but buys us some convenience later.

        Each row is a single bounding box.

        Column 0 is batch index.
        Columns 1 - 4 are bounding box top left and bottom right coordinates.
        Column 5 is score for that box.
        Columns 6-10 are offset values.
    //
    """
    def __init__(self, pretrained=True,
                 min_face_size=20, min_detection_size=12,
                 score_thresholds=(.6, .7, .8), iou_thresholds=(.7, .7, .7),
                 factor=math.sqrt(.5)):

        super().__init__()

        self.pnet = PNet()
        self.rnet = RNet()
        self.onet = ONet()

        if pretrained:
            with resources.path("torchvision.models.mtcnn", "mtcnn.pth") as model_path:
                state_dict = torch.load(model_path)

            self.load_state_dict(state_dict)

        self.min_face_size = min_face_size
        self.min_detection_size = min_detection_size

        self.score_thresholds = score_thresholds
        self.iou_thresholds = iou_thresholds

        self.factor = factor

    def _pnet_helper(self, image, scale):
        """Apply PNet to single scaling of batch and return NMS boxes."""
        n, c, h, w = image.size()

        resized = nn.functional.interpolate(image,
                                            scale_factor=scale,
                                            mode='bilinear',
                                            align_corners=False)

        offsets, scores = self.pnet(resized)
        scores = scores[:, 1, :, :]
        boxes = compute_boxes(scores, offsets, scale, self.score_thresholds[0])
        # boxes: [batch_index, (4: bbox), score, (4: offset)]

        if boxes is None:
            return None

        kept = batched_nms(boxes, n, .5, mode='union')

        return kept

    def run_pnet(self, image, scales):
        """Run PNet at multiple scales, then return NMS-suppressed boxes."""
        n, c, h, w = image.size()

        bounding_boxes = []

        for s in scales:
            boxes = self._pnet_helper(image, scale=s)
            bounding_boxes.append(boxes)

        bounding_boxes = [i for i in bounding_boxes if i is not None]

        if len(bounding_boxes) == 0:
            return torch.tensor([], dtype=torch.long)

        bounding_boxes = torch.cat(bounding_boxes, dim=0)

        bounding_boxes = batched_nms(bounding_boxes, n,
                                     self.iou_thresholds[0],
                                     mode='union')

        bounding_boxes = adjust_boxes(bounding_boxes)
        bounding_boxes = boxes_to_square(bounding_boxes)
        bounding_boxes[:, 1:5] = torch.round(bounding_boxes[:, 1:5])

        return bounding_boxes

    def run_rnet(self, image, bounding_boxes):
        """Run RNet and return bounding boxes."""
        if len(bounding_boxes) == 0:
            return torch.tensor([], dtype=torch.float32)

        n, c, h, w = image.size()

        crops = crop_boxes(image, bounding_boxes, size=24)
        offsets, scores = self.rnet(crops)

        keep = (scores[:, 1] > self.score_thresholds[1]).nonzero(as_tuple=True)[0]
        bounding_boxes = bounding_boxes[keep, :]
        bounding_boxes[:, 5] = scores[keep, 1].view(-1)

        bounding_boxes = batched_nms(bounding_boxes, n,
                                     self.iou_thresholds[1],
                                     mode='union')

        bounding_boxes = adjust_boxes(bounding_boxes)
        bounding_boxes = boxes_to_square(bounding_boxes)
        bounding_boxes[:, 1:5] = torch.round(bounding_boxes[:, 1:5])

        return bounding_boxes

    def run_onet(self, image, bounding_boxes):
        """Run ONet and return bounding boxes."""
        n, c, h, w = image.size()

        crops = crop_boxes(image, bounding_boxes, size=48)

        if len(crops) == 0:
            return []

        landmarks, offsets, scores = self.onet(crops)

        keep = (scores[:, 1] > self.score_thresholds[2]).nonzero(as_tuple=True)[0]
        bounding_boxes = bounding_boxes[keep, :]
        bounding_boxes[:, 5] = scores[keep, 1].view(-1)
        landmarks = landmarks[keep]

        # Rescale landmarks
        width = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
        height = bounding_boxes[:, 4] - bounding_boxes[:, 2] + 1.0

        x_min, y_min = bounding_boxes[:, 1], bounding_boxes[:, 2]

        landmarks[:, 0:5] = x_min.unsqueeze(1) + width.unsqueeze(1) * landmarks[:, 0:5]
        landmarks[:, 5:10] = y_min.unsqueeze(1) + height.unsqueeze(1) * landmarks[:, 5:10]

        bounding_boxes = adjust_boxes(bounding_boxes)
        bounding_boxes = torch.cat((bounding_boxes, landmarks), dim=1)

        bounding_boxes = batched_nms(bounding_boxes, n,
                                     self.iou_thresholds[2], mode='min')

        return bounding_boxes

    def forward(self, image):
        n, c, w, h = image.size()
        min_length = min(h, w)

        scales = []

        m = self.min_detection_size / self.min_face_size
        min_length *= m

        factor_count = 0
        while min_length > self.min_detection_size:
            scales.append(m * self.factor ** factor_count)
            min_length *= self.factor
            factor_count += 1

        bounding_boxes = self.run_pnet(image, scales)
        bounding_boxes = self.run_rnet(image, bounding_boxes)
        bounding_boxes = self.run_onet(image, bounding_boxes)

        return bounding_boxes
