from ultralytics.nn.modules import Detect
import torch

from ultralytics.utils.tal import make_anchors
from ultralytics.nn.modules.conv import Conv
from torch import Tensor, Value, nn

class NonMaxSuppression(torch.autograd.Function):
    """NMS block for YOLO-fused model for TensorRT."""

    @staticmethod
    def forward(
        ctx,
        boxes, # [num_batches, spatial_dimension, 4].
        scores, # [num_batches, num_classes, spatial_dimension]
        max_output_boxes_per_class: float = 10,
        iou_threshold: float = 0.65,
        score_threshold: float = 0.25,
        center_point_box: int = 1, # [x_center, y_center, width, height]
    ) -> Tensor:
        _, num_classes, _ = scores.shape
        selected_indices = torch.randint(0, num_classes, (max_output_boxes_per_class * num_classes, 3), dtype=torch.int64)

        return selected_indices #[num_selected_indices, [batch_index, class_index, box_index]]

    @staticmethod
    def symbolic(
        g,
        boxes,
        scores,
        max_output_boxes_per_class: float = 10,
        iou_threshold: float = 0.65,
        score_threshold: float = 0.25,
        center_point_box: int = 1,
    ) -> Value:
        return g.op(
            'NonMaxSuppression',
            boxes,
            scores,
            torch.tensor(max_output_boxes_per_class, dtype=torch.int64),
            torch.tensor(iou_threshold, dtype=torch.float),
            torch.tensor(score_threshold, dtype=torch.float),
            center_point_box_i=center_point_box
        )

# class ORT_NMS(torch.autograd.Function):
#     '''ONNX-Runtime NMS operation'''
#     @staticmethod
#     def forward(ctx,
#                 boxes,
#                 scores,
#                 max_output_boxes_per_class=torch.tensor([100]),
#                 iou_threshold=torch.tensor([0.45]),
#                 score_threshold=torch.tensor([0.25])):
#         device = boxes.device
#         batch = scores.shape[0]
#         num_det = random.randint(0, 100)
#         batches = torch.randint(0, batch, (num_det,)).sort()[0].to(device)
#         idxs = torch.arange(100, 100 + num_det).to(device)
#         zeros = torch.zeros((num_det,), dtype=torch.int64).to(device)
#         selected_indices = torch.cat([batches[None], zeros[None], idxs[None]], 0).T.contiguous()
#         selected_indices = selected_indices.to(torch.int64)
#         return selected_indices

#     @staticmethod
#     def symbolic(g, boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold):
#         return g.op("NonMaxSuppression", boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold)


class UltralyticsDetect(Detect):
    """Ultralytics Detect head for detection models."""

    max_det = 10
    iou_thres = 0.65
    conf_thres = 0.25

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
          self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
          self.shape = shape

        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides
        cls = cls.sigmoid()
        dbox = dbox.transpose(1, 2)
        return dbox, cls, NonMaxSuppression.apply(
          dbox,
          cls,
          self.max_det,
          self.iou_thres,
          self.conf_thres,
        )

class UltralyticsPose(UltralyticsDetect):
    """YOLOv8 Pose head for keypoints models."""

    def __init__(self, nc=80, kpt_shape=(4, 3), ch=()):
        """Initialize YOLO network with default parameters and Convolutional Layers."""
        super().__init__(nc, ch)
        self.kpt_shape = kpt_shape  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
        self.nk = kpt_shape[0] * kpt_shape[1]  # number of keypoints total

        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)

    def forward(self, x):
        """Perform forward pass through YOLO model and return predictions."""
        bs = x[0].shape[0]  # batch size
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)  # (bs, 17*3, h*w)
        
        box, cls, selected_indices = UltralyticsDetect.forward(self, x)
        # print(selected_indices.size())
        pred_kpt = self.kpts_decode(bs, kpt).transpose(1, 2)
        

        class_index, box_index = selected_indices[:, 1], selected_indices[:, 2]
        detection_boxes = box[:, box_index, :]
        detection_classes = class_index.unsqueeze(0).type(torch.int32)
        detection_scores = cls[:, class_index, box_index]
        detection_keypoints = pred_kpt[:, box_index, :]
        num_detections = torch.tensor(selected_indices.size()[0], dtype=torch.int32).unsqueeze(0).unsqueeze(0)
        return num_detections, detection_boxes, detection_scores, detection_classes, detection_keypoints

    def kpts_decode(self, bs, kpts):
        """Decodes keypoints."""
        ndim = self.kpt_shape[1]
        if self.export:  # required for TFLite export to avoid 'PLACEHOLDER_FOR_GREATER_OP_CODES' bug
            y = kpts.view(bs, *self.kpt_shape, -1)
            a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                y[:, 2::3] = y[:, 2::3].sigmoid()  # sigmoid (WARNING: inplace .sigmoid_() Apple MPS bug)
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
            return y