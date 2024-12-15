import torch
import torchvision.ops as ops

boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=torch.float32).cuda()
scores = torch.tensor([0.9, 0.8], dtype=torch.float32).cuda()
iou_threshold = 0.5

result = ops.nms(boxes, scores, iou_threshold)
print("NMS result:", result)
