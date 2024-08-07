import sys
from copy import deepcopy
import onnx
import onnxsim
import torch
from ultralytics import YOLO
from ultralytics.utils.checks import check_imgsz

from typing import Optional
from pathlib import Path
from head import UltralyticsPose

from loguru import logger

logger.configure(handlers=[{'sink': sys.stdout, 'colorize': True, 'format': "<level>[{level.name[0]}]</level> <level>{message}</level>"}])

OUTPUT_NAMES = ['num_dets', 'det_boxes', 'det_scores', 'det_classes', 'det_kpts']

def export_onnx(
  weights: str,
  output: str,
  imgsz: Optional[int] = 640,
  batch: Optional[int] = 1,
  opset_version: Optional[int] = 11,
  max_boxes: Optional[int] = 10,
  iou_thres: Optional[float] = 0.65,
  conf_thres: Optional[float] = 0.25,
) -> None:
  
  logger.info("Starting export with Pytorch.")
  model = YOLO(model=weights).model
  model = deepcopy(model).to(torch.device("cpu"))

  for m in model.modules():
    class_name = m.__class__.__name__
    if class_name == "Pose":
      detect_head = UltralyticsPose
      detect_head.dynamic = False
      detect_head.max_det = max_boxes
      detect_head.iou_thres = iou_thres
      detect_head.conf_thres = conf_thres
      m.__class__ = detect_head
      break

  imgsz = check_imgsz(imgsz, stride=model.stride, min_dim=2)

  im = torch.zeros(batch, 3, *imgsz).to(torch.device("cpu"))

  for p in model.parameters():
    p.requires_grad = False
  model.eval()
  model.float()
  for _ in range(2):  # Warm-up run
    model(im)

  output_path = Path(output)
  output_path.mkdir(parents=True, exist_ok=True)
  onnx_filepath = output_path / (Path(weights).stem + ".onnx")

  torch.onnx.export(
      model=model,
      args=im,
      f=str(onnx_filepath),
      opset_version=opset_version,
      input_names=['images'],
      output_names=OUTPUT_NAMES,
      dynamic_axes=None,
  )

  model_onnx = onnx.load(onnx_filepath)
  onnx.checker.check_model(model_onnx)

  logger.success(f"Simplifying with onnxsim {onnxsim.__version__}...")
  model_onnx, check = onnxsim.simplify(model_onnx)
  assert check, "Simplified ONNX model could not be validated"

  onnx.save(model_onnx, onnx_filepath)

  logger.success(f'Export complete, results saved to {output}, visualize at https://netron.app')

if __name__ == "__main__":
  export_onnx(
    weights="best.pt",
    output="best")