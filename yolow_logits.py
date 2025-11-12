import numpy as np
import os
import torch
from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect

try:
    from ultralytics.models.yolo.world.head import WorldDetect
except ImportError:
    WorldDetect = None

from ultralytics.utils import ops, nms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from PIL import Image
import json

JSON_OUTPUT_PATH = "logits.json"
MODEL_PATH = "/home/coloranto/Documents/PhD/yolow_logits/yolo-world-hooks/yolov8s-world.pt"
IMAGE_PATH = "/home/coloranto/Documents/PhD/yolow_logits/yolo-world-hooks/image"

class SaveIO:
    """Simple PyTorch hook to save the output of a nn.module."""
    def __init__(self):
        self.input = None
        self.output = None
        
    def __call__(self, module, module_in, module_out):
        self.input = module_in
        self.output = module_out



def load_and_prepare_model(model_path):
    """
    Load YOLO-World model and register hooks.
    YOLO-World may have a different architecture, so we need to be more flexible.
    """
    model = YOLO(model_path)
    model.model.eval()
    
    detect = None
    cv2_hooks = None
    cv3_hooks = None
    detect_hook = SaveIO()
    
    # Debug: print all module types to see what we're working with
    # print("Model architecture modules:")
    # for i, module in enumerate(model.model.modules()):
    #     module_type = type(module).__name__
    #     if 'detect' in module_type.lower() or 'head' in module_type.lower():
    #         print(f"  {i}: {module_type}")
    
    # Find the detection head - support both YOLOv8 (Detect) and YOLO-World (WorldDetect)
    for i, module in enumerate(model.model.modules()):
        # Check if it's a Detect or WorldDetect module
        if isinstance(module, Detect) or (WorldDetect and isinstance(module, WorldDetect)):
            module_type = type(module).__name__
            print(f"\nFound detection module: {module_type}")
            module.register_forward_hook(detect_hook)
            detect = module
            
            # Check if it has the expected attributes
            if hasattr(module, 'nl'):
                print(f"  Number of detection layers (nl): {module.nl}")
                cv2_hooks = [SaveIO() for _ in range(module.nl)]
                cv3_hooks = [SaveIO() for _ in range(module.nl)]
                print("cv2 hooks", cv2_hooks)
                print("cv3 hooks", cv3_hooks)
                
                if hasattr(module, 'cv2') and hasattr(module, 'cv3'):
                    for i in range(module.nl):
                        module.cv2[i].register_forward_hook(cv2_hooks[i])
                        module.cv3[i].register_forward_hook(cv3_hooks[i])
                    print("  Registered hooks to cv2 and cv3 layers")
                else:
                    print("  WARNING: Module doesn't have cv2/cv3 attributes!")
                    print(f"  Available attributes: {[attr for attr in dir(module) if not attr.startswith('_')]}")
            else:
                print("  WARNING: Module doesn't have 'nl' attribute!")
            break
    
    if detect is None:
        raise RuntimeError("Could not find Detect module in model!")
    
    input_hook = SaveIO()
    model.model.register_forward_hook(input_hook)

    hooks = [input_hook, detect, detect_hook, cv2_hooks, cv3_hooks]
    return model, hooks


def results_predict(img_path, model, hooks, threshold=0.1, iou=0.7, save_image=False, category_mapping=None):
    """
    Run prediction with YOLO-World model and extract logits.
    """
    input_hook, detect, detect_hook, cv2_hooks, cv3_hooks = hooks

    # Run inference
    print(f"\nRunning inference on {img_path}...")
    _ = model(img_path, verbose=False)
    
    # Debug: Check what the hooks captured
    print(f"detect_hook.input: {type(detect_hook.input)}")
    print(f"detect_hook.output: {type(detect_hook.output)}")
    
    if detect_hook.input is not None:
        print(f"  detect_hook.input length: {len(detect_hook.input) if detect_hook.input else 'None'}")
        if detect_hook.input and len(detect_hook.input) > 0:
            print(f"  detect_hook.input[0] type: {type(detect_hook.input[0])}")
            if isinstance(detect_hook.input[0], (list, tuple)) and len(detect_hook.input[0]) > 0:
                print(f"  detect_hook.input[0][0] shape: {detect_hook.input[0][0].shape if hasattr(detect_hook.input[0][0], 'shape') else 'no shape'}")
    
    if detect_hook.output is not None:
        print(f"  detect_hook.output length: {len(detect_hook.output) if isinstance(detect_hook.output, (list, tuple)) else 'not a list'}")
    
    # Check cv2/cv3 hooks
    if cv2_hooks:
        print(f"cv2_hooks outputs: {[type(h.output) for h in cv2_hooks]}")
    if cv3_hooks:
        print(f"cv3_hooks outputs: {[type(h.output) for h in cv3_hooks]}")
    
    # Validation
    if detect_hook.input is None:
        raise RuntimeError("detect_hook.input is None - hooks not capturing data!")
    
    if not isinstance(detect_hook.input, (list, tuple)) or len(detect_hook.input) == 0:
        raise RuntimeError(f"detect_hook.input has unexpected format: {type(detect_hook.input)}")
    
    if detect_hook.input[0] is None:
        raise RuntimeError("detect_hook.input[0] is None!")
    
    if not  isinstance(detect_hook.input[0], (list, tuple)) or len(detect_hook.input[0]) == 0:
        raise RuntimeError(f"detect_hook.input[0] has unexpected format: {type(detect_hook.input[0])}")
    

    # Now try to get the shape
    try:
        shape = detect_hook.input[0][0].shape  # Batch size, C number of channels, Height, Width
        print(f"Input shape: {shape}")
    except Exception as e:
        print(f"ERROR accessing shape: {e}")
        print(f"detect_hook.input[0][0] = {detect_hook.input[0][0]}")
        raise
    

    x = []
    
    for i in range(detect.nl):
        x.append(torch.cat((cv2_hooks[i].output, cv3_hooks[i].output), 1))
    x_cat = torch.cat([xi.view(shape[0], detect.no, -1) for xi in x], 2)
    box, cls = x_cat.split((detect.reg_max * 4, detect.nc), 1)

    batch_idx = 0
    xywh_sigmoid = detect_hook.output[0][batch_idx]
    all_logits = cls[batch_idx]

    img_shape = input_hook.input[0].shape[2:]
    orig_img_shape = model.predictor.batch[1][batch_idx].shape[:2]

    boxes = []
    for i in range(xywh_sigmoid.shape[-1]):
        x0, y0, x1, y1, *class_probs_after_sigmoid = xywh_sigmoid[:,i]
        x0, y0, x1, y1 = ops.scale_boxes(img_shape, np.array([x0.cpu(), y0.cpu(), x1.cpu(), y1.cpu()]), orig_img_shape)
        logits = all_logits[:,i]
        
        boxes.append({
            'image_id': img_path,
            'bbox': [x0.item(), y0.item(), x1.item(), y1.item()],
            'bbox_xywh': [(x0.item() + x1.item())/2, (y0.item() + y1.item())/2, x1.item() - x0.item(), y1.item() - y0.item()],
            'logits': logits.cpu().tolist(),
            'activations': [p.item() for p in class_probs_after_sigmoid]
        })

    # NMS
    boxes_for_nms = torch.stack([
        torch.tensor([*b['bbox_xywh'], *b['activations'], *b['activations'], *b['logits']]) for b in boxes
    ], dim=1).unsqueeze(0)
    
    nms_results = nms.non_max_suppression(boxes_for_nms, conf_thres=threshold, iou_thres=iou, nc=detect.nc)[0]
    
    boxes = []
    for b in range(nms_results.shape[0]):
        box = nms_results[b, :]
        x0, y0, x1, y1, conf, cls, *acts_and_logits = box
        activations = acts_and_logits[:detect.nc]
        logits = acts_and_logits[detect.nc:]
        box_dict = {
            'bbox': [x0.item(), y0.item(), x1.item(), y1.item()],
            'bbox_xywh': [(x0.item() + x1.item())/2, (y0.item() + y1.item())/2, x1.item() - x0.item(), y1.item() - y0.item()],
            'best_conf': conf.item(),
            'best_cls': cls.item(),
            'image_id': img_path,
            'activations': [p.item() for p in activations],
            'logits': [p.item() for p in logits]
        }
        boxes.append(box_dict)

    return boxes


# Test function
def main():
    SAVE_TEST_IMG = True
    threshold = 0.1
    nms_threshold = 0.7

    print("Loading model...")
    model, hooks = load_and_prepare_model(MODEL_PATH)
    
    print("\nRunning prediction...")
    results = results_predict(IMAGE_PATH, model, hooks, threshold=threshold, iou=nms_threshold)
    print("results follows: ", results[0])

    with open(JSON_OUTPUT_PATH, "w") as j_file:
        json.dump(results, j_file,  indent=4)

if __name__ == '__main__':
    main()