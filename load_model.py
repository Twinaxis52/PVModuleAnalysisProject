# Loading in detectron 2 model
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.modeling import build_model
import torch

def prepare_predictor(threshold, path):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    # If no GPU available, we use a CPU Detectron copy.
    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = 'cpu' 
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.MODEL.WEIGHTS = (path) # model output path
    predictor = DefaultPredictor(cfg)

    return predictor