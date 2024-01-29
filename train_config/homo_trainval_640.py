import sys
sys.path.append('/data/zjy/homography/train_config')
from train_config.base import cfg

cfg.DATASET.TRAINVAL_DATA_SOURCE = "/data/zjy/data/homography/Oxford-Paris/"

cfg.DATASET.IMG_RESIZE = (640, 480)
