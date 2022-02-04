from models.ssd_mobilenet import ssd_300
import cv2
from evaluation import model

if __name__ == '__main__':

    model.load_weights(args.weight_file)