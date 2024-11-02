import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('runs/train/exp-rt-detr-r18/weights/best.pt') # select your model.pt path
    model.predict(source='datasets/VisDrone2019/VisDrone2019-DET-test-dev/images',
                  project='runs/detect',
                  name='exp',
                  save=True,
                #   visualize=True # visualize model features maps
                  )