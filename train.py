import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr-PConv.yaml')
    # model.load('') # loading pretrain weights
    model.train(data='datasets/VisDrone.yaml',
                cache=False,
                imgsz=640,
                epochs=200,
                batch=4,
                workers=4,
                device='0',
                #resume='runs/train/rtdert-r18-focaler-ciou/weights/last.pt', # last.pt path
                project='runs/train',
                name='rtdetr-PConv',
                )