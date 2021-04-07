from mmdet.apis import init_detector, inference_detector
import mmcv
import time
import os

config_file = '/chase/dataset/chedou/models2/point_rend_r50_caffe_fpn_mstrain_1x_coco_chedou.py'
checkpoint_file = '/chase/dataset/chedou/models2/epoch_12.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = '/chase/dataset/chedou/images/1614671696.322689.jpg'  # or img = mmcv.imread(img), which will only load it once
# start = time.time()
# for i in range(10):
#     result = inference_detector(model, img)
# print((time.time() - start)/10)
# visualize the results in a new window
# model.show_result(img, result)
imgPath = '/chase/dataset/chedou/images3'
savePath = '/chase/dataset/chedou/draw2'
for imgName in os.listdir(imgPath):
    result = inference_detector(model, imgPath+'/'+imgName)
    model.show_result(imgPath+'/'+imgName, result, out_file=savePath+'/'+imgName)
