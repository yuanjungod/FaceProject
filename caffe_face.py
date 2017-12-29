import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import glob
import sys
import os
from  math import pow
from PIL import Image, ImageDraw, ImageFont
import random
import caffe
from tools.rect_tool import *

# caffe.set_device(0)
caffe.set_mode_gpu()
print(dir(caffe))
base_dir = os.getcwd()
print("1")
sys.path.append(base_dir)
print("2")


def face_detection(imgFile):
    deploy = os.path.join(base_dir, 'deploy_full_conv.prototxt')
    caffemodel = os.path.join(base_dir, 'alexnet_iter_50000_full_conv.caffemodel')
    print("load model begin")
    net_full_conv = caffe.Net(deploy, caffemodel, caffe.TEST)
    print("load model end")
    randNum = random.randint(1, 10000)

    scales = []
    factor = 0.793700526

    img = cv2.imread(imgFile)

    largest = min(2, 4000 / max(img.shape[0:2]))  # shape[0]:height,shape[1]:width

    scale = largest
    minD = largest * min(img.shape[0:2])

    while minD >= 227:
        scales.append(scale)
        scale *= factor
        minD *= factor

    total_boxes = []

    for scale in scales:

        # scale_img = cv2.resize(img,((int(img.shape[0] * scale), int(img.shape[1] * scale))))
        scale_img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
        scale_img_name = 'scale/scale_img_%f.jpg' % scale
        scale_img_path = os.path.join(base_dir, scale_img_name)
        cv2.imwrite(scale_img_path, scale_img)

        im = caffe.io.load_image(scale_img_path)
        # net_full_conv.blobs['data'].reshape(1,3,scale_img.shape[1],scale_img.shape[0])
        net_full_conv.blobs['data'].reshape(1, 3, scale_img.shape[0], scale_img.shape[1])
        transformer = caffe.io.Transformer({'data': net_full_conv.blobs['data'].data.shape})

        ilsvrc_2012_mean = os.path.join(base_dir, 'ilsvrc_2012_mean.npy')
        transformer.set_mean('data', np.load(ilsvrc_2012_mean).mean(1).mean(1))
        transformer.set_transpose('data', (2, 0, 1))
        transformer.set_channel_swap('data', (2, 1, 0))
        transformer.set_raw_scale('data', 255.0)

        out = net_full_conv.forward_all(data=np.asarray([transformer.preprocess('data', im)]))

        boxes = generateBoundingBox(out['prob'][0, 1], scale)
        if boxes:
            total_boxes.extend(boxes)

    boxes_nms = np.array(total_boxes)
    true_boxes = nms_average(boxes_nms, 1, 0.2)
    if not true_boxes == []:
        (x1, y1, x2, y2) = true_boxes[0][:-1]
        print(x1, y1, x2, y2)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), thickness=5)

        output = os.path.join(base_dir, '1.jpg')
        cv2.imwrite(output, img)


imgFile = os.path.join(base_dir, 'img/Aaron_Sorkin_0002.jpg')
img = plt.imread(imgFile)
plt.imshow(img)
plt.show()
face_detection(imgFile)

imgFile = os.path.join(base_dir, '1.jpg')
img = plt.imread(imgFile)
plt.imshow(img)
plt.show()

