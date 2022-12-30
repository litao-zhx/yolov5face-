# -*- coding: UTF-8 -*-
import argparse
import time
from pathlib import Path
import os
import cv2
import torch
import onnx
import onnxruntime
import copy
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, scale_coords, xyxy2xywh
from utils.torch_utils import  time_synchronized
import glob
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def load_model(weights, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model

def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7]] -= pad[1]  # y padding

    coords[:, :8] /= gain
    # coords[:, :10] /= gain #人脸
    #clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1  #.clamp_函数主要用于控制边界
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    # coords[:, 8].clamp_(0, img0_shape[1])  # x5
    # coords[:, 9].clamp_(0, img0_shape[0])  # y5
    return coords

def show_results(img, xywh, conf, landmarks,class_num):
    h,w,c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
    y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
    x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
    y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)

    cv2.rectangle(img, (x1,y1), (x2, y2), (0,255,0), thickness=tl, lineType=cv2.LINE_AA)
    # print("左上：",(x1,y1))

    clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0)]
    # clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)] #人脸

    rects=[]
    for i in range(4): #四个端点 (左上、右上、右下、左下)
    # for i in range(5): #左眼、右眼、鼻子、左嘴角、右嘴角
        point_x = int(landmarks[2 * i] * w) #1
        point_y = int(landmarks[2 * i + 1] * h)#1
        cv2.circle(img, (point_x, point_y), tl+1, clors[i], -1) #1
        rects.append([point_x,point_y])

    if len(rects)==0:
        print(imgpath + "\t No target detected!!!")
    tf = max(tl - 1, 1)  # font thickness
    # label = names[int(class_num)]+ str(conf)[:5]
    # cv2.putText(new_img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img #测试时


def detect_one(model, image_path, device,img_size,save_dir):


    # Load model
    img_size = img_size
    conf_thres = 0.3
    iou_thres = 0.5

    s=time.time()
    orgimg = cv2.imread(image_path)  # BGR
    e0=time.time()
    # print("读取图片时间：", e0 - s)

    img0 = copy.deepcopy(orgimg)
    e1=time.time()
    assert orgimg is not None, 'Image Not Found ' + image_path
    h0, w0 = orgimg.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # resize image to img_size
    s1=time.time()
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)
    # img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)))
    e1=time.time()
    # print("resize time :",e1-s1)

    imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size

    img = letterbox(img0, new_shape=imgsz)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416

    # Run inference
    t0 = time.time()

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    # print("66666:",type(img))
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img)[0]
    # Apply NMS
    pred = non_max_suppression_face(pred, conf_thres, iou_thres)
    t2 = time.time()

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        flag=0
        my_dict = {}

        gn = torch.tensor(orgimg.shape)[[1, 0, 1, 0]].to(device)  # normalization gain whwh
        gn_lks = torch.tensor(orgimg.shape)[[1, 0, 1, 0, 1, 0, 1, 0]].to(device)  # normalization gain landmarks
        # gn_lks = torch.tensor(orgimg.shape)[[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]].to(device)  # normalization gain landmarks 人脸
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()

            # Print results1
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class

            det[:, 5:13] = scale_coords_landmarks(img.shape[2:], det[:, 5:13], orgimg.shape).round()
            # det[:, 5:15] = scale_coords_landmarks(img.shape[2:], det[:, 5:15], orgimg.shape).round() #人脸

            for j in range(det.size()[0]):
                xywh = (xyxy2xywh(det[j, :4].view(1, 4)) / gn).view(-1).tolist()
                conf = det[j, 4].detach().cpu().numpy()
                # print(conf)
                landmarks = (det[j, 5:13].view(1, 8) / gn_lks).view(-1).tolist()
                # landmarks = (det[j, 5:15].view(1, 10) / gn_lks).view(-1).tolist() #人脸

                # class_num = det[j, 15].cpu().numpy()#人脸
                class_num = det[j, 13].cpu().numpy()

                # print(class_num)
                orgimg = show_results(orgimg, xywh, conf, landmarks,class_num) #测试

            imgname=Path(imgpath).name
            cv2.imwrite(save_dir+os.sep+imgname,orgimg)
            # orgimg,roi_img = show_results(orgimg, best_xywh, best_conf, best_landmarks,best_class_num) #测试


    return flag

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=r'./runs/best.pt', help='model.pt path(s)')
    parser.add_argument('--image', type=str, default=r'./images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    opt = parser.parse_args()
    # print(opt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #保存路径
    save_dir= r"./result"  #检测后的图
    os.makedirs(save_dir,exist_ok=True)

    img_dist=['jpg','png','peg']
    if opt.image[-3:] in img_dist: #单张
        start_time=time.time()

        model = load_model(opt.weights, device)

        detect_one(model, opt.image, device,opt.img_size,save_dir) #测试

        end_time=time.time()
        print("花费时间：",end_time-start_time)
    else:

        #pt模型
        model = load_model(opt.weights, device)
        #onnx推理
        # model=onnxruntime.InferenceSession(opt.weights)

        image_list=glob.glob(opt.image+os.sep+'*.jpg')
        #以下部分通用
        for imgpath in image_list:
            print("-*"*10+"\n",imgpath)
            start_time = time.time()
            flag = detect_one(model, imgpath, device, opt.img_size, save_dir)  # 测试
            end_time = time.time()
            print("花费时间：", end_time - start_time)
            print('--'*20)
