import sys
sys.path.insert(0, './yolov5')

from yolov5.utils.google_utils import attempt_download
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, \
    check_imshow
from yolov5.utils.plots import colors, plot_one_box
from yolov5.utils.torch_utils import select_device, time_synchronized
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from collections import deque
import numpy as np
import math

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
data_deque = {}
passenger_deque = {}
passenger_in = {}
passenger_out = {}

def Diff(li1, li2):
    return list(set(li1) - set(li2))

def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

'''
def draw_boxes(img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 +
                                 t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img
'''

def draw_boxes(img, bbox, identities=None, offset=(0, 0), trailslen=20):
    # print(len(identities))
    # print(type(identities))
    
    # remove tracked point from buffer if object is lost
    for key in list(data_deque):
      if key not in identities:
        data_deque.pop(key)

    for i, box in enumerate(bbox):
        # print("box muner", i)
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        
        # code to find center of bottom edge
        center = (int((x2+x1)/2), int((y1+y2)/2))
        # draw circle at center
        cv2.circle(img, center, 5, (0, 0, 255), -1)
        # pts.appendleft(center)
        # deque[key].appendleft(center)

        # get ID of object
        id = int(identities[i]) if identities is not None else 0

        # create new buffer for new object
        if id not in data_deque:  
          data_deque[id] = deque(maxlen= trailslen)

        color = compute_color_for_labels(id)

        # add center to buffer
        data_deque[id].appendleft(center)

        # print(data_deque[id])

        # draw trail
        for i in range(1, len(data_deque[id])):
            # check if on buffer value is none
            if data_deque[id][i - 1] is None or data_deque[id][i] is None:
                continue

            # generate dynamic thickness of trails
            thickness = int(np.sqrt(trailslen / float(i + i)) * 1.5)
            
            # draw trails
            cv2.line(img, data_deque[id][i - 1], data_deque[id][i], color, thickness)


        # box text and bar
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 +
                                 t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img



def detect(opt):
    out, source, weights, show_vid, save_vid, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.show_vid, opt.save_vid, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    attempt_download(deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(yolo_weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, vid_channel=vid_channel, show_mask=show_mask)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    save_path = str(Path(out))
    # extract what is in between the last '/' and last '.'
    txt_file_name = source.split('/')[-1].split('.')[0]
    txt_path = str(Path(out)) + '/' + txt_file_name + '.txt'

    '''
    ZONE_CH5_DETECT = [[300, 576], [300, 380], [380, 390], [550, 570], [550, 576]] #for inside the door
    ZONE_CH5_SILENT = [[250, 576], [250, 400], [360, 400], [460, 576]] #for dead zone
    roi_detect = np.array(ZONE_CH5_DETECT).reshape((-1,1,2)).astype(np.int32)     
    roi_silent = np.array(ZONE_CH5_SILENT).reshape((-1,1,2)).astype(np.int32)     
    '''
    #ZONE_CH3_SCREEN = [[0, 600], [0, 300], [150, 300], [150, 600]] #for inside the door    
    #ZONE_CH3_DETECT = [[0, 620], [0, 100], [100, 100], [200, 100], [200, 620], [100, 620]] #for inside the door    
    #ZONE_CH3_SILENT = [[0, 580], [0, 100], [2, 100], [2, 580]] #for dead zone
    ZONE_CH3_SCREEN = [[20, 560], [20, 150], [220, 250], [220, 460]] #for inside door area    
    ZONE_CH3_DETECT = [[0, 600], [0, 100], [100, 100], [250, 200], [250, 600], [100, 600]] #for close to door area    
    ZONE_CH3_SILENT = [[0, 520], [0, 120], [120, 220], [120, 520]] #for dead zone
    roi_screen = np.array(ZONE_CH3_SCREEN).reshape((-1,1,2)).astype(np.int32)     
    roi_detect = np.array(ZONE_CH3_DETECT).reshape((-1,1,2)).astype(np.int32)     
    roi_silent = np.array(ZONE_CH3_SILENT).reshape((-1,1,2)).astype(np.int32)     
    ppl_count, ppl_in, ppl_out = 0, 0, 0
    PHI0 = 45
    XV=[1, 0]
    trailslen = 24
    identities = []
    identities_prev = []
    bbox_xyxy = []
    bbox_xyxy_prev = []

    for frame_idx, (path, img, im0s, vid_cap, status) in enumerate(dataset):
        if frame_idx % step_frames == 0:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(
                pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t2 = time_synchronized()

            # Process detections
            s = status
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, sa, im0 = path[i], '%g: ' % i, im0s[i].copy()
                else:
                    p, sa, im0 = path, '', im0s

                s += sa + '%gx%g ' % img.shape[2:]  # print string
                save_path = str(Path(out) / Path(p).name)

                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string

                xywh_bboxs = []
                confs = []

                # Adapt detections to deep sort input format
                for *xyxy, conf, cls in det:
                    # to deep sort format
                    x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
                    xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                    xywh_bboxs.append(xywh_obj)
                    confs.append([conf.item()])

                xywhs = torch.Tensor(xywh_bboxs)
                confss = torch.Tensor(confs)

                # pass detections to deepsort
                outputs = deepsort.update(xywhs, confss, im0)

                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    draw_boxes(im0, bbox_xyxy, identities)
                    # to MOT format
                    tlwh_bboxs = xyxy_to_tlwh(bbox_xyxy)

                    # Write MOT compliant results to file
                    if save_txt:
                        for j, (tlwh_bbox, output) in enumerate(zip(tlwh_bboxs, outputs)):
                            bbox_top = tlwh_bbox[0]
                            bbox_left = tlwh_bbox[1]
                            bbox_w = tlwh_bbox[2]
                            bbox_h = tlwh_bbox[3]
                            identity = output[-1]
                            with open(txt_path, 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx, identity, bbox_top,
                                                            bbox_left, bbox_w, bbox_h, -1, -1, -1, -1))  # label format

                else:
                    deepsort.increment_ages()

                # Print time (inference + NMS)
                print('%sDone. (%.3fs)' % (s, t2 - t1))

                # Stream results
                if show_vid:
                    if show_roi: 
                        cv2.polylines(im0, [roi_detect], isClosed = True, color = (0, 0, 255), thickness = 1)                                                  
                        cv2.polylines(im0, [roi_silent], isClosed = True, color = (0, 0, 255), thickness = 1)                                                                          
                        cv2.polylines(im0, [roi_screen], isClosed = True, color = (255, 0, 0), thickness = 1)                                                                                                  
                    if ppl_count in list(passenger_deque):
                        (xn2, yn2, xn1, yn1) = passenger_deque[ppl_count]
                        displacement = np.sqrt((xn2-xn1)*(xn2-xn1)+(yn2-yn1)*(yn2-yn1))
                        cv2.arrowedLine(im0, (xn1, yn1), (xn2, yn2), (0, 255, 255), thickness=2, line_type=cv2.LINE_AA, tipLength=0.15)
                        cv2.putText(im0, f'{int(displacement)}', (int((xn1+xn2)/2), int((yn1+yn2)/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)        
                    #cv2.putText(im0, f'count: {ppl_count}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(im0, f'count: {ppl_count}, in: {ppl_in}, out: {ppl_out}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow(p, im0)
                    if frame_idx == 0: time.sleep(5)
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        raise StopIteration

                # Save results (image with detections)
                if save_vid:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'

                        #vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_vid:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5/weights/yolov5s.pt', help='model.pt path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default='0', help='source')
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=320, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.2, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.4, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--vid-channel', type=str, default='', help='select which video channel to use: ch2-front+side, ch3-back, ch5-front')
    parser.add_argument('--step-frames', type=int, default=1, help='decimate every step-frames to speed up processing')    
    parser.add_argument('--show-mask', action='store_true', help='display image non-detection mask region')    
    parser.add_argument('--show-roi', action='store_true', help='display tracking roi zone')    
    parser.add_argument('--show-det', action='store_true', help='display detection raw boxs')        
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', default=[0], type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)
    args.vid_channel = args.vid_channel.lower()
    #if args.vid_channel != 'ch2' and args.vid_channel != 'ch3' and args.vid_channel != 'ch5':
        #raise Exception(f'ERROR: videl channel label {args.vid_channel} is invalid, it has to be either ch2, ch3, or ch5!')
    #if not args.vid_channel in args.source:
        #raise Exception(f'ERROR: videl channel label {args.vid_channel} does not match {args.source} channel!')

    with torch.no_grad():
        detect(args)

'''
# Classes
nc: 80  # number of classes
names: [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]  # class names
'''

#python track.py --source "d:\\myData\\Bus\\SD1377CAM3.mp4" --show-vid --conf-thres 0.1 --iou-thres 0.2 --weights "D:\\CodeBucket\\Yolov5_DeepSort_Pytorch\\yolov5\\weights\\crowdhuman_yolov5m.pt"
#python track.py --source "d:\\myData\\Bus\\WH7555CAM3.mp4" --show-vid --conf-thres 0.1 --iou-thres 0.2 --weights "D:\\CodeBucket\\Yolov5_DeepSort_Pytorch\\yolov5\\weights\\crowdhuman_yolov5m.pt"
#python track.py --source "d:\\myData\\Bus\\SD1377CAM2.mp4" --show-vid --conf-thres 0.2 --iou-thres 0.4 --weights "D:\\CodeBucket\\Yolov5_DeepSort_Pytorch\\yolov5\\weights\\crowdhuman_yolov5m.pt" 
#python track.py --source "d:\\myData\\Bus\\WH7555CAM2.mp4" --show-vid --conf-thres 0.2 --iou-thres 0.4 --weights "D:\\CodeBucket\\Yolov5_DeepSort_Pytorch\\yolov5\\weights\\crowdhuman_yolov5m.pt" 

'''
09/10/2021
python track.py --source "d:\\myData\\Bus\\20210824_180000_ch5_test.mp4" --show-vid --conf-thres 0.1 --iou-thres 0.1 --weights "D:\\CodeBucket\\Yolov5_DeepSort_Pytorch\\yolov5\\weights\\yolov5l.pt" --img-size 320
DEEPSORT:
  REID_CKPT: "deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7"
  MAX_DIST: 0.2
  MIN_CONFIDENCE: 0.3
  NMS_MAX_OVERLAP: 0.2
  MAX_IOU_DISTANCE: 0.5
  MAX_AGE: 10
  N_INIT: 3
  NN_BUDGET: 100

python track.py --source "d:\\myData\\Bus\\20210824_180000_ch3_test.mp4" --show-vid --conf-thres 0.05 --iou-thres 0.2 --weights "D:\\CodeBucket\\Yolov5_DeepSort_Pytorch\\yolov5\\weights\\yolov5l.pt" --img-size 256 --rot90cc
DEEPSORT:
  REID_CKPT: "deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7"
  MAX_DIST: 0.2
  MIN_CONFIDENCE: 0.05
  NMS_MAX_OVERLAP: 0.5
  MAX_IOU_DISTANCE: 0.7
  MAX_AGE: 10
  N_INIT: 2
  NN_BUDGET: 100  
'''
'''
09/15/2021
python track.py --source "d:\\myData\\Bus\\20210824_180000_ch5_test.mp4" --vid-channel 'ch5' --show-vid --conf-thres 0.3 --iou-thres 0.6 --step-frames 1
python track.py --source "d:\\myData\\Bus\\20210830_080000_ch5_test.mp4" --vid-channel 'ch5' --show-vid --conf-thres 0.3 --iou-thres 0.6 --step-frames 1
DEEPSORT:
  REID_CKPT: "deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7"
  MAX_DIST: 0.5
  MIN_CONFIDENCE: 0.3
  NMS_MAX_OVERLAP: 0.3
  MAX_IOU_DISTANCE: 0.6
  MAX_AGE: 10
  N_INIT: 3
  NN_BUDGET: 100
'''  
'''  
09/23/2021
python track.py --source "d:\\myData\\Bus\\20210824_180000_ch5_test.mp4" --vid-channel 'ch5' --conf-thres 0.05 --iou-thres 0.4 --step-frames 2 --show-vid --img-size 320
python track.py --source "d:\\myData\\Bus\\20210830_080000_ch5_test.mp4" --vid-channel 'ch5' --conf-thres 0.05 --iou-thres 0.4 --step-frames 1 --show-vid --img-size 320
DEEPSORT:
  REID_CKPT: "deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7"
  MAX_DIST: 0.4
  MIN_CONFIDENCE: 0.05
  NMS_MAX_OVERLAP: 0.5
  MAX_IOU_DISTANCE: 0.7
  MAX_AGE: 10
  N_INIT: 2
  NN_BUDGET: 100
'''
'''
09/25/2021
python track.py --source "d:\\myData\\Bus\\20210824_180000_ch3_test.mp4" --vid-channel 'ch3' --conf-thres 0.05 --iou-thres 0.4 --step-frames 2 --show-vid --img-size 320 
DEEPSORT:
  REID_CKPT: "deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7"
  MAX_DIST: 0.4
  MIN_CONFIDENCE: 0.05
  NMS_MAX_OVERLAP: 0.5
  MAX_IOU_DISTANCE: 0.7
  MAX_AGE: 10
  N_INIT: 1
  NN_BUDGET: 100
'''  
'''  
10/13/2021
python track.py --source "d:\\myData\\Bus\\20210830_080000_ch3_test.mp4" --vid-channel 'ch3' --conf-thres 0.2 --iou-thres 0.2 --step-frames 1 --show-vid --img-size 320 
DEEPSORT:
  REID_CKPT: "deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7"
  MAX_DIST: 0.2
  MIN_CONFIDENCE: 0.2
  NMS_MAX_OVERLAP: 0.5
  MAX_IOU_DISTANCE: 0.7
  MAX_AGE: 10
  N_INIT: 2
  NN_BUDGET: 100
'''  