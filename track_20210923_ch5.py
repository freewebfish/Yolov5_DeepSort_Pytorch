import sys
sys.path.insert(0, './yolov5')

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

def xyxy_to_xywh(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

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
    out, source, weights, show_vid, save_vid, save_txt, imgsz, vid_channel, show_roi, show_mask, show_det, step_frames= \
        opt.output, opt.source, opt.weights, opt.show_vid, opt.save_vid, opt.save_txt, opt.img_size, \
        opt.vid_channel, opt.show_roi, opt.show_mask, opt.show_det, opt.step_frames
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
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
    txt_path = str(Path(out)) + '/results.txt'

    #ZONE_CH5 = [[200, 576], [200, 450], [300, 450], [300, 300], [715, 520], [715, 576], [550, 576]] #for outside the door
    #ZONE_CH5_OUT = [[300, 450], [300, 300], [715, 520], [715, 576], [550, 576]] #for outside the door
    #roi = np.array(ZONE_CH5).reshape((-1,1,2)).astype(np.int32)     
    #roi_out = np.array(ZONE_CH5_OUT).reshape((-1,1,2)).astype(np.int32)     
    ZONE_CH5_DETECT = [[300, 576], [300, 380], [380, 390], [550, 570], [550, 576]] #for inside the door
    ZONE_CH5_SILENT = [[250, 576], [250, 400], [360, 400], [460, 576]] #for dead zone
    roi_detect = np.array(ZONE_CH5_DETECT).reshape((-1,1,2)).astype(np.int32)     
    roi_silent = np.array(ZONE_CH5_SILENT).reshape((-1,1,2)).astype(np.int32)     
    ppl_count, ppl_in, ppl_out = 0, 0, 0
    PHI0 = 30
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

                    '''
                    # Show detection boxes
                    if show_vid:  # Add bbox to image, before feeding into deepsort tracker
                        for *xyxy, conf, cls in reversed(det):
                            cc = int(cls)  # integer class
                            label = f'{names[cc]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors(cc, True), line_thickness=2)
                    '''

                    # Adapt detections to deep sort input format
                    xywh_bboxs = []
                    confs = []
                    obj_cls = []
                    for *xyxy, conf, cls in det:
                        # to deep sort format
                        x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
                        xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                        xywh_bboxs.append(xywh_obj)
                        confs.append([conf.item()])
                        obj_cls.append([cls.item()])

                    # Filter detections before feeding into deepsort tracker
                    outputs = []
                    if len(xywh_bboxs) > 0 and len(confs) > 0 and len(obj_cls) > 0:
                        # take out irrelevant detections
                        idx_remove = []
                        for idx in range(len(xywh_bboxs)):
                            x_c, y_c = xywh_bboxs[idx][0], xywh_bboxs[idx][1]
                            bbox_w, bbox_h = xywh_bboxs[idx][2], xywh_bboxs[idx][3]
                            box_diagonal = np.sqrt(bbox_h*bbox_h+bbox_w*bbox_w)                                                   
                            pta = [int(x_c - bbox_w / 2), int(y_c + bbox_h / 2)]
                            ptb = [int(x_c - bbox_w / 2), int(y_c - bbox_h / 2)]                            
                            ptc = [int(x_c + bbox_w / 2), int(y_c - bbox_h / 2)]                                                        
                            ptd = [int(x_c + bbox_w / 2), int(y_c + bbox_h / 2)]                            
                            ptad0=[int(x_c), int(y_c + bbox_h / 2)]
                            dist_a2detect = cv2.pointPolygonTest(roi_detect, pta, True)
                            dist_b2detect = cv2.pointPolygonTest(roi_detect, ptb, True)
                            dist_c2detect = cv2.pointPolygonTest(roi_detect, ptc, True)                            
                            dist_d2detect = cv2.pointPolygonTest(roi_detect, ptd, True)                                                    
                            dist_ad02detect = cv2.pointPolygonTest(roi_detect, ptad0, True)
                            if ((dist_ad02detect >= 0 and dist_b2detect < 0 and dist_b2detect > -300 and dist_c2detect < 0 and dist_c2detect > -300 ) and\
                                (box_diagonal >= 120 and box_diagonal <= 600)):
                                if show_vid and show_det:
                                    cc = obj_cls[idx][0] 
                                    plot_one_box([ptb[0], ptb[1], ptd[0], ptd[1]], im0, \
                                    label=f'{names[int(cc)]} {conf:.2f}', color=colors(cc, True), line_thickness=2)
                            else:
                                idx_remove.append(idx)
                        xywh_bboxs = np.delete(xywh_bboxs, idx_remove, axis=0)
                        confs = np.delete(confs, idx_remove, axis=0)
                        obj_cls = np.delete(obj_cls, idx_remove, axis=0)                        
                       
                        # pass detections to deepsort
                        xywhs = torch.Tensor(xywh_bboxs)
                        confss = torch.Tensor(confs)
                        outputs = deepsort.update(xywhs, confss, im0)

                    # Count passengers
                    if len(outputs) > 0:
                        bbox_xyxy = outputs[:, :4]
                        identities = outputs[:, -1]
                    else:
                        bbox_xyxy = []
                        identities = []
                    
                    ids_emerging = Diff(identities, identities_prev)
                    if (len(ids_emerging) > 0):
                        for idx in range(len(ids_emerging)):
                            key = int(ids_emerging[idx]) 
                            [x1, y1, x2, y2] = bbox_xyxy[identities==key][0]
                            if key not in list(passenger_in): passenger_in[key] = deque(maxlen=1)
                            passenger_in[key]=(frame_idx, int((x1+x2)/2), int((y1+y2)/2), int(abs(x2-x1)), int(abs(y2-y1)))

                    ids_gone = Diff(identities_prev, identities)
                    if (len(ids_gone) > 0):
                        for idx in range(len(ids_gone)):
                            key = int(ids_gone[idx])
                            [x1, y1, x2, y2] = bbox_xyxy_prev[identities_prev==key][0]
                            if key not in list(passenger_out): passenger_out[key] = deque(maxlen=1)
                            passenger_out[key]=(frame_idx, int((x1+x2)/2), int((y1+y2)/2), int(abs(x2-x1)), int(abs(y2-y1)))
                            if key in list(passenger_in) and key in list(passenger_out):
                                (n1, xn1, yn1, wn1, hn1) = passenger_in[key]
                                (n2, xn2, yn2, wn2, hn2) = passenger_out[key]
                                passenger_in.pop(key)
                                passenger_out.pop(key)
                                ds = [xn2-xn1, yn2-yn1]
                                displacement=np.linalg.norm(ds, ord=1)
                                elapsed_frames = n2 - n1
                                ifound = False
                                if elapsed_frames >= 3 and displacement >= 30:
                                    dv=[ds[0]/displacement, ds[1]/displacement]                                
                                    dphi = int(math.acos(np.dot(dv, XV))/math.pi*180) 
                                    dist_n12silent = cv2.pointPolygonTest(roi_silent, [int(xn1), int(yn1)], True)
                                    dist_n22silent = cv2.pointPolygonTest(roi_silent, [int(xn2), int(yn2)], True)
                                    dist_n12detect = cv2.pointPolygonTest(roi_detect, [int(xn1), int(yn1)], True)
                                    dist_n22detect = cv2.pointPolygonTest(roi_detect, [int(xn2), int(yn2)], True)
                                    if dist_n12detect < 0 and dist_n22detect < 0:  #start and end points are outside the detect zone
                                        if dphi > PHI0: ppl_in += 1 
                                        else: ppl_out += 1  
                                        ppl_count += 1  
                                        if ppl_count not in list(passenger_deque): passenger_deque[ppl_count] = deque(maxlen=1)
                                        passenger_deque[ppl_count]=(int(xn2), int(yn2), int(xn1), int(yn1))
                                        ifound = True
                                    elif dist_n12detect < 0 and dist_n22detect >= 0:  #start point outside and end point inside the detect zone
                                        ppl_in += 1 
                                        ppl_count += 1                                          
                                        if ppl_count not in list(passenger_deque): passenger_deque[ppl_count] = deque(maxlen=1)
                                        passenger_deque[ppl_count]=(int(xn2), int(yn2), int(xn1), int(yn1))
                                        ifound = True                                        
                                    elif dist_n12detect >= 0 and dist_n22detect < 0:  #start point inside and end point outside the detect zone                                           
                                        ppl_out += 1  
                                        ppl_count += 1
                                        if ppl_count not in list(passenger_deque): passenger_deque[ppl_count] = deque(maxlen=1)
                                        passenger_deque[ppl_count]=(int(xn2), int(yn2), int(xn1), int(yn1))
                                        ifound = True                                        
                                    elif dist_n12detect >= 0 and dist_n22detect >= 0: #start and end point inside the detect zone
                                        if dist_n12silent >= 0 and dist_n22silent >= 0:
                                            if wn2 >= 1.33*hn2 or wn1 >= 1.33*hn1: # if both are in silent zone and box width too wide
                                                pass
                                            else: # if both are in silent zone and box width normal
                                                if dphi > PHI0: ppl_in += 1 
                                                else: ppl_out += 1  
                                                ppl_count += 1
                                                if ppl_count not in list(passenger_deque): passenger_deque[ppl_count] = deque(maxlen=1)
                                                passenger_deque[ppl_count]=(int(xn2), int(yn2), int(xn1), int(yn1))
                                                ifound = True                                                
                                        elif dist_n12silent >= 0 and dist_n22silent < 0:  
                                            ppl_out += 1  
                                            ppl_count += 1
                                            if ppl_count not in list(passenger_deque): passenger_deque[ppl_count] = deque(maxlen=1)
                                            passenger_deque[ppl_count]=(int(xn2), int(yn2), int(xn1), int(yn1))
                                            ifound = True                                            
                                        elif dist_n12silent < 0 and dist_n22silent >= 0:  
                                            ppl_in += 1 
                                            ppl_count += 1                                          
                                            if ppl_count not in list(passenger_deque): passenger_deque[ppl_count] = deque(maxlen=1)
                                            passenger_deque[ppl_count]=(int(xn2), int(yn2), int(xn1), int(yn1))
                                            ifound = True                                            
                                        else:
                                            if dphi > PHI0: ppl_in += 1 
                                            else: ppl_out += 1  
                                            ppl_count += 1
                                            if ppl_count not in list(passenger_deque): passenger_deque[ppl_count] = deque(maxlen=1)
                                            passenger_deque[ppl_count]=(int(xn2), int(yn2), int(xn1), int(yn1))
                                            ifound = True                                            

                    identities_prev = identities
                    bbox_xyxy_prev = bbox_xyxy 

                    if len(outputs) > 0:
                        im0 = draw_boxes(im0, bbox_xyxy, identities, trailslen=trailslen)
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
                                    f.write(('%g ' * 10 + '\n') % (frame_idx, identity, bbox_top, bbox_left, bbox_w, bbox_h, -1, -1, -1, -1))  # label format

                else:
                    deepsort.increment_ages()

                # Print time (inference + NMS)
                print('%sDone. (%.3fs)' % (s, t2 - t1))

                # Stream results
                if show_vid:
                    if show_roi: 
                        cv2.polylines(im0, [roi_detect], isClosed = True, color = (0, 0, 255), thickness = 2)                                                  
                        cv2.polylines(im0, [roi_silent], isClosed = True, color = (255, 0, 0), thickness = 2)                                                                          
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
    parser.add_argument('--weights', type=str, default='yolov5/weights/yolov5l.pt', help='model.pt path')
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
    # class 0 is person, 1 is bycicle, 2 is car, 32 is sports ball, ... 79 is oven
    parser.add_argument('--classes', nargs='+', default=0, type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
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