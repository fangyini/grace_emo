from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from pytorch_face_landmark.Retinaface.data import cfg_mnet, cfg_re50
from pytorch_face_landmark.Retinaface.layers.functions.prior_box import PriorBox
from pytorch_face_landmark.Retinaface.utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from pytorch_face_landmark.Retinaface.models.retinaface import RetinaFace
from pytorch_face_landmark.Retinaface.utils.box_utils import decode, decode_landm
import time


# some global configs
trained_model='pytorch_face_landmark/Retinaface/weights/mobilenet0.25_Final.pth'
network='mobile0.25'
confidence_threshold = 0.9
top_k = 5000
keep_top_k = 750
nms_threshold = 0.3
vis_thres = 0.5
resize = 1
cpu=True
if torch.cuda.is_available():
    device = "cuda:0"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    #print('Missing keys:{}'.format(len(missing_keys)))
    #print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    #print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    #print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path):
    #print('Loading pretrained model from {}'.format(pretrained_path))
    pretrained_dict = torch.load(pretrained_path, map_location=device)
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


class Retinaface:
    def __init__(self, timer_flag=False):
        torch.set_grad_enabled(False)
        '''
        if network == "mobile0.25":
            cfg = cfg_mnet
        elif network == "resnet50":
            cfg = cfg_re50
        '''
        self.cfg = cfg_mnet    
        # net and model
        net = RetinaFace(cfg=self.cfg, phase = 'test')
        self.net = load_model(net, trained_model)
        self.net.eval().to(device)
        #print('Finished loading model!')
        #print(net)
        #cudnn.benchmark = True
        self.timer_flag = timer_flag
        self.device = device

    def process(self, frames):
        frames, scale, im_shape = self.preprocess_images(frames)
        all_faces = self.process_batch(frames, scale, im_shape)
        return all_faces

    def preprocess_images(self, frames):
        # Convert frames to float32 and subtract mean like in __call__
        frames = [np.float32(frame) - (104, 117, 123) for frame in frames]
        frames = np.stack(frames)
        
        # Transpose like in __call__ (HWC to CHW)
        frames = frames.transpose(0, 3, 1, 2)
        
        # Convert to torch tensor and move to device
        frames = torch.from_numpy(frames).float().to(self.device)
        
        # Get scale like in __call__
        scale = torch.Tensor([frames[0].shape[2], frames[0].shape[1], 
                            frames[0].shape[2], frames[0].shape[1]]).to(self.device)
        im_shape = (frames[0].shape[1], frames[0].shape[2])
        
        return frames, scale, im_shape

    def process_batch(self, frames, scale, im_shape):
        locs, confs, landmss = self.net(frames)
        all_faces = []
        for i in range(locs.size(0)):
            loc = locs[i]
            conf = confs[i]
            landms = landmss[i]
            priorbox = PriorBox(self.cfg, image_size=im_shape)
            priors = priorbox.forward()
            priors = priors.to(self.device)
            prior_data = priors.data
            boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
            boxes = boxes * scale / resize
            boxes = boxes.cpu().numpy()
            scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
            landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
            scale1 = torch.Tensor([im_shape[0], im_shape[1], im_shape[0], im_shape[1],
                                   im_shape[0], im_shape[1], im_shape[0], im_shape[1],
                                   im_shape[0], im_shape[1]])
            scale1 = scale1.to(self.device)
            landms = landms * scale1 / resize
            landms = landms.cpu().numpy()

            # ignore low scores
            inds = np.where(scores > confidence_threshold)[0]
            boxes = boxes[inds]
            landms = landms[inds]
            scores = scores[inds]

            # keep top-K before NMS
            order = scores.argsort()[::-1][:top_k]
            boxes = boxes[order]
            landms = landms[order]
            scores = scores[order]

            # do NMS
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = py_cpu_nms(dets, nms_threshold)
            dets = dets[keep, :]
            landms = landms[keep]

            # keep top-K faster NMS
            dets = dets[:keep_top_k, :]
            det_bboxes = []
            for b in dets:
                if b[4] > vis_thres:
                    xmin, ymin, xmax, ymax, score = b[0], b[1], b[2], b[3], b[4]
                    bbox = [xmin, ymin, xmax, ymax, score]
                    det_bboxes.append(bbox)
            all_faces.append(det_bboxes)
        return all_faces


    def __call__(self, img_):
        img_raw = img_.copy()

        img = np.float32(img_raw)

        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)

        tic = time.time()
        loc, conf, landms = self.net(img)  # forward pass
        #print('net forward time: {:.4f}'.format(time.time() - tic))

        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(self.device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:keep_top_k, :]
        #landms = landms[:args.keep_top_k, :]

        #dets = np.concatenate((dets, landms), axis=1)
        
        if self.timer_flag:
            print('Detection: {:d}/{:d} forward_pass_time: {:.4f}s misc: {:.4f}s'.format(1, 1, _t[
                'forward_pass'].average_time, _t['misc'].average_time))

        # filter using vis_thres
        det_bboxes = []
        for b in dets:
            if b[4] > vis_thres:
                xmin, ymin, xmax, ymax, score = b[0], b[1], b[2], b[3], b[4]
                w = xmax - xmin + 1
                h = ymax - ymin + 1
                bbox = [xmin, ymin, xmax, ymax, score]
                det_bboxes.append(bbox)

        return det_bboxes


