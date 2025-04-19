import torch
import cv2
import numpy as np
from pytorch_face_landmark.common.utils import BBox
from pytorch_face_landmark.models.mobilefacenet import MobileFaceNet
from pytorch_face_landmark.Retinaface import Retinaface
from pytorch_face_landmark.utils.align_trans import get_reference_facial_points
from tqdm import tqdm

mean = np.asarray([ 0.485, 0.456, 0.406 ])
std = np.asarray([ 0.229, 0.224, 0.225 ])

crop_size= 112
scale = crop_size / 112.
reference = get_reference_facial_points(default_square = True) * scale

if torch.cuda.is_available():
    map_location = "cuda:0"
elif torch.backends.mps.is_available():
    map_location = "mps"
else:
    map_location = "cpu"

model = MobileFaceNet([112, 112],136)
checkpoint = torch.load('pytorch_face_landmark/checkpoint/mobilefacenet_model_best.pth.tar', map_location=map_location)
print('Use MobileFaceNet as backbone')
model.load_state_dict(checkpoint['state_dict'])
model = model.eval().to(map_location)

# Initialize RetinaFace detector once
retinaface = Retinaface.Retinaface()

def get_bbox(face, height, width):
    x1 = face[0]
    y1 = face[1]
    x2 = face[2]
    y2 = face[3]
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    size = int(min([w, h]) * 1.2)
    cx = x1 + w // 2
    cy = y1 + h // 2
    x1 = cx - size // 2
    x2 = x1 + size
    y1 = cy - size // 2
    y2 = y1 + size

    dx = max(0, -x1)
    dy = max(0, -y1)
    x1 = max(0, x1)
    y1 = max(0, y1)

    edx = max(0, x2 - width)
    edy = max(0, y2 - height)
    x2 = min(width, x2)
    y2 = min(height, y2)
    new_bbox = list(map(int, [x1, x2, y1, y2]))
    return new_bbox, dx, dy, edx, edy


# New batch processing functions
def detect_faces_batch(frames, batch_size=8):
    """
    Detect faces in a batch of frames using RetinaFace
    """
    all_faces = []
    all_bboxes = []
    all_dx_dy_edx_edy = []
    
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i+batch_size]
        batch_faces = []
        batch_bboxes = []
        batch_dx_dy_edx_edy = []
        all_faces = retinaface.process(batch_frames)
        
        for i in range(len(batch_frames)):
            height, width, _ = batch_frames[i].shape
            faces = all_faces[i]
            #faces = retinaface(batch_frames[i])
            
            if len(faces) == 0:
                print(f"Warning: No face detected in frame")
                # Use a default face box if no face is detected
                faces = [[0, 0, width-1, height-1]]
            
            if len(faces) > 1:
                '''for face in faces:
                    cv2.rectangle(batch_frames[i], (int(face[0]), int(face[1])), (int(face[2]), int(face[3])), color=(255, 0, 0),
                                  thickness=2)
                print(faces)
                cv2.imshow('img', batch_frames[i])
                cv2.waitKey(0)'''
                print(f"Warning: Multiple faces detected in frame, using the first one")
            
            face = faces[0]
            new_bbox, dx, dy, edx, edy = get_bbox(face, height, width)
            
            batch_faces.append(face)
            batch_bboxes.append(BBox(new_bbox))
            batch_dx_dy_edx_edy.append((dx, dy, edx, edy))
        
        all_faces.extend(batch_faces)
        all_bboxes.extend(batch_bboxes)
        all_dx_dy_edx_edy.extend(batch_dx_dy_edx_edy)
    
    return all_faces, all_bboxes, all_dx_dy_edx_edy

def crop_and_resize_faces_batch(frames, bboxes, dx_dy_edx_edy_list, out_size=112):
    """
    Crop and resize faces from frames based on bounding boxes
    """
    cropped_faces = []
    #save_path = '/Users/xiaokeai/Documents/HKUST/projects/grace/ETFG/MEAD_features/M003/fear/level_2'
    
    for i, (frame, bbox, (dx, dy, edx, edy)) in enumerate(zip(frames, bboxes, dx_dy_edx_edy_list)):
        cropped = frame[bbox.top:bbox.bottom, bbox.left:bbox.right]
        
        if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
            cropped = cv2.copyMakeBorder(cropped, int(dy), int(edy), int(dx), int(edx), cv2.BORDER_CONSTANT, 0)
        
        cropped_face = cv2.resize(cropped, (out_size, out_size))
        
        if cropped_face.shape[0] <= 0 or cropped_face.shape[1] <= 0:
            print(f'ERROR: cropped face is zero in frame {i}')
            # Create a blank image as fallback
            cropped_face = np.zeros((out_size, out_size, 3), dtype=np.uint8)
        
        # Save the cropped face
        #save_file = f'{save_path}/frame_{i:04d}.jpg'
        #cv2.imwrite(save_file, cropped_face)
        
        cropped_faces.append(cropped_face)
    
    return cropped_faces

def extract_features_batch(cropped_faces, bboxes, batch_size=32):
    """
    Extract facial features and landmarks from cropped faces in batches
    """
    all_landmarks = []
    all_facial_feats = []
    
    for i in range(0, len(cropped_faces), batch_size):
        batch_faces = cropped_faces[i:i+batch_size]
        batch_bboxes = bboxes[i:i+batch_size]
        
        batch_input = np.stack(batch_faces)
        batch_input = torch.from_numpy(batch_input).float().permute(0, 3, 1, 2)
        batch_input = batch_input / 255.0
        batch_input = torch.autograd.Variable(batch_input).to(map_location)
        
        # Process batch
        with torch.no_grad():
            landmarks, facial_feats = model(batch_input, returnFeatureAndLdmk=True)
        
        # Process results
        landmarks = landmarks.cpu().numpy()
        facial_feats = facial_feats.cpu().numpy()
        
        # Reproject landmarks
        for j, (landmark, bbox) in enumerate(zip(landmarks, batch_bboxes)):
            landmark = landmark.reshape(-1, 2)
            reprojected_landmark = bbox.reprojectLandmark(landmark)
            all_landmarks.append(reprojected_landmark)
            all_facial_feats.append(facial_feats[j])
    
    return all_landmarks, all_facial_feats

def extract_facial(video_path, detection_bs, feat_bs):
    ldmk_stack, face_embed_stack = process_video_frames(video_path, detection_bs, feat_bs)
    return ldmk_stack, face_embed_stack

def get_num_of_frames(video_path):
    num_of_frames = 0
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        num_of_frames += 1

    cap.release()

    if num_of_frames == 0:
        raise ValueError(f"No frames found in video: {video_path}")
    return num_of_frames

def process_video_frames(video_path, detection_bs=16, feat_bs=64):
    """
    Process video frames in batches for faster facial feature extraction
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    all_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    
    cap.release()
    
    if not all_frames:
        raise ValueError(f"No frames found in video: {video_path}")

    faces, bboxes, dx_dy_edx_edy = detect_faces_batch(all_frames, batch_size=detection_bs)
    cropped_faces = crop_and_resize_faces_batch(all_frames, bboxes, dx_dy_edx_edy)
    _, new_bboxes, _ = detect_faces_batch(cropped_faces, batch_size=detection_bs)
    landmarks, facial_feats = extract_features_batch(cropped_faces, new_bboxes, batch_size=feat_bs)
    
    # Stack results
    landmarks_stack = np.stack(landmarks)
    facial_feats_stack = np.stack(facial_feats)
    
    return landmarks_stack, facial_feats_stack


