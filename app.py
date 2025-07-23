

# SECTION 1: INSTALLATION & IMPORTS
print("Step 1: Forcibly upgrading torch and torchvision for compatibility...")
!pip install --upgrade -q torch torchvision

print("\nStep 2: Installing assignment dependencies...")
!pip install -q ultralytics
!pip install -q 'git+https://github.com/facebookresearch/segment-anything.git'
!pip install -q supervision
print("✅ Dependencies installed.")

import os
import sys
import json
import cv2
import torch
import numpy as np
import supervision as sv
import shutil
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from IPython.display import display, JSON, HTML
from base64 import b64encode

# SECTION 2: SETUP FOR KAGGLE ENVIRONMENT


KAGGLE_INPUT_DIR = "/kaggle/input/driving-video-with-object-tracking/"


print("\n--- Searching for video file in Kaggle dataset ---")
VIDEO_PATH = None
for root, dirs, files in os.walk(KAGGLE_INPUT_DIR):
    for file in files:
        if file.endswith(('.mp4', '.mov')):
            VIDEO_PATH = os.path.join(root, file)
            break
    if VIDEO_PATH:
        break

assert VIDEO_PATH is not None, f"🛑 ERROR: Could not find any .mp4 or .mov file inside {KAGGLE_INPUT_DIR} or its subdirectories."
VIDEO_FILENAME = os.path.basename(VIDEO_PATH)
print(f"✅ Video file found automatically at: {VIDEO_PATH}")


print("\n--- Model Loading ---")
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

yolo_model = YOLO('yolov8n.pt')
print("✅ YOLOv8 model loaded.")

SAM_CHECKPOINT_DIR = "/kaggle/working/sam_model"
os.makedirs(SAM_CHECKPOINT_DIR, exist_ok=True)
SAM_CHECKPOINT_PATH = os.path.join(SAM_CHECKPOINT_DIR, "sam_vit_h_4b8939.pth")
SAM_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
if not os.path.exists(SAM_CHECKPOINT_PATH):
    !wget -q -O {SAM_CHECKPOINT_PATH} {SAM_CHECKPOINT_URL}
print("✅ SAM models loaded to device.")
MODEL_TYPE = "vit_h"
sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
sam_predictor = SamPredictor(sam)

# SECTION 3: HELPER FUNCTION
def generate_mask_and_polygon(image, bbox_xyxy):
    sam_predictor.set_image(image)
    masks, _, _ = sam_predictor.predict(box=bbox_xyxy, multimask_output=False)
    polygons = sv.mask_to_polygons(masks[0])
    return masks[0], polygons[0] if len(polygons) > 0 else np.array([])

# SECTION 4: MAIN PROCESSING PIPELINE (Frames-to-Video Method)

print("\n🚀 Starting the fully automated annotation pipeline...")

OUTPUT_VIDEO_PATH = "/kaggle/working/output_annotated_video.mp4"
OUTPUT_JSON_PATH = "/kaggle/working/output_polygons.json"

CONFIDENCE_THRESHOLD = 0.25
PROCESSING_STRIDE = 5

OUTPUT_FRAMES_DIR = "/kaggle/working/output_frames"
if os.path.exists(OUTPUT_FRAMES_DIR):
    shutil.rmtree(OUTPUT_FRAMES_DIR)
os.makedirs(OUTPUT_FRAMES_DIR)

TARGET_CLASSES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck', 0: 'person'}
TARGET_CLASS_IDS = list(TARGET_CLASSES.keys())

video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)
frame_generator = sv.get_video_frames_generator(VIDEO_PATH, stride=PROCESSING_STRIDE)

box_annotator = sv.BoxAnnotator(thickness=2)
mask_annotator = sv.MaskAnnotator(opacity=0.5)
label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_CENTER)

final_annotations = {"video_name": VIDEO_FILENAME, "annotations": {}}
total_frames = video_info.total_frames // PROCESSING_STRIDE
for frame_idx, frame in enumerate(frame_generator):
    if frame_idx % 10 == 0:
        print(f"  > Processing and saving frame {frame_idx + 1}/{total_frames}...")

    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    yolo_results = yolo_model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(yolo_results)
    mask = np.array([(cid in TARGET_CLASS_IDS) and (conf > CONFIDENCE_THRESHOLD) for cid, conf in zip(detections.class_id, detections.confidence)])
    detections = detections[mask]

    annotated_frame = frame.copy()
    polygons_for_json = []

    if len(detections) > 0:
        detections.mask = np.array([generate_mask_and_polygon(image=frame, bbox_xyxy=bbox)[0] for bbox in detections.xyxy])
        for i in range(len(detections)):
            polygons = sv.mask_to_polygons(detections.mask[i])
            if polygons:
                class_name = TARGET_CLASSES.get(detections.class_id[i], 'unknown').replace('person', 'pedestrian')
                polygons_for_json.append({"object_index": i, "class": class_name, "polygon": polygons[0].tolist()})

        labels = [f"{TARGET_CLASSES.get(cid, '?')} {conf:0.2f}" for cid, conf in zip(detections.class_id, detections.confidence)]
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

    final_annotations["annotations"][f"frame_{frame_idx*PROCESSING_STRIDE}"] = polygons_for_json
    frame_filename = os.path.join(OUTPUT_FRAMES_DIR, f"frame_{frame_idx:05d}.png")
    cv2.imwrite(frame_filename, annotated_frame)

print("\n✅ Frame processing complete. Stitching frames into final video...")
output_fps = video_info.fps / PROCESSING_STRIDE
!ffmpeg -y -loglevel quiet -framerate {output_fps} -i {OUTPUT_FRAMES_DIR}/frame_%05d.png -c:v libx264 -pix_fmt yuv420p {OUTPUT_VIDEO_PATH}

print(f"\n✅ Processing complete. Annotated video saved to: {OUTPUT_VIDEO_PATH}")
with open(OUTPUT_JSON_PATH, 'w') as f: json.dump(final_annotations, f, indent=2)
print(f"✅ Polygon annotations saved to: {OUTPUT_JSON_PATH}")

shutil.rmtree(OUTPUT_FRAMES_DIR)
print("✅ Temporary frames directory cleaned up.")

# SECTION 5: DISPLAY RESULTS

print("\n--- Sample of output_polygons.json ---")
first_annotated_frame_key = next((k for k, v in final_annotations["annotations"].items() if v), None)
if first_annotated_frame_key:
    display(JSON({"video_name": final_annotations["video_name"],
                  first_annotated_frame_key: final_annotations["annotations"][first_annotated_frame_key]}))
else:
    print("No objects were detected in the processed frames.")

print("\n--- Generated Annotated Video ---")
def show_video(video_path, width=800):
    video_file = open(video_path, "rb").read()
    video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"
    return HTML(f"""<video width={width} controls autoplay loop><source src="{video_url}"></video>""")

display(show_video(OUTPUT_VIDEO_PATH))
