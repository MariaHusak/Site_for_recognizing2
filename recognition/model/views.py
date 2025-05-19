import os
import torch
import tempfile
from django.conf import settings
from django.shortcuts import render, redirect, get_object_or_404
from uploading_data.models import UploadedFile
from django.contrib.auth.decorators import login_required
from django.http import FileResponse
from PIL import Image
import cv2
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.WEIGHTS = os.path.join(settings.BASE_DIR, 'model_final.pth')
cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
predictor = None


def get_predictor():
    global predictor
    if predictor is None:
        print(f"Initializing Detectron2 predictor on device: {cfg.MODEL.DEVICE}")
        predictor = DefaultPredictor(cfg)
    return predictor


def segment_image(image_path):
    image = cv2.imread(image_path)
    outputs = get_predictor()(image)

    masks = outputs["instances"].pred_masks.cpu().numpy()
    if len(masks) == 0:
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
    scores = outputs["instances"].scores.cpu().numpy()
    result = image.copy()

    colors = [
        (0, 255, 0),
        (255, 0, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
    ]
    for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
        color = colors[i % len(colors)]

        colored_mask = np.zeros_like(image, dtype=np.uint8)
        colored_mask[mask] = color
        alpha = 0.5
        result = cv2.addWeighted(result, 1, colored_mask, alpha, 0)

        x1, y1, x2, y2 = map(int, box)
        thickness = 3
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

        confidence = int(score * 100)
        text = f"person {confidence}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        text_thickness = 2

        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)

        text_x = x1
        text_y = y1 - 10
        if text_y - text_height < 0:
            text_y = y2 + text_height + 10

        cv2.rectangle(result,
                      (text_x - 5, text_y - text_height - baseline - 5),
                      (text_x + text_width + 5, text_y + baseline + 5),
                      color, -1)

        cv2.putText(result, text, (text_x, text_y), font, font_scale, (255, 255, 255), text_thickness)

    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(result_rgb)

    return pil_image


def segment_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    pred = get_predictor()

    colors = [
        (0, 255, 0),
        (255, 0, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
    ]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        outputs = pred(frame)
        masks = outputs["instances"].pred_masks.cpu().numpy()

        if len(masks) > 0:
            boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
            scores = outputs["instances"].scores.cpu().numpy()

            result = frame.copy()

            for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
                color = colors[i % len(colors)]

                colored_mask = np.zeros_like(frame, dtype=np.uint8)
                colored_mask[mask] = color

                alpha = 0.5
                result = cv2.addWeighted(result, 1, colored_mask, alpha, 0)

                x1, y1, x2, y2 = map(int, box)
                thickness = 3
                cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

                confidence = int(score * 100)
                text = f"person {confidence}%"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                text_thickness = 2

                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)

                text_x = x1
                text_y = y1 - 10
                if text_y - text_height < 0:
                    text_y = y2 + text_height + 10

                cv2.rectangle(result,
                              (text_x - 5, text_y - text_height - baseline - 5),
                              (text_x + text_width + 5, text_y + baseline + 5),
                              color, -1)

                cv2.putText(result, text, (text_x, text_y), font, font_scale, (255, 255, 255), text_thickness)
        else:
            result = frame

        out.write(result)

    cap.release()
    out.release()


@login_required
def select_file(request):
    files = UploadedFile.objects.filter(user=request.user)
    return render(request, 'model/select_file.html', {'files': files})


@login_required
def segment_file(request, file_id):
    file = get_object_or_404(UploadedFile, id=file_id, user=request.user)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'__{file.filename}')
    temp_file.write(file.file_data)
    temp_file.close()

    try:
        file_path = temp_file.name
        ext = os.path.splitext(file.filename)[1].lower() if file.filename else ''

        output_name = f"segmented_{file.filename}" if file.filename else f"segmented_file_{file_id}"
        output_path = os.path.join(settings.MEDIA_ROOT, 'segmented', output_name)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if ext in ['.jpg', '.jpeg', '.png']:
            result = segment_image(file_path)
            result.save(output_path)

            return render(request, 'model/result.html', {
                'result_url': os.path.join(settings.MEDIA_URL, 'segmented', output_name)
            })

        elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
            segment_video(file_path, output_path)
            return render(request, 'model/result_video.html', {
                'result_url': os.path.join(settings.MEDIA_URL, 'segmented', output_name)
            })

        else:
            return render(request, 'model/unsupported.html')

    finally:
        os.unlink(temp_file.name)