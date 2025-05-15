import os
from django.conf import settings
from django.shortcuts import render, redirect, get_object_or_404
from uploading_data.models import UploadedFile
from django.contrib.auth.decorators import login_required

"""
from django.http import FileResponse
from PIL import Image
import cv2
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# === Ініціалізація Detectron2 ===
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.WEIGHTS = os.path.join(settings.BASE_DIR, 'model_final.pth')
cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

predictor = DefaultPredictor(cfg)

def segment_image(image_path):
    image = cv2.imread(image_path)
    outputs = predictor(image)

    masks = outputs["instances"].pred_masks.cpu().numpy()
    if len(masks) == 0:
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # повертаємо оригінал, якщо немає об'єктів

    combined_mask = np.any(masks, axis=0).astype(np.uint8) * 255
    mask_3ch = np.stack([combined_mask]*3, axis=-1)
    segmented = cv2.bitwise_and(image, mask_3ch)

    return Image.fromarray(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))

# === Функція сегментації відео ===
def segment_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        outputs = predictor(frame)
        masks = outputs["instances"].pred_masks.cpu().numpy()

        if len(masks) > 0:
            combined_mask = np.any(masks, axis=0).astype(np.uint8) * 255
            mask_3ch = np.stack([combined_mask]*3, axis=-1)
            segmented = cv2.bitwise_and(frame, mask_3ch)
        else:
            segmented = np.zeros_like(frame)

        out.write(segmented)

    cap.release()
    out.release()"""


@login_required
def select_file(request):
    files = UploadedFile.objects.filter(user=request.user)
    return render(request, 'model/select_file.html', {'files': files})


@login_required
def segment_file(request, file_id):
    file = get_object_or_404(UploadedFile, id=file_id, user=request.user)
    file_path = file.file.path
    ext = os.path.splitext(file_path)[1].lower()

    output_name = f"segmented_{os.path.basename(file_path)}"
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
