import os
import torch
import tempfile
import subprocess
import cv2
import numpy as np
from PIL import Image
from django.conf import settings
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo


class Model:
    def __init__(self):
        self.cfg = None
        self.predictor = None
        self._initialize_config()

    def _initialize_config(self):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        self.cfg.MODEL.WEIGHTS = os.path.join(settings.BASE_DIR, 'model_final.pth')
        self.cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    def get_predictor(self):
        if self.predictor is None:
            self.predictor = DefaultPredictor(self.cfg)
        return self.predictor


class Photo:
    def __init__(self, model_instance):
        self.model = model_instance
        self.colors = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
        ]

    def segment_image(self, image_path):
        image = cv2.imread(image_path)
        outputs = self.model.get_predictor()(image)

        masks = outputs["instances"].pred_masks.cpu().numpy()
        if len(masks) == 0:
            return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
        scores = outputs["instances"].scores.cpu().numpy()
        result = image.copy()

        for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
            result = self._apply_mask_and_box(result, mask, box, score, i)

        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result_rgb)

    def _apply_mask_and_box(self, image, mask, box, score, index):
        color = self.colors[index % len(self.colors)]

        colored_mask = np.zeros_like(image, dtype=np.uint8)
        colored_mask[mask] = color
        alpha = 0.5
        result = cv2.addWeighted(image, 1, colored_mask, alpha, 0)

        x1, y1, x2, y2 = map(int, box)
        thickness = 3
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

        confidence = int(score * 100)
        text = f"person {confidence}%"
        result = self._add_text_to_image(result, text, x1, y1, y2, color)

        return result

    def _add_text_to_image(self, image, text, x1, y1, y2, color):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        text_thickness = 2

        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)

        text_x = x1
        text_y = y1 - 10
        if text_y - text_height < 0:
            text_y = y2 + text_height + 10

        cv2.rectangle(image,
                      (text_x - 5, text_y - text_height - baseline - 5),
                      (text_x + text_width + 5, text_y + baseline + 5),
                      color, -1)

        cv2.putText(image, text, (text_x, text_y), font, font_scale, (255, 255, 255), text_thickness)

        return image


class Video:
    def __init__(self, model_instance):
        self.model = model_instance
        self.colors = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
        ]

    def segment_video(self, video_path, output_path):
        if not output_path.endswith('.mp4'):
            output_path = os.path.splitext(output_path)[0] + '.mp4'

        temp_dir = tempfile.mkdtemp(prefix="video_processing_")
        frames_dir = os.path.join(temp_dir, "frames")

        try:
            if self._check_ffmpeg():
                return self._process_with_ffmpeg(video_path, output_path, frames_dir)
            else:
                return self._process_with_opencv(video_path, output_path, temp_dir)
        except Exception as e:
            return None
        finally:
            self._cleanup_temp_dir(temp_dir)

    def _check_ffmpeg(self):
        try:
            subprocess.run(['ffmpeg', '-version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def _process_with_ffmpeg(self, video_path, output_path, frames_dir):
        video_info = self._segment_frames(video_path, frames_dir)

        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-framerate', str(video_info['fps']),
            '-i', os.path.join(frames_dir, 'frame_%06d.jpg'),
            '-c:v', 'libx264',
            '-profile:v', 'high',
            '-pix_fmt', 'yuv420p',
            '-preset', 'medium',
            '-crf', '23',
            '-movflags', '+faststart',
            output_path
        ]

        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path
        else:
            return None

    def _process_with_opencv(self, video_path, output_path, temp_dir):
        temp_output = os.path.join(temp_dir, "output_opencv.avi")

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

        if not out.isOpened():
            return None

        pred = self.model.get_predictor()
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            processed_frame = self._process_frame(frame, pred)
            out.write(processed_frame)

        cap.release()
        out.release()

        with open(temp_output, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                f_out.write(f_in.read())

        return output_path

    def _segment_frames(self, input_path, frames_dir):
        os.makedirs(frames_dir, exist_ok=True)

        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        pred = self.model.get_predictor()
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            result = self._process_frame(frame, pred)
            cv2.imwrite(os.path.join(frames_dir, f"frame_{frame_count:06d}.jpg"), result)

        cap.release()

        return {
            'frame_count': frame_count,
            'fps': fps,
            'width': width,
            'height': height,
        }

    def _process_frame(self, frame, predictor):
        outputs = predictor(frame)
        masks = outputs["instances"].pred_masks.cpu().numpy()

        if len(masks) == 0:
            return frame

        boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
        scores = outputs["instances"].scores.cpu().numpy()
        result = frame.copy()

        for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
            color = self.colors[i % len(self.colors)]

            colored_mask = np.zeros_like(frame, dtype=np.uint8)
            colored_mask[mask] = color

            alpha = 0.5
            result = cv2.addWeighted(result, 1, colored_mask, alpha, 0)

            x1, y1, x2, y2 = map(int, box)
            thickness = 3
            cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

            confidence = int(score * 100)
            text = f"person {confidence}%"
            result = self._add_text_to_frame(result, text, x1, y1, y2, color)

        return result

    def _add_text_to_frame(self, frame, text, x1, y1, y2, color):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        text_thickness = 2

        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)

        text_x = x1
        text_y = y1 - 10
        if text_y - text_height < 0:
            text_y = y2 + text_height + 10

        cv2.rectangle(frame,
                      (text_x - 5, text_y - text_height - baseline - 5),
                      (text_x + text_width + 5, text_y + baseline + 5),
                      color, -1)

        cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), text_thickness)

        return frame

    def _cleanup_temp_dir(self, temp_dir):
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except SystemExit:
            pass


class RecognizeFacade:
    def __init__(self):
        self._model_initial = Model()
        self._video_seg = Video(self._model_initial)
        self._photo_seg = Photo(self._model_initial)

    def process_image(self, image_path):
        return self._photo_seg.segment_image(image_path)

    def process_video(self, video_path, output_path):
        return self._video_seg.segment_video(video_path, output_path)
