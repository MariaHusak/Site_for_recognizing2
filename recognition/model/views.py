import os
import torch
import tempfile
import subprocess
from django.conf import settings
from django.shortcuts import render, redirect, get_object_or_404
from uploading_data.models import UploadedFile
from django.contrib.auth.decorators import login_required
from django.http import FileResponse, HttpResponse
from PIL import Image
import cv2
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# Конфігурація Detectron2
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


def check_ffmpeg():
    """Перевіряє наявність FFmpeg в системі"""
    try:
        subprocess.run(['ffmpeg', '-version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def has_ffmpeg_python():
    """Перевіряє наявність бібліотеки ffmpeg-python"""
    try:
        import ffmpeg
        return True
    except ImportError:
        return False


def segment_frames(input_path, frames_dir):
    """Обробляє кадри відео та зберігає їх в директорію"""
    # Створюємо директорію для кадрів, якщо вона не існує
    os.makedirs(frames_dir, exist_ok=True)

    # Відкриваємо відео
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Отримуємо предиктор
    pred = get_predictor()

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processing frame {frame_count}/{total_frames}")

        # Отримуємо передбачення
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

        # Зберігаємо оброблений кадр
        cv2.imwrite(os.path.join(frames_dir, f"frame_{frame_count:06d}.jpg"), result)

    cap.release()

    return {
        'frame_count': frame_count,
        'fps': fps,
        'width': width,
        'height': height,
    }


def segment_video(video_path, output_path):
    """Обробка відео з покращеним алгоритмом для сумісності з веб-браузерами"""
    print(f"Processing video: {video_path}")
    print(f"Output will be saved to: {output_path}")

    # Переконуємось, що ми використовуємо розширення .mp4
    if not output_path.endswith('.mp4'):
        output_path = os.path.splitext(output_path)[0] + '.mp4'

    temp_dir = tempfile.mkdtemp(prefix="video_processing_")
    frames_dir = os.path.join(temp_dir, "frames")

    try:
        # Метод 1: Використання FFmpeg безпосередньо через subprocess
        if check_ffmpeg():
            print("Using FFmpeg for video processing")

            # Крок 1: Обробка відео з використанням OpenCV та збереження кадрів
            video_info = segment_frames(video_path, frames_dir)

            # Крок 2: Збирання кадрів у відео з використанням FFmpeg
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-framerate', str(video_info['fps']),
                '-i', os.path.join(frames_dir, 'frame_%06d.jpg'),
                '-c:v', 'libx264',
                '-profile:v', 'high',
                '-pix_fmt', 'yuv420p',
                '-preset', 'medium',  # Баланс між якістю та швидкістю
                '-crf', '23',  # Гарна якість, 0 - без втрат, 51 - найгірша
                '-movflags', '+faststart',  # Для швидкого запуску відео у браузері
                output_path
            ]

            process = subprocess.run(ffmpeg_cmd,
                                     check=True,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)

            print("FFmpeg finished processing")
            print(f"Output saved to: {output_path}")

            # Перевіряємо чи файл був створений та має розмір
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f"File created successfully, size: {os.path.getsize(output_path)} bytes")
                return output_path
            else:
                print("File was not created or is empty")

        # Метод 2: Використання бібліотеки ffmpeg-python, якщо вона доступна
        elif has_ffmpeg_python():
            print("Using ffmpeg-python library")
            import ffmpeg

            # Обробка відео з використанням OpenCV та збереження кадрів
            video_info = segment_frames(video_path, frames_dir)

            # Збирання кадрів у відео з використанням ffmpeg-python
            (
                ffmpeg
                .input(os.path.join(frames_dir, 'frame_%06d.jpg'), framerate=video_info['fps'])
                .output(output_path,
                        vcodec='libx264',
                        pix_fmt='yuv420p',
                        preset='medium',
                        crf=23,
                        movflags='+faststart')
                .overwrite_output()
                .run()
            )

            print(f"ffmpeg-python finished processing")
            print(f"Output saved to: {output_path}")

            return output_path

        # Метод 3: Запасний варіант - використання OpenCV для створення відео
        else:
            print("FFmpeg not available, using OpenCV VideoWriter")

            # Створюємо директорію для проміжного відео
            temp_output = os.path.join(temp_dir, "output_opencv.avi")

            # Використовуємо OpenCV для обробки та збереження відео
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Використання кодеку, який працює майже на всіх системах
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

            if not out.isOpened():
                print("Failed to open VideoWriter")
                return None

            pred = get_predictor()

            colors = [
                (0, 255, 0),
                (255, 0, 0),
                (0, 0, 255),
                (255, 255, 0),
                (255, 0, 255),
                (0, 255, 255),
            ]

            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % 100 == 0:
                    print(f"Processing frame {frame_count}/{total_frames}")

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

            print(f"OpenCV finished processing, temp output at: {temp_output}")

            # Копіюємо файл в цільове місце
            with open(temp_output, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    f_out.write(f_in.read())

            print(f"File copied to final destination: {output_path}")

            return output_path

    except Exception as e:
        print(f"Error during video processing: {e}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        # Очищення тимчасових файлів
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except:
            pass


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

        if ext in ['.jpg', '.jpeg', '.png']:
            output_name = f"segmented_{file.filename}" if file.filename else f"segmented_file_{file_id}.jpg"
            output_path = os.path.join(settings.MEDIA_ROOT, 'segmented', output_name)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            result = segment_image(file_path)
            result.save(output_path)

            return render(request, 'model/result.html', {
                'result_url': os.path.join(settings.MEDIA_URL, 'segmented', output_name)
            })

        elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
            # Завжди зберігаємо як .mp4 для веб-сумісності
            base_name = os.path.splitext(file.filename)[0] if file.filename else f"segmented_file_{file_id}"
            output_name = f"segmented_{base_name}.mp4"
            output_path = os.path.join(settings.MEDIA_ROOT, 'segmented', output_name)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            final_output_path = segment_video(file_path, output_path)

            if final_output_path:
                # Перевіряємо чи файл був створений та має розмір
                if os.path.exists(final_output_path) and os.path.getsize(final_output_path) > 0:
                    # Отримуємо відносний URL від MEDIA_ROOT
                    result_url = os.path.join(settings.MEDIA_URL, 'segmented', output_name)
                else:
                    return render(request, 'model/error.html', {
                        'error_message': 'Помилка при обробці відео. Створений файл порожній або відсутній.'
                    })
            else:
                return render(request, 'model/error.html', {
                    'error_message': 'Не вдалося обробити відео. Перевірте логи сервера для деталей.'
                })

            return render(request, 'model/result_video.html', {
                'result_url': result_url,
                'original_filename': file.filename
            })

        else:
            return render(request, 'model/unsupported.html')

    finally:
        # Clean up the temporary file
        try:
            os.unlink(temp_file.name)
        except:
            pass


@login_required
def download_video(request, file_path):
    """Надає відео для завантаження з правильними заголовками"""
    full_path = os.path.join(settings.MEDIA_ROOT, file_path)

    if os.path.exists(full_path) and os.path.isfile(full_path):
        with open(full_path, 'rb') as f:
            response = HttpResponse(f.read(), content_type='video/mp4')
            response['Content-Disposition'] = f'attachment; filename="{os.path.basename(file_path)}"'
            return response
    else:
        return HttpResponse("Файл не знайдено", status=404)