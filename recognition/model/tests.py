import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from django.test import TestCase, Client, override_settings
from django.contrib.auth.models import User
from django.urls import reverse
from django.conf import settings
from PIL import Image
import numpy as np
import cv2

from uploading_data.models import UploadedFile
from .recognize import Model, Photo, Video, RecognizeFacade


class MockDetectron2Tests:
    @staticmethod
    def mock_get_cfg():
        mock_cfg = Mock()
        mock_cfg.merge_from_file = Mock()
        mock_cfg.MODEL = Mock()
        mock_cfg.MODEL.ROI_HEADS = Mock()
        mock_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        mock_cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        mock_cfg.MODEL.WEIGHTS = "/fake/path/model.pth"
        mock_cfg.MODEL.DEVICE = 'cpu'
        return mock_cfg

    @staticmethod
    def mock_predictor_output():
        mock_output = MagicMock()
        mock_instances = Mock()

        mock_masks = np.array([
            np.random.choice([True, False], size=(100, 100)),
            np.random.choice([True, False], size=(100, 100))
        ])
        mock_instances.pred_masks = Mock()
        mock_instances.pred_masks.cpu.return_value.numpy.return_value = mock_masks

        mock_boxes = Mock()
        mock_boxes.tensor = Mock()
        mock_boxes.tensor.cpu.return_value.numpy.return_value = np.array([
            [10, 10, 50, 50],
            [60, 60, 100, 100]
        ])
        mock_instances.pred_boxes = mock_boxes

        mock_instances.scores = Mock()
        mock_instances.scores.cpu.return_value.numpy.return_value = np.array([0.9, 0.8])

        mock_output.__getitem__ = MagicMock(return_value=mock_instances)
        return mock_output


@override_settings(MEDIA_ROOT=tempfile.mkdtemp())
class ModelTests(TestCase):

    @patch('model.recognize.model_zoo')
    @patch('model.recognize.get_cfg')
    @patch('model.recognize.torch')
    def setUp(self, mock_torch, mock_get_cfg, mock_model_zoo):
        mock_torch.cuda.is_available.return_value = False
        mock_get_cfg.return_value = MockDetectron2Tests.mock_get_cfg()
        mock_model_zoo.get_config_file.return_value = "/fake/config.yaml"

        self.model = Model()

    @patch('model.recognize.DefaultPredictor')
    def test_get_predictor_creates_predictor(self, mock_predictor_class):
        mock_predictor_instance = Mock()
        mock_predictor_class.return_value = mock_predictor_instance

        predictor = self.model.get_predictor()

        self.assertIsNotNone(predictor)
        mock_predictor_class.assert_called_once_with(self.model.cfg)

    @patch('model.recognize.DefaultPredictor')
    def test_get_predictor_returns_same_instance(self, mock_predictor_class):
        mock_predictor_instance = Mock()
        mock_predictor_class.return_value = mock_predictor_instance

        predictor1 = self.model.get_predictor()
        predictor2 = self.model.get_predictor()

        self.assertEqual(predictor1, predictor2)
        mock_predictor_class.assert_called_once()


class PhotoTests(TestCase):

    def setUp(self):
        self.mock_model = Mock()
        self.photo = Photo(self.mock_model)

        self.test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.test_image_fd, self.test_image_path = tempfile.mkstemp(suffix='.jpg')
        os.close(self.test_image_fd)
        cv2.imwrite(self.test_image_path, self.test_image)

    def tearDown(self):
        if os.path.exists(self.test_image_path):
            os.unlink(self.test_image_path)

    @patch('model.recognize.cv2.imread')
    def test_segment_image_no_masks(self, mock_imread):
        mock_imread.return_value = self.test_image
        mock_predictor = Mock()
        mock_output = MagicMock()
        mock_instances = Mock()
        mock_instances.pred_masks = Mock()
        mock_instances.pred_masks.cpu.return_value.numpy.return_value = np.array([])
        mock_output.__getitem__ = MagicMock(return_value=mock_instances)
        mock_predictor.return_value = mock_output

        self.mock_model.get_predictor.return_value = mock_predictor

        result = self.photo.segment_image(self.test_image_path)

        self.assertIsInstance(result, Image.Image)
        mock_imread.assert_called_once_with(self.test_image_path)

    @patch('model.recognize.cv2.imread')
    def test_segment_image_with_masks(self, mock_imread):
        mock_imread.return_value = self.test_image

        mock_predictor = Mock()
        mock_predictor.return_value = MockDetectron2Tests.mock_predictor_output()
        self.mock_model.get_predictor.return_value = mock_predictor

        result = self.photo.segment_image(self.test_image_path)

        self.assertIsInstance(result, Image.Image)
        mock_imread.assert_called_once_with(self.test_image_path)

    def test_apply_mask_and_box(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.random.choice([True, False], size=(100, 100))
        box = [10, 10, 50, 50]
        score = 0.9
        index = 0

        result = self.photo._apply_mask_and_box(image, mask, box, score, index)

        self.assertEqual(result.shape, image.shape)
        self.assertEqual(result.dtype, image.dtype)

    def test_add_text_to_image(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        text = "person 90%"
        x1, y1, y2 = 10, 10, 50
        color = (0, 255, 0)

        result = self.photo._add_text_to_image(image, text, x1, y1, y2, color)

        self.assertEqual(result.shape, image.shape)
        self.assertEqual(result.dtype, image.dtype)


class VideoTests(TestCase):

    def setUp(self):
        self.mock_model = Mock()
        self.video = Video(self.mock_model)

        self.temp_dir = tempfile.mkdtemp()
        self.input_video = os.path.join(self.temp_dir, 'input.mp4')
        self.output_video = os.path.join(self.temp_dir, 'output.mp4')
        with open(self.input_video, 'wb') as f:
            f.write(b'fake video content')

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('model.recognize.subprocess.run')
    def test_check_ffmpeg_available(self, mock_run):
        mock_run.return_value = Mock()

        result = self.video._check_ffmpeg()

        self.assertTrue(result)
        mock_run.assert_called_once()

    @patch('model.recognize.subprocess.run')
    def test_check_ffmpeg_not_available(self, mock_run):
        mock_run.side_effect = FileNotFoundError()

        result = self.video._check_ffmpeg()

        self.assertFalse(result)

    def test_has_ffmpeg_python_not_available(self):
        with patch.dict('sys.modules', {'ffmpeg': None}):
            result = self.video._has_ffmpeg_python()

    @patch('model.recognize.cv2.VideoCapture')
    def test_process_frame(self, mock_video_capture):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        mock_predictor = Mock()
        mock_predictor.return_value = MockDetectron2Tests.mock_predictor_output()

        result = self.video._process_frame(frame, mock_predictor)

        self.assertEqual(result.shape, frame.shape)
        self.assertEqual(result.dtype, frame.dtype)

    @patch('model.recognize.cv2.VideoCapture')
    def test_process_frame_no_masks(self, mock_video_capture):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        mock_predictor = Mock()
        mock_output = MagicMock()
        mock_instances = Mock()
        mock_instances.pred_masks = Mock()
        mock_instances.pred_masks.cpu.return_value.numpy.return_value = np.array([])
        mock_output.__getitem__ = MagicMock(return_value=mock_instances)
        mock_predictor.return_value = mock_output

        result = self.video._process_frame(frame, mock_predictor)

        np.testing.assert_array_equal(result, frame)

    def test_cleanup_temp_dir(self):
        test_dir = tempfile.mkdtemp()
        test_file = os.path.join(test_dir, 'test.txt')
        with open(test_file, 'w') as f:
            f.write('test')

        self.assertTrue(os.path.exists(test_dir))
        self.video._cleanup_temp_dir(test_dir)
        self.assertFalse(os.path.exists(test_dir))


class RecognizeFacadeTests(TestCase):

    @patch('model.recognize.Model')
    def setUp(self, mock_model_class):
        self.mock_model_instance = Mock()
        mock_model_class.return_value = self.mock_model_instance
        self.facade = RecognizeFacade()

    @patch('model.recognize.Photo.segment_image')
    def test_process_image(self, mock_segment_image):
        mock_result = Mock(spec=Image.Image)
        mock_segment_image.return_value = mock_result

        result = self.facade.process_image('/fake/path.jpg')

        self.assertEqual(result, mock_result)
        mock_segment_image.assert_called_once_with('/fake/path.jpg')

    @patch('model.recognize.Video.segment_video')
    def test_process_video(self, mock_segment_video):
        mock_segment_video.return_value = '/fake/output.mp4'

        result = self.facade.process_video('/fake/input.mp4', '/fake/output.mp4')

        self.assertEqual(result, '/fake/output.mp4')
        mock_segment_video.assert_called_once_with('/fake/input.mp4', '/fake/output.mp4')


@override_settings(MEDIA_ROOT=tempfile.mkdtemp())
class ViewTests(TestCase):

    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username='testuser', password='testpass')
        self.other_user = User.objects.create_user(username='other', password='pass')

        self.image_file = UploadedFile.objects.create(
            user=self.user,
            filename='test.jpg',
            file_data=b'fake image data',
            content_type='image/jpeg'
        )

        self.video_file = UploadedFile.objects.create(
            user=self.user,
            filename='test.mp4',
            file_data=b'fake video data',
            content_type='video/mp4'
        )

        self.unsupported_file = UploadedFile.objects.create(
            user=self.user,
            filename='test.txt',
            file_data=b'fake text data',
            content_type='text/plain'
        )

    def tearDown(self):
        if os.path.exists(settings.MEDIA_ROOT):
            shutil.rmtree(settings.MEDIA_ROOT, ignore_errors=True)

    def test_select_file_requires_login(self):
        response = self.client.get(reverse('select_file'))
        self.assertEqual(response.status_code, 302)

    def test_select_file_shows_user_files(self):
        self.client.login(username='testuser', password='testpass')
        response = self.client.get(reverse('select_file'))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'test.jpg')
        self.assertContains(response, 'test.mp4')
        self.assertTemplateUsed(response, 'model/select_file.html')

    def test_select_file_shows_only_user_files(self):
        other_file = UploadedFile.objects.create(
            user=self.other_user,
            filename='other.jpg',
            file_data=b'other data',
            content_type='image/jpeg'
        )

        self.client.login(username='testuser', password='testpass')
        response = self.client.get(reverse('select_file'))

        self.assertContains(response, 'test.jpg')
        self.assertNotContains(response, 'other.jpg')

    def test_segment_file_requires_login(self):
        response = self.client.get(reverse('segment_file', args=[self.image_file.id]))
        self.assertEqual(response.status_code, 302)

    def test_segment_file_not_owner(self):
        self.client.login(username='other', password='pass')
        response = self.client.get(reverse('segment_file', args=[self.image_file.id]))
        self.assertEqual(response.status_code, 404)

    @patch('model.views.recognize_facade.process_image')
    @patch('model.views.tempfile.NamedTemporaryFile')
    def test_segment_image_file(self, mock_temp_file, mock_process_image):
        mock_file = Mock()
        mock_file.name = '/tmp/test_image.jpg'
        mock_temp_file.return_value = mock_file

        mock_result = Mock(spec=Image.Image)
        mock_process_image.return_value = mock_result
        segmented_dir = os.path.join(settings.MEDIA_ROOT, 'segmented')
        os.makedirs(segmented_dir, exist_ok=True)

        self.client.login(username='testuser', password='testpass')
        response = self.client.get(reverse('segment_file', args=[self.image_file.id]))

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'model/result.html')
        mock_process_image.assert_called_once()
        mock_result.save.assert_called_once()

    @patch('model.views.recognize_facade.process_video')
    @patch('model.views.tempfile.NamedTemporaryFile')
    def test_segment_video_file_success(self, mock_temp_file, mock_process_video):
        mock_file = Mock()
        mock_file.name = '/tmp/test_video.mp4'
        mock_temp_file.return_value = mock_file

        segmented_dir = os.path.join(settings.MEDIA_ROOT, 'segmented')
        os.makedirs(segmented_dir, exist_ok=True)
        output_path = os.path.join(segmented_dir, 'segmented_test.mp4')

        with open(output_path, 'wb') as f:
            f.write(b'fake processed video')

        mock_process_video.return_value = output_path

        self.client.login(username='testuser', password='testpass')
        response = self.client.get(reverse('segment_file', args=[self.video_file.id]))

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'model/result_video.html')
        mock_process_video.assert_called_once()

    @patch('model.views.recognize_facade.process_video')
    @patch('model.views.tempfile.NamedTemporaryFile')
    def test_segment_video_file_failure(self, mock_temp_file, mock_process_video):
        mock_file = Mock()
        mock_file.name = '/tmp/test_video.mp4'
        mock_temp_file.return_value = mock_file

        mock_process_video.return_value = None

        template_dir = os.path.join(settings.BASE_DIR, 'model', 'templates', 'model')
        os.makedirs(template_dir, exist_ok=True)
        error_template_path = os.path.join(template_dir, 'error.html')

        with open(error_template_path, 'w') as f:
            f.write('<html><body><h1>Error</h1><p>{{ error_message }}</p></body></html>')

        self.client.login(username='testuser', password='testpass')

        try:
            response = self.client.get(reverse('segment_file', args=[self.video_file.id]))
            self.assertEqual(response.status_code, 200)
            self.assertTemplateUsed(response, 'model/error.html')
            self.assertContains(response, 'Failed to process video')
        finally:
            if os.path.exists(error_template_path):
                os.unlink(error_template_path)

    def test_segment_unsupported_file(self):
        self.client.login(username='testuser', password='testpass')
        response = self.client.get(reverse('segment_file', args=[self.unsupported_file.id]))

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'model/unsupported.html')

    def test_download_video_existing_file(self):
        segmented_dir = os.path.join(settings.MEDIA_ROOT, 'segmented')
        os.makedirs(segmented_dir, exist_ok=True)
        video_path = os.path.join(segmented_dir, 'test_video.mp4')

        with open(video_path, 'wb') as f:
            f.write(b'fake video content')

        self.client.login(username='testuser', password='testpass')
        response = self.client.get(reverse('download_video', args=['segmented/test_video.mp4']))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'video/mp4')
        self.assertIn('attachment', response['Content-Disposition'])
        self.assertEqual(response.content, b'fake video content')

    def test_download_video_nonexistent_file(self):
        self.client.login(username='testuser', password='testpass')
        response = self.client.get(reverse('download_video', args=['nonexistent.mp4']))

        self.assertEqual(response.status_code, 404)
        self.assertContains(response, 'File not found', status_code=404)

    def test_download_video_requires_login(self):
        response = self.client.get(reverse('download_video', args=['test.mp4']))
        self.assertEqual(response.status_code, 302)


class IntegrationTests(TestCase):

    @patch('model.recognize.Model')
    def setUp(self, mock_model_class):
        self.client = Client()
        self.user = User.objects.create_user(username='testuser', password='testpass')
        self.mock_model_instance = Mock()
        mock_model_class.return_value = self.mock_model_instance

        self.test_file = UploadedFile.objects.create(
            user=self.user,
            filename='integration_test.jpg',
            file_data=b'test image data',
            content_type='image/jpeg'
        )

    @patch('model.views.recognize_facade')
    @override_settings(MEDIA_ROOT=tempfile.mkdtemp())
    def test_full_image_processing_workflow(self, mock_facade):
        mock_result = Mock(spec=Image.Image)
        mock_facade.process_image.return_value = mock_result
        os.makedirs(os.path.join(settings.MEDIA_ROOT, 'segmented'), exist_ok=True)

        self.client.login(username='testuser', password='testpass')

        select_response = self.client.get(reverse('select_file'))
        self.assertEqual(select_response.status_code, 200)
        self.assertContains(select_response, 'integration_test.jpg')

        segment_response = self.client.get(reverse('segment_file', args=[self.test_file.id]))
        self.assertEqual(segment_response.status_code, 200)

        mock_facade.process_image.assert_called_once()
        mock_result.save.assert_called_once()


class PerformanceTests(TestCase):

    def setUp(self):
        self.user = User.objects.create_user(username='testuser', password='testpass')

    def test_multiple_files_handling(self):
        files = []
        for i in range(10):
            file = UploadedFile.objects.create(
                user=self.user,
                filename=f'test_{i}.jpg',
                file_data=b'test data',
                content_type='image/jpeg'
            )
            files.append(file)

        self.client.login(username='testuser', password='testpass')
        response = self.client.get(reverse('select_file'))

        self.assertEqual(response.status_code, 200)
        for file in files:
            self.assertContains(response, file.filename)

    @patch('model.views.recognize_facade.process_image')
    def test_large_file_processing_simulation(self, mock_process_image):
        large_file = UploadedFile.objects.create(
            user=self.user,
            filename='large_image.jpg',
            file_data=b'x' * 1024 * 1024,
            content_type='image/jpeg'
        )

        mock_result = Mock(spec=Image.Image)
        mock_process_image.return_value = mock_result

        with override_settings(MEDIA_ROOT=tempfile.mkdtemp()):
            os.makedirs(os.path.join(settings.MEDIA_ROOT, 'segmented'), exist_ok=True)

            self.client.login(username='testuser', password='testpass')
            response = self.client.get(reverse('segment_file', args=[large_file.id]))

            self.assertEqual(response.status_code, 200)
            mock_process_image.assert_called_once()


if __name__ == '__main__':
    import django
    from django.conf import settings
    from django.test.utils import get_runner

    if not settings.configured:
        settings.configure(
            DEBUG=True,
            DATABASES={
                'default': {
                    'ENGINE': 'django.db.backends.sqlite3',
                    'NAME': ':memory:',
                }
            },
            INSTALLED_APPS=[
                'django.contrib.auth',
                'django.contrib.contenttypes',
                'uploading_data',
                'model',
            ],
            MEDIA_ROOT=tempfile.mkdtemp(),
            MEDIA_URL='/media/',
        )

    django.setup()
    TestRunner = get_runner(settings)
    test_runner = TestRunner()
    failures = test_runner.run_tests(["__main__"])