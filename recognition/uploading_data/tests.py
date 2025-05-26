import io
from django.contrib.auth.models import User
from django.urls import reverse
from django.test import TestCase, Client
from .models import UploadedFile
from django.core.files.uploadedfile import SimpleUploadedFile


class FileViewTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username='testuser', password='testpass')
        self.upload_url = reverse('upload_file')
        self.file_list_url = reverse('file_list')
        self.main_url = reverse('main')

    def test_main_view_returns(self):
        response = self.client.get(self.main_url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'main.html')

    def test_upload_file_requires_login(self):
        response = self.client.get(self.upload_url)
        self.assertEqual(response.status_code, 302)

    def test_file_list_requires_login(self):
        response = self.client.get(self.file_list_url)
        self.assertEqual(response.status_code, 302)

    def test_upload_file_valid(self):
        self.client.login(username='testuser', password='testpass')
        png_content = (
            b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'
            b'\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01'
            b'\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
        )

        file_data = SimpleUploadedFile(
            "test.png",
            png_content,
            content_type="image/png"
        )

        response = self.client.post(self.upload_url, {
            'file': file_data,
        })
        if response.status_code != 302:
            print(f"Response status: {response.status_code}")
            print(f"Response content: {response.content.decode()}")
            if hasattr(response, 'context') and response.context:
                form = response.context.get('form')
                if form and form.errors:
                    print(f"Form errors: {form.errors}")

        self.assertEqual(response.status_code, 302)
        self.assertRedirects(response, self.file_list_url)
        self.assertEqual(UploadedFile.objects.count(), 1)
        uploaded = UploadedFile.objects.first()
        self.assertEqual(uploaded.filename, "test.png")
        self.assertEqual(uploaded.user, self.user)

    def test_upload_file_invalid_type(self):
        self.client.login(username='testuser', password='testpass')
        file_data = SimpleUploadedFile(
            "test.txt", b"Test content", content_type="text/plain"
        )
        response = self.client.post(self.upload_url, {
            'file': file_data,
        })

        self.assertEqual(response.status_code, 200)
        self.assertEqual(UploadedFile.objects.count(), 0)

    def test_file_list_shows_only_user_files(self):
        other_user = User.objects.create_user(username='other', password='pass')
        UploadedFile.objects.create(user=other_user, filename='other.txt', file_data=b'xxx', content_type='text/plain')
        UploadedFile.objects.create(user=self.user, filename='mine.txt', file_data=b'abc', content_type='text/plain')

        self.client.login(username='testuser', password='testpass')
        response = self.client.get(self.file_list_url)

        self.assertEqual(response.status_code, 200)
        files = response.context['files']
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].filename, 'mine.txt')

    def test_delete_file(self):
        self.client.login(username='testuser', password='testpass')
        file = UploadedFile.objects.create(user=self.user, filename='del.txt', file_data=b'data',
                                           content_type='text/plain')
        delete_url = reverse('delete_file', args=[file.id])

        response = self.client.post(delete_url)
        self.assertRedirects(response, self.file_list_url)
        self.assertEqual(UploadedFile.objects.count(), 0)

    def test_delete_file_not_owner(self):
        other_user = User.objects.create_user(username='other', password='pass')
        file = UploadedFile.objects.create(user=other_user, filename='other.txt', file_data=b'data',
                                           content_type='text/plain')
        self.client.login(username='testuser', password='testpass')

        delete_url = reverse('delete_file', args=[file.id])
        response = self.client.post(delete_url)
        self.assertEqual(response.status_code, 404)
        self.assertEqual(UploadedFile.objects.count(), 1)

    def test_download_file(self):
        self.client.login(username='testuser', password='testpass')
        file = UploadedFile.objects.create(user=self.user, filename='download.txt', file_data=b'hello',
                                           content_type='text/plain')
        download_url = reverse('download_file', args=[file.id])

        response = self.client.get(download_url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b'hello')
        self.assertEqual(response['Content-Type'], 'text/plain')
        self.assertEqual(response['Content-Disposition'], 'attachment; filename="download.txt"')

    def test_download_file_not_owner(self):
        other_user = User.objects.create_user(username='other', password='pass')
        file = UploadedFile.objects.create(user=other_user, filename='forbidden.txt', file_data=b'data',
                                           content_type='text/plain')
        self.client.login(username='testuser', password='testpass')
        download_url = reverse('download_file', args=[file.id])

        response = self.client.get(download_url)
        self.assertEqual(response.status_code, 404)

