from django import forms
from .models import UploadedFile
from django.core.exceptions import ValidationError


class UploadFileForm(forms.ModelForm):
    class Meta:
        model = UploadedFile
        fields = ['file']

    def clean_file(self):
        file = self.cleaned_data['file']
        content_type = file.content_type

        allowed_types = [
            'image/jpeg', 'image/png', 'image/gif',
            'video/mp4', 'video/x-msvideo', 'video/quicktime', 'video/x-matroska'
        ]

        if content_type not in allowed_types:
            raise ValidationError('Unsupported file type. Please upload a valid image or video file.')

        return file
