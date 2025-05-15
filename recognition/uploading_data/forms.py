from django import forms
from django.core.exceptions import ValidationError


class UploadFileForm(forms.Form):
    file = forms.FileField()

    def clean_file(self):
        file = self.cleaned_data.get('file')
        if not file:
            raise ValidationError('No file uploaded.')

        allowed_types = [
            'image/jpeg', 'image/png', 'image/gif',
            'video/mp4', 'video/x-msvideo', 'video/quicktime', 'video/x-matroska'
        ]
        if file.content_type not in allowed_types:
            raise ValidationError('Unsupported file type.')
        return file
