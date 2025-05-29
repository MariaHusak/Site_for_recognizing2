import os
from django.conf import settings
from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse
from .models import UploadedFile
from .forms import UploadFileForm
from django.contrib.auth.decorators import login_required


def main(request):
    return render(request, 'main.html')


def _create_uploaded_file_instance(user, uploaded_file):
    return UploadedFile(
        user=user,
        filename=uploaded_file.name,
        file_data=uploaded_file.read(),
        content_type=uploaded_file.content_type
    )


def _handle_file_upload_post(request):
    form = UploadFileForm(request.POST, request.FILES)
    if form.is_valid():
        uploaded_file_instance = _create_uploaded_file_instance(
            request.user,
            request.FILES['file']
        )
        uploaded_file_instance.save()
        return redirect('file_list')
    return form


def _handle_file_upload_get():
    return UploadFileForm()


@login_required
def upload_file(request):
    if request.method == 'POST':
        form_or_redirect = _handle_file_upload_post(request)
        if isinstance(form_or_redirect, HttpResponse):
            return form_or_redirect
        form = form_or_redirect
    else:
        form = _handle_file_upload_get()

    return render(request, 'upload.html', {'form': form})


@login_required
def file_list(request):
    files = UploadedFile.objects.filter(user=request.user)
    return render(request, 'file_list.html', {'files': files})


@login_required
def delete_file(request, file_id):
    file = get_object_or_404(UploadedFile, id=file_id, user=request.user)
    file.delete()
    return redirect('file_list')


@login_required
def download_file(request, file_id):
    file = get_object_or_404(UploadedFile, id=file_id, user=request.user)
    response = HttpResponse(file.file_data, content_type=file.content_type)
    response['Content-Disposition'] = f'attachment; filename="{file.filename}"'
    return response