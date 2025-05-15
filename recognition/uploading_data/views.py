import os
from django.conf import settings
from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse
from .models import UploadedFile
from .forms import UploadFileForm
from django.contrib.auth.decorators import login_required


def main(request):
    return render(request, 'main.html')


@login_required
def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = UploadedFile(
                user=request.user,
                filename=request.FILES['file'].name,
                file_data=request.FILES['file'].read(),
                content_type=request.FILES['file'].content_type
            )
            uploaded_file.save()
            return redirect('file_list')
    else:
        form = UploadFileForm()
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