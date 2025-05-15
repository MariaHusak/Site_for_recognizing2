import os
from django.conf import settings
from django.shortcuts import render, redirect, get_object_or_404
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
            uploaded_file = form.save(commit=False)
            uploaded_file.user = request.user
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
    file_path = file.file.path
    file.delete()
    if os.path.exists(file_path):
        os.remove(file_path)
    return redirect('file_list')
