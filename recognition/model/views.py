import os
import tempfile
from django.conf import settings
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from uploading_data.models import UploadedFile
from .recognize import RecognizeFacade


recognize_facade = RecognizeFacade()


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
            return _process_image(request, file, file_id, file_path)
        elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
            return _process_video(request, file, file_id, file_path)
        else:
            return render(request, 'model/unsupported.html')

    finally:
        _cleanup_temp_file(temp_file.name)


def _process_image(request, file, file_id, file_path):
    output_name = f"segmented_{file.filename}" if file.filename else f"segmented_file_{file_id}.jpg"
    output_path = os.path.join(settings.MEDIA_ROOT, 'segmented', output_name)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    result = recognize_facade.process_image(file_path)
    result.save(output_path)

    return render(request, 'model/result.html', {
        'result_url': os.path.join(settings.MEDIA_URL, 'segmented', output_name)
    })


def _process_video(request, file, file_id, file_path):
    base_name = os.path.splitext(file.filename)[0] if file.filename else f"segmented_file_{file_id}"
    output_name = f"segmented_{base_name}.mp4"
    output_path = os.path.join(settings.MEDIA_ROOT, 'segmented', output_name)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    final_output_path = recognize_facade.process_video(file_path, output_path)

    if final_output_path and os.path.exists(final_output_path) and os.path.getsize(final_output_path) > 0:
        result_url = os.path.join(settings.MEDIA_URL, 'segmented', output_name)
        return render(request, 'model/result_video.html', {
            'result_url': result_url,
            'original_filename': file.filename
        })
    else:
        error_message = ('Error processing video. The created file is empty or missing.'
                        if final_output_path
                        else 'Failed to process video.')
        return render(request, 'model/error.html', {'error_message': error_message})


def _cleanup_temp_file(file_path):
    try:
        os.unlink(file_path)
    except:
        pass


@login_required
def download_video(request, file_path):
    full_path = os.path.join(settings.MEDIA_ROOT, file_path)

    if os.path.exists(full_path) and os.path.isfile(full_path):
        with open(full_path, 'rb') as f:
            response = HttpResponse(f.read(), content_type='video/mp4')
            response['Content-Disposition'] = f'attachment; filename="{os.path.basename(file_path)}"'
            return response
    else:
        return HttpResponse("File not found", status=404)