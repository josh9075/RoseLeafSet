import io
import os
from django.shortcuts import render
from django.conf import settings
from django.http import JsonResponse
from PIL import Image

from .forms import UploadImageForm
from .model_utils import ModelEnsemble

MODEL_DIR = os.path.abspath(os.path.join(settings.BASE_DIR, '.'))

# lazy load the ensemble
ensemble = None


def index(request):
    form = UploadImageForm()
    return render(request, 'predictor/index.html', {'form': form})


def predict_view(request):
    global ensemble
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=400)

    form = UploadImageForm(request.POST, request.FILES)
    if not form.is_valid():
        return JsonResponse({'error': 'invalid form'}, status=400)

    imgfile = request.FILES['image']
    img = Image.open(imgfile).convert('RGB')

    if ensemble is None:
        ensemble = ModelEnsemble(model_dir=MODEL_DIR)

    preds = ensemble.predict(img)

    # If the form was submitted via classic form submit, render a page.
    if 'text/html' in request.META.get('HTTP_ACCEPT', ''):
        return render(request, 'predictor/result.html', {'result': preds})

    return JsonResponse({'predictions': preds})


def api_predict(request):
    """Simple API endpoint that accepts multipart/form-data with 'image' and returns JSON."""
    global ensemble
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=400)

    imgfile = request.FILES.get('image')
    if not imgfile:
        return JsonResponse({'error': 'no image provided'}, status=400)

    img = Image.open(imgfile).convert('RGB')
    if ensemble is None:
        ensemble = ModelEnsemble(model_dir=MODEL_DIR)

    preds = ensemble.predict(img)
    return JsonResponse(preds)
