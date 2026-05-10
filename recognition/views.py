from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .model import predict_image   # your AI function

@csrf_exempt
def predict(request):
    if request.method == "POST":
        image = request.FILES.get("image")

        if not image:
            return JsonResponse({"error": "No image uploaded"}, status=400)

        # Call AI model
        results = predict_image(image)

        return JsonResponse({
            "predictions": results
        })

    return JsonResponse({"error": "Invalid request"})