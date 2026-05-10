import torch
from torchvision import models, transforms
from PIL import Image
import json
import urllib.request

# Load pretrained model
model = models.resnet18(pretrained=True)
model.eval()

# Load ImageNet labels
LABELS_URL = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
class_idx = json.load(urllib.request.urlopen(LABELS_URL))

# Transform image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_image(image_file):
    img = Image.open(image_file).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    # Get top 3 predictions
    top3 = torch.topk(probabilities, 3)

    results = []
    for i in range(3):
        idx = top3.indices[i].item()
        label = class_idx[str(idx)][1]
        confidence = top3.values[i].item()

        results.append({
            "label": label,
            "confidence": confidence
        })

    return results