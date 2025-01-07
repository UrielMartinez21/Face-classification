import torch
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics import classification_report
import pillow_heif


pillow_heif.register_heif_opener()

transform = transforms.Compose([
    transforms.Resize((224, 224)),          # Rezise the image to 224x224 pixels
    transforms.ToTensor(),                  # Convert to tensor
    transforms.Normalize(
        [0.485, 0.456, 0.406], 
        [0.229, 0.224, 0.225])              # Normalize the image
])


def preprocess_image(image_path):
    """Preprocess an image by loading, converting to RGB, applying transformations, and adding batch dimension."""
    try:
        with Image.open(image_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            input_tensor = transform(img).unsqueeze(0)  # Apply transformation and add batch dimension
        return input_tensor
    except Exception as e:
        print(f"[-] Error processing image {image_path}: {e}")
        return None


def load_model(path, num_classes, device: str):
    # Use the recommended way to load an uninitialized model (no pretrained weights)
    model = models.resnet50(weights=None)

    # Modify the final fully connected (FC) layer
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    # Load the model weights with security fix
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))

    # Move model to device (CPU/GPU)
    model = model.to(device)
    model.eval()
    
    return model


def predict_image(model, image_tensor, class_names, device: str):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted_class = torch.max(output, 1)
    return class_names[predicted_class.item()]


def calculate_accuracy(model, dataloader, device: str):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    return accuracy


def evaluate_classification_report(model, dataloader, class_names, device: str):
    all_labels = []
    all_preds = []
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print(report)