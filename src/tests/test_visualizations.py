import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(model, dataloader, class_names, device):
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
    
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    # Plot the matrix
    fig, ax = plt.subplots(figsize=(10, 10))  # Ajusta el tamaño de la figura
    disp.plot(cmap='Blues', ax=ax)
    
    # Rotate the x-axis labels
    plt.xticks(rotation=90)
    plt.tight_layout()  # Ajusta los márgenes para evitar solapamientos
    
    plt.show()
    return cm