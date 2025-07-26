import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, classification_report, accuracy_score
import torch




def f1_score_fn(labels, y_predicted):

    f1 = f1_score(labels, y_predicted, average='weighted')

    return f1


def plot_confusion_matrix(y_true, y_predicted, class_names, save_path=None):

    # Compute Confusion Matrix
    cm = confusion_matrix(y_true, y_predicted)

    # Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Confusion matrix saved to {save_path}")


    plt.close(fig)

    return fig

def eval_model(model, test_loader, device, class_names, save_path):

    # Switch to evaluation mode (disable dropout & batch norm updates)
    model.eval()

    print(f"Model set to eval mode: {not model.training}")  # Debug print

    # Lists to store predictions and true labels
    y_pred = []
    y_true = [] 

    with torch.no_grad():
        for tinput, tlabels in test_loader:
            tinput, tlabels = tinput.to(device), tlabels.to(device)

            # Check if model switched back to train mode (should NOT happen)
            # print(f"Inside loop - Model in training mode? {model.training}")

            toutputs = model(tinput) # Forward pass

            # Get predicted class indices
            predicted = torch.argmax(toutputs, dim=1)

            # # Debugging step
            # print(f"tlabels shape: {tlabels.shape}")  # Check tensor dimensions
            # print(f"tlabels shape: {tlabels.dim()}")  # Add this to debug

            # Fix based on tensor shape
            if tlabels.dim() == 1:
                true_labels = tlabels  # No need for argmax
            else:
                true_labels = torch.argmax(tlabels, dim=1)  # If 2D, apply argmax



            # Stroe prdiction and true label
            y_pred.extend(predicted.cpu().numpy()) # Convert to NumPy
            y_true.extend(true_labels.cpu().numpy())
    
    # Compute Test Accuracy
    test_accuracy = accuracy_score(y_true, y_pred) * 100
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    # Compute F1-Score
    f1 = f1_score_fn(y_true, y_pred)

    # Compute Classification Report
    class_report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    plot_confusion_matrix(y_true, y_pred, class_names=class_names, save_path=save_path)


    metrics ={
        "Test accuracy": test_accuracy,
        "f1_score": f1,
        "classificaton report": class_report
    }
    return metrics

# **Save the figure**
# save_path = "/mnt/New Volume/Deep Learning. DR Mostafa/Group_activity_project/models/base_line1/outputs/confusion_matrix.png"  # Change to your desired path

# group_activity_clases_b1 = ["r_set", "r_spike", "r-pass", "r_winpoint", "l_winpoint", "l-pass", "l-spike", "l_set"]

