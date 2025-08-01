import sys
import os
import torch
torch.cuda.empty_cache()
# Automatically find the project root (assuming train.py is inside models/base_line1/)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Add the project root to sys.path
sys.path.append(project_root)

# Now, you can import your modules normally
from data.data_loader import Group_Activity_DataSet

from torch.utils.data import DataLoader
import torchvision.transforms as transforms



from model import b1_classifier


# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


# Step1: Data Augmentation and Transformations
train_transformers = transforms.Compose(
    [
    transforms.Resize(224),            # Resize shorter side to 256     
    transforms.RandomResizedCrop(224),  # Randomly crop and resize to 224x224
    transforms.RandomRotation(degrees=5),                   # Randomly rotate images within ±5 degrees
    transforms.ToTensor(),                                   # Convert PIL images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406],        # Normalize using ImageNet mean and std values
                         std=[0.229, 0.224, 0.225]),         # (mean and std are the same used during ResNet pre-training)
])

val_transformers = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),                                   # Convert PIL images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406],        # Normalize using ImageNet mean and std values
                         std=[0.229, 0.224, 0.225]), 
])

test_transformers = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406],        # Normalize using ImageNet mean and std values
                         std=[0.229, 0.224, 0.225])
])


PROJECT_ROOT = r"/mnt/New Volume/Deep Learning. DR Mostafa/05- Project 1 - CV/volleyball-datasets"
sys.path.append(PROJECT_ROOT)
group_activity_clases = ["r_set", "r_spike", "r-pass", "r_winpoint", "l_winpoint", "l-pass", "l-spike", "l_set"]
group_activity_labels = {class_name:i for i, class_name in enumerate(group_activity_clases)}
# print("class labels name", group_activity_labels)

# Step 2: prepare dataset (make it read for dataloader)
train_dataset = Group_Activity_DataSet(
    videos_path= f"{PROJECT_ROOT}/volleyball_/videos",
    annot_path= f"{PROJECT_ROOT}/annot_all.pkl",
    labels=group_activity_labels,
    split=[1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54],
    transform=train_transformers

)

val_dataset = Group_Activity_DataSet(
    videos_path= f"{PROJECT_ROOT}/volleyball_/videos",
    annot_path= f"{PROJECT_ROOT}/annot_all.pkl",
    labels=group_activity_labels,
    split=[0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51],
    transform=val_transformers,

)

test_dataset = Group_Activity_DataSet(
    videos_path= f"{PROJECT_ROOT}/volleyball_/videos",
    annot_path= f"{PROJECT_ROOT}/annot_all.pkl",
    labels=group_activity_labels,
    split=[4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47],
    transform=test_transformers,

)




# Step 3 DataLoader
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True

)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True

)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)


# Step 4 define loss function and optimizer
# Loss Function
loss_fun = torch.nn.CrossEntropyLoss()
# Optimizer
# optimizer = torch.optim.AdamW(model.prameters())

# define Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Model
model = b1_classifier(num_classes=8)
# Training loop

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0
    last_loss = 0
    correct_predictions = 0
    total_samples = 0

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting

    for i, data in enumerate(train_loader):
        # data.to(device)
        input, labels = data
        input, labels = input.to(device), labels.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001,  weight_decay=0.0001)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make prediction for this batch
        model.to(device)      #  Move model to GPU (if available)
        # print(device)
        outputs = model(input)
        
        # Compute the loss and its gradients
        loss = loss_fun(outputs, labels)
        loss.backward()

        # Adjaust learning weights
        optimizer.step()


        

        # Gather data and report
        running_loss += loss.item()

        # Compute accuracy
        predicted = outputs.argmax(1)  # Get the predicted class
        class_labels = labels.argmax(1)  # Ensure labels have shape [batch_size]
        

        correct_predictions += (predicted == class_labels).sum().item()
        total_samples += labels.size(0)

        if i % 50 == 49:
            last_loss = running_loss / 50 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    # Calculate final accuracy
    train_accuracy = 100.0 * correct_predictions / total_samples
    print(f"Epoch {epoch_index + 1} Summary")
        #   Train Accuracy: {train_accuracy:.2f}%")

    return last_loss, train_accuracy


# Per-Epoch Activity

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_dir = os.path.expanduser('/mnt/New Volume/Deep Learning. DR Mostafa/Group_activity_project/models/models/baseline 1//outputs/tensorbord')
writer = SummaryWriter(log_dir.format(timestamp))
epoch_number = 0

Epochs = 5

best_vloss = 1_000_000.

# Free up GPU memory before starting training
torch.cuda.empty_cache()

for epoch in range(Epochs):
    print('Epochs {}:'.format(epoch_number + 1))


    # Make sure gradient tracking is on, and do a pss over the data
    model.train(True)

    avg_loss, train_accuracy = train_one_epoch(epoch_number, writer)

    running_vloss = 0.0

    # Set the model to evaluation mode, disabling dropout and using
    # population statistics for batch normalization.
    model.eval()
    vcorrect_predictions = 0
    vtotal_samples = 0
    # Disable gradient computation and reduce memort consumpation.
    with torch.no_grad():
        for i, vdata in enumerate(val_loader):
            # vdata.to(device)
            vinputs, vlabels = vdata
            vinputs, vlabels = vinputs.to(device), vlabels.to(device) 

            voutputs = model(vinputs)
            vloss = loss_fun(voutputs, vlabels)
            running_vloss += vloss

    # Compute validation accuracy
    vpredicted = voutputs.argmax(1)  # Get predicted class
    vclass_labels = vlabels.argmax(1)  # Ensure labels have shape [batch_size]
    vcorrect_predictions += (vpredicted == vclass_labels).sum().item()
    vtotal_samples += vlabels.size(0)

    val_accuracy = 100.0 * vcorrect_predictions / vtotal_samples  # Compute accuracy percentage
    print(f'Accuracy train {train_accuracy:.2f}% Accuracy valid {val_accuracy:.2f}%')


    avg_vloss = running_vloss / (i + 1)
    print("LOSS train {} valid {}".format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars("Training vs. Validation Loss",
                       {'Training' : avg_loss, "Validation": avg_vloss},
                       epoch_number + 1)
    # Log training and validation accuracy
    writer.add_scalars('Training vs. Validation Accuracy',
                       {'Training': train_accuracy, 'Validation': val_accuracy},
                       epoch_number + 1)
    writer.flush() 

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
    # if True:
        # model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        model_save_dir = "/mnt/New Volume/Deep Learning. DR Mostafa/Group_Activity_Recognition/models/baseline1/outputs/trained_model/"
        os.makedirs(model_save_dir, exist_ok=True)

        # Save the model with timestamp and epoch number
        model_path = os.path.join(model_save_dir, f"model_{timestamp}_epoch{epoch_number}")
        torch.save(model.state_dict(), model_path)
    
    epoch_number += 1
