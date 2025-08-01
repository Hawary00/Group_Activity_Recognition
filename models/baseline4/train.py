import os 
import sys
# Automatically find the project root (assuming train.py is inside models/base_line1/)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Add the project root to sys.path
sys.path.append(project_root)
# Now, you can import your modules normally
from data.data_loader import Group_Activity_DataSet

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import GroupTemporalClassifier





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


# Step 2: prepare dataset (make it read for dataloader)
train_dataset = Group_Activity_DataSet(
    videos_path= f"{PROJECT_ROOT}/volleyball_/videos",
    annot_path= f"{PROJECT_ROOT}/annot_all.pkl",
    labels=group_activity_labels,
    split=[1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54],
    transform=train_transformers,
    crops=False,
    seq=True,

)

val_dataset = Group_Activity_DataSet(
    videos_path= f"{PROJECT_ROOT}/volleyball_/videos",
    annot_path= f"{PROJECT_ROOT}/annot_all.pkl",
    labels=group_activity_labels,
    split=[0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51],
    transform=val_transformers,
    crops=False,
    seq=True,

)

test_dataset = Group_Activity_DataSet(
    videos_path= f"{PROJECT_ROOT}/volleyball_/videos",
    annot_path= f"{PROJECT_ROOT}/annot_all.pkl",
    labels=group_activity_labels,
    split=[4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47],
    transform=test_transformers,
    crops=False,
    seq=True,

)




# Step 3 DataLoader
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    pin_memory=True

)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    pin_memory=True

)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)


# Step 4 define loss function and optimizer
# Loss Function
loss_fn = torch.nn.CrossEntropyLoss()
# Optimizer
# optimizer = torch.optim.AdamW(model.prameters())

# define Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Model
model = GroupTemporalClassifier(num_classes=8,
                                 input_size=2048, hidden_size=512, num_layers=1)
model.to(device)



optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

# Training loop
def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(train_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)
        # outputs = outputs.mean(dim=1)      # average over time → [B, C]


        # Compute the loss and its gradients
        # Ensure labels are 1D class indices for CrossEntropyLoss
        if labels.ndim == 2 and labels.shape[1] == 1:
            labels = labels.squeeze(1)
        elif labels.ndim == 2 and labels.shape[1] > 1:
            labels = torch.argmax(labels, dim=1)  # assume one-hot or multi-label
        elif labels.ndim > 2:
            raise ValueError(f"Unexpected label shape: {labels.shape}")
        labels = labels.long()

        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 100 == 99:
            last_loss = running_loss / 100 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss



timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Define log directory for TensorBoard
log_dir = f"/mnt/New Volume/Deep Learning. DR Mostafa/Group_Activity_Recognition/models/baseline4/outputs/tensorbord/baseline4_{timestamp}"

# Create the directory if it doesn't exist
os.makedirs(log_dir, exist_ok=True)

# Initialize SummaryWriter
writer = SummaryWriter(log_dir)

epoch_number = 0

EPOCHS = 35

best_vloss = 1_000_000.

# Free up GPU memory before starting training
torch.cuda.empty_cache()

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)


    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(val_loader):
            vinputs, vlabels = vdata


            vinputs, vlabels = vinputs.to(device), vlabels.to(device)
            voutputs = model(vinputs)

            if vlabels.ndim == 2 and vlabels.shape[1] == 1:
                vlabels = vlabels.squeeze(1)
            elif vlabels.ndim == 2 and vlabels.shape[1] > 1:
                vlabels = torch.argmax(vlabels, dim=1)
            elif vlabels.ndim > 2:
                raise ValueError(f"Unexpected label shape: {vlabels.shape}")
            vlabels = vlabels.long()

            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        # model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        # torch.save(model.state_dict(), model_path)
        # Define model save directory
        model_save_dir = "/mnt/New Volume/Deep Learning. DR Mostafa/Group_Activity_Recognition/models/baseline4/outputs/trained_model"
        os.makedirs(model_save_dir, exist_ok=True)

        # Save the model with timestamp and epoch number
        model_path = os.path.join(model_save_dir, f"model_{timestamp}_epoch{epoch_number}")
        torch.save(model.state_dict(), model_path)

    epoch_number += 1