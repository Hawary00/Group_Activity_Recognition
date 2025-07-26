import os 
import sys
# Automatically find the project root (assuming train.py is inside models/base_line1/)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Add the project root to sys.path
sys.path.append(project_root)
# Now, you can import your modules normally
from data.data_loader import Person_Activity_DataSet

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import PersonActivityTemporalClassifier
import albumentations as A
from albumentations.pytorch import ToTensorV2






# Step1: Data Augmentation and Transformations
train_transformers = A.Compose([
        A.Resize(224, 224),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7)),
            A.ColorJitter(brightness=0.2),
            A.RandomBrightnessContrast(),
            A.GaussNoise()
        ], p=0.55),
        A.OneOf([
            A.HorizontalFlip(),
            A.VerticalFlip(),
        ], p=0.05),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

val_transformers = A.Compose([
        A.Resize(224, 224),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

# test_transformers = A.Compose([
    #     A.Resize(224, 224),
    #     A.Normalize(
    #         mean=[0.485, 0.456, 0.406],
    #         std=[0.229, 0.224, 0.225]
    #     ),
    #     ToTensorV2()
    # ])

PROJECT_ROOT = r"/mnt/New Volume/Deep Learning. DR Mostafa/05- Project 1 - CV/volleyball-datasets"
sys.path.append(PROJECT_ROOT)
person_activity_clases = ["Waiting", "Setting", "Digging", "Falling", "Spiking", "Blocking", "Jumping", "Moving", "Standing"]
person_activity_labels = {class_name.lower():i for i, class_name in enumerate(person_activity_clases)}


# Step 2: prepare dataset (make it read for dataloader)
train_dataset = Person_Activity_DataSet(
    videos_path=f"{PROJECT_ROOT}/volleyball_/videos",
    annot_path=f"{PROJECT_ROOT}/annot_all.pkl",
    seq=True,
    split=[1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54],
    labels=person_activity_labels,
    transform=train_transformers,
    
)


val_dataset = Person_Activity_DataSet(
    videos_path= f"{PROJECT_ROOT}/volleyball_/videos",
    annot_path= f"{PROJECT_ROOT}/annot_all.pkl",
    seq=True,
    labels=person_activity_labels,
    split=[0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51],
    transform=val_transformers,
   
    

)
from model import person_collate_fn

# Step 3 DataLoader
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    collate_fn=person_collate_fn
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    collate_fn=person_collate_fn

)


# Step 4 define loss function and optimizer
# Loss Function
loss_fun = torch.nn.CrossEntropyLoss(ignore_index=-100)
# Optimizer
# optimizer = torch.optim.AdamW(model.prameters())

# define Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Model
model = PersonActivityTemporalClassifier(num_classes=9)
model.to(device)



optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0001)


# Automatic Mixed Precision (AMP) to reduce memory usage.
scaler = torch.amp.GradScaler('cuda')  # Add this before training loop


# The Training Loop
def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0
    last_loss = 0
    correct_prediction = 0
    total_samples = 0

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting

    for i, data in enumerate(train_loader):
        # Every data instance is an input + label pair
        input, labels = data
        input, labels = input.to(device), labels.to(device)

        # Zero gradints for every batch
        optimizer.zero_grad()

        # with torch.cuda.amp.autocast():  # Enable AMP
        with torch.amp.autocast('cuda'):  # Enable AMP
            outputs = model(input)
            loss = loss_fun(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Compute Accuracy 
        predicted = outputs.argmax(1) # Get the predicted class
        class_labels = labels.argmax(1) # Ensure labels have shape [batch_size]

        correct_prediction += (predicted == class_labels).sum().item()
        total_samples += labels.size(0) 

        # Gather data and report
        running_loss +=loss.item()
        if i % 500 ==499:
            last_loss = running_loss / 500 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    # Calculate final accuracy
    train_accuracy = 100.0 * correct_prediction / total_samples
    print(f"Epoch {epoch_index + 1} Summary")

    return last_loss, train_accuracy




timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Define log directory for TensorBoard
log_dir = f"/mnt/New Volume/Deep Learning. DR Mostafa/Group_Activity_Recognition/models/baseline5/outputs/tensorbord/baseline4_{timestamp}"

# Create the directory if it doesn't exist
os.makedirs(log_dir, exist_ok=True)

# Initialize SummaryWriter
writer = SummaryWriter(log_dir)

epoch_number = 0

EPOCHS = 20

best_vloss = 1_000_000.

# Free up GPU memory before starting training
torch.cuda.empty_cache()

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    
    avg_loss, _ = train_one_epoch(epoch_number, writer)


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

            vloss = loss_fun(voutputs, vlabels)
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    # writer.add_scalars('Training vs. Validation Loss',
    #                 { 'Training' : float(avg_loss[0]), 'Validation' : avg_vloss.item() },
    #                 epoch_number + 1)
    writer.add_scalars('Training vs. Validation Loss',
                { 'Training' : float(avg_loss), 'Validation' : avg_vloss.item()},
                epoch_number + 1)

    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        # model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        # torch.save(model.state_dict(), model_path)
        # Define model save directory
        model_save_dir = "/mnt/New Volume/Deep Learning. DR Mostafa/Group_Activity_Recognition/models/baseline5/outputs/trained_model"
        os.makedirs(model_save_dir, exist_ok=True)

        # Save the model with timestamp and epoch number
        model_path = os.path.join(model_save_dir, f"model_{timestamp}_epoch{epoch_number}")
        torch.save(model.state_dict(), model_path)

    epoch_number += 1