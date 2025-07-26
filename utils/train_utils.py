import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import os


# # Step 4 define loss function and optimizer
# # Loss Function
# loss_fun = torch.nn.CrossEntropyLoss()
# # Optimizer
# # optimizer = torch.optim.AdamW(model.prameters())

# # define Device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001,  weight_decay=0.0001)


# # Define Model
# model = b1_classifier(num_classes=8)
# # Training loop

def train_one_epoch(epoch_index, model, loss_fun, optimizer, device, train_loader,tb_writer, use_AMP=False):
    running_loss = 0
    last_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    torch.cuda.empty_cache()
    scaler = torch.amp.GradScaler('cuda')  # Add this before training loop

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting

    for i, data in enumerate(train_loader):
        # data.to(device)
        input, labels = data
        input, labels = input.to(device), labels.to(device)


        # Zero your gradients for every batch!
        optimizer.zero_grad()

        if use_AMP:
            with torch.amp.autocast('cuda'):  # Enable AMP
                outputs = model(input)
                loss = loss_fun(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        else:
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
        class_labels = labels  # Ensure labels have shape [batch_size]
        

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
# log_dir = os.path.expanduser('/mnt/New Volume/Deep Learning. DR Mostafa/Group_activity_project/models/models/baseline 1//outputs/tensorbord')
# writer = SummaryWriter(log_dir.format(timestamp))
# epoch_number = 0

# Epochs = 5

# best_vloss = 1_000_000.

# Free up GPU memory before starting training
torch.cuda.empty_cache()


def trian(model, Epochs, val_loader, device, loss_fun, optimizer,
           train_loader, save_dir, writer, use_AMP, epoch_number = 0, ):
    torch.cuda.empty_cache()
    print("Training started")

    for epoch in range(Epochs):
        print('Epochs {}:'.format(epoch_number + 1))


        # Make sure gradient tracking is on, and do a pss over the data
        model.train(True)

        avg_loss, train_accuracy = train_one_epoch(epoch_number, model, loss_fun,
                                                    optimizer, device, train_loader,writer, use_AMP)

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
        vclass_labels = vlabels  # Ensure labels have shape [batch_size]
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
        # if avg_vloss < best_vloss:
        #     best_vloss = avg_vloss
        if True:
            model_path = os.path.join(save_dir, "model_{}_{}.pth".format(timestamp, epoch_number))

            torch.save(model.state_dict(), model_path)
        
        epoch_number += 1

    return val_accuracy, avg_vloss, avg_loss, train_accuracy