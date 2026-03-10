# Implementation-of-Transfer-Learning
## Aim
To Implement Transfer Learning for classification using VGG-19 architecture.
## Problem Statement and Dataset
Include the problem statement and Dataset
### Problem Statement

To classify images into 10 object categories using Transfer Learning with a pre-trained VGG-19 model.
The objective is to:
1.Use a pre-trained VGG-19 model (trained on ImageNet)
2.Freeze convolution layers
3.Modify the final fully connected layer
4.Train only the classifier part
5.Evaluate performance using loss plot, confusion matrix, and classification report
### Dataset Used: CIFAR-10
We used the CIFAR-10 dataset, which is automatically downloaded in Google Colab using PyTorch.
Dataset Details:
Total Images: 60,000
Training Images: 50,000
Test Images: 10,000
Image Size: 32×32 (Resized to 224×224 for VGG-19)
Number of Classes: 10


## DESIGN STEPS
### STEP 1:
Load and preprocess the dataset (resize to 224×224, normalize images, create DataLoaders).

### STEP 2:
Load the pretrained VGG-19 model and freeze the convolutional layers.

### STEP 3:
Modify the final fully connected layer to match the number of dataset classes.

### STEP 4:
Define the loss function (CrossEntropyLoss) and optimizer (Adam).

### STEP 5:
Train the model, evaluate using validation data, and compute performance metrics (Loss plot, Confusion Matrix, Classification Report).

## PROGRAM
Include your code here
```python
model = models.vgg19(pretrained=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

from torchsummary import summary
summary(model, input_size=(3, 224, 224))

# Modify final layer
num_classes = len(train_dataset.classes)

model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

model = model.to(device)

summary(model, input_size=(3, 224, 224))


for param in model.features.parameters():
    param.requires_grad = False

# Define Loss Function
criterion = nn.CrossEntropyLoss()

# Define Optimizer (ONLY classifier parameters)
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
def train_model(model, train_loader,test_loader,num_epochs=10):
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        # Compute validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_losses.append(val_loss / len(test_loader))
        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    # Plot training and validation loss
    print("Name:  SHIVASRI      ")
    print("Register Number:   212224220098     ")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()



```

## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot
Include your plot here

<img width="935" height="752" alt="image" src="https://github.com/user-attachments/assets/700457ff-04ac-4830-9427-0b3dbca89680" />



### Confusion Matrix
Include confusion matrix here

<img width="840" height="607" alt="image" src="https://github.com/user-attachments/assets/420fcaf5-3b3b-4732-ba45-5c36a04fd499" />


### Classification Report
Include Classification Report here

<img width="761" height="225" alt="image" src="https://github.com/user-attachments/assets/529692fd-2042-44a7-9787-c389c114e458" />



### New Sample Prediction

<img width="506" height="799" alt="image" src="https://github.com/user-attachments/assets/d84841cb-9be7-49c1-aa15-85a512aeb512" />

>

## RESULT
The pre-trained ResNet18 model was successfully fine-tuned using transfer learning for CIFAR-10 classification. The modified final layer enabled classification into 10 classes. The model achieved good performance with reduced training time due to freezing pre-trained layers.
