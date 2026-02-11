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
# Load Pretrained Model and Modify for Transfer Learning
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
model.fc = nn.Linear(model.fc.in_features, 10)

model = model.to(device)



# Modify the final fully connected layer to match the dataset classes
transform = transforms.Compose([
    transforms.Resize((128, 128)),   # Smaller than 224 → Faster
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)

test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class_names = train_dataset.classes

# data procession:
transform = transforms.Compose([
    transforms.Resize((128, 128)),   # Smaller than 224 → Faster
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)

test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class_names = train_dataset.classes



# Include the Loss function and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)



# Train the model
train_loss_list = []
val_loss_list = []

epochs = 3   # Reduced for speed

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    train_loss = running_loss / len(train_loader)
    train_loss_list.append(train_loss)
    
    # Validation
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    
    val_loss = val_loss / len(test_loader)
    val_loss_list.append(val_loss)
    
    print(f"Epoch [{epoch+1}/{epochs}] "
          f"Train Loss: {train_loss:.4f} "
          f"Val Loss: {val_loss:.4f}")



```

## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot
Include your plot here
<img width="1025" height="781" alt="image" src="https://github.com/user-attachments/assets/a0de1b1a-335c-4bd7-8736-796d3d459efc" />


### Confusion Matrix
Include confusion matrix here
<img width="893" height="758" alt="image" src="https://github.com/user-attachments/assets/47e2c25b-2b99-40bb-a032-020512eda883" />


### Classification Report
Include Classification Report here
<img width="1054" height="459" alt="image" src="https://github.com/user-attachments/assets/442cb7aa-0597-493b-a8a0-df3ae903a48a" />


### New Sample Prediction
<img width="1745" height="655" alt="image" src="https://github.com/user-attachments/assets/0cac03a7-1b9f-48f6-acee-16488049c3e5" />
>

## RESULT
The pre-trained ResNet18 model was successfully fine-tuned using transfer learning for CIFAR-10 classification. The modified final layer enabled classification into 10 classes. The model achieved good performance with reduced training time due to freezing pre-trained layers.
