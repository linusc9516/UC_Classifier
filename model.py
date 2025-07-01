import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.onnx
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from PIL import Image

# Data augmentation for better generalization
data_transforms = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
}

data_dir = './2class/'
all_data = datasets.ImageFolder(data_dir, data_transforms['train'])

images_paths = [item[0] for item in all_data.imgs]
labels = [item[1] for item in all_data.imgs]

train_paths, test_paths, train_labels, test_labels = train_test_split(images_paths, labels, test_size=0.2, stratify=labels, random_state=42)

# # Custom Dataset Class
# This class will handle loading images and their corresponding labels
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx])
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Run the trainin code
if __name__ == "__main__":
    train_dataset = CustomDataset(train_paths, train_labels, transform=data_transforms['train'])
    val_dataset = CustomDataset(test_paths, test_labels, transform=data_transforms['val'])

    class_counts = [6105, 5166]
    weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = [weights[label] for _, label in train_dataset]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    batch_size = 32

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=sampler,
        shuffle=False,
        num_workers=4
    )

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    # Unfreeze the last two layers (layer3 and layer4) for fine-tuning
    for param in model.parameters():
        param.requires_grad = False

    for param in model.layer3.parameters():
        param.requires_grad = True

    for param in model.layer4.parameters():
        param.requires_grad = True 

    num_classes = 2
    
    # Modify the final layer to match the number of classes
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )

    # Custom Focal Loss implementation for better handling of class imbalance
    # Unused in the final code, but can be used if needed
    class FocalLoss(nn.Module):
        def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
            super(FocalLoss, self).__init__()
            self.gamma = gamma
            self.alpha = alpha
            self.reduction = reduction

        def forward(self, inputs, targets):
            ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
            
            if self.reduction == 'mean':
                return focal_loss.mean()
            elif self.reduction == 'sum':
                return focal_loss.sum()
            else:
                return focal_loss

    # Lower learning rate for fine-tuning
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                           lr=1e-4, weight_decay=1e-3)  # Reduced LR and weight decay
    
    # Use simpler, more stable scheduler
    criterion = nn.CrossEntropyLoss()
    # criterion = FocalLoss(gamma=2.0, alpha=0.25)  # Try this if class imbalance is severe (ie. in 4classifier model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # More epochs and patience for better training
    num_epochs = 20
    best_val_acc = 0.0
    patience = 7
    patience_counter = 0
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            
            # Add gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            
            # Track training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        scheduler.step()
        
        # Calculate training accuracy
        train_acc = 100 * correct_train / total_train
        avg_loss = running_loss / len(train_loader)
        
        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print(f'Training - Loss: {avg_loss:.4f}, Accuracy: {train_acc:.2f}%')
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        class_names = ["Normal", "Ulcerative Colitis"]  # More descriptive names
        
        val_acc = 100 * correct / total
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'Validation - Loss: {avg_val_loss:.4f}, Accuracy: {val_acc:.2f}%')
        print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
        print(f'Learning Rate: {scheduler.get_last_lr()[0]}')
        print("-" * 50)
        
        # Flexible early stopping condition
        improvement_threshold = 0.1  # Minimum improvement to reset patience
        if val_acc > best_val_acc + improvement_threshold:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
                'val_acc': val_acc,
                'train_acc': train_acc,
            }, 'best_model_checkpoint.pth')
            print(f"✓ New best validation accuracy: {best_val_acc:.2f}%")
        else:
            patience_counter += 1
            print(f"No significant improvement. Patience: {patience_counter}/{patience}")
            
        if patience_counter >= patience:
            print("Early stopping triggered!")
            break

    # Saving
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, 'model_checkpoint.pth')

    model.eval()

    example = torch.zeros(1, 3, 224, 224).to(device) 
    traced_script_module = torch.jit.trace(model, example)

    # Exporting the model
    traced_script_module.save("2classifier_rn18_all_unfreeze.pt")
    # torch.onnx.export(model, example, "2classifier.onnx") # onnx is not used in this script, but can be uncommented if needed