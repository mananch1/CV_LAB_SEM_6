import matplotlib
matplotlib.use("Agg")
from livenessnet import LivenessNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

class FaceDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        img = self.data[idx]
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
    ap.add_argument("-m", "--model", type=str, required=True, help="path to trained model")
    ap.add_argument("-l", "--le", type=str, required=True, help="path to label encoder")
    ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to output loss/accuracy plot")
    args = vars(ap.parse_args())

    INIT_LR = 1e-4
    BS = 8
    EPOCHS = 50

    print("[INFO] loading images...")
    imagePaths = glob.glob(os.path.join(args["dataset"], "*", "*.*"))
    data = []
    labels = []

    for imagePath in imagePaths:
        label = imagePath.split(os.path.sep)[-2]
        image = cv2.imread(imagePath)
        if image is None: continue
        image = cv2.resize(image, (32, 32))
        data.append(image)
        labels.append(label)

    data = np.array(data)

    le = LabelEncoder()
    labels = le.fit_transform(labels)

    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

    train_transforms = transforms.Compose([
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.85, 1.15), shear=15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    val_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = FaceDataset(trainX, trainY, transform=train_transforms)
    val_dataset = FaceDataset(testX, testY, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BS, shuffle=False)

    print("[INFO] compiling model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LivenessNet.build(width=32, height=32, depth=3, classes=len(le.classes_)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=INIT_LR, weight_decay=INIT_LR/EPOCHS)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    print(f"[INFO] training network for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = correct / total
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
                if epoch == EPOCHS - 1:
                    all_preds.extend(predicted.cpu().numpy())
                
        epoch_val_loss = val_loss / len(val_dataset)
        epoch_val_acc = val_correct / val_total
        
        train_losses.append(epoch_loss)
        val_losses.append(epoch_val_loss)
        train_accs.append(epoch_acc)
        val_accs.append(epoch_val_acc)
        
        print(f"Epoch {epoch+1}/{EPOCHS} - loss: {epoch_loss:.4f} - accuracy: {epoch_acc:.4f} - val_loss: {epoch_val_loss:.4f} - val_accuracy: {epoch_val_acc:.4f}")

    print("[INFO] evaluating network...")
    print(classification_report(testY, all_preds, target_names=le.classes_))

    print(f"[INFO] serializing network to '{args['model']}'...")
    torch.save(model.state_dict(), args["model"])

    with open(args["le"], "wb") as f:
        f.write(pickle.dumps(le))

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, EPOCHS), train_losses, label="train_loss")
    plt.plot(np.arange(0, EPOCHS), val_losses, label="val_loss")
    plt.plot(np.arange(0, EPOCHS), train_accs, label="train_acc")
    plt.plot(np.arange(0, EPOCHS), val_accs, label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(args["plot"])
