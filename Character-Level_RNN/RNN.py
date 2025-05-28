import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, random_split
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import string
import unicodedata
import glob
import os
import time
from io import open

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)
print(f"Using device: {device}")

HIDDEN_SIZE = 128
N_EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.15

allowed_characters = string.ascii_letters + " .,;'_"
n_letters = len(allowed_characters)

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
              if unicodedata.category(c) != 'Mn' and c in allowed_characters)

def letter_to_index(letter):
    return allowed_characters.find(letter) if letter in allowed_characters else allowed_characters.find('_')

def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letter_to_index(letter)] = 1
    return tensor

class NamesDataset(Dataset):
    def __init__(self, data_dir):
        self.labels = []
        self.names = []
        self.labels_uniq = []
        
        for filename in glob.glob(os.path.join(data_dir, '*.txt')):
            label = os.path.splitext(os.path.basename(filename))[0]
            if label not in self.labels_uniq:
                self.labels_uniq.append(label)
            with open(filename, 'r', encoding='utf-8') as f:
                names = [unicode_to_ascii(line.strip()) for line in f]
                self.names.extend(names)
                self.labels.extend([label]*len(names))
        
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):
        name_tensor = line_to_tensor(self.names[idx])
        label_idx = self.labels_uniq.index(self.labels[idx])
        return name_tensor, torch.tensor([label_idx], dtype=torch.long)

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CharRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[-1]) 
        return nn.functional.log_softmax(out, dim=1)

def train(model, dataset, epochs, batch_size, lr):
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    losses = []
    for epoch in range(1, epochs+1):
        total_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, labels.squeeze(1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataset)
        losses.append(avg_loss)
        print(f'Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}')
    
    return losses

def evaluate(model, dataset, classes):
    confusion = torch.zeros(len(classes), len(classes))
    correct = 0
    
    with torch.no_grad():
        for inputs, labels in dataset:
            output = model(inputs)
            _, predicted = torch.max(output, 1)
            confusion[labels.item()][predicted.item()] += 1
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / len(dataset)
    print(f'Accuracy: {accuracy:.2%}')
    
    for i in range(len(classes)):
        confusion[i] = confusion[i] / confusion[i].sum()
    
    fig, ax = plt.subplots()
    cax = ax.matshow(confusion.cpu().numpy())
    fig.colorbar(cax)
    
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90)
    ax.set_yticklabels(classes)
    
    plt.show()

if __name__ == "__main__":
    dataset = NamesDataset('data/names')  
    train_set, test_set = random_split(dataset, [0.8, 0.2])
    
    model = CharRNN(n_letters, HIDDEN_SIZE, len(dataset.labels_uniq)).to(device)
    
    print("Початок навчання...")
    losses = train(model, train_set, N_EPOCHS, BATCH_SIZE, LEARNING_RATE)
    
    print("\nОцінка на тестовому наборі:")
    evaluate(model, test_set, dataset.labels_uniq)
    
    def predict(name):
        tensor = line_to_tensor(name)
        output = model(tensor)
        _, predicted = torch.max(output, 1)
        return dataset.labels_uniq[predicted.item()]
    
    print("\nТестування на прикладах:")
    test_names = ["Zhang", "Sato", "Müller", "Nowak"]
    for name in test_names:
        print(f"{name} → {predict(name)}")