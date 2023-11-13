# This serves as a template which will guide you through the implementation of this task.
# First, we import necessary libraries:
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import os
import torch
from torchvision import transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import sys

from torchvision.models import resnet50, ResNet50_Weights
from sklearn.preprocessing import normalize


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def generate_embeddings():
    """
    Transform, resize and normalize the images and then use a pretrained model to extract
    the embeddings.
    """

    # Initialize ResNet50 with the best weights
    # preproc = ResNet50_Weights.DEFAULT

    # Try if  including transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1) gives
    # better score before the preproc

    # train_transforms = transforms.Compose([preproc, transforms.ToTensor()])
    train_transforms = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root="dataset/", transform=train_transforms)
    # Hint: adjust batch_size and num_workers to your PC configuration, so that you don't
    # run out of memory
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=32,
                              shuffle=False,
                              pin_memory=True, num_workers=2)

    # more info here: https://pytorch.org/vision/stable/models.html)

    # Load a pretrained model,
    #  model to access the embeddings the model generates

    # Select ResNet50 as the model
    model = resnet50(pretrained=True)
    # model.load_state_dict(torch.load('weights.pt'))

    # Delete the last layer to obtain the activation maps (AdaptiveAvgPool2d-173 output)
    #modules = list(model.children())[:-1]
    #model = torch.nn.Sequential(*modules)

    embedding_size = 1000  # Dummy variable, replace with the actual embedding size once you
    # pick your model
    num_images = len(train_dataset)
    embeddings = np.zeros((num_images, embedding_size))
    print('BEGINNING; ', embeddings.shape)
    model.eval()

    for i, (images, labels) in enumerate(train_loader):
        with torch.no_grad():
            if i == 0:

                embeddings = model(images).detach().numpy()
                print("i=0 ", embeddings.shape)
            else:
                embeddings = np.concatenate((embeddings, model(images).detach().numpy()), axis=0)
                print("i= ", i, ' EMB: ', embeddings.shape)
    return embeddings
    np.save('dataset/embeddings3.npy', embeddings)

    print("aaaaaa")
    print(embeddings.shape)


def get_data(file, train=True):
    """
    Load the triplets from the file and generate the features and labels.

    input: file: string, the path to the file containing the triplets
          train: boolean, whether the data is for training or testing

    output: X: numpy array, the features
            y: numpy array, the labels
    """
    triplets = []
    with open(file) as f:
        for line in f:
            triplets.append(line)

    # generate training data from triplets
    train_dataset = datasets.ImageFolder(root="dataset/",
                                         transform=None)
    filenames = [s[0].split('\\')[-1].replace('.jpg', '') for s in train_dataset.samples]
    embeddings = np.load('dataset/embeddings3.npy')
    embeddings = embeddings.squeeze()

    embeddings = normalize(embeddings, norm='l2', axis=1)

    file_to_embedding = {}
    for i in range(len(filenames)):
        file_to_embedding[filenames[i]] = embeddings[i]
    X = []
    y = []
    # use the individual embeddings to generate the features and labels for triplets
    for t in triplets:
        emb = [file_to_embedding[a] for a in t.split()]
        X.append(np.hstack([emb[0], emb[1], emb[2]]))
        y.append(1)
        # Generating negative samples (data augmentation)
        if train:
            X.append(np.hstack([emb[0], emb[2], emb[1]]))
            y.append(0)
    X = np.vstack(X)
    y = np.hstack(y)

    return X, y


# Hint: adjust batch_size and num_workers to your PC configuration, so that you don't run out of memory
def create_loader_from_np(X, y=None, train=True, batch_size=32, shuffle=False, num_workers=2):
    """
    Create a torch.utils.data.DataLoader object from numpy arrays containing the data.

    input: X: numpy array, the features
           y: numpy array, the labels

    output: loader: torch.data.util.DataLoader, the object containing the data
    """
    if train:
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float),
                                torch.from_numpy(y).type(torch.long))
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, lengths=(0.7, 0.3))
        tr_loader = DataLoader(dataset=train_dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            pin_memory=True, num_workers=num_workers)
        val_loader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            pin_memory=True, num_workers=num_workers)
        loader = [tr_loader, val_loader]

    else:
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float))

        loader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            pin_memory=True, num_workers=num_workers)
    return loader

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1000*3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        #self.fc3 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(512, 1)
        self.out = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5)
        x = self.fc3(x)
        x = F.sigmoid(x)
        return x


def train_model(loader):
    """
    The training procedure of the model; it accepts the training data, defines the model
    and then trains it.

    input: train_loader: torch.data.util.DataLoader, the object containing the training data

    output: model: torch.nn.Module, the trained model
    """
    model = Net()
    model.load_state_dict(torch.load("best_params.pt"))
    model.train()
    model.to(device)

    n_epochs = 20

    # Define a loss function, optimizer and proceed with training. Hint: use the part
    # of the training data as a validation split. After each epoch, compute the loss on the
    # validation split and print it out. This enables you to see how your model is performing
    # on the validation data before submitting the results on the server. After choosing the
    # best model, train it on the whole training data.

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.001)

    train_loss = 0
    val_loss = 0
    reference = 0.570

    tr_loader = loader[0]
    val_loader = loader[1]

    for epoch in range(n_epochs):
        print(epoch)
        train_loss = 0
        val_loss = 0

        for i, [X, y] in enumerate(tr_loader):
            optimizer.zero_grad()
            #print(X)
            y_pred = model.forward(X)
            y_pred = y_pred.squeeze()
            y = y.to(torch.float32)
            loss_t = criterion(y_pred, y)
            train_loss += loss_t.item()

            loss_t.backward()
            optimizer.step()
        print(f"Training loss : {train_loss / (i + 1)}")

        model.eval()

        for i, [X, y] in enumerate(val_loader):
            optimizer.zero_grad()
            #print(X)
            y_pred = model.forward(X)
            y_pred = y_pred.squeeze()
            y = y.to(torch.float32)
            loss_v = criterion(y_pred, y)
            val_loss += loss_v.item()

        print(f"Validation loss : {val_loss / (i + 1)}")
        avg_val_loss = val_loss / (i+1)
        if avg_val_loss < reference:
            print(f"New best val_loss : {avg_val_loss}")
            reference = avg_val_loss

            torch.save(model.state_dict(), "best_params.pt")


    model.load_state_dict(torch.load("best_params.pt"))

    return model


def test_model(model, loader):
    """
    The testing procedure of the model; it accepts the testing data and the trained model and
    then tests the model on it.

    input: model: torch.nn.Module, the trained model
           loader: torch.data.util.DataLoader, the object containing the testing data

    output: None, the function saves the predictions to a results.txt file
    """
    model.eval()
    predictions = []
    # Iterate over the test data
    with torch.no_grad():  # We don't need to compute gradients for testing
        for [x_batch] in loader:
            x_batch = x_batch.to(device)
            predicted = model(x_batch)
            predicted = predicted.cpu().numpy()
            # Rounding the predictions to 0 or 1
            predicted[predicted >= 0.5] = 1
            predicted[predicted < 0.5] = 0
            predictions.append(predicted)
        predictions = np.vstack(predictions)
    np.savetxt("results.txt", predictions, fmt='%i')


# %%
# Main function. You don't have to change this

if __name__ == '__main__':
    TRAIN_TRIPLETS = 'train_triplets.txt'
    TEST_TRIPLETS = 'test_triplets.txt'

    # generate embedding for each image in the dataset
    if (os.path.exists('dataset/embeddings3.npy') == False):
        generate_embeddings()

    # load the training and testing data
    X, y = get_data(TRAIN_TRIPLETS)
    X_test, _ = get_data(TEST_TRIPLETS, train=False)

    # Create data loaders for the training and testing data
    train_loader = create_loader_from_np(X, y, train=True, batch_size=32)
    test_loader = create_loader_from_np(X_test, train=False, batch_size=2048, shuffle=False)

    # define a model and train it
    model = train_model(train_loader)

    # test the model on the test data
    test_model(model, test_loader)
    print("Results saved to results.txt")


