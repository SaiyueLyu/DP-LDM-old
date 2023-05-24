import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import random
from tqdm import tqdm


def set_seeds(rank, seed):
    random.seed(rank + seed)
    torch.manual_seed(rank + seed)
    np.random.seed(rank + seed)
    torch.cuda.manual_seed(rank + seed)
    torch.cuda.manual_seed_all(rank + seed)
    torch.backends.cudnn.benchmark = True


class MLP(nn.Module):

    def __init__(self, img_dim=1024, num_classes=10):
        super(MLP, self).__init__()
        self.net = torch.nn.Sequential(
            nn.Linear(img_dim, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes),
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        return F.log_softmax(self.net(x), dim=1)

    def pred(self, x):
        x = x.reshape(x.shape[0], -1)
        return F.softmax(self.net(x), dim=1)


class LogReg(nn.Module):

    def __init__(self, img_dim=1024, num_classes=10):
        super(LogReg, self).__init__()
        self.net = torch.nn.Sequential(
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        return F.log_softmax(self.net(x), dim=1)

    def pred(self, x):
        x = x.reshape(x.shape[0], -1)
        return F.softmax(self.net(x), dim=1)



# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.fc1 = nn.Linear(32 * 8 * 8, 64)
#         self.fc2 = nn.Linear(64, 10)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = nn.functional.relu(x)
#         x = self.pool(x)
#         x = self.conv2(x)
#         x = nn.functional.relu(x)
#         x = self.pool(x)
#         x = x.view(-1, 32 * 8 * 8)
#         x = self.fc1(x)
#         x = nn.functional.relu(x)
#         x = self.fc2(x)
#         return x

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.model = torch.nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(32, 64, 3, 1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(64, 128, 3, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.model(x)



def train_cnn(loader1, loader2, loader3, loader4, device, max_epochs=50):
    model = CNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    objective = nn.CrossEntropyLoss()

    return train_model(loader1, loader2, loader3, loader4, device, model, optimizer, objective, max_epochs)


def train_mlp(loader1, loader2, loader3, loader4, device, max_epochs=50):
    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    objective = lambda x, y: F.nll_loss(x, y)

    return train_model(loader1, loader2, loader3, loader4, device, model, optimizer, objective, max_epochs)


def train_log_reg(loader1, loader2, loader3, loader4, device, max_epochs=50):
    model = LogReg().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    objective = lambda x, y: F.nll_loss(x, y)

    return train_model(loader1, loader2, loader3, loader4, device, model, optimizer, objective, max_epochs)


def train_model(loader1, loader2, loader3, loader4, device, model, optimizer, objective, max_epochs):
    best_acc = best_train_acc = best_test_acc = 0.

    for _ in tqdm(range(max_epochs)):
        for _, (train_x, train_y) in enumerate(loader1):

            x = train_x.to(device).to(torch.float32) * 2. - 1.
            y = train_y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = objective(outputs, y)
            loss.backward()
            optimizer.step()

        model.eval()
        acc, _ = compute_acc(model, loader2, device)
        if acc > best_acc:
            best_acc = acc
            best_train_acc, _ = compute_acc(model, loader3, device)
            best_test_acc, _ = compute_acc(model, loader4, device)
        model.train()

    return best_train_acc, best_test_acc
    # return best_test_acc


def compute_acc(model, loader, device):
    test_loss = 0
    correct = 0
    outputs = []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            data = data.to(torch.float32)
            data = data * 2. - 1.
            output = model(data)
            output = nn.functional.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().cpu().item()
            outputs.append(output)
    preds = torch.cat(outputs, dim=0)
    test_loss /= loader.dataset.__len__()
    acc = correct / loader.dataset.__len__()
    return acc, preds


def train_all_classifiers(train_set_loader, test_set_loader, device, batch_size):
    train_dataset = torchvision.datasets.MNIST(
        root='data/mnist/', train=True, download=True, transform=torchvision.transforms.Compose(
            [torchvision.transforms.Resize(32), torchvision.transforms.ToTensor()]))
    test_dataset = torchvision.datasets.MNIST(
        root='data/mnist/', train=False, download=True, transform=torchvision.transforms.Compose(
            [torchvision.transforms.Resize(32), torchvision.transforms.ToTensor()]))

    train_dataset_loader = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=False, batch_size=batch_size)
                                                       # pin_memory=True, num_workers=1)
    test_dataset_loader = torch.utils.data.DataLoader(dataset=test_dataset, shuffle=False, batch_size=batch_size)
                                                      # pin_memory=True, num_workers=1)

    train_cnn_acc, test_cnn_acc = train_cnn(
        train_set_loader, test_set_loader, train_dataset_loader, test_dataset_loader, device)
    # train_cnn_acc, test_cnn_acc = 0,0
    train_mlp_acc, test_mlp_acc = train_mlp(
        train_set_loader, test_set_loader, train_dataset_loader, test_dataset_loader, device)
    # train_mlp_acc, test_mlp_acc = 0,0
    train_log_rec_acc, test_log_rec_acc = train_log_reg(
        train_set_loader, test_set_loader, train_dataset_loader, test_dataset_loader, device)
    # train_log_rec_acc, test_log_rec_acc = 0,0
    return train_cnn_acc, test_cnn_acc, train_mlp_acc, test_mlp_acc, train_log_rec_acc, test_log_rec_acc

    # test_cnn_acc = train_cnn(train_set_loader, test_set_loader, train_dataset_loader, test_dataset_loader, device)
    # test_mlp_acc = train_mlp(train_set_loader, test_set_loader, train_dataset_loader, test_dataset_loader, device)
    # test_log_rec_acc = train_log_reg(train_set_loader, test_set_loader, train_dataset_loader, test_dataset_loader, device)
    # return test_cnn_acc,  test_mlp_acc,  test_log_rec_acc


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        super().__init__()

        self.path = path
        print("Compute the accuracy of : ", path)
        self.img = torch.load(self.path)["image"]
        self.label = torch.load(self.path)["class_label"]
        self.transform = transform

    def __getitem__(self, idx):
        image = self.img[idx]

        if self.transform is not None:
            image = self.transform(image)

        if self.label is not None:
            return image, self.label[idx]
        else:
            return image

    def __len__(self):
        return len(self.img)


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        super().__init__()

        self.path = path
        print("Compute the accuracy of : ", path)
        self.img = torch.load(self.path)["image"]
        self.label = torch.load(self.path)["class_label"]
        self.transform = transform

    def __getitem__(self, idx):
        image = self.img[idx]

        if self.transform is not None:
            image = self.transform(image)

        if self.label is not None:
            return image, self.label[idx]
        else:
            return image

    def __len__(self):
        return len(self.img)


def main(args):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'

    # path = "sampling/conditional_3_ch_mnist_samples.pt"
    # dataset = torch.load(path)
    # assert len(dataset["image"]) == len(dataset["class_label"])
    # split = int(len(dataset["image"])*2 /3) # split at 2/3 place
    # train_dataset = {"image": dataset["image"][: split],
    #                  "class_label": dataset["class_label"][: split] }
    # eval_dataset = {"image": dataset["image"][split:],
    #                  "class_label": dataset["class_label"][split:] }

    train_dataset = TrainDataset(path="sampling/conditional_1_ch_mnist_samples_train_50k.pt")
    eval_dataset = TestDataset(path="sampling/conditional_1_ch_mnist_samples_test_10k.pt")
    train_queue = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_queue = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=args.batch_size, shuffle=True)

    train_cnn_acc, test_cnn_acc, train_mlp_acc, test_mlp_acc, train_log_rec_acc, test_log_rec_acc = train_all_classifiers(
        train_queue, eval_queue, device, args.batch_size)
    # test_cnn_acc, test_mlp_acc,  test_log_rec_acc = train_all_classifiers(train_queue, eval_queue, device, args.batch_size)
    print('Log reg test acc: %.4f %.4f' % (train_log_rec_acc, test_log_rec_acc))
    print('MLP test acc: %.4f %.4f' % (train_mlp_acc, test_mlp_acc))
    print('CNN test acc: %.4f %.4f' % (train_cnn_acc, test_cnn_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', type=str, choices=['mnist28', 'fmnist28'], required=True)
    # parser.add_argument('--train_path', type=str, required=True)
    # parser.add_argument('--eval_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()

    set_seeds(0, 0)

    main(args)