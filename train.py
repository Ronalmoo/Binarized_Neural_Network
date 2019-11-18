import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from models.resnet import ResNet50
import torchvision
from torchvision.transforms import transforms
from tqdm import tqdm


def train(loss_fn, model, trn_dl, optimizer, device, epochs):
    # write your codes here
    if not model.training:
        model.train()

    for epoch in tqdm(range(epochs), total=epochs):
        tr_loss = 0
        correct_count = 0

        for step, (data, target) in tqdm(enumerate(trn_dl), total=len(trn_dl)):
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(data)

            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            correct_count += (output.max(dim=-1).indices == target).sum().item()
            tr_loss += loss.item()
        else:
            tr_acc = correct_count / len(trn_dl.dataset)
            tr_loss /= (step + 1)
            tqdm.write('epoch: {:3}, tr_acc: {:.2%}, tr_loss: {:.3f}'.format(epoch, tr_acc, tr_loss))


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='LeNet5')
parser.add_argument('--train_batch_size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_model', action='store_true', default=False,
                    help='For Saving the current Model')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
args = parser.parse_args()
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

model = ResNet50()
device = "cuda:0" if args.cuda else "cpu"
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
criterion = nn.CrossEntropyLoss()
tr_dl = trainloader


if __name__ == '__main__':
    train(criterion, model, tr_dl, optimizer, device, 5)

