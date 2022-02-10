import argparse
import logging
import nni
import torch
import torch.nn.functional as F
import torchvision

logger = logging.getLogger('mnist_AutoML')


class Net(torch.nn.Module):
    def __init__(self, hidden_size):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 20, 5, 1)
        self.conv2 = torch.nn.Conv2d(20, 50, 5, 1)
        self.fc1 = torch.nn.Linear(4*4*50, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for ind_batch, (data_i, label_i) in enumerate(train_loader, start=1):
        data_i, label_i = data_i.to(device), label_i.to(device)
        optimizer.zero_grad()
        predict = model(data_i)
        loss = F.cross_entropy(predict, label_i)
        loss.backward()
        optimizer.step()
        if (ind_batch%args['print_freq']==0) or ind_batch==len(train_loader):
            logger.info(f'[epoch={epoch}][{ind_batch}/{len(train_loader)}] train loss={loss:.4}\n')


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data_i, label_i in test_loader:
            data_i, label_i = data_i.to(device), label_i.to(device)
            predict = model(data_i)
            test_loss += F.cross_entropy(predict, label_i).item()*len(data_i)
            correct += (predict.argmax(dim=1)==label_i).sum().item()
    test_loss = test_loss / len(test_loader.dataset)
    accuracy = 100 * correct / len(test_loader.dataset)
    logger.info(f'[epoch={epoch}] validation loss={test_loss:.4}, accuracy={accuracy:.4}\n')
    return accuracy


def get_params():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument("--data_dir", type=str, default='./data', help="data directory")
    parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument("--hidden_size", type=int, default=512, metavar='N', help='hidden layer size (default: 512)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--print_freq', type=int, default=100, metavar='N', help='how many batches to wait before logging training status')
    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    try:
        args = nni.get_next_parameter()
        logger.debug(args)
        args = vars(nni.utils.merge_parameter(get_params(), args))
        print(args)

        use_cuda = torch.cuda.is_available()
        device = torch.device('cuda' if use_cuda else 'cpu')
        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
        data_dir = args['data_dir']
        tmp0 = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
        trainset = torchvision.datasets.MNIST(data_dir, train=True, download=True, transform=tmp0)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args['batch_size'], shuffle=True, **kwargs)
        tmp0 = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
        testset = torchvision.datasets.MNIST(data_dir, train=False, transform=tmp0)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=True, **kwargs)
        hidden_size = args['hidden_size']

        model = Net(hidden_size=hidden_size).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'])

        for epoch in range(1, args['epochs'] + 1):
            train(args, model, device, train_loader, optimizer, epoch)
            test_acc = test(args, model, device, test_loader)

            nni.report_intermediate_result(test_acc)
            logger.debug(f'test accuracy {test_acc}')

        nni.report_final_result(test_acc)
        logger.debug('Final result is %g', test_acc)

    except Exception as exception:
        logger.exception(exception)
        raise
