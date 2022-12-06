#https://qiskit.org/textbook/ch-machine-learning/machine-learning-qiskit-pytorch.html
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision
import qiskit

plt.ion()

TORCH_DATA_ROOT = os.path.expanduser('~/torch_data')

class QuantumCircuit:
    def __init__(self, n_qubits, backend, shots):
        self._circuit = qiskit.QuantumCircuit(n_qubits)

        all_qubits = list(range(n_qubits))
        self.theta = qiskit.circuit.Parameter('theta')

        self._circuit.h(all_qubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, all_qubits)

        self._circuit.measure_all()

        self.backend = backend
        self.shots = shots

    def run(self, thetas):
        t_qc = qiskit.transpile(self._circuit, self.backend)
        qobj = qiskit.assemble(t_qc, shots=self.shots, parameter_binds=[{self.theta: x} for x in thetas])
        result = self.backend.run(qobj).result().get_counts()
        expectation = np.array([sum(int(k)*v for k,v in result.items())/self.shots])
        return expectation


aer_simulator = qiskit.Aer.get_backend('aer_simulator')

circuit = QuantumCircuit(1, aer_simulator, 100)
print('Expected value for rotation pi:', circuit.run([np.pi])[0])
# circuit._circuit.draw()


class HybridFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, quantum_circuit, shift):
        """ Forward pass computation """
        ctx.shift = shift
        ctx.quantum_circuit = quantum_circuit

        expectation_z = ctx.quantum_circuit.run(input[0].tolist())
        result = torch.tensor([expectation_z])
        ctx.save_for_backward(input, result)

        return result

    @staticmethod
    def backward(ctx, grad_output):
        """ Backward pass computation """
        input, expectation_z = ctx.saved_tensors
        input_list = np.array(input.tolist())

        shift_right = input_list + np.ones(input_list.shape) * ctx.shift
        shift_left = input_list - np.ones(input_list.shape) * ctx.shift

        gradients = []
        for i in range(len(input_list)):
            expectation_right = ctx.quantum_circuit.run(shift_right[i])
            expectation_left  = ctx.quantum_circuit.run(shift_left[i])

            gradient = torch.tensor([expectation_right]) - torch.tensor([expectation_left])
            gradients.append(gradient)
        gradients = np.array([gradients]).T
        return torch.tensor([gradients]).float() * grad_output.float(), None, None

class Hybrid(torch.nn.Module):
    """ Hybrid quantum - classical layer definition """

    def __init__(self, backend, shots, shift):
        super(Hybrid, self).__init__()
        self.quantum_circuit = QuantumCircuit(1, backend, shots)
        self.shift = shift

    def forward(self, input):
        return HybridFunction.apply(input, self.quantum_circuit, self.shift)


# Concentrating on the first 100 samples
n_samples = 100
tmp0 = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
X_train = torchvision.datasets.MNIST(root=TORCH_DATA_ROOT, train=True, download=True, transform=tmp0)
# Leaving only labels 0 and 1
idx = np.concatenate([np.where(X_train.targets==x)[0][:n_samples] for x in (0,1)])
X_train.data = X_train.data[idx]
X_train.targets = X_train.targets[idx]
train_loader = torch.utils.data.DataLoader(X_train, batch_size=1, shuffle=True)


n_samples_show = 6
fig, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(10, 3))
for ind0,(image,label) in zip(range(n_samples_show),train_loader):
    axes[ind0].imshow(image[0,0].numpy(), cmap='gray') #(0,0) for (batch,channel)
    axes[ind0].set_xticks([])
    axes[ind0].set_yticks([])
    axes[ind0].set_title(f"Labeled: {label.item()}")


n_samples = 50
tmp0 = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
X_test = torchvision.datasets.MNIST(root=TORCH_DATA_ROOT, train=False, download=True, transform=tmp0)
idx = np.concatenate([np.where(X_test.targets==x)[0][:n_samples] for x in (0,1)])
X_test.data = X_test.data[idx]
X_test.targets = X_test.targets[idx]
test_loader = torch.utils.data.DataLoader(X_test, batch_size=1, shuffle=True)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(6, 16, kernel_size=5)
        self.dropout = torch.nn.Dropout2d()
        self.fc1 = torch.nn.Linear(256, 64)
        self.fc2 = torch.nn.Linear(64, 1)
        self.hybrid = Hybrid(aer_simulator, 100, np.pi / 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = x.view(1, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.hybrid(x)
        return torch.cat((x, 1 - x), -1)


model = Net()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_func = torch.nn.NLLLoss()

epochs = 20
loss_list = []

model.train()
for epoch in range(epochs):
    total_loss = []
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())
    loss_list.append(sum(total_loss)/len(total_loss))
    print('Training [{:.0f}%]\tLoss: {:.4f}'.format(
        100. * (epoch + 1) / epochs, loss_list[-1]))


fig,ax = plt.subplots()
ax.plot(loss_list)
ax.set_title('Hybrid NN Training Convergence')
ax.set_xlabel('Training Iterations')
ax.set_ylabel('Neg Log Likelihood Loss')


model.eval()
with torch.no_grad():

    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        output = model(data)

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        loss = loss_func(output, target)
        total_loss.append(loss.item())

    print('Performance on test data:\n\tLoss: {:.4f}\n\tAccuracy: {:.1f}%'.format(
        sum(total_loss) / len(total_loss),
        correct / len(test_loader) * 100)
        )


n_samples_show = 6
count = 0
fig, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(10, 3))

model.eval()
with torch.no_grad():
    for batch_idx, (data, target) in enumerate(test_loader):
        if count == n_samples_show:
            break
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        axes[count].imshow(data[0].numpy().squeeze(), cmap='gray')
        axes[count].set_xticks([])
        axes[count].set_yticks([])
        axes[count].set_title('Predicted {}'.format(pred.item()))
        count += 1
