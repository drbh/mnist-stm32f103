import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from datetime import datetime


class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.fc = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        return self.fc(x)


def save_weights_to_c_header(model, filename="mnist_weights.h"):
    weights = model.fc.weight.data.cpu().numpy()
    biases = model.fc.bias.data.cpu().numpy()

    with open(filename, "w") as f:
        # write header guard
        header_guard = "MNIST_WEIGHTS_H"
        f.write(f"#ifndef {header_guard}\n")
        f.write(f"#define {header_guard}\n\n")

        # write timestamp and info
        f.write(f"// Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("// Author: David Holtz @drbh\n")
        f.write("// MNIST model weights for digit classification\n")
        f.write("// Input: 28x28 grayscale image (784 pixels), normalized to [0,1]\n")
        f.write("// Output: 10 classes (digits 0-9)\n\n")

        # write constants
        f.write("#define IMAGE_SIZE (28 * 28)\n")
        f.write("#define NUM_CLASSES 10\n\n")

        # write weights
        f.write("// Model weights (NUM_CLASSES x IMAGE_SIZE)\n")
        f.write("const float fc1_weights[NUM_CLASSES][IMAGE_SIZE] = {\n")
        for i in range(10):
            f.write("    {")
            for j in range(784):
                if j % 8 == 0 and j != 0:
                    f.write("\n     ")
                f.write(f"{weights[i][j]:8.6f}f")
                if j < 783:
                    f.write(", ")
            f.write("},\n")
        f.write("};\n\n")

        # write biases
        f.write("// Model biases (NUM_CLASSES)\n")
        f.write("const float fc1_bias[NUM_CLASSES] = {\n    ")
        for i in range(10):
            f.write(f"{biases[i]:8.6f}f")
            if i < 9:
                f.write(", ")
            if i % 5 == 4 and i != 9:
                f.write("\n    ")
        f.write("\n};\n\n")

        # close header guard
        f.write("#endif // " + header_guard + "\n")


def main():
    # training settings
    batch_size = 64
    epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load MNIST dataset
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    train_dataset = datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST("./data", train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    # initialize model
    model = SimpleClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # training loop
    best_accuracy = 0
    best_model = None

    for epoch in range(1, epochs + 1):
        # training phase
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(
                    f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                    f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
                )

        # testing phase
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = 100.0 * correct / len(test_loader.dataset)
        test_loss /= len(test_loader)
        print(
            f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n"
        )

        # save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model.state_dict().copy()

    # load best model
    model.load_state_dict(best_model)
    print(f"\nBest model accuracy: {best_accuracy:.2f}%")

    # save weights to header file
    save_weights_to_c_header(model)
    print("\nWeights saved to mnist_weights.h")


if __name__ == "__main__":
    main()
