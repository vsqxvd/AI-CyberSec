import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

import torchvision
import torchvision.transforms.v2 as transforms
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

# Завантаження датасетів
train_set = torchvision.datasets.MNIST("./mnist/", train=True, download=True)
valid_set = torchvision.datasets.MNIST("./mnist/", train=False, download=True)
# print(train_set)
# print(valid_set)

def show_img(x_0, y_0):
    plt.imshow(x_0, cmap='gray')
    plt.title(f"Label: {y_0}")
    plt.show()

# Дослідження датасету
x_0, y_0 = train_set[0]
# show_img(x_0, y_0)
# print(type(x_0))
# print(type(y_0))
# print(y_0)

# Тензори
trans = transforms.Compose([transforms.ToTensor()]) # Змінна діапозону з [0, 255] до [0.0, 1.0]
x_0_tensor = trans(x_0)
# print(x_0_tensor.dtype)
# print(x_0_tensor.min())
# print(x_0_tensor.max())
# print(x_0_tensor.size())
# print(x_0_tensor)
# print(x_0_tensor.device)

image = F.to_pil_image(x_0_tensor) # Конвертація тензорів
# show_img(image, y_0)

# Підготовка даних
trans = transforms.Compose([transforms.ToTensor()])
train_set.transform = trans
valid_set.transform = trans

batch_size = 32 # Лімітація навчання

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size)

# Створення моделі
test_matrix = torch.tensor(
    [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]
)
# print(test_matrix)
# print(nn.Flatten()(test_matrix)) # Однакові результати
batch_test_matrix = test_matrix[None, :]
# print(batch_test_matrix)
# print(nn.Flatten()(batch_test_matrix))

input_size = 1 * 28 * 28
n_classes = 10

layers = [
    nn.Flatten(), # Згладжування
    nn.Linear(input_size, 512),  # Вхідний шар
    nn.ReLU(),  # Активація вхідного шару
    nn.Linear(512, 512),  # Прихований шар
    nn.ReLU(),  # Активація прихованого шару
    nn.Linear(512, n_classes)  # Вихідний шар
]
# print(layers)

model = nn.Sequential(*layers) # Послідовний запис у модель
# model = torch.compile(model) # Компіляція моделі для прискорення
# print(model)

# Тренування моделі
loss_function = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters())

train_N = len(train_loader.dataset)
valid_N = len(valid_loader.dataset)

def get_batch_accuracy(output, y, N): # Обчислення точності моделі
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(y.view_as(pred)).sum().item()
    return correct / N

def train(): # Тренування моделі
    loss = 0
    accuracy = 0

    model.train()
    for x, y in train_loader:
        # x, y = x.to(device), y.to(device)
        output = model(x)
        optimizer.zero_grad()
        batch_loss = loss_function(output, y)
        batch_loss.backward()
        optimizer.step()

        loss += batch_loss.item()
        accuracy += get_batch_accuracy(output, y, train_N)
    train_ls.append(loss)
    train_acc.append(accuracy)
    print('Train - Loss: {:.4f} Accuracy: {:.4f}'.format(loss, accuracy))

def validate(): # Валідація моделі
    loss = 0
    accuracy = 0

    model.eval()
    with torch.no_grad(): # Не запам'ятовує обчислення
        for x, y in valid_loader:
            # x, y = x.to(device), y.to(device)
            output = model(x)

            loss += loss_function(output, y).item()
            accuracy += get_batch_accuracy(output, y, valid_N)
    valid_ls.append(loss)
    valid_acc.append(accuracy)
    print('Valid - Loss: {:.4f} Accuracy: {:.4f}'.format(loss, accuracy))

epochs = 5
train_ls, valid_ls, train_acc, valid_acc = [], [], [], []

for epoch in range(epochs):
        print('Epoch: {}'.format(epoch))
        train()
        validate()

prediction = model(x_0_tensor)
print(prediction)

epochs_range = range(epochs)

plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_ls, label='Train Loss')
plt.plot(epochs_range, valid_ls, label='Validation Loss')
plt.title("Loss degression")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_acc, label='Train Accuracy')
plt.plot(epochs_range, valid_acc, label='Valid Accuracy')
plt.title("Accuracy Progression")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.show()