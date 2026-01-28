from torchvision import datasets
from torchvision.transforms import ToTensor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

train_data = datasets.MNIST(
    root='data',
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.MNIST(
    root='data',
    download=True,
    train=False,
    transform=ToTensor()
)

train_X = []
train_y = []
for img, labels in train_data:
    train_X.append(img.view(-1).numpy())
    train_y.append(labels)

train_X = np.array(train_X)
train_y = np.array(train_y)

test_X = []
test_y = []
for img, labels in test_data:
    test_X.append(img.view(-1).numpy())
    test_y.append(labels)

test_X = np.array(test_X)
test_y = np.array(test_y)

model = RandomForestClassifier(n_estimators=300, criterion='log_loss')
model.fit(train_X, train_y)

y_pred = model.predict(test_X)
accuracy = accuracy_score(test_y, y_pred)


print(f'Accuracy: {accuracy}')

result = model.predict(test_X[69].reshape(1,-1))