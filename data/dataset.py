from torchvision.datasets import CIFAR10

train_data  = CIFAR10(root='./data',train = True,download = True)
test_data = CIFAR10(root='/data',train=False,download = True)