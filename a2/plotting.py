import matplotlib.pyplot as plt
import numpy as np


def plot(epochs, train_loss, val_loss, test_loss, train_acc, val_acc, test_acc):

    x = [i for i in range(1, epochs + 1)]

    plt.plot(x, train_loss, label="Training loss")
    plt.plot(x, val_loss, label="Validation loss")
    plt.plot(x, test_loss, label="Test loss")

    plt.xlabel('Batch')
    plt.ylabel('Loss')

    plt.legend()
    plt.title('Loss Curves')
    plt.show()

    plt.plot(x, train_acc, label="Training Accuracy")
    plt.plot(x, val_acc, label="Validation Accuracy")
    plt.plot(x, test_acc, label="Test Accuracy")

    plt.xlabel('Batch')
    plt.ylabel('Accuracy')

    plt.legend()
    plt.title('Accuracy Curves')

    plt.show()



