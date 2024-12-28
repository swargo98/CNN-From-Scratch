import sys

import numpy as np
import os
import math
import pickle
import copy
import cv2
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


# Took help from https://blog.ca.meron.dev/Vectorized-CNN/
class ConvolutionalLayer:
    def __init__(self, num_output_channel, kernel_size, stride=1, padding=0):
        self.num_output_channel = num_output_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = np.zeros(self.num_output_channel)

        self.weights = None
        self.X_tmp = None
        self.windows = None

    def xavier_initialization(self, X):
        s = math.sqrt(2 / (self.kernel_size * self.kernel_size * X.shape[3]))
        self.weights = np.random.uniform(-s, s, (self.num_output_channel, X.shape[3], self.kernel_size, self.kernel_size))

    def getWindows(self, input, output_size, kernel_size, padding, stride, dilate=0):
        working_input = input
        working_pad = padding
        if dilate != 0:
            working_input = np.insert(working_input, range(1, input.shape[2]), 0, axis=2)
            working_input = np.insert(working_input, range(1, input.shape[3]), 0, axis=3)

        if working_pad != 0:
            pad_width = ((0,), (0,), (working_pad,), (working_pad,))
            working_input = np.pad(working_input, pad_width=pad_width, mode='constant', constant_values=(0.,))

        in_b, in_c, out_h, out_w = output_size
        out_b, out_c = input.shape[:2]
        batch_str, channel_str, kern_h_str, kern_w_str = working_input.strides

        shape = (out_b, out_c, out_h, out_w, kernel_size, kernel_size)
        strides = (batch_str, channel_str, stride * kern_h_str, stride * kern_w_str, kern_h_str, kern_w_str)
        windows = np.lib.stride_tricks.as_strided(working_input, shape=shape, strides=strides)

        return windows


    def forward(self, X):
        X_tmp = X.copy()
        if self.weights is None:
            self.xavier_initialization(X)
        X_tmp = np.transpose(X_tmp, (0, 3, 1, 2))
        n, c, h, w = X_tmp.shape
        out_dim = (h - self.kernel_size + 2 * self.padding) // self.stride + 1

        windows = self.getWindows(
            X_tmp, (n, c, out_dim, out_dim), self.kernel_size, self.padding, self.stride)

        self.X_tmp = X_tmp
        self.windows = windows

        out = np.einsum('bihwkl,oikl->bohw', windows, self.weights)

        out += self.bias[None, :, None, None]

        out = np.transpose(out, (0, 2, 3, 1))

        return out

    def backward(self, delta, lr):
        delta = np.transpose(delta, (0, 3, 1, 2))

        padding = self.padding
        if padding == 0:
            padding = self.kernel_size - 1

        x = self.X_tmp
        windows = self.windows

        dout_windows = self.getWindows(delta, x.shape, self.kernel_size, padding=padding, stride=1, dilate=self.stride - 1)

        db = np.sum(delta, axis=(0, 2, 3))

        dw = np.einsum('bihwkl,bohw->oikl', windows, delta)

        rot_kern = np.rot90(self.weights, 2, axes=(2, 3))
        dx = np.einsum('bohwkl,oikl->bihw', dout_windows, rot_kern)

        self.weights -= lr * dw
        self.bias -= lr * db

        dx = np.transpose(dx, (0, 2, 3, 1))
        return dx

class ActivationLayer():
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, delta, lr=0.1):
        return delta * (self.x > 0)

# Took help from https://stackoverflow.com/questions/61954727/max-pooling-backpropagation-using-numpy
class MaxPoolingLayer:
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, X):
        x = np.transpose(X, (0, 3, 1, 2))
        n_batch, ch_x, h_x, w_x = x.shape

        out_dim = int((h_x - self.kernel_size) / self.stride) + 1

        shape = (n_batch, ch_x, out_dim, out_dim, self.kernel_size, self.kernel_size)
        strides = (x.strides[0], x.strides[1], self.stride * x.strides[2], self.stride * x.strides[3], x.strides[2], x.strides[3])

        windows = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

        out = np.max(windows, axis=(4, 5))

        maxs = out.repeat(2, axis=2)
        maxs = maxs.repeat(2, axis=3)

        x_window = x[:, :, :out_dim * self.stride, :out_dim * self.stride]
        mask = np.equal(x_window, maxs)
        mask = mask.astype(int)

        self.x = x
        self.mask = mask

        out = np.transpose(out, (0, 2, 3, 1))
        return out

    def backward(self, delta, lr):
        x = self.x
        mask = self.mask

        dA_prev = np.transpose(delta, (0, 3, 1, 2))

        dA = dA_prev.repeat(self.kernel_size, axis=2)
        dA = dA.repeat(self.kernel_size, axis=3)
        dA = dA * mask

        pad = np.zeros(x.shape)
        pad[:, :, :dA.shape[2], :dA.shape[3]] = dA

        pad = np.transpose(pad, (0, 2, 3, 1))
        return pad


class FlatteningLayer():
    def __init__(self):
        self.x_shape = None

    def forward(self, x):
        self.x_shape = x.shape
        k = x.reshape(x.shape[0], -1)
        return np.transpose(k)

    def backward(self, delta, lr=0.1):
        return delta.reshape(self.x_shape)

class FullyConnectedLayer:
    def __init__(self, output_size):
        self.output_size = output_size
        self.bias = np.zeros((self.output_size, 1))

        self.weights = None
        self.X = None

    def xaiver_initialization(self):
            s = np.sqrt(6. / (self.X.shape[0] + self.output_size))
            self.weights = np.random.uniform(-s, s, (self.output_size, self.X.shape[0]))

    def forward(self, X):
        self.X = X.copy()

        if self.weights is None:
            self.xaiver_initialization()

        v = np.dot(self.weights, X)
        v += self.bias
        return v

    def backward(self, delta, lr):
        dZ = delta.copy()

        X_T = self.X.transpose()
        dW = np.dot(dZ, X_T) / dZ.shape[1]

        db = np.reshape(np.mean(dZ, axis=1), (dZ.shape[0], 1))

        W_T = self.weights.transpose()
        dX = np.dot(W_T, dZ)

        self.weights -= lr * dW
        self.bias -= lr * db

        return dX


class SoftmaxLayer():
    def __init__(self):
        self.X = None

    def forward(self, X):
        self.X = X - np.max(X, axis=0, keepdims=True)
        self.v = np.exp(self.X) / np.sum(np.exp(self.X), axis=0, keepdims=True)
        return self.v

    def backward(self, delta, lr):
        return delta


class Model:
    def __init__(self):
        self.components = []

        #using LeNet
        self.components.append(ConvolutionalLayer(6, 5, 1, 2))
        self.components.append(ActivationLayer())
        self.components.append(MaxPoolingLayer(2, 2))
        self.components.append(ConvolutionalLayer(16, 5, 1, 0))
        self.components.append(ActivationLayer())
        self.components.append(MaxPoolingLayer(2, 2))
        self.components.append(FlatteningLayer())
        self.components.append(FullyConnectedLayer(120))
        self.components.append(ActivationLayer())
        self.components.append(FullyConnectedLayer(84))
        self.components.append(ActivationLayer())
        self.components.append(FullyConnectedLayer(10))
        self.components.append(SoftmaxLayer())

    def train(self, X, y_true, lr):
        X_copy = copy.deepcopy(X)
        for i in range(len(self.components)):
            X_copy = self.components[i].forward(X_copy)

        delta = X_copy - y_true

        for i in reversed(range(len(self.components))):
            delta = self.components[i].backward(delta, lr)


    def predict(self, X):
        temp = copy.deepcopy(X)
        for i in range(len(self.components)):
            temp = self.components[i].forward(temp)
        return temp

    def set_weights(self, w1 , b1, w2 , b2, w3 , b3, w4 , b4, w5 , b5):
        self.components[0].weights = w1
        self.components[0].bias = b1
        self.components[3].weights = w2
        self.components[3].bias = b2
        self.components[7].weights = w3
        self.components[7].bias = b3
        self.components[9].weights = w4
        self.components[9].bias = b4
        self.components[11].weights = w5
        self.components[11].bias = b5

def print_scores(y_true, y_pred, train=False, val=False, test=False):
    cross_entropy_loss = np.sum(-1 * np.sum(y_true * np.log(y_pred), axis=0))

    y_pred_temp = np.argmax(y_pred, axis=0)
    y_true_temp = np.argmax(y_true, axis=0)

    accuracy = np.sum(y_pred_temp == y_true_temp) / y_true_temp.shape[0]
    f1 = f1_score(y_true_temp, y_pred_temp, average='macro')

    # print according to the given format
    if train:
        print("Training Results")
    elif val:
        print("Validation Results")
    elif test:
        print("Test Results")

    print("Cross Entropy Loss: {:.4f}".format(cross_entropy_loss))
    print("Accuracy: {:.4f}".format(accuracy))
    print("F1 Score: {:.4f}".format(f1))

    return cross_entropy_loss, accuracy, f1


def plot_metrices(x_list, y_list, x_label, y_label, title, file_name):
    plt.clf()
    plt.plot(x_list, y_list)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(file_name)


def plot_heatmap(cm, title, file_name):
    plt.clf()
    sns.heatmap(cm, annot=True, fmt="d" , cmap=plt.cm.Blues)
    plt.title(title)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.savefig(file_name)

def load_data_from_dir(image_dir):
    image_list = []
    name_list = []
    for filename in os.listdir(image_dir):
        if filename.endswith(".png"):
            if len(image_list) > 2000:
                break
            image = cv2.imread(os.path.join(image_dir, filename), cv2.IMREAD_GRAYSCALE)


            image = 255 - image

            image = cv2.resize(image, (28, 28))

            image = image.reshape(28, 28, 1)
            image = image.astype(np.float32) / 255.0


            image_list.append(image)
            name_list.append(filename)

    image_list = np.array(image_list)
    name_list = np.array(name_list)
    return image_list, name_list


def train(x_train, y_train, x_val, y_val, batch_size=32, num_classes=10, lr=0.01, epochs=10):
    model = Model()
    best_model = Model()
    best_f1_score = 0

    y_train_true = np.eye(num_classes)[y_train.reshape(-1)].transpose()

    y_true_validation = np.eye(num_classes)[y_val.reshape(-1)].transpose()

    x_axis = [0] * epochs
    y_axis_train_loss = [0] * epochs
    y_axis_train_accuracy = [0] * epochs
    y_axis_train_f1_score = [0] * epochs
    y_axis_validation_loss = [0] * epochs
    y_axis_validation_accuracy = [0] * epochs
    y_axis_validation_f1_score = [0] * epochs

    for epoch in range(epochs):
        print("Epoch: {}".format(epoch + 1))
        for i in (range(0, len(x_train), batch_size)):

            X = x_train[i:i + batch_size]
            y_true = y_train[i:i + batch_size]

            y_true = np.eye(num_classes)[y_true.reshape(-1)].transpose()
            model.train(X, y_true, lr)

        y_pred = model.predict(x_train)
        train_loss, train_acc, train_f1 = print_scores(y_true=y_train_true, y_pred=y_pred, train=True)

        y_pred_validation = model.predict(x_val)
        validation_loss, validation_acc, validation_f1 = print_scores(y_true=y_true_validation, y_pred=y_pred_validation, val=True)

        if validation_f1 > best_f1_score:
            best_f1_score = validation_f1
            best_model = copy.deepcopy(model)

        x_axis[epoch] = epoch + 1
        y_axis_train_loss[epoch] = train_loss
        y_axis_train_accuracy[epoch] = train_acc
        y_axis_train_f1_score[epoch] = train_f1
        y_axis_validation_loss[epoch] = validation_loss
        y_axis_validation_accuracy[epoch] = validation_acc
        y_axis_validation_f1_score[epoch] = validation_f1


    plot_metrices(x_axis, y_axis_train_loss, 'Epoch', 'Training Loss', 'Training Loss vs Epoch for Learning Rate = ' + str(lr), 'Figures2/train_loss_lr_' + str(lr) + '.png')
    plot_metrices(x_axis, y_axis_validation_accuracy, 'Epoch', 'Validation Accuracy(%)', 'Validation Accuracy vs Epoch for Learning Rate = ' + str(lr), 'Figures2/valid_accuracy_lr_' + str(lr) + '.png')
    plot_metrices(x_axis, y_axis_validation_f1_score, 'Epoch', 'Validation F1 Score', 'Validation F1 Score vs Epoch for Learning Rate = ' + str(lr), 'Figures2/valid_f1_lr_' + str(lr) + '.png')
    plot_metrices(x_axis, y_axis_validation_loss, 'Epoch', 'Validation Loss', 'Validation Loss vs Epoch for Learning Rate = ' + str(lr), 'Figures2/valid_loss_lr_' + str(lr) + '.png')


    y_pred = best_model.predict(x_val)
    y_pred = np.argmax(y_pred, axis=0)
    y_true = y_val.reshape(-1)
    cm = confusion_matrix(y_true, y_pred)
    plot_heatmap(cm, 'Confusion Matrix for Learning Rate = ' + str(lr), 'Figures2/confusion_matrix_lr_' + str(lr) + '.png')
    return best_f1_score, best_model


def test(model, test_images, test_labels, num_class=10):
    y_true = np.eye(num_class)[test_labels.reshape(-1)]
    y_true = y_true.transpose()
    y_pred = model.predict(test_images)
    test_loss, test_acc, test_f1 = print_scores(y_true=y_true, y_pred=y_pred, test=True)
    return test_loss, test_acc, test_f1


if __name__ == "__main__":
    num_classes = 10

    #load data
    x_test, filename = load_data_from_dir(sys.argv[1])

    model = Model()

    with open('1705051_model.pkl', 'rb') as f:
        w1 , b1, w2 , b2, w3 , b3, w4 , b4, w5 , b5 = pickle.load(f)


    model.set_weights(w1 , b1, w2 , b2, w3 , b3, w4 , b4, w5 , b5)

    # predict and output to csv with filename
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=0)

    with open('1705051_prediction.csv', 'w') as f:
        f.write('FileName,Digit\n')
        for i in range(len(filename)):
            f.write('{},{}\n'.format(filename[i], y_pred[i]))


