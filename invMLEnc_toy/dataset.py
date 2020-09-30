
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import math
from IPython import embed
from torchvision import datasets, transforms
from mpl_toolkits.mplot3d import Axes3D
import sklearn
from sklearn import manifold, datasets

from generator.samples_generator_star import make_swiss_roll, make_s_curve

import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import integrate
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def Remove(data, label, center, r):

    mask = torch.zeros((data.shape[0]))
    for i in range(data.shape[0]):
        if torch.norm(data[i]-torch.tensor(center, device=device)) < r:
            mask[i] = 0
        else:
            mask[i] = 1
    mask = mask.bool()
    data = data[mask]
    label = label[mask]

    return data, label


def make_new_swiss_roll(n_samples=100, noise=0.0, random_state=None, remove=False, scale=1.0):

    center = np.zeros((5, 2))
    length, _ = integrate.quad(f, 1.5 * np.pi, 4.5 * np.pi)
    center = np.array([[length / 16 * 2, 10.5],
                       [length / 16 * 5, 10.5],
                       [length / 16 * 8, 10.5],
                       [length / 16 * 11, 10.5],
                       [length / 16 * 14, 10.5]])
    points = []
    T = []
    num = 0

    while num < n_samples:
        t = 1.5 * np.pi * (1 + 2 * np.random.rand())
        x = t * np.cos(t)
        y = 21 * np.random.rand()
        z = t * np.sin(t)
        v, _ = integrate.quad(f, 1.5 * np.pi, t)

        Flag = 0
        for i in range(5):
            if ((y - center[i, 1]) ** 2 + (v - center[i, 0]) ** 2) ** 0.5 < 21 * 0.5 \
                    and ((y - center[i, 1]) ** 2 + (v - center[i, 0]) ** 2) ** 0.5 > 21 * 0.15:
                Flag = 1

        if Flag == 1:
            points.append(np.array([x, y, z]))
            T.append(t)
            num += 1

    points = np.array(points)
    T = np.array(T)

    return points, T


def new_swiss_roll(n_samples=1500, plot=False):

    x, y = make_new_swiss_roll(n_samples=n_samples)

    if plot:
        ax = plt.subplot(111, projection='3d')
        ax.scatter(x[:, 0], x[:, 1], x[:, 2], s=1, c=y, cmap='rainbow_r')
        plt.show()
    
    return x, y


def CW_rotate_X(angle, x, y, z):
    angle = math.radians(angle)
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    new_x = x
    new_y = y*math.cos(angle) + z*math.sin(angle)
    new_z = -(y*math.sin(angle)) + z*math.cos(angle)
    return new_x, new_y, new_z


def CW_rotate_Y(angle, x, y, z):
    angle = math.radians(angle)
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    new_x = x*math.cos(angle) - z*math.sin(angle)
    new_y = y
    new_z = x*math.sin(angle) + z*math.cos(angle)
    return new_x, new_y, new_z


def CW_rotate_Z(angle, x, y, z):
    angle = math.radians(angle)
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    new_x = x*math.cos(angle) + y*math.sin(angle)
    new_y = -(x*math.sin(angle)) + y*math.cos(angle)
    new_z = z
    return new_x, new_y, new_z


def swiss_roll(n=50, seed=42, plot=False):
    data = make_swiss_roll(n_samples=n, noise=0.0,
                           random_state=0, remove=True, scale=1.2)
    X = data[0]
    y = data[1]

    X[:, 0] = X[:, 0] - np.mean(X[:, 0])
    X[:, 1] = X[:, 1] - np.mean(X[:, 1])
    X[:, 2] = X[:, 2] - np.mean(X[:, 2])

    for i in range(len(y)):
        y[i] = (y[i] - np.min(y))/(np.max(y)-np.min(y))

    scale = 15 / max(np.max(X[:, 0]) - np.min(X[:, 0]), np.max(X[:, 1]) -
                     np.min(X[:, 1]), np.max(X[:, 2]) - np.min(X[:, 2]))
    X = X * scale

    if plot:
        ax = plt.subplot(111, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y)
        plt.show()

    return X, y


def s_curve(n=50, seed=42, plot=False):
    data = make_s_curve(n_samples=n, noise=0.0,
                        random_state=0, remove=True, scale=0.11)
    X = data[0]
    y = data[1]

    X[:, 0] = X[:, 0] - np.mean(X[:, 0])
    X[:, 1] = X[:, 1] - np.mean(X[:, 1])
    X[:, 2] = X[:, 2] - np.mean(X[:, 2])

    for i in range(len(y)):
        y[i] = (y[i] - np.min(y))/(np.max(y)-np.min(y))

    scale = 18 / max(np.max(X[:, 0]) - np.min(X[:, 0]), np.max(X[:, 1]) -
                     np.min(X[:, 1]), np.max(X[:, 2]) - np.min(X[:, 2]))
    X = X * scale

    if plot:
        ax = plt.subplot(111, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y)
        plt.show()

    return X, y


def spiral_curve(n=50, seed=42, plot=False):
    X = np.zeros((n, 3))
    y = np.zeros(n)

    theta = np.linspace(-8 * np.pi, 8 * np.pi, n)
    X[:, 2] = np.linspace(-2, 2, n) + np.random.randn(n)*0.01
    r = X[:, 2]**2 + 1
    X[:, 0] = r * np.sin(theta) + np.random.randn(n)*0.01
    X[:, 1] = r * np.cos(theta) + np.random.randn(n)*0.01

    for i in range(n):
        y[i] = i

    scale = 15 / max(np.max(X[:, 0]) - np.min(X[:, 0]), np.max(X[:, 1]) -
                     np.min(X[:, 1]), np.max(X[:, 2]) - np.min(X[:, 2]))
    X = X * scale

    if plot:
        ax = plt.subplot(111, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y)
        plt.show()
    return X, y


def dsphere(n=100, d=2, r=1, noise=None, ambient=None):
    """
    Sample `n` data points on a d-sphere.

    Parameters
    -----------
    n : int
        Number of data points in shape.
    r : float
        Radius of sphere.
    ambient : int, default=None
        Embed the sphere into a space with ambient dimension equal to `ambient`. The sphere is randomly rotated in this high dimensional space.
    """
    data = np.random.randn(n, d+1)

    # Normalize points to the sphere
    data = r * data / np.sqrt(np.sum(data**2, 1)[:, None])

    if noise:
        data += noise * np.random.randn(*data.shape)

    if ambient:
        assert ambient > d, "Must embed in higher dimensions"
        data = embed(data, ambient)

    return data


def create_sphere_dataset(n_samples=500, d=100, n_spheres=11, r=5, plot=False, seed=42):
    np.random.seed(seed)

    variance = 10/np.sqrt(d)

    shift_matrix = np.random.normal(0, variance, [n_spheres, d+1])

    spheres = []
    n_datapoints = 0
    for i in np.arange(n_spheres-1):
        sphere = dsphere(n=n_samples, d=d, r=r)
        spheres.append(sphere + shift_matrix[i, :])
        n_datapoints += n_samples

    # Additional big surrounding sphere:
    n_samples_big = 10*n_samples  # int(n_samples/2)
    big = dsphere(n=n_samples_big, d=d, r=r*5)
    spheres.append(big)
    n_datapoints += n_samples_big

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        colors = matplotlib.cm.rainbow(np.linspace(0, 1, n_spheres))
        for data, color in zip(spheres, colors):
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=[color])

    # Create Dataset:
    dataset = np.concatenate(spheres, axis=0)

    labels = np.zeros(n_datapoints)
    label_index = 0
    for index, data in enumerate(spheres):
        n_sphere_samples = data.shape[0]
        labels[label_index:label_index + n_sphere_samples] = index
        label_index += n_sphere_samples

    index_seed = np.linspace(0, 10000, num=20, dtype='int16', endpoint=False)
    arr = np.array([], dtype='int16')
    for i in range(500):
        arr = np.concatenate((arr, index_seed+int(i)))
    
    print(arr.shape)

    dataset = dataset[arr]
    labels = labels[arr]

    return dataset/22+0.5, labels


def create_sphere_dataset5500(n_samples=500, d=100, n_spheres=11, r=5, plot=False, seed=42):
    np.random.seed(seed)

    variance = 10/np.sqrt(d)

    shift_matrix = np.random.normal(0, variance, [n_spheres, d+1])

    spheres = []
    n_datapoints = 0
    for i in np.arange(n_spheres-1):
        sphere = dsphere(n=n_samples, d=d, r=r)
        spheres.append(sphere + shift_matrix[i, :])
        n_datapoints += n_samples

    n_samples_big = 1*n_samples  # int(n_samples/2)
    big = dsphere(n=n_samples_big, d=d, r=r*5)
    spheres.append(big)
    n_datapoints += n_samples_big

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        colors = matplotlib.cm.rainbow(np.linspace(0, 1, n_spheres))
        for data, color in zip(spheres, colors):
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=[color])

    # Create Dataset:
    dataset = np.concatenate(spheres, axis=0)

    labels = np.zeros(n_datapoints)
    label_index = 0
    for index, data in enumerate(spheres):
        n_sphere_samples = data.shape[0]
        labels[label_index:label_index + n_sphere_samples] = index
        label_index += n_sphere_samples

    arr = np.arange(dataset.shape[0])
    np.random.shuffle(arr)
    # rng.shuffle(arr)
    dataset = dataset[arr]
    labels = labels[arr]

    return dataset/22+0.5, labels



def LoadData(
    dataname='mnist',
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    train_number=1500,
    test_number=1500,
    randomstate=0,
    noise=0.0,
    remove='circle'
):

    all_number = train_number+test_number

    if dataname == 'mnist':

        train_data = datasets.MNIST(
            './data', train=True, download=True,
            transform=transforms.ToTensor()
        ).data.float().view(-1, 28*28)/255
        train_label = datasets.MNIST(
            './data', train=True, download=True,
            transform=transforms.ToTensor()
        ).targets
        test_data = datasets.MNIST(
            './data', train=False,
            transform=transforms.ToTensor()
        ).data.float().view(-1, 28*28)/255

        test_labels = datasets.MNIST(
            './data', train=False,
            transform=transforms.ToTensor()
        ).targets

    if dataname == 'Fmnist':

        train_data = datasets.FashionMNIST(
            './data', train=True, download=True,
            transform=transforms.ToTensor()
        ).data.float()/255
        train_label = datasets.FashionMNIST(
            './data', train=True, download=True,
            transform=transforms.ToTensor()
        ).targets
        test_data = datasets.FashionMNIST(
            './data', train=False,
            transform=transforms.ToTensor()
        ).data.float()/255

        test_labels = datasets.FashionMNIST(
            './data', train=False,
            transform=transforms.ToTensor()
        ).targets
        # print(train_data.max())

    if dataname == 'sphere':

        train_data, train_label = create_sphere_dataset5500(seed=42)
        test_data, test_labels = create_sphere_dataset5500(seed=42)
        print("sphere dataset shape={}".format(train_data.shape))
        train_test_split = train_data.shape[0] * 5//10
        train_data = torch.tensor(train_data).to(device)[:train_test_split]
        train_label = torch.tensor(train_label).to(device)[:train_test_split]
        test_data = torch.tensor(test_data).to(device)[train_test_split:]
        test_labels = torch.tensor(test_labels).to(device)[train_test_split:]
    
    if dataname == 'SwissRoll':

        train_data, train_label = make_swiss_roll(
            n_samples=train_number, noise=noise, random_state=randomstate)
        test_data, test_labels = make_swiss_roll(
            n_samples=test_number * 10, noise=noise, random_state=randomstate+1,
            remove=remove,
            center=[10, 10],
            r=8
        )

        import scipy
        n = 20
        train_data = train_data / n
        test_data = test_data / n

        train_data = torch.tensor(train_data).to(device)
        train_label = torch.tensor(train_label).to(device)
        test_data = torch.tensor(test_data).to(device)
        test_labels = torch.tensor(test_labels).to(device)
    
    if dataname == 'SCurve':
        train_data, train_label = make_s_curve(
            n_samples=train_number, random_state=randomstate)
        test_data, test_labels = make_s_curve(
            n_samples=test_number * 10, random_state=randomstate,
            remove=False)
        
        train_data = train_data / 2
        test_data = test_data / 2

        train_data = torch.tensor(train_data).to(device)
        train_label = torch.tensor(train_label).to(device)
        test_data = torch.tensor(test_data).to(device)
        test_labels = torch.tensor(test_labels).to(device)

    return train_data, train_label, test_data, test_labels


if __name__ == "__main__":
    pass