import numpy as np
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

class MultiViewDataset(Dataset):
    def __init__(self, data_name, data_X, data_Y):
        super(MultiViewDataset, self).__init__()
        self.data_name = data_name

        self.X = dict()
        self.num_views = len(data_X)
        for v in range(self.num_views):
            self.X[v] = self.normalize(data_X[v])

        self.Y = data_Y
        self.Y = np.squeeze(self.Y)
        if np.min(self.Y) == 1:
            self.Y = self.Y - 1
        self.Y = self.Y.astype(dtype=np.int64)
        self.num_classes = len(np.unique(self.Y))
        self.dims = self.get_dims()

    def __getitem__(self, index):
        data = dict()
        for v_num in range(len(self.X)):
            data[v_num] = (self.X[v_num][index]).astype(np.float32)
        target = self.Y[index]
        return data, target, index

    def __len__(self):
        return len(self.X[0])

    def get_dims(self):
        dims = []
        for view in range(self.num_views):
            dims.append([self.X[view].shape[1]])
        return np.array(dims)

    @staticmethod
    def normalize(x, min=0):
        if min == 0:
            scaler = MinMaxScaler((0, 1))
        else:  # min=-1
            scaler = MinMaxScaler((-1, 1))
        norm_x = scaler.fit_transform(x)
        return norm_x

def handwritten0():
    # 1024 300
    data_path = "data/handwritten0.mat"
    data = sio.loadmat(data_path)
    data_X = data['X'][0]
    data_Y = data['gt']
    for v in range(len(data_X)):
        data_X[v] = data_X[v].T
    return MultiViewDataset("handwritten0", data_X, data_Y)

def PIE():
    data_path = "data/PIE.mat"
    data = sio.loadmat(data_path)
    data_X = data['X'][0]
    data_Y = data['gt']
    for v in range(len(data_X)):
        data_X[v] = data_X[v].T
    return MultiViewDataset("PIE", data_X, data_Y)

def YaleB():
    data_path = "data/YaleB.mat"
    data = sio.loadmat(data_path)
    data_X = data['X'].flatten()
    data_Y = data['Y']-1
    return MultiViewDataset("YaleB", data_X, data_Y)

def Scene():
    data_path = "data/Scene.mat"
    data = sio.loadmat(data_path)
    data_X = data['X'][0]
    data_Y = data['Y']-1
    return MultiViewDataset("Scene", data_X, data_Y)

def Caltech101all():
    data_path = "data/Caltech101-7.mat"
    data = sio.loadmat(data_path)
    data_X = data['X'].flatten()
    data_Y = data['y']
    return MultiViewDataset("Caltech101-7", data_X, data_Y)

def ALOI100():
    data_path = "data/ALOI_100.mat"
    data = sio.loadmat(data_path)
    data_X = data['X'][0]
    data_Y = data['Y']-1
    return MultiViewDataset("ALOI_100", data_X, data_Y)

def leaves100():
    data_path = "data/100leaves.mat"
    data = sio.loadmat(data_path)
    data_X = data['X'][0]
    data_Y = data['Y']-1
    return MultiViewDataset("100leaves", data_X, data_Y)

def sources3():
    data_path = "data/3sources.mat"
    data = sio.loadmat(data_path)
    data_X = data['X'][0]
    data_Y = data['Y']-1
    return MultiViewDataset("3sources", data_X, data_Y)

def MNIST10k():
    data_path = "data/MNIST-10k.mat"
    data = sio.loadmat(data_path)
    data_X = data['X'].flatten()
    data_Y = data['y']
    return MultiViewDataset("MNIST-10k", data_X, data_Y)

def Cora():
    data_path = "data/Cora.mat"
    data = sio.loadmat(data_path)
    data_X = data['X'].flatten()
    data_Y = data['y']
    return MultiViewDataset("Cora", data_X, data_Y)

def Reuters1200():
    data_path = "data/Reuters-1200.mat"
    data = sio.loadmat(data_path)
    data_X = data['X'].flatten()
    data_Y = data['y']
    return MultiViewDataset("Reuters-1200", data_X, data_Y)

def Wikipedia():
    data_path = "data/Wikipedia.mat"
    data = sio.loadmat(data_path)
    data_X = data['X'].flatten()
    data_Y = data['y']
    return MultiViewDataset("Wikipedia", data_X, data_Y)