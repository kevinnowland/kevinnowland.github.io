---
layout: code-post
title: Experimenting with PyTorch
description: This is a super basic intro to PyTorch since it's new to me
tags: [neural nets]
---

So I've never played with PyTorch before, so I'm going to play around with 
some simple multilayer perceptrons to get started. Much of this code is adapted
from PyTorchs' [60 minute bliz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) tutorial.
This is a very simple walkthrough of creating a `DataSet` and then training a neural net on it. There's also
an example of a class allowing for some variable architecture.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

## Creating a simple MLP

In previous notebooks, we were playing around with neural nets that have the following
architecture. The linear layers all have biases by default, but we an also set
`bias=False` to remove those.

```python
class Net(nn.Module):

    def __init__(self, random_state=47):
        super(Net, self).__init__()
        torch.manual_seed(random_state)
        self.fc1 = nn.Linear(2, 9)
        self.fc2 = nn.Linear(9, 9)
        self.fc3 = nn.Linear(9, 9)
        self.fc4 = nn.Linear(9, 9)
        self.fc5 = nn.Linear(9, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = torch.sigmoid(self.fc5(x))
        return x

```

```python
net = Net()
```

Let's see this `Net`'s parameters.

```python
list(net.parameters())
```




    [Parameter containing:
     tensor([[-0.6321, -0.6365],
             [-0.0457,  0.5313],
             [ 0.0793,  0.4220],
             [ 0.6728, -0.3561],
             [-0.4993, -0.0926],
             [ 0.2811,  0.5492],
             [-0.3340, -0.3312],
             [-0.5127, -0.0552],
             [ 0.3450, -0.6575]], requires_grad=True),
     Parameter containing:
     tensor([-0.5059, -0.1334,  0.0482,  0.1219, -0.4993, -0.2885, -0.3198,  0.3339,
              0.5823], requires_grad=True),
     Parameter containing:
     tensor([[-0.1811, -0.2940, -0.1910, -0.0098,  0.1716, -0.0090, -0.2765,  0.2010,
               0.1643],
             [-0.0962, -0.0630, -0.3145, -0.0620, -0.1136, -0.1233,  0.0276, -0.1000,
              -0.1335],
             [-0.0280, -0.1061, -0.0684, -0.1077, -0.0805, -0.2235, -0.3329,  0.0497,
              -0.1480],
             [-0.0531,  0.1734, -0.1789, -0.1540, -0.0209,  0.2476, -0.0416,  0.2050,
              -0.2495],
             [-0.1266,  0.1427,  0.2202, -0.1647,  0.1948, -0.3059, -0.0061, -0.1883,
               0.2310],
             [ 0.1373,  0.2318, -0.0628,  0.0181,  0.0913, -0.1387,  0.1729, -0.2129,
              -0.3316],
             [-0.2938, -0.1869,  0.0955,  0.1379, -0.2259, -0.1054, -0.0654,  0.2578,
               0.2550],
             [ 0.0694,  0.0563,  0.2023, -0.2261, -0.3202, -0.3294, -0.1565,  0.2440,
               0.0995],
             [-0.0856, -0.2808,  0.2253, -0.2386, -0.1659,  0.2512, -0.1394,  0.2062,
               0.2192]], requires_grad=True),
     Parameter containing:
     tensor([ 0.0364,  0.0177, -0.0671, -0.0629,  0.2630,  0.1507,  0.1259,  0.2261,
              0.2479], requires_grad=True),
     Parameter containing:
     tensor([[ 0.0164,  0.2709,  0.1827, -0.0299, -0.3086, -0.0359, -0.0419, -0.1654,
               0.1594],
             [-0.2598,  0.1595, -0.1115, -0.2831,  0.0947, -0.2015,  0.0895, -0.1406,
              -0.1099],
             [ 0.2737,  0.2374, -0.2414,  0.0338, -0.2378,  0.2125,  0.2424, -0.3081,
              -0.1013],
             [-0.1592,  0.3064, -0.3091,  0.0482, -0.1604, -0.1640,  0.0284, -0.1429,
              -0.0736],
             [-0.0347, -0.1987, -0.2923,  0.1233, -0.1856, -0.0249,  0.1784, -0.0934,
              -0.2156],
             [-0.2519, -0.1175,  0.2238, -0.0720,  0.2375, -0.2586, -0.3317, -0.0766,
              -0.0132],
             [-0.1333,  0.1775, -0.0158,  0.0148,  0.1501,  0.0394, -0.0820,  0.0427,
              -0.2443],
             [-0.1119, -0.2423, -0.1671,  0.1403, -0.1500,  0.3107,  0.2377, -0.1625,
              -0.1476],
             [ 0.0048, -0.2982, -0.1838, -0.0436,  0.2403,  0.3223, -0.1673,  0.3087,
               0.2547]], requires_grad=True),
     Parameter containing:
     tensor([-0.2039,  0.1192,  0.2852, -0.0418,  0.1649,  0.1144,  0.0551, -0.1212,
              0.2805], requires_grad=True),
     Parameter containing:
     tensor([[-0.2038,  0.2893, -0.2738,  0.1934,  0.2750,  0.1062,  0.1668,  0.0617,
              -0.2509],
             [ 0.1988,  0.0586, -0.2811,  0.1281,  0.0124,  0.1549, -0.1912, -0.2525,
               0.2383],
             [ 0.2656,  0.0522,  0.3332, -0.0548, -0.0110,  0.0477, -0.0600,  0.0831,
               0.2949],
             [ 0.2794,  0.0648, -0.3260, -0.1705, -0.2997,  0.0134, -0.2370,  0.1158,
              -0.2011],
             [ 0.1341, -0.0625,  0.2552,  0.2959, -0.2620, -0.2471, -0.2127,  0.2490,
              -0.2640],
             [ 0.1694,  0.1014,  0.0625,  0.3167,  0.0342,  0.0496,  0.1624, -0.0332,
               0.2074],
             [-0.2492,  0.2484,  0.1048,  0.2532, -0.0296,  0.1778, -0.0860,  0.2848,
               0.1285],
             [-0.2977,  0.1723, -0.1206, -0.3330, -0.1580, -0.0370,  0.2655,  0.0722,
               0.2115],
             [-0.2625, -0.1298,  0.0753,  0.3242,  0.1949,  0.3222, -0.2359,  0.0084,
               0.1131]], requires_grad=True),
     Parameter containing:
     tensor([-0.3060,  0.2340,  0.2761,  0.0770, -0.1074,  0.2764,  0.0608,  0.1932,
             -0.2089], requires_grad=True),
     Parameter containing:
     tensor([[-0.0924, -0.1799, -0.2314,  0.0416, -0.0254, -0.1123, -0.1199,  0.1387,
               0.3063]], requires_grad=True),
     Parameter containing:
     tensor([0.0768], requires_grad=True)]



And let's make a prediction.

```python
net(torch.tensor([[1., 2.], [2., 3.]]))
```




    tensor([[0.4689],
            [0.4699]], grad_fn=<SigmoidBackward>)



## Loading Data

The cleanest way to use data with PyTorch -- I'm guessing here -- is to
use the classes native to `torch.utils.data`, principally the `Dataset`
(instead of say a numpy `Array` or a pandas `DataFrame`) and to load
data from the dataset using a `DataLoader`. We find our data initially
loaded into a pandas `DataFrame`, so we'll try to make a `Dataset` from
it.

```python
import numpy as np
import pandas as pd

def normalize_data(df, col_names=['x_1', 'x_2']):
    """ return normalized x_1 and x_2 columns """
    return (df[col_names] - df[col_names].mean()) / df[col_names].std()

def train_data(random_seed=3):

    np.random.seed(random_seed)
    
    def rad_sq(array):
        return array[:, 0]**2 + array[:, 1]**2

    data_pos_ = np.random.normal(0, .75, size=(100, 2))
    data_pos = data_pos_[rad_sq(data_pos_) < 4]

    data_neg_ = np.random.uniform(-5, 5, size=(1000, 2))
    data_neg = data_neg_[(rad_sq(data_neg_) > 6.25) & (rad_sq(data_neg_) < 16)]

    data = np.concatenate((data_pos, data_neg), axis=0)
    y = np.concatenate((np.ones(data_pos.shape[0]), np.zeros(data_neg.shape[0])), axis=0)

    df = pd.DataFrame({
        'x_1': data[:, 0]
        ,'x_2': data[:, 1]
        ,'const': 1.0
        ,'y': y
    })
    
    df[['x_1_norm', 'x_2_norm']] = normalize_data(df)
    
    return df
```

```python
train_df = train_data()
train_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x_1</th>
      <th>x_2</th>
      <th>const</th>
      <th>y</th>
      <th>x_1_norm</th>
      <th>x_2_norm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.341471</td>
      <td>0.327382</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.595074</td>
      <td>0.205659</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.072373</td>
      <td>-1.397620</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>-0.001979</td>
      <td>-0.650942</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.208041</td>
      <td>-0.266069</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>-0.133901</td>
      <td>-0.089037</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.062056</td>
      <td>-0.470251</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>-0.065222</td>
      <td>-0.190429</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.032864</td>
      <td>-0.357914</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>-0.051488</td>
      <td>-0.134645</td>
    </tr>
  </tbody>
</table>
</div>



The `TrainDataset` class will subclass `Dataset`. We are required to fill
in the `__init__`, `__len__`, and `__getitem__` functions. We can then 
call `DataLoader` with this dataset, and it will know how to batch
and shuffle through the dataset. We are relying on the fact that `DataFrame`
can retrieve data by relying on numerical indexes. If we do not have indexes
but instead only an iterable list, we could have used `IterableDataset` instead.

```python
from torch.utils.data import Dataset, DataLoader

class TrainDataset(Dataset):
    
    def __init__(self, df):
        """ df contains the data that we want to use
        it should have x_1_norm, x_2_norm, and y columns """
        self.df = df[['x_1_norm', 'x_2_norm', 'y']].copy()
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        samples = self.df[['x_1_norm', 'x_2_norm']].iloc[idx].values
        labels = self.df[['y']].iloc[idx].values
        
        return samples, labels
```

```python
trainset = TrainDataset(train_df)
```

Let's print out a random sample from this.

```python
trainloader_4 = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
trainiter = iter(trainloader)

print('length of dataset:', len(trainset))
trainiter.next()
```

    length of dataset: 393





    [tensor([[ 0.6871,  1.7872],
             [ 1.0827, -0.3425],
             [ 1.1661, -0.5120],
             [ 0.6863, -1.4712]], dtype=torch.float64),
     tensor([[0.],
             [0.],
             [0.],
             [0.]], dtype=torch.float64)]



## Train the Neural Net

With a neural net and training dataset ready to go, let's train the neural net
using the Adam optimizer, which we've imported from `torch.optim`. We will
use the Mean Squared Error loss. We'll replace the trainloader from above
with one which ha batch size as the max size possible.

```python
from torch.optim import Adam

net = Net()
trainloader = DataLoader(trainset, batch_size=50, shuffle=True, num_workers=0)
optimizer = Adam(net.parameters())
criterion = nn.MSELoss()

for epoch in range(1000): 

    running_loss = 0.0
    num_wrong = 0.0
    for i, data in enumerate(trainloader, 0):
        samples, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(samples.float())
        loss = criterion(outputs.float(), labels.float())
        loss.backward()
        optimizer.step()

        predictions = 1 * (outputs >= 0.5) + 0 * (outputs < 0.5)
        
        num_wrong += torch.sum(torch.abs(predictions - labels))
        running_loss += loss.item()
        
    class_error = num_wrong / len(trainset)
    print('epoch: {} - loss: {:.3f} - classification error: {:.3f}'.format(epoch+1, loss, class_error))
    
    if class_error == 0.0:
        break

print('Finished Training')

```

    epoch: 1 - loss: 0.240 - classification error: 0.249
    epoch: 2 - loss: 0.231 - classification error: 0.249
    epoch: 3 - loss: 0.226 - classification error: 0.249
    epoch: 4 - loss: 0.219 - classification error: 0.249
    epoch: 5 - loss: 0.246 - classification error: 0.249
    epoch: 6 - loss: 0.214 - classification error: 0.249
    epoch: 7 - loss: 0.215 - classification error: 0.249
    epoch: 8 - loss: 0.205 - classification error: 0.249
    epoch: 9 - loss: 0.215 - classification error: 0.249
    epoch: 10 - loss: 0.238 - classification error: 0.249
    epoch: 11 - loss: 0.168 - classification error: 0.249
    epoch: 12 - loss: 0.184 - classification error: 0.249
    epoch: 13 - loss: 0.191 - classification error: 0.249
    epoch: 14 - loss: 0.169 - classification error: 0.249
    epoch: 15 - loss: 0.166 - classification error: 0.249
    epoch: 16 - loss: 0.187 - classification error: 0.249
    epoch: 17 - loss: 0.174 - classification error: 0.249
    epoch: 18 - loss: 0.139 - classification error: 0.249
    epoch: 19 - loss: 0.173 - classification error: 0.249
    epoch: 20 - loss: 0.161 - classification error: 0.249
    epoch: 21 - loss: 0.204 - classification error: 0.249
    epoch: 22 - loss: 0.166 - classification error: 0.249
    epoch: 23 - loss: 0.152 - classification error: 0.249
    epoch: 24 - loss: 0.176 - classification error: 0.249
    epoch: 25 - loss: 0.175 - classification error: 0.249
    epoch: 26 - loss: 0.207 - classification error: 0.249
    epoch: 27 - loss: 0.148 - classification error: 0.249
    epoch: 28 - loss: 0.166 - classification error: 0.249
    epoch: 29 - loss: 0.175 - classification error: 0.249
    epoch: 30 - loss: 0.163 - classification error: 0.249
    epoch: 31 - loss: 0.113 - classification error: 0.249
    epoch: 32 - loss: 0.098 - classification error: 0.249
    epoch: 33 - loss: 0.115 - classification error: 0.249
    epoch: 34 - loss: 0.205 - classification error: 0.249
    epoch: 35 - loss: 0.146 - classification error: 0.249
    epoch: 36 - loss: 0.134 - classification error: 0.249
    epoch: 37 - loss: 0.151 - classification error: 0.249
    epoch: 38 - loss: 0.114 - classification error: 0.249
    epoch: 39 - loss: 0.165 - classification error: 0.249
    epoch: 40 - loss: 0.132 - classification error: 0.249
    epoch: 41 - loss: 0.102 - classification error: 0.249
    epoch: 42 - loss: 0.092 - classification error: 0.249
    epoch: 43 - loss: 0.092 - classification error: 0.249
    epoch: 44 - loss: 0.114 - classification error: 0.249
    epoch: 45 - loss: 0.072 - classification error: 0.249
    epoch: 46 - loss: 0.088 - classification error: 0.249
    epoch: 47 - loss: 0.107 - classification error: 0.176
    epoch: 48 - loss: 0.067 - classification error: 0.132
    epoch: 49 - loss: 0.074 - classification error: 0.097
    epoch: 50 - loss: 0.082 - classification error: 0.069
    epoch: 51 - loss: 0.069 - classification error: 0.056
    epoch: 52 - loss: 0.060 - classification error: 0.046
    epoch: 53 - loss: 0.045 - classification error: 0.038
    epoch: 54 - loss: 0.065 - classification error: 0.025
    epoch: 55 - loss: 0.047 - classification error: 0.020
    epoch: 56 - loss: 0.034 - classification error: 0.020
    epoch: 57 - loss: 0.041 - classification error: 0.015
    epoch: 58 - loss: 0.057 - classification error: 0.015
    epoch: 59 - loss: 0.046 - classification error: 0.013
    epoch: 60 - loss: 0.041 - classification error: 0.010
    epoch: 61 - loss: 0.039 - classification error: 0.008
    epoch: 62 - loss: 0.036 - classification error: 0.005
    epoch: 63 - loss: 0.028 - classification error: 0.005
    epoch: 64 - loss: 0.030 - classification error: 0.003
    epoch: 65 - loss: 0.028 - classification error: 0.000
    Finished Training


And there we have it, we finished training in 65 epochs when we achieved 0 classification error.

## Variable architecture Net

Can we create a class that allows us to specify the number and width of
layers upon creation? Apparently we have to use the `add_module` function 
to do this, as the layers are not connected explicitly to `self`.

```python
class ReLuNet(nn.Module):
    
    
    def __init__(self, layer_widths, random_state=47):
        """ layer_widths should include the input and out layer widths."""
        torch.manual_seed(random_state)
        super(ReLuNet, self).__init__()
        self.layers = [
            nn.Linear(layer_widths[i], layer_widths[i+1])
            for i in range(len(layer_widths)-1)
        ]
        for i in range(len(self.layers)):
            self.add_module("hidden layer " + str(i), self.layers[i])
        
        
    def forward(self, x):
        for i in range(len(self.layers)-1):
            x = F.relu(self.layers[i](x))
        x = torch.sigmoid(self.layers[-1](x))
        return x
        
```

```python
relu_net = ReLuNet([2, 9, 9, 9, 9, 1])
```

```python
list(relu_net.parameters())
```




    [Parameter containing:
     tensor([[-0.6321, -0.6365],
             [-0.0457,  0.5313],
             [ 0.0793,  0.4220],
             [ 0.6728, -0.3561],
             [-0.4993, -0.0926],
             [ 0.2811,  0.5492],
             [-0.3340, -0.3312],
             [-0.5127, -0.0552],
             [ 0.3450, -0.6575]], requires_grad=True),
     Parameter containing:
     tensor([-0.5059, -0.1334,  0.0482,  0.1219, -0.4993, -0.2885, -0.3198,  0.3339,
              0.5823], requires_grad=True),
     Parameter containing:
     tensor([[-0.1811, -0.2940, -0.1910, -0.0098,  0.1716, -0.0090, -0.2765,  0.2010,
               0.1643],
             [-0.0962, -0.0630, -0.3145, -0.0620, -0.1136, -0.1233,  0.0276, -0.1000,
              -0.1335],
             [-0.0280, -0.1061, -0.0684, -0.1077, -0.0805, -0.2235, -0.3329,  0.0497,
              -0.1480],
             [-0.0531,  0.1734, -0.1789, -0.1540, -0.0209,  0.2476, -0.0416,  0.2050,
              -0.2495],
             [-0.1266,  0.1427,  0.2202, -0.1647,  0.1948, -0.3059, -0.0061, -0.1883,
               0.2310],
             [ 0.1373,  0.2318, -0.0628,  0.0181,  0.0913, -0.1387,  0.1729, -0.2129,
              -0.3316],
             [-0.2938, -0.1869,  0.0955,  0.1379, -0.2259, -0.1054, -0.0654,  0.2578,
               0.2550],
             [ 0.0694,  0.0563,  0.2023, -0.2261, -0.3202, -0.3294, -0.1565,  0.2440,
               0.0995],
             [-0.0856, -0.2808,  0.2253, -0.2386, -0.1659,  0.2512, -0.1394,  0.2062,
               0.2192]], requires_grad=True),
     Parameter containing:
     tensor([ 0.0364,  0.0177, -0.0671, -0.0629,  0.2630,  0.1507,  0.1259,  0.2261,
              0.2479], requires_grad=True),
     Parameter containing:
     tensor([[ 0.0164,  0.2709,  0.1827, -0.0299, -0.3086, -0.0359, -0.0419, -0.1654,
               0.1594],
             [-0.2598,  0.1595, -0.1115, -0.2831,  0.0947, -0.2015,  0.0895, -0.1406,
              -0.1099],
             [ 0.2737,  0.2374, -0.2414,  0.0338, -0.2378,  0.2125,  0.2424, -0.3081,
              -0.1013],
             [-0.1592,  0.3064, -0.3091,  0.0482, -0.1604, -0.1640,  0.0284, -0.1429,
              -0.0736],
             [-0.0347, -0.1987, -0.2923,  0.1233, -0.1856, -0.0249,  0.1784, -0.0934,
              -0.2156],
             [-0.2519, -0.1175,  0.2238, -0.0720,  0.2375, -0.2586, -0.3317, -0.0766,
              -0.0132],
             [-0.1333,  0.1775, -0.0158,  0.0148,  0.1501,  0.0394, -0.0820,  0.0427,
              -0.2443],
             [-0.1119, -0.2423, -0.1671,  0.1403, -0.1500,  0.3107,  0.2377, -0.1625,
              -0.1476],
             [ 0.0048, -0.2982, -0.1838, -0.0436,  0.2403,  0.3223, -0.1673,  0.3087,
               0.2547]], requires_grad=True),
     Parameter containing:
     tensor([-0.2039,  0.1192,  0.2852, -0.0418,  0.1649,  0.1144,  0.0551, -0.1212,
              0.2805], requires_grad=True),
     Parameter containing:
     tensor([[-0.2038,  0.2893, -0.2738,  0.1934,  0.2750,  0.1062,  0.1668,  0.0617,
              -0.2509],
             [ 0.1988,  0.0586, -0.2811,  0.1281,  0.0124,  0.1549, -0.1912, -0.2525,
               0.2383],
             [ 0.2656,  0.0522,  0.3332, -0.0548, -0.0110,  0.0477, -0.0600,  0.0831,
               0.2949],
             [ 0.2794,  0.0648, -0.3260, -0.1705, -0.2997,  0.0134, -0.2370,  0.1158,
              -0.2011],
             [ 0.1341, -0.0625,  0.2552,  0.2959, -0.2620, -0.2471, -0.2127,  0.2490,
              -0.2640],
             [ 0.1694,  0.1014,  0.0625,  0.3167,  0.0342,  0.0496,  0.1624, -0.0332,
               0.2074],
             [-0.2492,  0.2484,  0.1048,  0.2532, -0.0296,  0.1778, -0.0860,  0.2848,
               0.1285],
             [-0.2977,  0.1723, -0.1206, -0.3330, -0.1580, -0.0370,  0.2655,  0.0722,
               0.2115],
             [-0.2625, -0.1298,  0.0753,  0.3242,  0.1949,  0.3222, -0.2359,  0.0084,
               0.1131]], requires_grad=True),
     Parameter containing:
     tensor([-0.3060,  0.2340,  0.2761,  0.0770, -0.1074,  0.2764,  0.0608,  0.1932,
             -0.2089], requires_grad=True),
     Parameter containing:
     tensor([[-0.0924, -0.1799, -0.2314,  0.0416, -0.0254, -0.1123, -0.1199,  0.1387,
               0.3063]], requires_grad=True),
     Parameter containing:
     tensor([0.0768], requires_grad=True)]



And yeah, this works and we can create predictions. (The prediction is the 
same as the initial state of the `net` created above since we used
the same random state.)

```python
relu_net(torch.tensor([1., 2.]))
```




    tensor([0.4689], grad_fn=<SigmoidBackward>)



And the random state of course can be different and produces
different results.

```python
relu_net_2 = ReLuNet([2, 9, 9, 9, 9, 1], random_state=1)
relu_net_2(torch.tensor([1., 2.]))
```




    tensor([0.4216], grad_fn=<SigmoidBackward>)


