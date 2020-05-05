---
layout: code-post
title: Training neural nets - Backpropagation
tags: [neural nets]
---

Following our foray into the expressive power of shallow neural
networks, let's train a shallow neural network instead of 
designing them by hand.


Outline:

0. Generating data
1. Backpropagation
2. Examples

## 0. Generating Data

Before we really get going, I'm going to set up some training data. For
this notebook, I'm going to uniformly generate points in a 10 x 10 square
with the bottom left corner at the origin. The points will be classified
by where the fall with respect to input functions. We'll mostly be using
linear functions to carve up our test space, so we'll create some helper
functions for those as well.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import HTML
from matplotlib.animation import FuncAnimation
import logging
```

```python
# Functions for lines
class SlopeLine():
    """ a line defined by slope and intercept """
    def __init__(self, m, b):
        self.m = m
        self.b = b
        
    def y(self, x):
        return self.m * x + self.b
    
    def x(self, y):
        return (y - self.b) / self.m

def get_slope_line(point_1, point_2):
    """returns slope line object with line determined by the points
    where point_i = (x, y)"""
    
    m = (point_2[1] - point_1[1]) / (point_2[0] - point_1[0])
    b = point_1[1] - m * point_1[0]
    
    return SlopeLine(m, b)

# Functions to generate points
class Inequality():
    """
    given a function f(x) of one variable.
    we will test y < f(x) or y > f(x)
    for a point (x, y)
    """
    def __init__(self, func, ineq_type):
        self.func = func
        
        if ineq_type == '<':
            self.eval = self.less_than
        elif ineq_type == '>':
            self.eval = self.greater_than
        else:
            raise Exception("ineq_type must be '>' or '<'")
        
    def less_than(self, point):
        return point[:,1] < self.func(point[:,0])
    
    def greater_than(self, point):
        return point[:,1] > self.func(point[:,0])
    

def generate_points(n, inequality_lists, random_seed=47):
    """ get n points in the 10x10 square classified by
    inequalities.
    
    inequality_lists should be a list of lists of inequalities.
    
    Points are evaluated to the positive class (1) by the
    interior lists if all the inequalities in that list
    are satisfied. Then, if any of the the interior lists
    return true, the point is classified as true.
    
    """
    np.random.seed(random_seed)
    
    data_x = np.random.uniform(0, 10, (n, 2))
    
    def evaluate_list(inequality_list):
        
        evals = np.array([ineq.eval(data_x) for ineq in inequality_list]) \
                  .transpose()
        return np.array([all(p) for p in evals])
    
    all_evals = np.array([evaluate_list(il) for il in inequality_lists]) \
                  .transpose()
    
    data_y = np.array([1 if any(p) else -1 for p in all_evals])
    
    data = {
        'x_1': data_x[:,0]
        ,'x_2': data_x[:,1]
        ,'y': data_y
    }
    return pd.DataFrame(data)
```

## 1. Backpropagation

Even though neural networks are not an example of convex optimization,
it has still proven useful to train the networks via stochastic gradient
descent (SGD). Even if the found minima are only local or not even
minima, this does not preclude their usefulness.

Let $f(x, W)$ be the output of the neural net with weights $W$ at
the input $x$. Previously, we had normalized the output by using the
sign function to map to $\pm1$. In this case we will not do this, leaving
the output layer to have the identity function as its activation function.
Since the desired values are $\pm1$, we will stick with the hinge loss,
which is defined as $\ell(x, y, W) = \max\{0, 1 - y f(x, W)\}$. As before,
if $1 - y f(x, W) \leq 0$, i.e., if the correct sign is being predicted,
then the hinge loss has the zero vetor as a subgradient (with respect to
$W$). Otherwise a subgradient is $- y\partial_W f(x, W)$. Thus in the 
update step we have to calculate the gradient of the neural network with
respect to the weights.

The backpropagation algorithm is used to calculate this gradient. As noted in 
the [activation functions](https://kevinnowland.com/code/2020/04/19/activation-functions.html)
post, feed forward neural networks are compositions of affine linear 
transformations (of which the weights are parameters) and component-wise 
activation functions. The structure of the composition suggests the
backpropagation algorithm, where derivatives with respect to the weights
in layers closer to the output layer can be used to calculate the
derivatives with respect to weights closer to the input layer.
The backpropagation algorithm actually consists of two parts, a forward
pass through the neural network to calculate inputs and outputs of
each layer and the backward pass to calculate the components of the
gradient.

The typical scheme is to divide the training set into roughly equally
sized _batches_. Once every sample in the training set has been seen
once, that is the end of the first _epoch_. The batch size and the
number of epochs are hyperparameters to tune alongside the learning
rate.

I'm going to upgrade the `SimpleNN` class I used previously to work in
layers. The first step that I'll do is to create a `Neuron` class
and then a `Layer` class that takes will use the `Nuerons`.

```python
class Neuron():
    """ a neuron with an activation function and its derivative """
    def __init__(self, sigma, sigma_prime):
        self._sigma = sigma
        self._sigma_prime = sigma_prime
        
    @property
    def sigma(self):
        return self._sigma
    
    @property
    def sigma_prime(self):
        return self._sigma_prime
        
    def __str__(self):
        return "Neuron()"
    
class Layer():
    """ a layer of a neural network. The layer assumes that
    every neuron has the same activation function except
    for potentially a constant neuron at the end. If
    being used as an input layer, sigma_prime can be
    left as None. If has_const==True, then the nth
    neuron will be the constant neuron"""
    
    def __init__(self, n, sigma, sigma_prime=None, has_const=False):
        self._neurons = [Neuron(sigma, sigma_prime) for i in range(n)]
        self._has_const = has_const
        
        if has_const:
            self._neurons[-1] = Neuron(const, const_prime)
        
    def __str__(self):
        return "Layer() of {} neurons".format(self.width)
    
    def __repr__(self):
        return self.__str__()
    
    @property
    def neurons(self):
        return self._neurons
    
    @property
    def has_const(self):
        return self._has_const
    
    @property
    def width(self):
        return len(self._neurons)
        
    def output(self, x):
        assert len(x) == self.width, "input has bad size"
        return np.array([self._neurons[i].sigma(x[i]) for i in range(len(x))])
    
    def output_prime(self, x):
        assert len(x) == self.width, "input has bad size"
        return np.array([self._neurons[i].sigma_prime(x[i]) for i in range(len(x))])
        
```

The next block of code for the `NeuralNet` class is significantly larger.
The input to the class is a list of `Layer`s. One large improvement over
the `SimpleNeuralNet` is that the weights between layers are automatically
sized and initialized. The Kaiming He initializaiton is used, which is
what is typically used for ReLU activators, but the weights can also be
set by hand.

The bulk of the code is for the forward pass, the backward pass, and the
`fit()` function, which is where stochastic gradient descent with
backpropagation is used. We did not give flexibility in terms of the loss
function, and currently the hinge loss, as explained above, is hard-coded
in.

```python
class NeuralNet():
    """ a neural net that is constructed from the layers
    it is given. Should be given both an input layer 
    and an output layer. Weights will be initialized
    with the proper shape, but as all ones and zeros. They
    can either be initiated using the function included
    here or set by hand.
    
    With T hidden layers, we have T+2 layers of neurons
    as we have the input and output layers. There are only
    T+1 layers of weights, however, as there are no
    weights from the output layer.
    
    We're going to initialize with Kaiming initializaiton.
    If non-ReLU / ELU is used, set w by hand before training."""
    
    def __init__(self, layer_list, check_w_before_set=True, random_seed=None):
        np.random.seed(random_seed)
        
        self._layers = layer_list
        self._checks = check_w_before_set
        
        def get_layer_architecture(l, l_next):
            arch = np.ones(shape=(l_next.width, l.width))
            if l_next.has_const:
                arch[-1,:] = 0
            return arch
            
        self._architecture = [
            get_layer_architecture(self._layers[i], self._layers[i+1])
            for i in range(len(self._layers) - 1)
        ]
        
        self._w = [
            np.random.normal(0, np.sqrt(2.0 / a.shape[1]), size=a.shape) * a
            for a in self._architecture
        ]
        
    def __str__(self):
        num_hidden = len(self._layers) - 2
        width = np.max([l.width for l in self._layers])
        return "NeuralNet() with {0} hidden layer(s) and width {1}".format(num_hidden, width)
        
    @property
    def layers(self):
        return self._layers
    
    @property
    def architecture(self):
        return self._architecture
        
    @property
    def w(self):
        return self._w
    
    @property
    def check_w_before_set(self):
        return self._checks
    
    def _check_architecture(self, w, a):
        """checks that w has zeros wherever a does """
        inds = np.where(a==0)
        if len(inds[0]) == 0:
            return True
        else:
            return all([w[inds[0][i], inds[1][i]] == 0 for i in range(len(inds[0]))])
    
    @w.setter
    def w(self, w_val):
        if self._checks:
            # check that w_val has proper sizes
            assert len(w_val) == len(self._w), "proposed w has bad length"
            assert all([w_val[i].shape == self._w[i].shape for i in range(len(self._w))]), \
                "proposed w has bad shapes somewhere"
            assert all([self._check_architecture(w_val[i], self._architecture[i]) for i in range(len(self._w))]), \
                "proposed w has bad architecture"
        self._w = w_val
        
    def copy_weights(self):
        return [w.copy() for w in self._w]
        
    def _clean_x(self, x):
        """ take data of form (n_samples, n_features)
        that is either a pandas DataFrame, list, or numpy array
        and add a row of 1s then convert to a numpy array and
        transpose. """
        if type(x) == np.ndarray:
            # check if it's already clean
            if x.shape[0] == self._layers[0].width:
                if (x[-1,:] == 1).all():
                    return x
            else:
                return np.append(x, np.ones((x.shape[0], 1)), axis=1).transpose()
        elif type(x) == pd.DataFrame:
            xx = x.copy()
            xx['const'] = 1
            return xx.values.transpose()
        elif type(x) == list:
            return np.array([row + [1] for row in x]).transpose()
    
    def _clean_y(self, y):
        """ take a column vector and make sure it's a numpy column vector """
        if type(y) == np.ndarray:
            return y.reshape(-1, 1)
        elif type(y) == pd.Series:
            return y.values.reshape(-1, 1)
        elif type(y) == list:
            return np.array(y).reshape(-1, 1)
        
        
    def raw_output(self, x):
        """ gets the raw output of the neural network
        
        x needs to have shape (n_samples, n_features)
        without the constant feature appended."""
        
        raw_output, _, _ = self._forward_pass(self._clean_x(x))
        
        return raw_output
    
    def predict(self, x):
        """ gets the -1 or +1 prediction of the neural network """
        return sign(self.raw_output(x))
    
    def _forward_pass(self, x):
        """ forward pass through the neural network recording
        the outputs and inputs as we go. As an internal
        function, the input x will already be an numpy array
        with shape (n_features+1, n_smples). The +1 is
        for the constant feature."""
        
        # every layer has an input
        a = [
            np.zeros((l.width, x.shape[0]))
            for l in self._layers
        ]
                
        # every layer has an output
        o = [
            np.zeros((l.width, x.shape[0]))
            for l in self._layers
        ]
        
        a[0] = x
        o[0] = self._layers[0].output(a[0])
        
        for i in range(1, len(self._layers)):
            a[i] = np.matmul(self._w[i-1], o[i-1])
            o[i] = self._layers[i].output(a[i])
        
        raw_output = o[-1].transpose()
        
        return raw_output, a, o
    
    def _backward_pass(self, a, o, y):
        """ backward pass through the neural network
        using the forward pass results to calculate the gradient
        as well as the true values which in this case
        are +1 or -1 only. y should be a row vector of
        these values with the same number of columns
        as a[-1] and o[-1], which should have indenticial shapes.
        
        The output is the average gradient for the samples given."""
        
        num_samples = y.shape[1]
        
        # first calculate the sigma_prime values for each hidden + output layer
        sigma_prime = [None] + \
            [self._layers[i].output_prime(a[i]) for i in range(1, len(self._layers))]
        
        # now we start the backward pass for real
        delta = [None] + [np.zeros(a[i].shape) for i in range(1, len(self._layers))]
        z = 1 - y * o[-1]
        delta[-1] = z * (z>0)
        
        for i in range(-2, -len(delta)-1, -1):
            # this is w[i+1] bc len(w) = len(delta) - 1
            delta[i] = np.matmul(self._w[i+1].transpose(), (sigma_prime[i+1]) * delta[i+1])
        
        # calculate the average gradient across all samples
        return [
            np.matmul((sigma_prime[i+1] * delta[i+1]), o[i].transpose()) / num_samples
            for i in range(len(self._w))
        ]
    
    def fit(self, x, y, eta=0.1, max_error=0.1, 
            max_epochs=5, batch_size=100, max_iter=None,
            save_data=False, random_seed=None):
        """ use stochastic gradient descent with backpropagation
        to fit the network to the given training data x which
        should be of size (n_samples, n_features),
        y should be of size (n_samples, 1)"""
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        x_ = self._clean_x(x) # (n_features + 1, n_samples) matrix
        y_ = self._clean_y(y) # column vector
        num_samples = y_.shape[0]
        
        # some samples might be left behind
        batches_per_epoch = np.floor(num_samples / batch_size)
        saved_data = []
        
        w = self.copy_weights()
        
        def calculate_train_error():
            current_predictions = self.predict(x_)
            wrong = np.sum(y_ * current_predictions < 0)
            return 1.0 * wrong / num_samples
                    
        curr_iter = 1
        curr_epoch = 1
        curr_batch_number = 0
        batch_indexes = np.arange(num_samples)
        train_error = calculate_train_error()
        
        def get_save_data():
            max_w = np.max([np.max(np.abs(w)) for w in self._w])
            min_w = np.min([np.min(np.abs(w)) for w in self._w])
            current_output = self.raw_output(x_)
            z = 1 - current_output * y_
            avg_loss = np.mean(z * (z>0))

            return {
                'epoch': curr_epoch
                ,'avg_loss': avg_loss
                ,'train_error': train_error
                ,'max_w': max_w
                ,'min_w': min_w
                ,'w': self.copy_weights()
            }
            
        
        if save_data:
            saved_data = [get_save_data()]  
            
        keep_training = True   
        while keep_training:
            
            if curr_batch_number == 0:
                # re-shuffle indexes as neded
                logging.debug("NeuralNet.fit(): starting epoch {}".format(curr_epoch))
                np.random.shuffle(batch_indexes)
                
            batch_ind = batch_indexes[curr_batch_number * batch_size:(curr_batch_number + 1) * batch_size]
            
            x_batch = x_[:, batch_ind]
            y_batch = y_[batch_ind,:]
            
            # forward pass
            _, a, o = self._forward_pass(x_batch)
            
            # backward pass
            grad = self._backward_pass(a, o, y_batch.transpose())
            
            w = [w[i] - eta * grad[i] for i in range(len(w))]
            self.w = [
                (curr_iter * self._w[i] + w[i]) / (curr_iter + 1)
                for i in range(len(self._w))
            ]
            
            train_error = calculate_train_error()
            curr_iter += 1
            curr_batch_number = int((curr_batch_number + 1) % batches_per_epoch)
            
            if curr_batch_number == 0:
                curr_epoch += 1
                
                if save_data:
                    saved_data += [get_save_data()]
                    
            keep_training = train_error >= max_error and curr_epoch <= max_epochs
            if max_iter is not None:
                keep_training = keep_training and curr_iter <= max_iter
            
            
        if curr_epoch > max_epochs:
            logging.warning("NeuralNet.fit():no convergence, train_error above max_error")
        else:
            logging.warning("NeuralNet.fit(): converged during epoch {}.".format(curr_epoch-1))
        
        if save_data:
            return saved_data
        else:
            return None
        
```

Now that we have that implemented, let's code up some activation
functions that we might want to play with in the
future.. We will probably stick
with the ReLU functions, but maybe we'll get ambitious. 

```python
# here are some activation functions and derivatives we might use
def const(x):
    if type(x) == np.ndarray:
        return np.ones(x.shape)
    elif type(x) == list:
        return np.ones(len(x))
    else:
        return 1
    
def const_prime(x):
    if type(x) == np.ndarray:
        return np.zeros(x.shape)
    elif type(x) == list:
        return np.zeros(len(x))
    else:
        return 0

def relu(x):
    return x * (x>0) + 0 * (x<=0)

def relu_prime(x):
    return 1 * (x>0) + 0 * (x<=0)

def sign(x):
    return 1 * (x > 0) + -1 * (x <= 0)
```

## 2. Training

Now that we're set up, let's get training on some data.

### Setup Training Data

Now let's setup the training data that we want the neural net to learn.
We'll go simple by having two lines, $x_2=-x_1 + 10$ and $x_2=x_1$ and the positive
class will be below both of these lines. All of our data will be in the square
$[0, 10]\times[0, 10]$, such that the positive class forms a triangle of base 10
and height 5.

```python
lines = [SlopeLine(1, 0), SlopeLine(-1, 10)]
ineqs = [Inequality(lines[0].y, '<'), Inequality(lines[1].y, '<')]
train_data = generate_points(1000, [ineqs])

fig, ax = plt.subplots(1, 1)
lims = np.array([0, 10])

ax.set_xlim(lims)
ax.set_ylim(lims)

c_tr = train_data['y'] == 1
ax.plot(train_data[c_tr]['x_1'], train_data[c_tr]['x_2'], 'o', label='pos')
ax.plot(train_data[~c_tr]['x_1'], train_data[~c_tr]['x_2'], 'o', label='neg')
ax.legend(loc='upper center', bbox_to_anchor=(1.1, 1))

plt.show()
```


![png](/assets/images/Backpropogation_files/Backpropogation_10_0.png)


Note that this means that if every prediction is positive, the training
error is ~75% whereas if every prediction is negative than the trianing
error is ~25%. Any error significantly under 25% is good.

### Training attempt no. 1 - random initialization

Now, let's try training a neural network. We know from the 
[activation functions](https://kevinnowland.com/code/2020/04/19/activation-functions.html)
post that we only need one-hidden layer of width 2 using ReLU activation functions in order
to learn this shape, so let's not give ourselves any more freedom than is
required. At least initially.

You may see that "no. 1" up there and think that something might go wrong. 
Have some faith! We're going to set `save_data=True` so that we can analyze 
what happens after each epoch. You know, just in case this doesn't work out.

```python
input_layer = Layer(3, lambda x: x, None, True)
hidden_layer = Layer(2, relu, relu_prime, False)
output_layer = Layer(1, lambda x: x, const, False)

nn_1 = NeuralNet([input_layer, hidden_layer, output_layer], random_seed=12)

data = nn_1.fit(train_data[['x_1', 'x_2']], train_data['y'],
              eta=.001,
              max_error=0.1,
              max_epochs=100,
              batch_size=100,
              save_data=True,
              random_seed=6)
```

    WARNING:root:NeuralNet.fit():no convergence, train_error above max_error


So with 100 epochs, the training error stayed above the specified maximum error of 10%.
Let's plot what the initial and final predictions would have been as well as the
evolution of the loss and error over time.

```python
def plot_training_info(train_data, fit_data, nn):
    
    predictions = train_data[['x_1', 'x_2', 'y']].copy()
    predictions['final'] = nn.predict(train_data[['x_1', 'x_2']])
    nn.w = fit_data[0]['w']
    predictions['initial'] = nn.predict(train_data[['x_1', 'x_2']])
    nn.w = fit_data[-1]['w'] # restore the current trained state
    
    fig, ax = plt.subplots(2, 2, figsize=(16, 12))

    ax[0, 0].set_xlim([0, 10])
    ax[0, 0].set_ylim([0, 10])
    ax[0, 1].set_xlim([0, 10])
    ax[0, 1].set_ylim([0, 10])
    
    ax[0, 0].plot([0, 5], [0, 5], color="black")
    ax[0, 0].plot([5, 10], [5, 0], color="black")
    ax[0, 1].plot([0, 5], [0, 5], color="black")
    ax[0, 1].plot([5, 10], [5, 0], color="black")

    c = predictions['initial'] == 1
    ax[0, 0].plot(predictions[c]['x_1'], predictions[c]['x_2'], 'o', label='pos')
    ax[0, 0].plot(predictions[~c]['x_1'], predictions[~c]['x_2'], 'o', label='neg')
    ax[0, 0].set_title('initial predictions with train error {}'.format(data[0]['train_error']))

    c = predictions['final'] == 1
    ax[0, 1].plot(predictions[c]['x_1'], predictions[c]['x_2'], 'o', label='pos')
    ax[0, 1].plot(predictions[~c]['x_1'], predictions[~c]['x_2'], 'o', label='neg')
    ax[0, 1].set_title('final predictions with train error {}'.format(data[-1]['train_error']))

    ax[1, 0].plot([d['train_error'] for d in data])
    ax[1, 0].set_title('Train Error')
    ax[1, 0].set_xlabel('epoch')

    ax[1, 1].plot([d['avg_loss'] for d in data])
    ax[1, 1].set_title('Average Loss (hinge loss)')
    ax[1, 1].set_xlabel('epoch')   
    
    return fig, ax

fig, ax = plot_training_info(train_data, data, nn_1)

print("minimum training error:", min([d['train_error'] for d in data]))
plt.show()
```

    minimum training error: 0.186



![png](/assets/images/Backpropogation_files/Backpropogation_15_1.png)


The training error very quickly gets fixed as every point is predicted to be
negative, but the hinge loss actually keeps decreasing. This suggests
that the hinge-loss, which is what the stochastic gradient descent procedure is
actually trying to minimize, might not be the correct loss function.

Examining the weights, we can see that initially the weights in $W^1$
from the hidden layer to the output layer have different signs. The ReLU
activations will only ever put out non-negative values, so this is a
requirement for the shallow neural network to predict both positive
and negative values.

```python
print(data[0]['w'][1])
```

    [[ 0.00512708 -0.12022767]]


However, as soon as the third epoch, both of these weights are negative,
and the neural network will now only predict negative values. this remains true
for every epoch afterwards as well.

```python
print(data[3]['w'][1])
```

    [[-0.00248362 -0.12259217]]


### Training attempt no. 2 - explosion!

For the second attempt, let's increase the learning parameter `eta` and see
what happens if we start making bigger jumps.

```python
np.seterr('raise')

nn_2 = NeuralNet([input_layer, hidden_layer, output_layer], random_seed=12)

data = nn_2.fit(train_data[['x_1', 'x_2']], train_data['y'],
              eta=.01,
              max_error=0.1,
              max_epochs=100,
              batch_size=100,
              save_data=True,
              random_seed=6)
```


    ---------------------------------------------------------------------------

    FloatingPointError                        Traceback (most recent call last)

    <ipython-input-15-a16173c45498> in <module>
          9               batch_size=100,
         10               save_data=True,
    ---> 11               random_seed=6)
    

    <ipython-input-4-5116cb148eac> in fit(self, x, y, eta, max_error, max_epochs, batch_size, max_iter, save_data, random_seed)
        253 
        254             # backward pass
    --> 255             grad = self._backward_pass(a, o, y_batch.transpose())
        256 
        257             w = [w[i] - eta * grad[i] for i in range(len(w))]


    <ipython-input-4-5116cb148eac> in _backward_pass(self, a, o, y)
        176         for i in range(-2, -len(delta)-1, -1):
        177             # this is w[i+1] bc len(w) = len(delta) - 1
    --> 178             delta[i] = np.matmul(self._w[i+1].transpose(), (sigma_prime[i+1]) * delta[i+1])
        179 
        180         # calculate the average gradient across all samples


    FloatingPointError: overflow encountered in matmul


Well this is pretty easy to interpret. What ends up happening is that
we are running into the _exploding gradients_ problem that is caused by
the recursive nature of the backpropagation algorithm. Since our fit
algorithm updates the neural network in place as it goes (sorry all you
functional programmers out there, it hurts me a bit too), we can
examine the weights at the moment that things exploded. Let's print
out the largest absolute value of any of the weights:

```python
print(np.max([np.max(np.abs(w)) for w in nn_2.w]))
```

    9.142136893410605e+151


Ways to prevent this that I have come across include putting in a simple maximum
weight value or being more careful with the value of `eta` that is used.
I'm not going to do either of these in this notebook.

### Training attempt no. 3 - start smarter

For the third training attempt, I'm going to start the neural network
from an advantageous position by fixing the weights. As discussed above,
we need the weights in $W^1$ to have different signs, so we'll initialize them
to be $+1$ and $-1$. With that done, our work in the activation functions
post informs us that if the initial weights in $W^0$ did nothing to
transfrom the inputs, that the predicted output would be positive in
the fourth quadrant as well as below the line $x_2=x_1$ in the 
first quadrant and negative elswhere. We can in fact design 
an affine linear transformation $W^0$ by hand that collapses
the vector $(1, 1)$ onto the $x_1$-axis and then rotates the
picture by $-\pi/4$ degrees to make the shape point that we
want. We can then shift the point into the 
$[0, 10]\times[0, 10]$ box where are points lie. Working
all that out, one might initialize the neural net as
follows:

```python
input_layer = Layer(3, lambda x: x, None, True)
hidden_layer = Layer(2, relu, relu_prime, False)
output_layer = Layer(1, lambda x: x, const, False)

nn_3 = NeuralNet([input_layer, hidden_layer, output_layer])

initial_weights = [
    np.array([[1, -1, 1], [2, 0, -15]])
    ,np.array([[1, -1]])
]
nn_3.w = initial_weights

data = nn_3.fit(train_data[['x_1', 'x_2']], train_data['y'],
              eta=.001,
              max_error=0.1,
              max_epochs=100,
              batch_size=100,
              save_data=True,
              random_seed=6)
```

    WARNING:root:NeuralNet.fit(): converged during epoch 10.


Hey hey, we got down below the threshold! Let's take a look at
what the initial predictions would have been and the final
step predictions.

```python
fig, ax = plot_training_info(train_data, data, nn_3)
print("minimum training error:", min([d['train_error'] for d in data]))
plt.show()
```

    minimum training error: 0.098



![png](/assets/images/Backpropogation_files/Backpropogation_27_1.png)


Looks like the triangle of positivity shifted more closely into position!
However, the good results here are also the result of choosing an `eta` value
that worked as well as a good number of epochs. This result is not robust.
We can recreate an exploding gradient error by setting `max_error` to be
`0.05` and increasing `max_epochs` to 200.

### Training attempt no. 4 - train more

Now let's lower the `max_error` and see what happens as training continues.

```python
nn_4 = NeuralNet([input_layer, hidden_layer, output_layer])

initial_weights = [
    np.array([[1, -1, 1], [2, 0, -15]])
    ,np.array([[1, -1]])
]
nn_4.w = initial_weights

data = nn_4.fit(train_data[['x_1', 'x_2']], train_data['y'],
              eta=.001,
              max_error=0.05,
              max_epochs=100,
              batch_size=100,
              save_data=True,
              random_seed=6)
```

    WARNING:root:NeuralNet.fit():no convergence, train_error above max_error


```python
fig, ax = plot_training_info(train_data, data, nn_4)
print("minimum training error:", min([d['train_error'] for d in data]))
plt.show()
```

    minimum training error: 0.054



![png](/assets/images/Backpropogation_files/Backpropogation_30_1.png)


This is interesting, Here we see that the training error and hinge loss do 
mirror each other as we would hope. However, it appears that we are 
overshooting the minimum. Let's look at a video evolution of the
training and see what happens.

```python
nn_ = NeuralNet([input_layer, hidden_layer, output_layer])

fig, ax = plt.subplots()

ax.set_xlim([0, 10])
ax.set_ylim([0, 10])
ax.plot([0, 5], [0, 5], color='black')
ax.plot([5, 10], [5, 0], color='black')

def init():
    return []

def animate(i):
    nn_.w = data[i]['w']
    s_predict = nn_.predict(train_data[['x_1', 'x_2']])
    c = s_predict == 1
    ax.set_title("epoch: {}".format(i+1))
    animList = ax.plot(train_data[c]['x_1'], train_data[c]['x_2'], 'C0o',
                       train_data[~c]['x_1'], train_data[~c]['x_2'], 'C1o')
    return animList


anim = FuncAnimation(fig, animate, init_func=init,
                     frames=100, interval=100, blit=True)

HTML(anim.to_html5_video())
```




<video width="432" height="288" controls autoplay loop>
  <source type="video/mp4" src="data:video/mp4;base64,AAAAHGZ0eXBNNFYgAAACAGlzb21pc28yYXZjMQAAAAhmcmVlAAI87W1kYXQAAAKgBgX//5zcRem9
5tlIt5Ys2CDZI+7veDI2NCAtIGNvcmUgMTUyIC0gSC4yNjQvTVBFRy00IEFWQyBjb2RlYyAtIENv
cHlsZWZ0IDIwMDMtMjAxNyAtIGh0dHA6Ly93d3cudmlkZW9sYW4ub3JnL3gyNjQuaHRtbCAtIG9w
dGlvbnM6IGNhYmFjPTEgcmVmPTMgZGVibG9jaz0xOjA6MCBhbmFseXNlPTB4MzoweDExMyBtZT1o
ZXggc3VibWU9NyBwc3k9MSBwc3lfcmQ9MS4wMDowLjAwIG1peGVkX3JlZj0xIG1lX3JhbmdlPTE2
IGNocm9tYV9tZT0xIHRyZWxsaXM9MSA4eDhkY3Q9MSBjcW09MCBkZWFkem9uZT0yMSwxMSBmYXN0
X3Bza2lwPTEgY2hyb21hX3FwX29mZnNldD0tMiB0aHJlYWRzPTkgbG9va2FoZWFkX3RocmVhZHM9
MSBzbGljZWRfdGhyZWFkcz0wIG5yPTAgZGVjaW1hdGU9MSBpbnRlcmxhY2VkPTAgYmx1cmF5X2Nv
bXBhdD0wIGNvbnN0cmFpbmVkX2ludHJhPTAgYmZyYW1lcz0zIGJfcHlyYW1pZD0yIGJfYWRhcHQ9
MSBiX2JpYXM9MCBkaXJlY3Q9MSB3ZWlnaHRiPTEgb3Blbl9nb3A9MCB3ZWlnaHRwPTIga2V5aW50
PTI1MCBrZXlpbnRfbWluPTEwIHNjZW5lY3V0PTQwIGludHJhX3JlZnJlc2g9MCByY19sb29rYWhl
YWQ9NDAgcmM9Y3JmIG1idHJlZT0xIGNyZj0yMy4wIHFjb21wPTAuNjAgcXBtaW49MCBxcG1heD02
OSBxcHN0ZXA9NCBpcF9yYXRpbz0xLjQwIGFxPTE6MS4wMACAAAChfmWIhAA///73aJ8Cm1pDeoDk
lcUl20+B/6tncHyP6QMAAAMAAK6tMz+4SbviBhhXv2X4dM6rWNTXiJfp1SSnzR+E1sLUtW0leAQo
e9COiQd5JOdSIx6KZ9i2907WQxLe3trEJVC3ZsNw2Yyw/zFkfFfE7K3VSK9atBmw36HR6OqU+1/d
maPSruDyASGgIGBOvtoX23l2jJ2OJ7kimh7YOAjOaiKhbBlzbiibHaWGabtFTg4HgazOh+9jAlBb
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
XHRraGQAAAADAAAAAAAAAAAAAAABAAAAAAAAJxAAAAAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAA
AAAAAAABAAAAAAAAAAAAAAAAAABAAAAAAbAAAAEgAAAAAAAkZWR0cwAAABxlbHN0AAAAAAAAAAEA
ACcQAAAIAAABAAAAAAYgbWRpYQAAACBtZGhkAAAAAAAAAAAAAAAAAAAoAAABkABVxAAAAAAALWhk
bHIAAAAAAAAAAHZpZGUAAAAAAAAAAAAAAABWaWRlb0hhbmRsZXIAAAAFy21pbmYAAAAUdm1oZAAA
AAEAAAAAAAAAAAAAACRkaW5mAAAAHGRyZWYAAAAAAAAAAQAAAAx1cmwgAAAAAQAABYtzdGJsAAAA
s3N0c2QAAAAAAAAAAQAAAKNhdmMxAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAAAAbABIABIAAAASAAA
AAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAGP//AAAAMWF2Y0MBZAAV/+EA
GGdkABWs2UGwloQAAAMABAAAAwBQPFi2WAEABmjr48siwAAAABx1dWlka2hA8l8kT8W6OaUbzwMj
8wAAAAAAAAAYc3R0cwAAAAAAAAABAAAAZAAABAAAAAAUc3RzcwAAAAAAAAABAAAAAQAAAtBjdHRz
AAAAAAAAAFgAAAAIAAAIAAAAAAEAAAwAAAAAAQAABAAAAAABAAAIAAAAAAEAAAwAAAAAAQAABAAA
AAABAAAMAAAAAAEAAAQAAAAAAQAACAAAAAABAAAMAAAAAAEAAAQAAAAAAQAADAAAAAABAAAEAAAA
AAEAABAAAAAAAgAABAAAAAABAAAMAAAAAAEAAAQAAAAAAQAAEAAAAAACAAAEAAAAAAEAABQAAAAA
AQAACAAAAAABAAAAAAAAAAEAAAQAAAAAAQAAFAAAAAABAAAIAAAAAAEAAAAAAAAAAQAABAAAAAAB
AAAMAAAAAAEAAAQAAAAAAQAADAAAAAABAAAEAAAAAAEAABAAAAAAAgAABAAAAAABAAAMAAAAAAEA
AAQAAAAAAQAAEAAAAAACAAAEAAAAAAEAABQAAAAAAQAACAAAAAABAAAAAAAAAAEAAAQAAAAAAQAA
FAAAAAABAAAIAAAAAAEAAAAAAAAAAQAABAAAAAABAAAMAAAAAAEAAAQAAAAAAQAAFAAAAAABAAAI
AAAAAAEAAAAAAAAAAQAABAAAAAABAAAQAAAAAAIAAAQAAAAAAQAAFAAAAAABAAAIAAAAAAEAAAAA
AAAAAQAABAAAAAABAAAUAAAAAAEAAAgAAAAAAQAAAAAAAAABAAAEAAAAAAEAAAwAAAAAAQAABAAA
AAABAAAUAAAAAAEAAAgAAAAAAQAAAAAAAAABAAAEAAAAAAEAABQAAAAAAQAACAAAAAABAAAAAAAA
AAEAAAQAAAAAAQAADAAAAAABAAAEAAAAAAEAABQAAAAAAQAACAAAAAABAAAAAAAAAAEAAAQAAAAA
AQAACAAAAAABAAAUAAAAAAEAAAgAAAAAAQAAAAAAAAABAAAEAAAAAAEAABQAAAAAAQAACAAAAAAB
AAAAAAAAAAEAAAQAAAAAAQAADAAAAAABAAAEAAAAABxzdHNjAAAAAAAAAAEAAAABAAAAZAAAAAEA
AAGkc3RzegAAAAAAAAAAAAAAZAAApCYAAEFTAAAzsgAAKEEAAB5EAAAZbgAAEcYAAAyCAAAOlAAA
ASsAAAp5AAAJpgAAAM8AAAhQAAAAtQAABdUAAAbDAAAAXgAABJoAAABfAAAFSgAAAKUAAABwAAAD
ywAAAHAAAATCAAAAcAAAAFcAAAYwAAAAowAAAFcAAABlAAAD7QAAAIEAAABUAAAASwAAA4sAAABf
AAAC2QAAAFwAAARJAAAArQAAAF8AAAJnAAAAXgAAAd0AAABGAAAASwAAA6sAAACMAAAAUgAAAEQA
AAKbAAAAZgAAAEkAAABDAAAB0AAAAFgAAAUUAAAAqgAAAGkAAABlAAAC0gAAAE0AAABaAAADqQAA
AIcAAABOAAAAZAAAAgAAAABbAAAAUgAAAEYAAAHgAAAAVwAAA6kAAACKAAAAbwAAAGYAAAPoAAAA
aQAAAFoAAABOAAACQQAAAGQAAAJQAAAAYAAAAHUAAABOAAABMQAAAf0AAABwAAAAUAAAAEUAAAFp
AAAAYgAAAE4AAABRAAABvQAAAHIAAAAUc3RjbwAAAAAAAAABAAAALAAAAGJ1ZHRhAAAAWm1ldGEA
AAAAAAAAIWhkbHIAAAAAAAAAAG1kaXJhcHBsAAAAAAAAAAAAAAAALWlsc3QAAAAlqXRvbwAAAB1k
YXRhAAAAAQAAAABMYXZmNTguMjkuMTAw
">
  Your browser does not support the video tag.
</video>



It looks like it just overshoots, as if `eta` is too large or perhaps the `batch_size` is
too small? 

### Training attempt no. 5 - slower

Let's train the same starting position but cut `eta` in half to slow down
the learning.

```python
nn_5 = NeuralNet([input_layer, hidden_layer, output_layer])

initial_weights = [
    np.array([[1, -1, 1], [2, 0, -15]])
    ,np.array([[1, -1]])
]
nn_5.w = initial_weights

data = nn_5.fit(train_data[['x_1', 'x_2']], train_data['y'],
              eta=.0005,
              max_error=0.05,
              max_epochs=100,
              batch_size=100,
              save_data=True,
              random_seed=6)


fig, ax = plot_training_info(train_data, data, nn_5)
print("minimum training error:", min([d['train_error'] for d in data]))
plt.show()
```

    WARNING:root:NeuralNet.fit():no convergence, train_error above max_error


    minimum training error: 0.054



![png](/assets/images/Backpropogation_files/Backpropogation_34_2.png)


I performed more experiments as well. The larger the batch size, the longer this more stable
period endured, but I could still increase the number of epochs until the average loss
and indeed the gradients eventually exploded. It seems like the behavior was basically
the same in all cases, but with a slower learning rate and a larger batch size, the
best possible behavior would lost longer, but not be fundamentally different. Perhaps
rounding errors in calculating the average values used in the weights eventually win out?
I'm honestly not sure how to describe the behavior. Note that we did achieve the same
minimum training error here as we did above.

### Training attempt no. 6 - wider

Finally, let's try something different. Previously we were training from an advantageous 
starting position because the random starting position did not seem to help.
What if we increase the width of this shallow neural net? I'll spare the
many training runs with different random seeds, but I found a very minor 
success with with a width 8 neural network without a constant term. Pretty much
everything blew up, and as you'll see below, nothing particularly good happened
here either.

```python
input_layer = Layer(3, lambda x: x, None, True)
hidden_layer = Layer(8, relu, relu_prime, False)
output_layer = Layer(1, lambda x: x, const, False)

nn_6 = NeuralNet([input_layer, hidden_layer, output_layer], random_seed=5)

data = nn_6.fit(train_data[['x_1', 'x_2']], train_data['y'],
              eta=.0005,
              max_error=0.1,
              max_epochs=200,
              batch_size=200,
              save_data=True,
              random_seed=6)

fig, ax = plot_training_info(train_data, data, nn_6)
print("minimum training error:", min([d['train_error'] for d in data]))
plt.show()
```

    WARNING:root:NeuralNet.fit():no convergence, train_error above max_error


    minimum training error: 0.144



![png](/assets/images/Backpropogation_files/Backpropogation_36_2.png)


### Training 7 - deeper

Finally, I added a couple layers that were decently wide to see if by pure
chance I could get something that worked. Nothing was spectacularly successful,
so I will have to reflect further. Below is a width 8 neural network with
3 hidden layers that did "okay, not great". It is not representative of
my experience, as most things blew up without finding a good regime.
In this sense it seems that intelligent neetwork design and luck are
needed to train well. Given how fertile the field is, more thoughtful
network design must be winning in the end (I'm not that much a skeptic).
I even traded out the hinge loss for the $\ell^2$ loss, but this did
not improve things too much overall.

```python
input_layer = Layer(3, lambda x: x, None, True)
hidden_layer_1 = Layer(6, relu, relu_prime, True)
hidden_layer_2 = Layer(8, relu, relu_prime, True)
hidden_layer_3 = Layer(6, relu, relu_prime, True)
output_layer = Layer(1, lambda x: x, const, False)

layers = [
    input_layer
    ,hidden_layer_1
    ,hidden_layer_2
    ,hidden_layer_3
    ,output_layer
]

nn_7 = NeuralNet(layers, random_seed=1)

data = nn_7.fit(train_data[['x_1', 'x_2']], train_data['y'],
              eta=.0005,
              max_error=0.05,
              max_epochs=100,
              batch_size=300,
              save_data=True,
              random_seed=6)

fig, ax = plot_training_info(train_data, data, nn_7)
print("minimum training error:", min([d['train_error'] for d in data]))
plt.show()
```

    WARNING:root:NeuralNet.fit():no convergence, train_error above max_error


    minimum training error: 0.108



![png](/assets/images/Backpropogation_files/Backpropogation_38_2.png)


## Backpropagation Details

Let's recall the setup. We have a layer $V\_t$ of $n\_t$ neurons. The
input to $V\_t$ is $a\_t$ and the output is $o\_t = \sigma\_t(a\_t)$. 
Also, $a\_t = W^{t-1}o\_{t-1}$, and $W^{t-1}$ is a matrix of size
$n\_t\times n\_{t-1}$. The activation function $s\_t$ will act
componentwise on the $n\_t$-dimensional vector $a\_t$. $W^0$ is the
weights from the input layer, and $W^T$ is the weights from
the last hidden layer to the output layer, which consists of a single neuron
that has the identity function as its activator.

Let's examine how (the positive part of) the loss $\ell(x, y, W)=1-y f(x, W)$ 
depends on $w^T\_k$, a weight from the last hidden layer to the output
layer. We calculate

$$
\begin{align*}
    \frac{\partial\ell}{\partial{w^T_k}}
    &= \frac{\partial}{\partial{w^T_k}}(1 - y\sigma_{T+1}'(W^To_T)) \\
    &= -y \sigma_{T+1}'(a_{T+1})o_{T-1, k}.
\end{align*}
$$

We get the $o\_{T-1, k}$ term because $a\_{T+1}=W^To\_T = \sum\_i w^T\_i o\_{T, i}$.
Now let's consider the weight $w^T\_{k, n}$ from neuron $n$ in $V\_{T-1}$ 
to neuron $k$ in $V\_T$. For this partial derivative we first find that

$$
\begin{align*}
    \frac{\partial\ell}{\partial{w^{T-1}_{k, n}}}
    &= \frac{\partial}{\partial{w^{T-1}_{k, n}}}(1 - y \sigma_{T+1}(W^T\sigma_T(W^{T-1}o_{T-1}))) \\
    &= -y \sigma_{T+1}'(a_{T+1})\frac{\partial}{\partial{w^{T-1}_{k, n}}} W^T\sigma_T(W^{T-1}o_{T-1})
\end{align*}
$$

Stopping for a moment to consider this, $w^{T-1}\_{k, n}$ only feeds neuron $k$ in $V\_T$,
so only the $k$th component of $\sigma\_T$ is affected. Since the output layer only has one neuron,
the only weight from this neuron is $w^T\_k$, so we end up with

$$
\begin{align*}
    \frac{\partial\ell}{\partial{w^{T-1}_{k, n}}}
    &= -y \sigma_{T+1}'(a_{T+1}) w^T_k \sigma_{T, k}'(a_T) o_{T-1, n}.
\end{align*}
$$

Note that we've repeated $-y\sigma\_{T+1}'(a\_{T+1})$. Let's get a little ahead 
of ourselves define

$$
    \delta_{T+1} := -y, \quad
    \delta_{T, k} := \delta_{T+1,k}\sigma_{T+1}'(a_{T+1})w^T_k.
$$

Such that

$$
    \frac{\partial\ell}{\partial{w^T_k}} = \delta_{T+1}\sigma_{T+1}'(a_{T+1})o_{T-1, k},\quad
    \frac{\partial\ell}{\partial{w^{T-1}_{k, n}}} = \delta_T\sigma_{T, k}'(a_T) o_{T-1, n}.
$$

This already shows how backpropogation might work. A first forward pass through the neural
net is necessary to calculate the $a\_i$ and $o\_i$, while on the backward pass we
calculate the derivates one layer at a time, from output to input, storing the $\delta$
values along the way to help calculate successive derivatives. However, since the output
layer only has one neuron, it's a little bit cheating to say that we are done, so
let's calculate a derivative with respect to 
$w^{T-2}\_{n, m}$ 
to get the full scope of
the behavior. What changes here is that while $w^{T-2}\_{n, m}$ only feeds neuron $n$
in $V\_{T-1}$, but neuron $n$ in $V\_{T-1}$ feeds every neuron in $V\_T$, of which there are
multiple. So now the calculate looks like

$$
\begin{align*}
    \frac{\partial\ell}{\partial{w^{T-2}_{n, m}}}
    &= -y \sigma_{T+1}'(a_{T+1})\sum_{i=1}^{n_T}w^T_i\sigma_{T, i}'(a_T)w^{T-1}_{i, n}\sigma_{T-1, n}'(a_{T-1})o_{T-2, m} \\
    &= \sum_{i=1}^{n_T}\delta_{T, i}\sigma_{T, i}'(a_T)w^{T-1}_{i, n}\sigma_{T-1, n}'(a_{T-1})o_{T-2, m}.
\end{align*}
$$

Therefore we define

$$
    \delta_{T-1, n} = \sum_{i=1}^{n_T}\delta_{T, i}\sigma_{T, i}'(a_T)w^{T-1}_{i, n}
$$

such that

$$
    \frac{\partial\ell}{\partial w^{T-2}_{n, m}} = \delta_{T-1, n}\sigma_{T-1, n}'(a_{T-1})o_{T-2, m}.
$$

What we've shown is that the backward pass can be summarized by

$$
\begin{align*}
    \delta_{T+1} &= -y, \\
    \delta_{t} &= {}^TW^t(\sigma_{t+1}'(a_{t+1})*\delta_{t+1}), \\
    \frac{\partial\ell}{\partial w^t} &= (\sigma_{t+1}'(a_{t+1})*\delta_{t+1})({}^To_t),
\end{align*}
$$

where ${}^TA$ is the transpose of $A$ and $(x*y)$ is the elementwise product. Looking
at the structure of the recursion, it becomes clear what each component is. 
The $\delta\_{t+1}$ vectors hold the information for all derivatives beyond 
(closer to the output) layer $t+1$. Then $\sigma'\_{t+1}$ encodes the derivatives at
layer $t+1$. Since the activation functions at a layer act diagonally (for now), 
you can take the elementwise product, as
you would get a diagonal matrix for the Jacobian of activation functions.
Then, the input $a\_{t+1}$ that feeds into layer $t+1$ depends has
the form $W^to\_t$. If you want to go another layer down the network before taking
a derivative with respcet to the weights at a lower layer, you have to take the
derivative with respect to $o\_t$ and you end up with a $W^t$ term. Since you are
going from a layer that has size $n\_{t+1}$ to a layer of size $n\_t$ you need
a matrix of sixe $n\_t\times n\_{t+1}$, which is why you get ${}^TW\_t$. 
Similarly, if you are at the layer you want and are taking the derivative with
respect to $W^t$, then you end up with ${}^To\_t$ instead.

In other words, we are taking derivatives of the loss due to layer
$t+1$, which we can denote $\ell\_{t+1}$ which takes in $\sigma\_{t+1}$
which in turn has $W^to\_t$ as its input. If we are goign further
down the chain through $\ell\_t$ we need a derivative with respect to
$o\_t$, if not we take a derivative with respect to $W^t$.
So we have decomposed the loss $\ell$ as $\ell = \ell\_{T+1}\circ\ell\_T\circ\cdots$
where each step looks like
$
 \ell\_{t+1}(\sigma\_{t+1}(W^to\_t)).
$

The recursive formula also tells us how we would need to edit backpropagation
if we change the form of the neural net. If the mapping between spaces is no
longer linear, than the $o\_t$ and $W^t$ terms would change depending on the
function used. If the activation functions were no longer elementwise,
then instead of $\sigma'\_{t+1}$ we would need a full Jacobian matrix.
This procedure also begs the question of what should be done if the 
activation functions are not differentiable / do not have a well-defined
subgradients. Although practically, piece-wise differentiability should
suffice.

Just what sort of generalizations would be useful would be the big question.
Currently, as someone building up intuition from scratch, I'd say that 
the reason we stick with linear mappings and elementwise activators is that
it's hard to understand anything more complex. Putting anything more complicated
in represents a form of prior knowledge of bias. Without that, it should
always be effective -- given a sufficiently long training period -- to 
increase the width and depth, as it seems from our investigations in the
activation functions notebook that this is all we need in order to represent
more complex functions. It has been remarked that all of mathematics is time
spent reducing to linear operations, and this is yet another example of that
phenomenon.
