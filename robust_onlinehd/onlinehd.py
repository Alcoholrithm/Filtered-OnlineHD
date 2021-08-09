import math
from typing import Union

import torch
import numpy as np

from . import spatial
from . import Encoder

from . import _fasthd

class OnlineHD(object):
    '''
    Hyperdimensional classification algorithm. OnlineHD utilizes a `(c, d)`
    sized tensor for the model initialized with zeros. Every `d`-sized vector on
    this matrix will be the high dimensional representation of each class,
    called class hypervector.

    Args:
        classes (int, > 0): The number of classes of the problem.

        features (int, > 0): Dimensionality of original data.

        dim (int, > 0): The target dimensionality of the high dimensional
            representation.

    Example:
        >>> import onlinehd
        >>> dim = 10000
        >>> n_samples = 1000
        >>> features = 100
        >>> clusters = 5
        >>> x = torch.randn(n_samples, features) # dummy data
        >>> y = torch.randint(0, classes, [n_samples]) # dummy data
        >>> model = onlinehd.OnlineHD(classes, features, dim=dim)
        >>> if torch.cuda.is_available():
        ...     print('Training on GPU!')
        ...     model = model.to('cuda')
        ...     x = x.to('cuda')
        ...     y = y.to('cuda')
        ...
        Training on GPU!
        >>> model.fit(x, y, epochs=10)
        >>> ypred = model(x)
        >>> ypred.size()
        torch.Size([1000])
    '''

    def __init__(self, kernel_size = 1, scaler = None, classes : int = 10, features : int = 784, dim : int = 4000, path = None):
        if path != None:
            self.load_model(path)
            return

        self.classes = classes
        self.dim = dim
        self.encoder = Encoder(features, dim)
        self.model = torch.zeros(self.classes, self.dim)
        self.criterias = []
        self.kernel_size = kernel_size
        self.scaler = scaler
        self.device = 'cpu'
        self.layer = torch.nn.MaxPool2d(kernel_size = self.kernel_size, stride = 1, padding = self.kernel_size // 2)

    def get_criteria_from_data(self, x, bins):
        dist = np.histogram(x, bins)[1]
        criterias = [[dist[i].item(), dist[i + 1].item(), dist[i].item()] for i in range(len(dist) - 1)]
        criterias[-1][-1] = criterias[-1][1]
        return criterias

    def set_criterias(self, x, bins = None, is_data = True):
            if is_data:
                self.criterias = self.get_criteria_from_data(x, bins)
            else:
                self.criterias = x

    def local_maximum(self, imgs):
        if self.kernel_size == 1:
            return imgs
            
        outs = imgs.clone().detach().permute((0, 3, 1, 2))
        outs = self.layer(outs).permute((0, 2, 3, 1))
        return outs

    def quantizing(self, imgs):
        outs = imgs.clone().detach()
        for c in self.criterias:
            outs[(outs >= c[0]) & (outs < c[1])] = c[2]
    
        return outs  
    
    def load_model(self, path):
        temp = torch.load(path)['model']

        self.classes = temp.classes
        self.dim = temp.dim
        self.encoder = temp.encoder
        self.model = temp.model
        self.criterias = temp.criterias
        self.kernel_size = temp.kernel_size
        self.scaler = temp.scaler
        self.device = temp.device
        self.layer = temp.layer
    
    

    def __call__(self, x : torch.Tensor, encoded : bool = False):
        '''
        Returns the predicted class of each data point in x.

        Args:
            x (:class:`torch.Tensor`): The data points to predict. Must
                have size `(n?, dim)` if `encoded=False`, otherwise must
                have size `(n?, features)`.

            encoded (bool): Specifies if input data is already encoded.

        Returns:
            :class:`torch.Tensor`: The predicted class of each data point.
            Has size `(n?,)`.
        '''

        return self.scores(x, encoded=encoded).argmax(1)

    def predict(self, x : torch.Tensor, encoded : bool = False):
        '''
        Returns the predicted cluster of each data point in x. See
        :func:`__call__` for details.
        '''

        return self(x, encoded=encoded)

    def probabilities(self, x : torch.Tensor, encoded : bool = False):
        '''
        Returns the probabilities of belonging to a certain class for each
        data point in x.

        Args:
            x (:class:`torch.Tensor`): The data points to use. Must
                have size `(n?, dim)` if `encoded=False`, otherwise must
                have size `(n?, features)`.

            encoded (bool): Specifies if input data is already encoded.

        Returns:
            :class:`torch.Tensor`: The class probability of each data point.
            Has size `(n?, classes)`.
        '''


        return self.scores(x, encoded=encoded).softmax(1)

    def scores(self, x : torch.Tensor, encoded : bool = False):
        r'''
        Returns pairwise cosine similarity between datapoints in `x` and
        each class hypervector. Calling `model.scores(x, encoded=True)` is
        the same as `spatial.cos_cdist(x, model.model)`.

        Args:
            x (:class:`torch.Tensor`): The data points to score. Must
                have size `(n?, dim)` if `encoded=False`, otherwise must
                have size `(n?, features)`.

            encoded (bool): Specifies if input data is already encoded.

        Returns:
            :class:`torch.Tensor`: The cosine similarity between encoded input
            data and class hypervectors.

        See Also:
            :func:`spatial.cos_cdist` for details.
        '''
        x = x.to(*self.device)
        if not encoded:
            x = self.local_maximum(x)
            x = self.quantizing(x)
            x = x.reshape((x.shape[0], -1))
            x = x.to('cpu')
            x = self.scaler.transform(x)
            x = torch.from_numpy(x).float().to(*self.device)
            h = self.encode(x)
        else:
            h = x

        return spatial.cos_cdist(h, self.model)

    def encode(self, x : torch.Tensor):
        '''
        Encodes input data

        See Also:
            :class:`onlinehd.Encoder` for more information.
        '''

        return self.encoder(x)

    def fit(self,
            x : torch.Tensor,
            y : torch.Tensor,
            encoded : bool = False,
            lr : float = 0.035,
            epochs : int = 120,
            batch_size : Union[int, None, float] = 1024,
            one_pass_fit : bool = True,
            bootstrap : Union[float, str] = 0.01):
        '''
        Starts learning process using datapoints `x` as input points and `y`
        as their labels.

        Args:
            x (:class:`torch.Tensor`): Input data points. Must
                have size `(n?, dim)` if `encoded=False`, otherwise must
                have size `(n?, features)`.

            encoded (bool): Specifies if input data is already encoded.

            lr (float, > 0): Learning rate.

            epochs (int, > 0): Max number of epochs allowed.

            batch_size (int, > 0 and <= n?, or float, > 0 and <= 1, or None):
                If int, the number of samples to use in each batch. If float,
                the fraction of the samples to use in each batch. If none the
                whole dataset will be used per epoch (same if used 1.0 or n?).

            one_pass_fit (bool): Whether to use onepass learning process or not.
                If true, iterative method will be used after one pass fit
                anyways for the number of epochs specified.

            bootstrap (float, > 0, <= 1 or 'single-per-class'): In order to
                initialize class hypervectors, OnlineHD does naive accumulation
                with a small fragment of data. This portion is determined by
                this argument. If 'single-per-class' is used, a single datapoint
                per class will be used as starting class hypervector.

        Warning:
            Using `one_pass_fit` is not advisable for very large data or
            while using GPU. It is expected to see high memory usage using
            this option and it does not benefit from paralellization.

        Returns:
            :class:`OnlineHD`: self
        '''
        x = x.to(*self.device)
        if encoded:
            h = x
        else:
            x = self.local_maximum(x)
            x = self.quantizing(x)
            x = x.reshape((x.shape[0], -1))
            x = x.to('cpu')
            self.scaler.fit(x)
            x = self.scaler.transform(x)
            x = torch.from_numpy(x).float().to(*self.device)
            h = self.encode(x)

        y = y.to(*self.device)
        if one_pass_fit:
            self._one_pass_fit(h, y, lr, bootstrap)
        self._iterative_fit(h, y, lr, epochs, batch_size)
        return self

    def to(self, *args):
        '''
        Moves data to the device specified, e.g. cuda, cpu or changes
        dtype of the data representation, e.g. half or double.
        Because the internal data is saved as torch.tensor, the parameter
        can be anything that torch accepts. The change is done in-place.

        Args:
            device (str or :class:`torch.torch.device`) Device to move data.

        Returns:
            :class:`OnlineHD`: self
        '''


        self.model = self.model.to(*args)
        self.encoder = self.encoder.to(*args)
        self.device = args
        return self

    def _one_pass_fit(self, h, y, lr, bootstrap):
        # initialize class hypervectors from a single datapoint
        if bootstrap == 'single-per-class':
            # get binary mask containing whether one datapoint belongs to each
            # class
            idxs = y == torch.arange(self.classes, device=h.device).unsqueeze_(1)
            # choses first datapoint for every class
            # banned will store already seen data to avoid using it later
            banned = idxs.byte().argmax(1)
            self.model.add_(h[banned].sum(0), alpha=lr)
        else:
            # will accumulate data from 0 to cut
            cut = math.ceil(bootstrap*h.size(0))
            h_ = h[:cut]
            y_ = y[:cut]
            # updates each class hypervector (accumulating h_)
            for lbl in range(self.classes):
                self.model[lbl].add_(h_[y_ == lbl].sum(0), alpha=lr)
            # banned will store already seen data to avoid using it later
            banned = torch.arange(cut, device=h.device)

        # todo will store not used before data
        n = h.size(0)
        todo = torch.ones(n, dtype=torch.bool, device=h.device)
        todo[banned] = False

        # will execute one pass learning with data not used during model
        # bootstrap
        h_ = h[todo]
        y_ = y[todo]
        _fasthd.onepass(h_, y_, self.model, lr)

    def _iterative_fit(self, h, y, lr, epochs, batch_size):
        n = h.size(0)
        for epoch in range(epochs):
            for i in range(0, n, batch_size):
                h_ = h[i:i+batch_size]
                y_ = y[i:i+batch_size]
                scores = self.scores(h_, encoded=True)
                y_pred = scores.argmax(1)
                wrong = y_ != y_pred

                # computes alphas to update model
                # alpha1 = 1 - delta[lbl] -- the true label coefs
                # alpha2 = delta[max] - 1 -- the prediction coefs
                aranged = torch.arange(h_.size(0), device=h_.device)
                alpha1 = (1.0 - scores[aranged,y_]).unsqueeze_(1)
                alpha2 = (scores[aranged,y_pred] - 1.0).unsqueeze_(1)

                for lbl in y_.unique():
                    m1 = wrong & (y_ == lbl) # mask of missed true lbl
                    m2 = wrong & (y_pred == lbl) # mask of wrong preds
                    self.model[lbl] += lr*(alpha1[m1]*h_[m1]).sum(0)
                    self.model[lbl] += lr*(alpha2[m2]*h_[m2]).sum(0)
