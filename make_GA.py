import numpy as np
import torch
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from torch.autograd import Variable
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from time import time
import sklearn.preprocessing
import numpy as np
import robust_onlinehd
from GenAttack import GenAttack
from tensorflow.keras.datasets import mnist, fashion_mnist
from torchvision.datasets import EMNIST
import pickle
import logging

dataset = "fashion_mnist"
seed = 1234
torch.manual_seed(seed)

logger = logging.getLogger("ga_gen")
logger.setLevel(logging.DEBUG)


stream_handler = logging.StreamHandler()
file_handler = logging.FileHandler(filename="GA_%s.log" % dataset)

# formatter 객체 생성
formatter = logging.Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# handler에 level 설정
stream_handler.setLevel(logging.DEBUG)
file_handler.setLevel(logging.DEBUG)

# handler에 format 설정
stream_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(stream_handler)
logger.addHandler(file_handler)

def load():
    if dataset == 'mnist':
        (x, y), (x_test, y_test) = mnist.load_data()
    elif dataset == 'fashion_mnist':
        (x, y), (x_test, y_test) = fashion_mnist.load_data()
    else:
        temp = EMNIST('./data/EMNIST', split = 'letters', train = True, download = True)
        x = temp.data.unsqueeze(3).numpy().transpose((0,2,1,3))
        y = temp.targets.numpy() - 1

        temp = EMNIST('./data/EMNIST', split = 'letters', train = False, download = True)
        x_test = temp.data.unsqueeze(3).numpy().transpose((0,2,1,3))
        y_test = temp.targets.numpy() - 1 

    # changes data to pytorch's tensors
    x = torch.from_numpy(x).float()   
    y = torch.from_numpy(y).long().squeeze()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long().squeeze()
    
    if len(x.shape) == 3:
        x = x.unsqueeze(3)
        x_test = x_test.unsqueeze(3)

    return x, x_test, y, y_test


logger.info('Loading...')
x, x_test, y, y_test = load()

kernel_size = 1
classes = y.unique().size(0)
epochs = 300
model = robust_onlinehd.OnlineHD(True, x[0].shape, kernel_size, sklearn.preprocessing.Normalizer(), classes, dim = 10000)
model.set_criterias([], is_data=False)

if torch.cuda.is_available():
    model = model.to(1)
    logger.info('Using GPU!')

logger.info('Training...')
t = time()

model = model.fit(x, y, bootstrap=.3, lr=0.095, epochs=epochs, batch_size=8196)
t = time() - t

logger.info('Validating...')
yhat = model(x).cpu()
yhat_test = model(x_test).cpu()
acc = (y == yhat).float().mean()
acc_test = (y_test == yhat_test).float().mean()
logger.info(f'{acc = :6f}')
logger.info(f'{acc_test = :6f}')
logger.info(f'{t = :6f}')

attacker = GenAttack(model, classes, 0.6, 'cuda:1')
N = 8                          # size of population to evolve
G = 5000                        # number of generations to evolve through
p = torch.FloatTensor([0.9])   # the parameter for Bernoulli distribution used in mutation
alpha = torch.FloatTensor([1.0]) # the parameter controlling mutation amount (step-size in the original paper)
delta = torch.FloatTensor([.9]) # the parametr controlling mutation amount (norm threshold in the original paper)


targets = torch.randint(0, classes, y.shape)
for i in tqdm(range(len(y))):
    while targets[i] == y[i]:
        targets[i] = torch.randint(0,classes, (1,)).item()
unif = torch.ones(targets.shape[0])
indices = unif.multinomial(10000)

x = x[indices]
y = y[indices]

train_label = []
train_data = []

i = 0
success = 0
while success != 1000:
    temp = attacker.attack(x[i], targets[i], delta, alpha, p, N, G)
    if temp[1]:
        train_data.append(temp[0][(model(temp[0]) == targets[i]).nonzero()[0].item()].numpy())
        logger.info('example for train %d / %d completed' % (success + 1, 1000))
        train_label.append(y[i])
        success = success + 1
    i = i + 1
train_data = torch.Tensor(train_data).float()

targets = torch.randint(0, classes, y_test.shape)
for i in tqdm(range(len(y_test))):
    while targets[i] == y_test[i]:
        targets[i] = torch.randint(0,classes, (1,)).item()

unif = torch.ones(targets.shape[0])
indices = unif.multinomial(10000)

x_test = x_test[indices]
y_test = y_test[indices]

test_label = []
test_data = []

i = 0
success = 0
while success != 1000:
    temp = attacker.attack(x_test[i], targets[i], delta, alpha, p, N, G)
    if temp[1]:
        test_data.append(temp[0][(model(temp[0]) == targets[i]).nonzero()[0].item()].numpy())
        logger.info('example for test %d / %d completed' % (success + 1, 1000))
        test_label.append(y_test[i])
        success = success + 1
    i = i + 1

test_data = torch.Tensor(test_data).float()

cache = {'train_data' : train_data, \
         'train_label' : torch.Tensor(train_label).long(), \
         'test_data' : test_data, \
         'test_label' : torch.Tensor(test_label).long()}

torch.save(cache, 'GA_%s.pt' % dataset)

logger.info('Completed')
