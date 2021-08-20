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


dataset = "emnist"
algo = [ 'JSMA', 'DeepFool', 'FSGM']
sampling = 0.9
kernel_size = 2
do_maxpool = True
seed = 54



torch.manual_seed(seed)


# loads simple mnist dataset
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


print('Loading...')
x, x_test, y, y_test = load()

if dataset == "fashion_mnist":
    dataset = "FMNIST"
elif dataset == "mnist":
    dataset = "MNIST"
else:
    dataset = "EMNIST"

examples = {}

for a in algo:
    if a == "JSMA":
        data = pickle.load(open('/workspace/shared/HDC_data/%s_%s.pickle' % (dataset, a), 'rb'))
        examples['%s train_x' % a] = data['train_data']
        examples['%s train_y' % a] = data['train_label']
        examples['%s test_x' % a] = data['test_data']
        examples['%s test_y' % a] = data['test_label']
    elif a == "DeepFool":
        data = pickle.load(open('/workspace/shared/HDC_data/%s_%s.pickle' % (dataset, a), 'rb'))
        examples['%s train_x' % a] = torch.from_numpy(data['train_data'])
        examples['%s train_y' % a] = data['train_label']
        examples['%s test_x' % a] = torch.from_numpy(data['test_data'])
        examples['%s test_y' % a] = data['test_label']
    else:
        with open('/workspace/shared/HDC_data/%s_%s.pickle' % (dataset, a), 'rb') as f:
            examples['%s train_x' % a] = pickle.load(f).clone().detach()
            examples['%s train_y' % a] = pickle.load(f).clone().detach().long()
            examples['%s test_x' % a] = pickle.load(f).clone().detach()
            examples['%s test_y' % a] = pickle.load(f).clone().detach().long()

    examples['%s train_x' % a] = examples['%s train_x' % a].unsqueeze(3)
    examples['%s test_x' % a] = examples['%s test_x' % a].unsqueeze(3)
    examples['%s train_y' % a] = examples['%s train_y' % a].squeeze()
    examples['%s test_y' % a] = examples['%s test_y' % a].squeeze()

new_x = x.clone().detach()
new_y = y.clone().detach()
new_x = torch.cat((new_x, *[examples['%s train_x' %a] for a in algo]))
new_y = torch.cat((new_y, *[examples['%s train_y' %a] for a in algo]))

new_x_test = x_test.clone().detach()
new_y_test = y_test.clone().detach()
new_x_test = torch.cat((new_x_test, *[examples['%s test_x' %a] for a in algo]))
new_y_test = torch.cat((new_y_test, *[examples['%s test_y' %a] for a in algo]))

if dataset == "EMNIST":
    unif = torch.ones(new_x.shape[0])
    indices = unif.multinomial(int(new_x.shape[0] * sampling))
else:
    indices = range(new_x.shape[0])


classes = y.unique().size(0)
model = robust_onlinehd.OnlineHD(do_maxpool, x[0].shape, kernel_size, sklearn.preprocessing.Normalizer(), classes, dim = 10000)
if dataset == "FMNIST":
    model.set_criterias(x, 10)
else:
    model.set_criterias(x, 8)


if torch.cuda.is_available():
    model = model.to("cuda:1")
    print('Using GPU!')

print('Training...')
t = time()

model = model.fit(new_x[indices], new_y[indices], bootstrap=.3, lr=0.095, epochs=300, batch_size=8196)
t = time() - t

print('Validating...')
yhat = model(new_x).cpu()
yhat_test = model(new_x_test).cpu()
acc = (new_y == yhat).float().mean()
acc_test = (new_y_test == yhat_test).float().mean()
print(f'{acc = :6f}')
print(f'{acc_test = :6f}')
print(f'{t = :6f}')


test_pred = model(x_test).cpu()
print('origin', (test_pred == y_test).float().mean().item(), '\n')
for a in algo:
    test_pred = model(examples['%s test_x' % a]).cpu()
    print(a, (test_pred == examples['%s test_y' % a]).float().mean().item(), '\n')