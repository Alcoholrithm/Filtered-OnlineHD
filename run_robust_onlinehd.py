import argparse
import torch
from tensorflow.keras.datasets import mnist, fashion_mnist
from torchvision.datasets import EMNIST
import robust_onlinehd
import sklearn.preprocessing
from time import time
from GenAttack import GenAttack
from tqdm import tqdm
import numpy as np

# loads simple mnist dataset
def load(dataset):
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

def do_GenAttack(x, x_test, y, y_test, model, args):
    preds = model(x_test).cpu().numpy()
    targets = torch.randint(0, 10, preds.shape)

    for i in tqdm(range(len(preds))):
        while targets[i] == preds[i]:
            targets[i] = torch.randint(0,10, (1,)).item()

    unif = torch.ones(targets.shape[0])
    while True:
        indices = unif.multinomial(100)
        for idx in indices:
            if targets[idx] == y_test[idx]:
                break
        if idx == indices[-1] and targets[idx] != y_test[idx]:
            break
        else:
            indices = unif.multinomial(100)

    attacker = GenAttack(model, model.classes, 0.3, args.device)
    N = args.N                         # size of population to evolve
    G = args.G                      # number of generations to evolve through
    p = torch.FloatTensor([args.p])   # the parameter for Bernoulli distribution used in mutation
    alpha = torch.FloatTensor([args.alpha]) # the parameter controlling mutation amount (step-size in the original paper)
    delta = torch.FloatTensor([args.delta]) # the parametr controlling mutation amount (norm threshold in the original paper)

    pops = []
    results = []

    t = time()
    for i in tqdm(indices):
        temp = attacker.attack(x_test[i], targets[i], delta, alpha, p, N, G)
        pops.append(temp[0].numpy())
        results.append(temp[1])
    t = time() - t

    print('t = {:0.6f}'.format(t))

    pops = np.array(pops)
    sample_preds = preds[indices]

    new_preds = []
    for i in range(100):
        new_preds.append(model(torch.tensor(pops[i])).cpu().numpy())

    success = 0
    success_idx = []
    for i in range(100):
        if targets[indices[i]].item() in new_preds[i]:
            success_idx.append((indices[i].item(), (i, np.where(new_preds[i] == targets[indices[i]].item())[0][0])))
            success += 1
    print(success)

    if args.do_save == True:
        cache = {
            'indices' : indices,
            'sample_preds' : sample_preds,
            'pops' : np.array(pops),
            'hyper_parameter' : [N, G, p, alpha, delta],
            'success_idx' : success_idx,
            'model' : model, 
            'targets' : targets,
            'results' : results
        }

        torch.save(cache, 'robust_onlinehd_%s.pt' % args.dataset)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default=None, type=str, required=True)
    parser.add_argument("--kernel_size", default=2, type=int, required=True)
    parser.add_argument("--criterias_from_data", default=None, type=lambda s: s.lower() in ['true', '1'], required=True)
    parser.add_argument("--criterias", default=None, type=str, required=True)
    parser.add_argument("--seed", default=None, type=int, required=True)
    parser.add_argument("--do_maxpool", default=False, type=lambda s: s.lower() in ['true', '1'], required=True)

    parser.add_argument("--device", default='cuda', type=str, required=False)

    parser.add_argument("--dim", default=10000, type=int, required=False)
    parser.add_argument("--bootstrap", default=0.3, type=float, required=False)
    parser.add_argument("--lr", default=0.095, type=float, required=False)
    parser.add_argument("--epochs", default=300, type=int, required=False)
    parser.add_argument("--batch_size", default=8196, type=int, required=False)


    parser.add_argument("--do_gen_attack", default=True, type=lambda s: s.lower() in ['true', '1'], required=False)

    parser.add_argument("--N", default=8, type=int, required=False)
    parser.add_argument("--G", default=5000, type=int, required=False)
    parser.add_argument("--p", default=0.9, type=float, required=False)
    parser.add_argument("--alpha", default=1.0, type=float, required=False)
    parser.add_argument("--delta", default=0.9, type=float, required=False)
    
    parser.add_argument("--do_save", default=False, type=lambda s: s.lower() in ['true', '1'], required=False)

    args = parser.parse_args()



    torch.manual_seed(args.seed)

    print('Loading...')
    x, x_test, y, y_test = load(args.dataset)

    classes = y.unique().size(0)
    model = robust_onlinehd.OnlineHD(args.do_maxpool, x[0].shape, args.kernel_size, sklearn.preprocessing.Normalizer(), classes, dim = args.dim)

    if args.criterias_from_data == True:
        model.set_criterias(x, int(args.criterias))
    else:
        model.set_criterias(eval(args.criterias), is_data=False)
    
    if torch.cuda.is_available():
        model = model.to(args.device)
        print('Using GPU!')

    print('Training...')
    t = time()

    model = model.fit(x, y, bootstrap=args.bootstrap, lr=args.lr, epochs=args.epochs, batch_size=args.batch_size)
    t = time() - t

    print('Validating...')
    yhat = model(x).cpu()
    yhat_test = model(x_test).cpu()
    acc = (y == yhat).float().mean()
    acc_test = (y_test == yhat_test).float().mean()
    print('acc = {:0.6f}'.format(acc.item()))
    print('acc_test = {:0.6f}'.format(acc_test.item()))
    print('t = {:0.6f}'.format(t))

    if args.do_gen_attack:
        do_GenAttack(x, x_test, y, y_test ,model, args)
    else:
        if args.do_save == True:
            cache = {
                'model' : model
            }
            torch.save(cache, 'robust_onlinehd_%s.pt' % args.dataset)
        exit()

if __name__ =='__main__':
    main()
