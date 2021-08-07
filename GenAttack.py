import torch
from skimage.metrics import structural_similarity as ssim
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np

class GenAttack(object):
    def __init__(self, model, num_labels, img_size, scaler, threshold = 0.95, device = 'cuda'):
        self.model = model.to(device)
        self.device = device

        self.num_labels = num_labels
        self.tlab = None
        self.img_size = img_size
        self.scaler = scaler
        self.threshold = threshold

    def set_tlab(self, target):
        self.tlab = torch.zeros((1, self.num_labels)).to(self.device)
        self.tlab[:,target] = 1

    def get_mutation(self, shape, alpha, delta, bernoulli):
        N, h, w, nchannels = shape
        U = torch.FloatTensor(N * nchannels * h * w).uniform_().to(self.device) * 2 * alpha * delta - alpha * delta
        mask = bernoulli.sample((N * nchannels * h * w,)).squeeze()
        mutation = mask * U
        mutation = mutation.view(N, h, w, nchannels)
        return mutation
    
    def crossover(self, parents, fitness, population):
        _, h, w, nchannels = population.shape
        fitness_pairs = fitness[parents.long()].view(-1, 2)
        prob = fitness_pairs[:, -0] / fitness_pairs.sum()
        parental_bernoulli = torch.distributions.Bernoulli(prob)
        inherit_mask = parental_bernoulli.sample((nchannels * h * w,))
        inherit_mask = inherit_mask.view(-1, h, w, nchannels)
        parent_features = population[parents.long()]
        children = torch.FloatTensor(inherit_mask.shape).to(device=self.device)
        
        children = self.where(inherit_mask, parent_features[::2], parent_features[1::2])
        return children
        
    def where(self, cond, x_1, x_2):
        '''
        Pytorch 0.3.1 does not have torch.where
        '''
        return (cond.float() * x_1) + ((1-cond).float() * x_2)

    def get_fitness(self, population, target):
        
        pop_preds = self.model.probabilities(population)
        all_preds = torch.argmax(pop_preds, dim = 1)
        #print(pop_preds)

        success_pop = (all_preds == target).clone().detach()
        success_pop = success_pop.to(self.device).int()
        success = torch.max(success_pop, dim = 0)

        target_scores = torch.sum(self.tlab * pop_preds, dim = 1)
        sum_others = torch.sum((1 - self.tlab) * pop_preds, dim = 1)
        max_others = torch.max((1 - self.tlab) * pop_preds, dim = 1)

        # the goal is to maximize this loss
        fitness = torch.log(sum_others + 1e-30) - torch.log(target_scores + 1e-30)

        return fitness

    def attack(self, x, target, delta, alpha, p, N, G):
        self.set_tlab(target)
        x = x.to(self.device)
        target = target.to(self.device)
        delta = delta.to(self.device)
        alpha = alpha.to(self.device)
        p = p.to(self.device)
        
        bernoulli = torch.torch.distributions.Bernoulli(p)
        softmax = torch.nn.Softmax(0).to(device=self.device)

        # generate starting population
        h, w, nchannels = x.shape
        mutation = self.get_mutation([N, h, w, nchannels], alpha, delta, bernoulli)

        # init current population
        Pcurrent = x[None, :, :, :].expand(N, -1, -1, -1) + mutation
        Pnext = torch.zeros_like(Pcurrent)

        # init previous population with original example
        Pprev = x[None, :, :, :].expand(N, -1, -1, -1)
        # compute constraints to ensure permissible distance from the original example
        lo = x.min() - alpha[0]*delta[0]
        hi = x.max() + alpha[0]*delta[0]
        
        # start evolution
        for g in range(G):
            # measure fitness with MSE between descriptors
            fitness = self.get_fitness(Pcurrent, target)  # [N]
            
            # check SSIM
            ssimm = np.zeros(N)
            for i in range(N):
                ssimm[i] = ssim(x.cpu().numpy(),
                                Pcurrent[i].cpu().numpy(),
                                multichannel=True)  # [N]
            #survivors = ssimm >= 0.95  # [N]
            survivors = ssimm >= self.threshold

            if survivors.sum() == 0:
                print('All candidates died at generation', g)
                print('Target = ', target)
                return Pprev.cpu(), False
                
            if target in self.model(Pcurrent):
                print('Attack Success at generation', g)
                return Pcurrent.cpu(), True

            # choose the best fit candidate among population
            _, best = torch.min(fitness, 0)  # get idx of the best fitted candidate
            # ensure the best candidate gets a place in the next population
            Pnext[0] = Pcurrent[best]

            # generate next population
        #print(pop_preds)
            probs = softmax(Variable(torch.FloatTensor(survivors).to(self.device)) * Variable(fitness)).data
            cat = torch.distributions.Categorical(probs[None, :].expand(2 * (N-1), -1))
            parents = cat.sample()  # sample 2 parents per child, total number of children is N-1
            children = self.crossover(parents, fitness, Pcurrent)  # [(N-1) x nchannels x h x w]
            mutation = self.get_mutation([N-1, h, w, nchannels], alpha, delta, bernoulli)
            children = children + mutation
            Pnext[1:] = children
            Pprev = Pcurrent  # update previous generation
            Pcurrent = Pnext  # update current generation
            # clip to ensure the distance constraints
            Pcurrent = torch.clamp(Pcurrent, lo, hi)

        print('All', 5000, 'generations failed.')
        return Pcurrent.cpu(), False
    
    def to(self, device):
        self.device = device
        self.model.to(self.device)
        
        