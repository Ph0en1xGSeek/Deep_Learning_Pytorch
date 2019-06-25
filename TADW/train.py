import numpy as np
import random
import torch
from utils import load_new_data
import argparse
import networkx as nx
from model import TADW
import matplotlib.pyplot as plt
from sklearn import manifold

parser = argparse.ArgumentParser()

parser.add_argument('--prefix', type=str, default='.', help='dir_prefix')
parser.add_argument('--dataset', type=str, default='cora', help='dataset')
parser.add_argument('-weight_decay', type=float, default=0., help='weight_decay')
parser.add_argument('--k', type=int, default=64, help='struct dimension')
parser.add_argument('--f_t', type=int, default=64, help='feat dimension')

parser.add_argument('--epochs', type=int, default=200, help='epochs')
parser.add_argument('-lr', type=float, default=0.06, help='learning rate')


args = parser.parse_args()


def train(train_data):
    G = train_data[0]
    feat_data = train_data[1]
    id2idx = train_data[2]
    labels = train_data[3]
    idx2id = {idx: id for id, idx in id2idx.items()}
    A = nx.adjacency_matrix(G)
    A = A.todense()
    A = A + np.eye(*A.shape)
    A = (np.matmul(A, A) + A) / 2
    A = A / np.max(A)
    print(feat_data.shape)

    lr = args.lr
    lr_chaning_set = set([20, 70, 120])

    model = TADW(M=A,
                 T=feat_data,
                 k=args.k,
                 f_t=feat_data.shape[1])
    cnt = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
            cnt += 1
    print(cnt)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    for epoch in range(args.epochs):
        if epoch in lr_chaning_set:
            print('Change learning rate from {} to {}'.format(lr, lr / 2))
            lr /= 2
            for p in optimizer.param_groups:
                p['lr'] = lr
        optimizer.zero_grad()
        loss = model.loss()
        loss.backward()
        torch.nn.utils.clip_grad_value_(filter(lambda p: p.requires_grad, model.parameters()), 5)
        optimizer.step()
        print("epoch {:04}\tloss: {:.5f}".format(epoch, loss.data.item()))

    _, W, H_T = model.forward()
    W = W.cpu().data.numpy()
    H_T = H_T.cpu().data.numpy()

    print(W.shape)
    print(H_T.shape)
    output = np.concatenate((W, H_T), axis=1)
    print(output.shape)
    visualize(output[::3], labels[::3], "output")
    visualize(W[::3], labels[::3], "W")
    visualize(H_T[::3], labels[::3], "H_T")

    output_file = '{}/embedding/{}_new_tadw.txt'.format(args.prefix, args.dataset)
    with open(output_file, 'w') as f:
        print(len(output), len(output[0]), file=f)
        print('output dim', len(output), len(output[0]))
        for i, node_id in idx2id.items():
            print(node_id, end='', file=f)
            for item in output[i]:
                print(' {}'.format(item), file=f, end='')
            print('', file=f)
        f.close()



def visualize(features, labels, title):
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    features_2D = tsne.fit_transform(features)
    plt.scatter(features_2D[:, 0], features_2D[:, 1], c=labels, marker='.', cmap=plt.cm.rainbow)
    plt.title(title)
    plt.show()




def main():
    np.random.seed(1)
    random.seed(1)
    torch.manual_seed(1)
    print("Loading data...")
    train_data = load_new_data(args.prefix, args.dataset)
    print("Loading completed. Training starts")
    train(train_data)


if __name__ == "__main__":
    main()