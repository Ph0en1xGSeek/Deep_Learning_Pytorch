import numpy as np
import networkx as nx


def load_new_data(prefix, dataset_str):
    print('loading directed data')
    feats = np.loadtxt('{}/data_new/{}/feature.txt'.format(prefix, dataset_str))
    labels = np.loadtxt('{}/data_new/{}/group.txt'.format(prefix, dataset_str))[:, 1].astype(int)
    graph = np.loadtxt('{}/data_new/{}/graph.txt'.format(prefix, dataset_str)).astype(int)


    G = nx.Graph()
    G = G.to_undirected()
    for id in range(feats.shape[0]):
        G.add_node(id)
    for item in graph:
        G.add_edge(item[0], item[1])
        G[item[0]][item[1]]['weight'] = 1.0
    id_map = {i: i for i in range(feats.shape[0])}
    # feats /= np.max(feats)
    # feats = prep.StandardScaler().fit_transform(feats)
    # struct_init = prep.StandardScaler().fit_transform(struct_init)
    # ##===========================output line input
    # with open('{}_line_direct_input.txt'.format(dataset_str), 'w') as f:
    #     for item in G.edges():
    #         print('{}\t{}\t1'.format(item[0], item[1]), file=f)
    #     f.close()
    return G, feats, id_map, labels

if __name__ == "__main__":
    load_new_data(".", "cora")