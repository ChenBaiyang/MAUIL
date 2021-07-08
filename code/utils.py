# -*- coding: utf-8 -*-
import pickle
import json
import numpy as np
import networkx as nx
from scipy.sparse import coo_matrix
from sklearn.neighbors import KDTree
import scipy.sparse as sp

class dataset(object):
    def __init__(self,anchors):
        assert type(anchors) == type(dict())
        data = np.array(list(anchors.items()))
        #data.sort(axis=0) #这里的排序会导致anchors的对应关系被破坏。
        self.data=data
    def get(self,dset,n=100,seed=0):
        if 0<n<1:
        	n = int(len(self.data)*n)
        np.random.seed(seed)
        data = self.data.copy()
        np.random.shuffle(data)
        if dset=='train':
            return data[:n]
        elif dset=='test':
            return data[-n:]
        
def get_sim(embed, embed2 = None, sim_measure = "euclidean", top_k = 10):
    n_nodes, dim = embed.shape
    if embed2 is None:
        embed2 = embed
    kd_sim = kd_align(embed, embed2, distance_metric = sim_measure, top_k = top_k)
    return kd_sim

def kd_align(emb1, emb2, normalize=False, distance_metric = "euclidean", top_k = 10):
    kd_tree = KDTree(emb2, metric = distance_metric)    
        
    row = np.array([])
    col = np.array([])
    data = np.array([])
    
    dist, ind = kd_tree.query(emb1, k = top_k)

    row = np.array([])
    for i in range(emb1.shape[0]):
        row = np.concatenate((row, np.ones(top_k)*i))
    col = ind.flatten()
    data = np.exp(-dist).flatten()
    sparse_align_matrix = coo_matrix((data, (row, col)), shape=(emb1.shape[0], emb2.shape[0]))
    return sparse_align_matrix.tocsr()

def hit_precision(sim_matrix, top_k =10, anchors=None):

    if anchors is None:
        n_nodes = sim_matrix.shape[0]
        nodes = list(range(n_nodes))
    else:
        n_nodes = len(anchors)
        nodes = list(anchors.keys())
    
    score = 0
    for test_x in nodes:
        if anchors is None:
            test_y = test_x
        else:
            test_y = int(anchors[test_x])
        assert sp.issparse(sim_matrix)
      
        row_idx, col_idx, values = sp.find(sim_matrix[test_x])
        sorted_idx = col_idx[values.argsort()][-top_k:][::-1]
        
        h_x = 0
        for pos,idx in enumerate(sorted_idx):
            if idx == test_y:
                hit_x = pos+1
                h_x = (top_k-hit_x+1)/top_k
                break
        score += h_x
    score /= n_nodes 
    return score
        
def toABNE(g1,g2,anchors):
    # ABNE中anchor结点的标签为 网络中的 0-len(anchors), 需要重新标记网络中的结点。.
    # 默认输出训练集比例为：10% -90%
    mapping1 = dict()
    newid_anchor, newid_nonanchor = 0, len(anchors.keys())
    for i in range(len(g1)):
        if i in anchors.keys():
            mapping1.update({i:newid_anchor})
            newid_anchor +=1
        else:
            mapping1.update({i:newid_nonanchor})
            newid_nonanchor +=1
    #print(newid_anchor,newid_nonanchor)
    assert newid_anchor == len(anchors.keys()) and newid_nonanchor == len(g1)

    mapping2 = dict()
    for k,v in mapping1.items():
        if k in anchors.keys():
            mapping2.update({anchors[k]:v}) # 确保两个网络的anchors结点采用同样的标签
    assert len(mapping2) == len(anchors)

    newid_nonanchor = len(anchors.values())
    for i in g2.nodes():
        if i not in anchors.values():
            mapping2.update({i:newid_nonanchor})
            newid_nonanchor +=1
    print(newid_nonanchor,len(g2))
    assert newid_nonanchor == len(g2)
    
    new_anchors = dict([(mapping1[k],mapping2[v]) for k,v in anchors.items()])
    #print( max(list(new_anchors.values())),min(list(new_anchors.values())))
    assert max(list(new_anchors.values()))==len(anchors.values())-1 and min(list(new_anchors.values())) == 0

    for k,v in new_anchors.items():
        if k != v:
            print('Anchors不匹配：',k,v)

    #re_mapping1 = dict(zip(mapping1.values(),mapping1.keys()))
    #re_mapping2 = dict(zip(mapping2.values(),mapping2.keys()))

# 写入网络关系数据文件
    g1 = nx.relabel_nodes(g1,mapping1)
    g2 = nx.relabel_nodes(g2,mapping2)
    
    f2 = '../data/forABNE/AcrossNetworkEmbeddingData/twitter/following.number'
    f3 = '../data/forABNE/AcrossNetworkEmbeddingData/foursquare/following.number'

    nx.write_edgelist(g1,f2) #,comments='%')
    nx.write_edgelist(g2,f3) #,comments='%')

    new = ''
    with open(f2, 'r') as f:
        for i in f.readlines():
            re_i = ' '.join(i.split(' ')[::-1][1:3])
            new_i = i.replace(' {}', '')
            new += new_i+re_i+'\n'

    with open(f2, 'w') as f:
        f.write(new)
        print('写入网络1关系文件...'+f2[-30:])

    new = ''
    with open(f3, 'r') as f:
        for i in f.readlines():
            re_i = ' '.join(i.split(' ')[::-1][1:3])
            new_i = i.replace(' {}', '')
            new += new_i+re_i+'\n'

    with open(f3, 'w') as f:
        f.write(new)
        print('写入网络2关系文件...'+f3[-30:])
    
# 写入训练集、测试集文件
    # # to string
    d = dataset(anchors)
    fname= '../data/forABNE/AcrossNetworkEmbeddingData/twitter_foursquare_groundtruth/groundtruth.{}.foldtrain.{}.number'
    #for n in [100,200,300,400,500,600,700,800]:
    for idx,n in enumerate([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]):        
         for seed in range(10):
            testsets = d.get('test',1-n,seed)
            trainsets = d.get('train',n,seed)

            for k,v in trainsets:
                assert  mapping1[k] == mapping2[v]
            
            trainset_n = '\n'.join([str(mapping1[k]) for k,v in trainsets])
            f1 = fname.format(str(idx*10+seed+1),'train')
            with open(f1,'w',encoding='utf-8') as f:
                f.write(trainset_n)
                print('写入训练数据文件：'+f1[73:])
            
            testset_n = '\n'.join([str(mapping1[k]) for k,v in testsets])
            f1 = fname.format(str(idx*10+seed+1),'test')
            with open(f1,'w',encoding='utf-8') as f:
                f.write(testset_n)
                print('写入测试数据文件：'+f1[73:])


def wd2ABNE():
    g1,g2 = pickle.load(open('../data/WeiboDouban/networks','rb'))
    anchors = dict(json.load(open('../data/WeiboDouban/anchors.txt','r')))
    print(len(anchors),len(g1),len(g2))
    toABNE(g1,g2,anchors)

def dblp2ABNE():
    g1,g2 = pickle.load(open('../data/dblp/networks','rb'))
    anchors = dict(json.load(open('../data/dblp/anchors.txt','r')))
    print(len(anchors),len(g1),len(g2))
    toABNE(g1,g2,anchors)
    
if __name__ == "__main__":
    wd2ABNE()


