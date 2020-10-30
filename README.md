# MAUIL
The code and dataset for paper "MAUIL: Multi-level Attribute Embedding for Semi-supervised User Identity Linkage"

## Datasets
1. Weibo-Douban (WD)

First, a small number of users on the Douban platform have posted their Weibo accounts on their homepages. These users on distinguish two platforms are real user identity links, which can be used as pre-aligned user pairs in the UIL task. Second, the original data is prepared by crawling users' information pages, including users' attributes and their follower/followee relations. A clear benefit of data crawling in the Weibo platform could not be directly identified in this step. Weibo allows only a small part (two hundred) of follower/followee relations to be returned by crawlers. Hence, the relations that come from traditional Weibo crawling methods are quite incomplete. On the other hand, the size of Weibo network is enormous. The empirical treatment is to extract a subnet with common characteristics from the original Weibo network. We repeatedly remove the nodes with the degrees less than 2 or more than 1000. Then, the community discovery algorithm is performed to find the subnets with the typical social network characteristics, including the approximate power-law degree distribution and the high aggregation coefficient. Similar operations are carried out on the Douban network.

2. DBLP17-19 (DBLP)

Each author in DBLP has a unique key, which can be used as the ground truth for the UIL problem. In this study, the DBLP network backups of different periods, i.e.,2017-12-1 and 2018-12-1, were used as the aligned subjects in the UIL experiments. We select the Turing Award winner Yoshua Bengio as the center node in each network, and then delete any nodes more than three steps away from the center. Besides, the size of two DBLPs is reduced by discovering network communities and repeatedly deleting leaf nodes. The final DBLPs also enjoy the characteristics of the power-law distribution and the high aggregation coefficient.

## Environment

* Python>=3.7
* Tensorflow>=1.13
* Scipy
* Numpy

## Acknowledgement
