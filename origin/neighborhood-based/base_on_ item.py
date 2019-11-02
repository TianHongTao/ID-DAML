#-*-coding:utf-8 -*-
from collections import defaultdict
class RankItem:
    def __init__(self, weight, reason):
        self.weight = weight
        self.reason = reason

def ItemSimilarity(train):
    C = defaultdict(lambda: defaultdict(lambda: 0))
    N = defaultdict(lambda: 0)
    for u, items in train.items():
        for i in users:
            N[i] += 1
            for j in users:
                if i == j:
                    continue
                C[i][j] += 1 / math.log(1 + len(items) * 1.0)
                
    W = defaultdict(lambda: defaultdict(lambda: 0))
    for i,related_items in C.items():
        for j, cij in related_items.items(): 
            W[u][v] = cij / math.sqrt(N[i] * N[j])
    return W

def Recommendation(train, user_id, W, K):
    rank = defaultdict(lambda: RankItem(0, defaultdict(lambda: 0)))
    ru = train[user_id]
    for i,pi in ru.items():
        for j, wj in sorted(W[i].items(), key=itemgetter(1), reverse=True)[0:K]:
            if j in ru:
                continue
            rank[j].weight += pi * wj
            rank[j].reason[i] = pi * wj 
    return rank