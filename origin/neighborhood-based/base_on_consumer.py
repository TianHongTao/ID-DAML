#-*-coding:utf-8 -*-
from collections import defaultdict
def UserSimilarity(train):
    item_users = defaultdict(lambda: set())
    for u, items in train.items():
        for i in items.keys():
            item_users[i].add(u)

    C = defaultdict(lambda: defaultdict(lambda: 0))
    N = defaultdict(lambda: 0)
    for i, users in item_users.items():
             for u in users:
                 N[u] += 1
                 for v in users:
                     if u == v:
                        continue

                    C[u][v] += 1 / math.log(1 + len(users))

    W = defaultdict(lambda: defaultdict(lambda: 0))
    for u, related_users in C.items():
        for v, cuv in related_users.items(): 
            W[u][v] = cuv / math.sqrt(N[u] * N[v])
    return W

def Recommend(user, train, W):
    rank = defaultdict(lambda: 0)
    interacted_items = train[user]
    for v, wuv in sorted(W[u].items, key=itemgetter(1))[0:K][::-1]:
        for i, rvi in train[v].items:
                    if i in interacted_items:
                        continue
                    rank[i] += wuv * rvi
    return rank