import os

entity2id = {}
id2entity = {}
relation2id = {}
id2relation = {}
entityTotal = 0
relationTotal = 0

train_triples = []
trainTotal = 0
# valid_triples = []
# validTotal = 0
#
# test_triples = []
# testTotal = 0

headRelation2Tail = {}
tailRelation2Head = {}

left_entity = {}
right_entity = {}
left_num = {}
right_num = {}
left_tot = {}
right_tot = {}

with open(os.path.join("../data", 'FB15k', 'entity2id.txt'), 'r', encoding='utf-8') as f:
    for line in f:
        entityStr, entityId = line.strip().split('\t')
        # print('name:', entityStr, 'id:', entityId)
        entity2id[entityStr] = int(entityId)
        id2entity[entityId] = entityStr
    entityTotal = len(entity2id)

with open(os.path.join("../data", 'FB15k', 'relation2id.txt'), 'r', encoding='utf-8') as f:
    for line in f:
        relationStr, relationId = line.strip().split('\t')
        # print('name:', relationStr, 'id:', relationId)
        relation2id[relationStr] = int(relationId)
        id2relation[relationId] = relationStr
    relationTotal = len(relation2id)

with open(os.path.join("../data", 'FB15k', 'train.txt'), 'r', encoding='utf-8') as f:
    for line in f:
        h, t, r = line.strip().split('\t')
        # print(h, r, t)
        h, t, r = entity2id[h], entity2id[t], relation2id[r]
        # train_triples.append((h, r, t))
        headRelation2Tail[(h, r)] = t
        tailRelation2Head[(t, r)] = h
        # if r == 0:
        #     print(h, t, r)

        if r not in left_entity:
            left_entity[r] = {}
        if h not in left_entity[r]:
            left_entity[r][h] = 0
        left_entity[r][h] += 1
        #
        if r not in right_entity:
            right_entity[r] = {}
        if t not in right_entity[r]:
            right_entity[r][t] = 0
        # print(h, r, t)
        right_entity[r][t] += 1

    trainTotal = len(train_triples)
    print("train_triple:", trainTotal)
    for relation in range(relationTotal):
        sum1, sum2 = 0.0, 0.0
        # print(relation)
        for head in left_entity[relation]:
            sum1 += 1
            sum2 += left_entity[relation][head]
        left_num[relation] = sum1
        left_tot[relation] = sum2
    for relation in range(relationTotal):
        sum1, sum2 = 0.0, 0.0
        mp = right_entity[relation]
        for tail in right_entity[relation]:
            sum1 += 1
            sum2 += right_entity[relation][tail]
        right_num[relation] = sum1
        right_tot[relation] = sum2

with open(os.path.join("../data", 'FB15k', 'left_right_entity_num.txt'), 'w', encoding='utf-8') as f:
    for i in range(relationTotal):
        f.write("%d\t%d\t%d\t%d\t%d\n" % (i, left_tot[i], left_num[i], right_tot[i], right_num[i]))






