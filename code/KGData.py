import os
# class KnowledgeGraph:
#     def __init__(self, data_dir):
#         self.data_dir = data_dir
#
#
# if __name__ == "__main__":


class LoadTrainData:
    def __init__(self, entity2id, id2entity, relation2id, id2relation, train_triples, valid_triples, test_triples,
                 headRelation2Tail, tailRelation2Head, left_entity, right_entity, left_num, right_num):
        self.entity2id = entity2id
        self.id2entity = id2entity
        self.relation2id = relation2id
        self.id2relation = id2relation
        self.entityTotal = 0
        self.relationTotal = 0

        self.train_triples = train_triples
        self.trainTotal = 0
        self.valid_triples = valid_triples
        self.validTotal = 0

        self.test_triples = test_triples
        self.testTotal = 0

        self.headRelation2Tail = headRelation2Tail
        self.tailRelation2Head = tailRelation2Head

        self.left_entity = left_entity
        self.right_entity = right_entity
        self.left_num = left_num
        self.right_num = right_num

        self.read_entity()
        print("entity_num:", self.entityTotal)
        self.read_relation()
        print("relation_num:", self.relationTotal)
        self.read_valid_triple()
        self.read_test_triple()
        self.read_train_triple()  #
        print("train_triple:", self.trainTotal)

    def read_entity(self):
        with open(os.path.join("../data", 'FB15k', 'entity2id.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                entityStr, entityId = line.strip().split('\t')
                # print('name:', entityStr, 'id:', entityId)
                self.entity2id[entityStr] = int(entityId)
                self.id2entity[entityId] = entityStr
            self.entityTotal = len(self.entity2id)

    def read_relation(self):
        with open(os.path.join("../data", 'FB15k', 'relation2id.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                relationStr, relationId = line.strip().split('\t')
                # print('name:', relationStr, 'id:', relationId)
                self.relation2id[relationStr] = int(relationId)
                self.id2relation[relationId] = relationStr
            self.relationTotal = len(self.relation2id)

    def read_valid_triple(self):
        with open(os.path.join("../data", 'FB15k', 'valid.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                h, t, r = line.strip().split('\t')
                self.valid_triples.append((self.entity2id[h], self.relation2id[r], self.entity2id[t]))
            self.validTotal = len(self.valid_triples)

    def read_test_triple(self):
        with open(os.path.join("../data", 'FB15k', 'test.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                h, t, r = line.strip().split('\t')
                self.test_triples.append((self.entity2id[h], self.relation2id[r], self.entity2id[t]))
            self.testTotal = len(self.test_triples)

    def read_train_triple(self):
        with open(os.path.join("../data", 'FB15k', 'train.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                h, t, r = line.strip().split('\t')
                # print(h, r, t)
                h, t, r = self.entity2id[h], self.entity2id[t], self.relation2id[r]
                self.train_triples.append((h, r, t))
                self.headRelation2Tail[(h, r)] = t
                self.tailRelation2Head[(t, r)] = h
                # if r == 0:
                #     print(h, t, r)

                if r not in self.left_entity:
                    self.left_entity[r] = {}
                if h not in self.left_entity[r]:
                    self.left_entity[r][h] = 0
                self.left_entity[r][h] += 1
                #
                if r not in self.right_entity:
                    self.right_entity[r] = {}
                if t not in self.right_entity[r]:
                    self.right_entity[r][t] = 0
                # print(h, r, t)
                self.right_entity[r][t] += 1

            self.trainTotal = len(self.train_triples)
            print("train_triple:", self.trainTotal)
            for relation in range(self.relationTotal):
                sum1, sum2 = 0.0, 0.0
                # print(relation)
                for head in self.left_entity[relation]:
                    sum1 += 1
                    sum2 += self.left_entity[relation][head]
                self.left_num[relation] = sum2 / sum1
            for relation in range(self.relationTotal):
                sum1, sum2 = 0.0, 0.0
                mp = self.right_entity[relation]
                for tail in self.right_entity[relation]:
                    sum1 += 1
                    sum2 += self.right_entity[relation][tail]
                self.right_num[relation] = sum2 / sum1

    def get_total(self):
        return self.entityTotal, self.relationTotal, self.trainTotal, self.validTotal, self.testTotal






# import os
#
# entity2id = {}
# id2entity = {}
# relation2id = {}
# id2relation = {}
#
# n_entity = 0
# n_relation = 0
#
# train_triples = set()
# valid_triples = set()
# test_triples = set()
# gold_triples = set()
# n_train_triples = 0
# n_valid_triples = 0
# n_test_triples = 0
#
# # class KnowledgeGraph:
# #     def __init__(self, data_dir):
# #         self.data_dir = data_dir
# #
# #
# # if __name__ == "__main__":
# with open(os.path.join("../data", 'FB15k', 'entity2id.txt'), 'r', encoding='utf-8') as f:
#     for line in f:
#         entityStr, entityId = line.strip().split('\t')
#         print('name:', entityStr, 'id:', entityId)
#         entity2id[entityStr] = entityId
#         id2entity[entityId] = entityStr
#
# with open(os.path.join("../data", 'FB15k', 'relation2id.txt'), 'r', encoding='utf-8') as f:
#     for line in f:
#         relationStr, relationId = line.strip().split('\t')
#         print('name:', relationStr, 'id:', relationId)
#         relation2id[relationStr] = relationId
#         id2relation[relationId] = relationStr
#
# with open(os.path.join("../data", 'FB15k', 'train.txt'), 'r', encoding='utf-8') as f:
#     for line in f:
#         h, t, r = line.strip().split('\t')
#         train_triples.add((entity2id[h], relation2id[r], entity2id[t]))
#
# with open(os.path.join("../data", 'FB15k', 'valid.txt'), 'r', encoding='utf-8') as f:
#     for line in f:
#         h, t, r = line.strip().split('\t')
#         valid_triples.add((entity2id[h], relation2id[r], entity2id[t]))
#
# with open(os.path.join("../data", 'FB15k', 'test.txt'), 'r', encoding='utf-8') as f:
#     for line in f:
#         h, t, r = line.strip().split('\t')
#         test_triples.add((entity2id[h], relation2id[r], entity2id[t]))
#
# gold_triples = train_triples | valid_triples | test_triples
# print('entity num:', len(entity2id))
# print('relation num:', len(relation2id))
# print('train triples:', len(train_triples))
# print('valid triples:', len(valid_triples))
# print('test triples:', len(test_triples))
# print('gold_triples', len(gold_triples))
#
# n_entity = len(entity2id)
# n_relation = len(relation2id)
# n_train_triples = len(train_triples)
# n_valid_triples = len(valid_triples)
# n_test_triples = len(test_triples)



# np.random.permutation(n_training_triple)









