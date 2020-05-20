
import numpy as np
import torch

# class TrainDataLoader(object):
#     def __init__(self, batch_size=None, nbatches=None):
#         self.batch = 0
#         self.nbatches = nbatches
#         self.batch_seq_size = nbatches * 2
#
#         self.batch_h = np.zeros(self.batch_seq_size, dtype=np.int64)  # 定义一个
#         self.batch_t = np.zeros(self.batch_seq_size, dtype=np.int64)
#         self.batch_r = np.zeros(self.batch_seq_size, dtype=np.int64)
#         self.batch_y = np.zeros(self.batch_seq_size, dtype=np.int64)
#
#         self.batch_triplet = np.zeros((self.batch_seq_size,3,1))
#         self.batch_triplet = np.zeros((self.batch_seq_size, 3, 1))
#
#
#
#     def __iter__(self):
#         return self
#
#     def __next__(self):
#         self.batch += 1   # 第一次进来batch =1, 第batches次进来是batches次
#         if self.batch > self.nbatches:
#             raise StopIteration()
#         return self.batch
#
#     def sample(self):
#         for i in range(self.batchnes):
#
#
#         return {'batch_h': self.batch_h, 'batch_t': self.batch_t , 'batch_r': self.batch_r}
#
#
# for data in TrainDataLoader():
#     print(data)
