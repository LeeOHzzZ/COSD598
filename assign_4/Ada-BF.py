import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from Bloom_filter import hashfunc



parser = argparse.ArgumentParser()
parser.add_argument('--data_path', action="store", dest="data_path", type=str, required=True,
                    help="path of the dataset")
parser.add_argument('--num_group_min', action="store", dest="min_group", type=int, required=True,
                    help="Minimum number of groups")
parser.add_argument('--num_group_max', action="store", dest="max_group", type=int, required=True,
                    help="Maximum number of groups")
parser.add_argument('--size_of_Ada_BF', action="store", dest="R_sum", type=int, required=True,
                    help="size of the Ada-BF")
parser.add_argument('--c_min', action="store", dest="c_min", type=float, required=True,
                    help="minimum ratio of the keys")
parser.add_argument('--c_max', action="store", dest="c_max", type=float, required=True,
                    help="maximum ratio of the keys")



results = parser.parse_args()
DATA_PATH = results.data_path
num_group_min = results.min_group
num_group_max = results.max_group
R_sum = results.R_sum
c_min = results.c_min
c_max = results.c_max


'''
Load the data and select training data
'''
data = pd.read_csv(DATA_PATH)
negative_sample = data.loc[(data['label']==-1)]
positive_sample = data.loc[(data['label']==1)]
train_negative = negative_sample.sample(frac = 0.3)



'''
Plot the distribution of scores
'''
plt.style.use('seaborn-deep')

x = data.loc[data['label']==1,'score']
y = data.loc[data['label']==-1,'score']
bins = np.linspace(0, 1, 25)

plt.hist([x, y], bins, log=True, label=['Keys', 'non-Keys'])
plt.legend(loc='upper right')
plt.savefig('./Score_Dist.png')
plt.show()


class Ada_BloomFilter():
    def __init__(self, n, hash_len, k_max):
        self.n = n
        self.hash_len = int(hash_len)
        self.h = []
        for i in range(int(k_max)):
            self.h.append(hashfunc(self.hash_len))
        self.table = np.zeros(self.hash_len, dtype=int)
    def insert(self, key, k):
        for j in range(int(k)):
            t = self.h[j](key)
            self.table[t] = 1
    def test(self, key, k):
        test_result = 0
        match = 0
        for j in range(int(k)):
            t = self.h[j](key)
            match += 1*(self.table[t] == 1)
        if match == k:
            test_result = 1
        return test_result



def Find_Optimal_Parameters(c_min, c_max, num_group_min, num_group_max, R_sum, train_negative, positive_sample):
    c_set = np.arange(c_min, c_max+10**(-6), 0.1)
    FP_opt = train_negative.shape[0]

    k_min = 0
    for k_max in range(num_group_min, num_group_max+1):
        for c in c_set:
            tau = sum(c ** np.arange(0, k_max - k_min + 1, 1))
            n = positive_sample.shape[0]
            hash_len = R_sum
            bloom_filter = Ada_BloomFilter(n, hash_len, k_max)
            thresholds = np.zeros(k_max - k_min + 1)
            thresholds[-1] = 1.1
            num_negative = sum(train_negative['score'] <= thresholds[-1])
            num_piece = int(num_negative / tau) + 1
            score = train_negative.loc[(train_negative['score'] <= thresholds[-1]), 'score']
            score = np.sort(score)
            
        '''
        Please finish the code to find bloom_filter_opt, thresholds_opt, k_max_opt
        '''

            
        '''
        Ends here
        '''
    return bloom_filter_opt, thresholds_opt, k_max_opt



'''
Implement Ada-BF
'''
if __name__ == '__main__':
    '''Stage 1: Find the hyper-parameters (spare 30% samples to find the parameters)'''
    bloom_filter_opt, thresholds_opt, k_max_opt = Find_Optimal_Parameters(c_min, c_max, num_group_min, num_group_max, R_sum, train_negative, positive_sample)

    
    '''Stage 2: Run Ada-BF on all the samples'''
    ### Test Queries
    ML_positive = negative_sample.loc[(negative_sample['score'] >= thresholds_opt[-2]), 'query']
    query_negative = negative_sample.loc[(negative_sample['score'] < thresholds_opt[-2]), 'query']
    score_negative = negative_sample.loc[(negative_sample['score'] < thresholds_opt[-2]), 'score']
    test_result = np.zeros(len(query_negative))
    ss = 0
    for score_s, query_s in zip(score_negative, query_negative):
        ix = min(np.where(score_s < thresholds_opt)[0])
        # thres = thresholds[ix]
        k = k_max_opt - ix
        test_result[ss] = bloom_filter_opt.test(query_s, k)
        ss += 1
    FP_items = sum(test_result) + len(ML_positive)
    FPR = FP_items/len(query_negative)
    print('False positive items: {}; FPR: {}; Size of quries: {}'.format(FP_items, FPR, len(query_negative)))
