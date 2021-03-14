# ADAPTIVE LEARNED BLOOM FILTER (ADA-BF)

The python files include the implementation of the 1) Bloom filter (BF), 2) partial implementation of learned BF, and 3) partial implementation of Ada-BF, and 4) printing the size of False Positives of the corresponding algorithm.

Your task is to complete function to search the optimal parameters for learned BF and Ada-BF. Specfically, you are required to use cross validataion to find the best threshold for the ML classifier output scores for learned BF in `learned_Bloom_filter.py`; and find the best number of groups (*g* in [1]) and the best threshold of density ratio (*c* in [1]) in `Ada_BF.py`. 

**Reference**

[1] Dai, Z. and Shrivastava, A., 2019. Adaptive learned Bloom filter (Ada-BF): Efficient utilization of the classifier. arXiv preprint arXiv:1910.09131.

## How to run?

**Input arguments**: 
- `--data_path`: a csv file includes the items, scores and labels; `--size_of_Ada_BF`: size of Bloom filter;
- (for learned Bloom filter) `--threshold_min` and `--threshold_max` provide the range of the score threshold (between `threshold_min` and `threshold_max`). Items with score larger than the threshold are identified as keys;
- (for Ada-BF) `--num_group_min` and `--num_group_max` give the range of number of groups to divide (range of *g*); `--c_min` and `--c_max` provide the range of *c* where *c=m_j/m_{j+1}* and note *c>1*.

**Commands**:
- Run Bloom filter:\
```python Bloom_filter.py --data_path ./Datasets/URL_data.csv --size_of_BF 200000```
- Run learned Bloom filter and search the best threshold in [0.5, 0.95]:\
-`python learned_Bloom_filter.py --data_path ./Datasets/URL_data.csv --size_of_LBF 200000  --threshold_min 0.5   --threshold_max 0.95`
- Run Ada-BF and search the best *g* in [8,12] and *c* in [1.6, 2.5]:\
-`python Ada-BF.py --data_path ./Datasets/URL_data.csv --size_of_Ada_BF 200000  --num_group_min 8  --num_group_max 12  --c_min 1.6  --c_max 2.5`

## Your Tasks:
- [ ] Examine the ML classification logit scores for the URL dataset in `/Datasets/` by plotting the scores as a histogram using `matplotlib.pyplot.hist`. Please refer to Figure 2 of [1].
- [ ] Implement the `Find_Optimal_Parameters` functions in `learned_Bloom_filter.py` and `Ada_BF.py`.
- [ ] Compare FPR of Bloom filter (BF), learned BF, and Ada-BF with different memory budgets (Hash table sizes) using the provided URL dataset. To change the memory budget, you can set it by `--size_of_Ada_BF`. Please refer to Figure 4 of [1] to select the suitable range for the memory budget and plot your results.
- [ ] You are required to submit a report to show the comparison results, discuss the results, and analyze the limitation of Ada-BF (i.e., explore the cases when Ada-BF losses its advantages).


