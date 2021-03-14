# Neural Architecture Search (NAS)
In this assignment, you are required to use an open source AutoML toolkit (NNI) for neural architecture search. You are ask to compare three different NAS methods:
Efficient Neural Architecture Search via Parameter Sharing (ENAS)[1] and DARTS: Differentiable Architecture Search[2].

### References 
[1] Pham, H., Guan, M., Zoph, B., Le, Q. and Dean, J., 2018, July. Efficient neural architecture search via parameters sharing. In International Conference on Machine Learning (pp. 4095-4104). PMLR.

[2] Liu, H., Simonyan, K. and Yang, Y., 2018. Darts: Differentiable architecture search. arXiv preprint arXiv:1806.09055.


# Your tasks
- [ ] Install toolkit NNI following the [instruction](https://nni.readthedocs.io/en/stable/Tutorial/InstallationLinux.html#installation) 
- [ ] Perform CNN architecture search for Cifar10 (training and testing datasets are automatically split). You need to compare ENAS and DARTS. Please record the top1 testing accuracy per GPU hours. You can present the results using a figure (referring to Figure 3 of [2]). Single-trial experiments will be fine. Please use the same GPU type for a fair comparison.
- [ ] Implement retrain.py for ENAS method. Please refer to [code](https://github.com/microsoft/nni/blob/master/examples/nas/darts/retrain.py). Please report the top 1 accuracy after retraining.
- [ ] Show the final architecture for each method. %%Bonus%% - Check the visualization method on NNI [Document](https://nni.readthedocs.io/en/stable/NAS/Visualization.html?highlight=visualizationhttps://nni.readthedocs.io/en/stable/NAS/Visualization.html?highlight=visualization)
- [ ] You are required to submit a report to show the comparison results, discuss the results, and analyze the limitation of ENAS and DARTS.

# How to run?
Python >= 3.6, Pytorch >= 1.7.0

## DARTS
### Search
```
# search the best architecture
cd examples/nas/darts
python3 search.py --v1 --visualization
```
### Retrain
```
# train the best architecture
python3 retrain.py --arc-checkpoint ${Your saved checkpoint}
```

## ENAS
### Search 
```
cd examples/nas/enas/

# search in micro search space, where each unit is a cell
python3 search.py --search-for micro --v1 --visualization
```
### Retrain
Your need to implement the retraining code first and run retrain. You can refer to [retrain.py](https://github.com/microsoft/nni/blob/master/examples/nas/darts/retrain.py) of DARTS or [this repo](https://github.com/zuimeiyujianni/ENAS_micro_retrain_Pytorch).


Remark: You may meet possible error on `.view` function in `utils.py.` Change it to `.reshape` will work.


