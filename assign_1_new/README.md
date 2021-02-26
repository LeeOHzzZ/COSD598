# Network Pruning
### Assignment 1 for COS598D: System and Machine Learning

In this assignment, you are required to evaluate three advanced neural network pruning methods, including SNIP [1], GraSP [2] and SynFlow [3], and compare with two baseline pruning methods,including random pruning and magnitude-based pruning. In `example/singleshot.py`, we provide an example to do singleshot global pruning without iterative training. In `example/multishot.py`, we provde an example to do multi-shot iterative training. This assignment focuses on the pruning protocal in `example/singleshot.py`. Your are going to explore various pruning methods on differnt hyperparameters and network architectures.

***References***

[1] Lee, N., Ajanthan, T. and Torr, P.H., 2018. Snip: Single-shot network pruning based on connection sensitivity. arXiv preprint arXiv:1810.02340.

[2] Wang, C., Zhang, G. and Grosse, R., 2020. Picking winning tickets before training by preserving gradient flow. arXiv preprint arXiv:2002.07376.

[3] Tanaka, H., Kunin, D., Yamins, D.L. and Ganguli, S., 2020. Pruning neural networks without any data by iteratively conserving synaptic flow. arXiv preprint arXiv:2006.05467.

### Additional reading materials:

A recent preprint [4] assessed [1-3].

[4] Frankle, J., Dziugaite, G.K., Roy, D.M. and Carbin, M., 2020. Pruning Neural Networks at Initialization: Why are We Missing the Mark?. arXiv preprint arXiv:2009.08576.

## Getting Started
First clone this repo, then install all dependencies
```
pip install -r requirements.txt
```

## How to Run 
Run `python main.py --help` for a complete description of flags and hyperparameters. You can also go to `main.py` to check all the parameters. 

Example: Initialize a VGG16, prune with SynFlow and train it to the sparsity of 10^-0.5 . We have sparsity = 10**(-float(args.compression)).
```
python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner synflow --compression 0.5
```

To save the experiment, please add `--expid {NAME}`. `--compression-list` and `--pruner-list` are not available for runing singleshot experiment. You can modify the souce code following `example/multishot.py` to run a list of parameters. `--prune-epochs` is also not available as it does not affect your pruning in singleshot setting. 

For magnitude-based pruning, please set `--pre-epochs 200`. You can reduce the epochs for pretrain to save some time. The other methods do pruning before training, thus they can use the default setting `--pre-epochs 0`.

Please use the default batch size, learning rate, optimizer in the following experiment. Please use the default training and testing spliting. Please monitor training loss and testing loss, and set suitable training epochs.

## You Tasks

### 1. Hyper-parameter tuning

#### Testing on different archietectures. Please fill the results table:
*Test accuracy (top 1)* of pruned models on CIFAR10 and MNIST (sparsity = 10%). `--compression 1` means sparsity = 10^-1.
```
python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner synflow --compression 1
# my commands
python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner-list {rand,snip,grasp,synflow} --compression 0.5 --expid t1_vgg_wo_mag --post-epochs 200 --gpu 
python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner mag --pre-epochs 200 --compression 0.5 --expid t1_vgg_mag --post-epochs 200 --gpu
```
```
python main.py --model-class default --model fc --dataset mnist --experiment singleshot --pruner synflow --compression 1
# my commands
python main.py --model-class default --model fc --dataset mnist --experiment singleshot --pruner-list {rand,snip,grasp,synflow} --compression 0.5 --expid t1_fc_wo_mag --post-epochs 200 --gpu
python main.py --model-class default --model fc --dataset mnist --experiment singleshot --pruner mag --pre-epochs 200 --compression 0.5 --expid t1_fc_mag --post-epochs 200 --gpu
```
|   Data  |   Arch |   Rand |  Mag |  SNIP |  GraSP | SynFlow       |   
|----------------|----------------|-------------|-------------|-------------|---------------|----------------|
|Cifar10 | VGG16 | 78.44%  |  89.21%    |    80.09%    |  22.38%   |   80.92%      |
|MNIST| FC |  96.21%  |   98.26%   |   97.07%     |   96.82%   |    94.97%     |


#### Tuning compression ratio. Please fill the results table:
Prune models on CIFAR10 with VGG16, please replace {} with sparsity 10^-a for a \in {0.05,0.1,0.2,0.5,1,2}. Feel free to try other sparsity values.

```
python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner synflow  --compression {}
python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner-list {rand,snip,grasp,synflow} --compression-list {0.05,0.1,0.2,0.5,1,2} -expid t2_wo_mag --post-epochs 200 --gpu

# separate mag workload, because of the huge pre-training...
python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner mag --pre-epochs 200 --compression-list {0.05,0.1,0.2,0.5,1,2} -expid t2_mag --post-epochs 200 --gpu

```
***Testing accuracy (top 1)***

|   Compression |   Rand |  Mag |  SNIP |  GraSP | SynFlow       |   
|----------------|-------------|-------------|-------------|---------------|----------------|
| 0.05|  79.34%  |  89.45%    |   78.31%     |  29.24%   |     78.01%    |
| 0.1| 80.59%   |   89.06%   |     78.51%    |  17.41%   |    81.25%     |
| 0.2|  79.65%  |    89.18%   |    79.93%    |  12.37%    |    78.63%    |
| 0.5|  75.82%  |   89.77%   |    77.16%    |   11.82%   |     80.85%     |
| 1|  10.0%  |  88.30%   |   76.43%     |   10.04%   |    81.04%     |
| 2|  10.0%  |  15.56%   |     53.53%   |   13.91%   |    10.00%     |

***Testing time***

|   Compression |   Rand |  Mag |  SNIP |  GraSP | SynFlow       |   
|----------------|-------------|-------------|-------------|---------------|----------------|
| 0.05| 2.24686   |   2.24146    |  2.24065      |  2.23529   |  2.27756       |
| 0.1|  2.23533  |   2.23204   |    2.23628    |  2.25550   |    2.23810     |
| 0.2|  2.23843  |   2.24448   |    2.25314    |   2.24408   |    2.25986     |
| 0.5|  2.24722  |   2.23186   |    2.24642    |    2.24707   |     2.26203    |
| 1|  2.22442  |   2.26428   |   2.24512     |   2.25322   |    2.26746     |
| 2|  2.23237  |   2.23331   |    2.23781    |   2.24748   |     2.24359     |


***FLOP***

|   Compression |   Rand |  Mag |  SNIP |  GraSP | SynFlow       |   
|----------------|-------------|-------------|-------------|---------------|----------------|
| 0.05|  279473230  |   297256726   |    301140357    |   256738852  |    297416253     |
| 0.1|  249017342  |   282390227   |    289955251    |   229925437  |    282917399     |
| 0.2|  197913366  |  255384262    |    245382304    |   179099186   |    257542273     |
| 0.5|  99270958  |   177116952   |   138207996     |   119043862   |    201378012     |
| 1|  31675248  |   74477668   |   61603571     |   54320754   |   143221205      |
| 2|  3405561  |   10436313   |    16049046    |   19449427   |    57801609    |

For better visualization, you are encouraged to transfer the above three tables into curves and present them as three figrues.
### 2. The compression ratio of each layer
Report the sparsity and draw the weight histograms of each layer using pruner Rand |  Mag |  SNIP |  GraSP | SynFlow with the following settings
`model = vgg16`, `dataset=cifar10`, `compression = 1`

***Bonus (optional)***

Report the FLOP of each layer using pruner Rand |  Mag |  SNIP |  GraSP | SynFlow with the following settings
`model = vgg16`, `dataset=cifar10`, `compression= 1`.
### 3. Explain your results and submit a short report.
Please describe the settings of your experiments. Please include the required results (described in Task 1 and 2). Please add captions to describe your figures and tables. It would be best to write brief discussions on your results, such as the patterns (what and why), conclusions, and any observations you want to discuss.  
