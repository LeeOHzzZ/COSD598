# Network Pruning
### Assignment 1 for COS598D: System and Machine Learning

In this assignment, you are required to evaluate three advanced neural network pruning methods, including SNIP [1], GraSP [2] and SynFlow [3], and compare with two baseline pruning methods,including random pruning and magnitude-based pruning. In `example/experiment.py`, we provide an example to do singleshot global pruning without iterative training. This assignment focuses on the pruning protocal in `experiment/example.py`. Your are going to explore various pruning methods on differnt hyperparameters and network architectures.

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
python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment example --pruner synflow --compression 0.5
```

To save the experiment, please add `--expid {NAME}`. `--compression-list` and `--pruner-list` are not available for runing example experiment. You can inmplement it following `experiment/singleshot.py` if you try to run a bunch of experiments. `--prune-epochs` is also not available as it does not affect your pruning. 

For magnitude-based pruning, please set `--pre-epochs 200`. The other methods do pruning before training, thus they can use the default setting `--pre-epochs 0`.

Please use the default batch size, learning rate, optimizer in the following experiment. Please use the default training and testing spliting. Please monitor training loss and testing loss, and set suitable training epochs.

## You Tasks

### 1. Hyper-parameter tuning

#### Testing on different archietectures. Please fill the results table:
*Test accuracy (top 1)* of pruned models on CIFAR10 and MNIST (sparsity = 10%). `--compression 1` means sparsity = 10^-1.
```
python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment example --pruner synflow --compression 1

# My command:
python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment example --pruner rand --compression 1 --expid task_1_rand --gpu 0 | tee ./log/task_1_rand.txt 

python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment example --pruner snip --compression 1 --expid task_1_snip --gpu 1 | tee ./log/task_1_snip.txt

python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment example --pruner grasp --compression 1 --expid task_1_grasp --gpu 2 | tee ./log/task_1_grasp.txt

python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment example --pruner synflow --compression 1 --expid task_1_synflow --gpu 3 | tee ./log/task_1_synflow.txt

python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment example --pruner mag --pre-epochs 200 --compression 1 --expid task_1_mag | tee ./log/task_1_rand.txt

```
```
python main.py --model-class default --model fc --dataset mnist --experiment example --pruner synflow --compression 1
# My command:
python main.py --model-class default --model fc --dataset mnist --experiment example --pruner rand --compression 1 --expid task_1_rand_fc --gpu 0

python main.py --model-class default --model fc --dataset mnist --experiment example --pruner snip --compression 1 --expid task_1_snip_fc --gpu 1

python main.py --model-class default --model fc --dataset mnist --experiment example --pruner grasp --compression 1 --expid task_1_grasp_fc --gpu 2

python main.py --model-class default --model fc --dataset mnist --experiment example --pruner synflow --compression 1 --expid task_1_synflow_fc --gpu 3
python main.py --model-class default --model fc --dataset mnist --experiment example --pruner mag --pre-epochs 200 --compression 1 --expid task_1_mag_fc 
```
|   Data  |   Arch |   Rand |  Mag |  SNIP |  GraSP | SynFlow       |   
|----------------|----------------|-------------|-------------|-------------|---------------|----------------|
|Cifar10 | VGG16 |  76.79%  |      |   10%     |   49.92%  |    80.61%     |
|MNIST| FC |  54.51%  |   43.54%   |     56.48%   |  55.53%    |    10.0%     |


#### Tuning compression ratio. Please fill the results table:
Prune models on CIFAR10 with VGG16, please replace {} with sparsity 10^-a for a \in {0.05,0.1,0.2,0.5,1,2}. Feel free to try other sparsity values.

```
python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment example --pruner synflow  --compression {}
```
***Testing accuracy (top 1)***

|   Compression |   Rand |  Mag |  SNIP |  GraSP | SynFlow       |   
|----------------|-------------|-------------|-------------|---------------|----------------|
| 0.05|    |      |        |     |         |
| 0.1|    |      |        |     |         |
| 0.2|    |      |        |      |         |
| 0.5|    |      |        |      |         |
| 1|    |      |        |      |         |
| 2|    |      |        |      |         |

***Testing time***

|   Compression |   Rand |  Mag |  SNIP |  GraSP | SynFlow       |   
|----------------|-------------|-------------|-------------|---------------|----------------|
| 0.05|    |      |        |     |         |
| 0.1|    |      |        |     |         |
| 0.2|    |      |        |      |         |
| 0.5|    |      |        |      |         |
| 1|    |      |        |      |         |
| 2|    |      |        |      |         |


***FLOP***

|   Compression |   Rand |  Mag |  SNIP |  GraSP | SynFlow       |   
|----------------|-------------|-------------|-------------|---------------|----------------|
| 0.05|    |      |        |     |         |
| 0.1|    |      |        |     |         |
| 0.2|    |      |        |      |         |
| 0.5|    |      |        |      |         |
| 1|    |      |        |      |         |
| 2|    |      |        |      |         |

For better visualization, you are encouraged to transfer the above three tables into curves and present them as three figrues.
### 2. The compression ratio of each layer
Report the sparsity and draw the weight histograms of each layer using pruner Rand |  Mag |  SNIP |  GraSP | SynFlow with the following settings
`model = vgg16`, `dataset=cifar10`, `compression = 1`

***Bonus (optional)***

Report the FLOP of each layer using pruner Rand |  Mag |  SNIP |  GraSP | SynFlow with the following settings
`model = vgg16`, `dataset=cifar10`, `compression= 1`.
### 3. Explain your results and submit a short report.
Please describe the settings of your experiments. Please include the required results (described in Task 1 and 2). Please add captions to describe your figures and tables. It would be best to write brief discussions on your results, such as the patterns (what and why), conclusions, and any observations you want to discuss.  
