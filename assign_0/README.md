# COSD598_Warmup -- MNIST Classification
This is a warmup task to let you get familiar with Azure Cloud Platform and running deep learning model using Pytorch.

## Task1

### Get started with Azura

To get started with Azure, I’d recommend you to view videos 1 – 5 in the Azure for Academic series found here:

<a href="https://channel9.msdn.com/Series/Azure-for-Academics">Azure for Academics | Channel 9 (msdn.com)</a>. You might find the rest of the series useful, but they can skip the 6th video as they will have already have an Azure account.

### Virtual Machine

As for resources, here’s step-by-step guided lessons from Microsoft Learn on virtual machines:
<a href="https://docs.microsoft.com/en-us/learn/modules/intro-to-azure-virtual-machines/">Introduction to Azure virtual machines - Learn | Microsoft Docs</a>

<a href="https://docs.microsoft.com/en-us/learn/modules/create-windows-virtual-machine-in-azure/">Create a Windows virtual machine in Azure - Learn | Microsoft Docs</a>

<a href="https://docs.microsoft.com/en-us/learn/modules/create-linux-virtual-machine-in-azure/">Create a Linux virtual machine in Azure - Learn | Microsoft Docs</a>
 
This video is focused on troubleshooting VMs: 

<a href="https://www.youtube.com/watch?v=MAJrN-2IPY8">How to troubleshoot common virtual machine issues | Azure Portal Series - YouTube</a>

## Task 2
Please run the model with the default setting in main.py. You do not need to specify any configuration. 

### Model and Dataset

In this experiment, you will train convolutional neural networks (CNN) for MNIST dataset classification. MNIST dataset contains hand-writing digits 0-9. When you run the code, data will be automatically downloaded and split into training and testing datasets. Training data will be used for CNN training and testing data will be used for evaluation. 

### CNN archietecture

You are required to learn how to build your own deep learning model. We offered a CNN with two Conv2D layers and two FC layers as defined below. Let us call the default model as 'model1'.
```
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
```
- [ ] Please add one more Conv2D layer with `channel size  (number of kernels) = 32` between layer `self.conv1` and layer `self.conv2`. You need to modify the `forward` function correspondingly. Let us call this model as 'model2'.
- [ ] Please add one more FC layer with `channel size  (number of kernels) = 128` between layer `self.fc1` and layer `self.fc2`. You need to modify the `forward` function correspondingly. Let us call this model as 'model3'.


### What to be included in your report?
Please plot `training loss vs. epoch`, `testing loss vs. epoch`, `training accuracy vs. epoch`, and `testing accuracy vs. epoch` for model1, model2 and model3. You need to write a function(s) to record the results in `main.py`.

### How to run?
```bash
pip install -r requirements.txt
python main.py
```
