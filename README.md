# GIMVC
This is the source code of the paper "Graph-Guided Imputation-Free Incomplete Multi-View Clustering". The paper has been accepted by the journal Expert Systems With Applications.

The GIMVC algorithm described in the paper is implemented using the Python language and the PyTorch framework. The algorithm needs to use a GPU when training the model, and if you want to train on the CPU, you need to modify the code.

The dataset directory contains the datasets used. To prevent the compression file from becoming too large, only a portion of the datasets has been uploaded. The code directory contains the code that implements the algorithm. In the construct_w directory is the similarity matrix corresponding to the dataset.

The main program file of the code is: train_duibi.py. Before running this program, make sure the dataset exists in the dataset directory.

Running environment: Python 3.8.18, torch 2.1.0+cu118.

If you have any questions, please feel free to contact me at yff235351@stu.xjtu.edu.cn.
