# GIMVC
这是论文“Graph-Guided Imputation-Free Incomplete Multi-View Clustering”的源代码。论文已经被Expert Systems With Applications期刊接收。

论文中的GIMVC算法采用python语言和pytorch框架实现。该算法在训练模型时需要用到显卡，如果想在CPU上进行训练，则需要修改代码。

dataset目录下是用到的数据集，为了防止压缩包体积过大，只上传了部分数据集。code目录下是实现算法的代码。construct_w目录下是数据集对应的相似度矩阵。

代码的主程序文件为：train_duibi.py。运行该程序前首先确保数据集存在于dataset目录下。

运行环境：Python 3.8.18，torch 2.1.0+cu118。

如果您有任何疑问，欢迎与我联系：yff235351@stu.xjtu.edu.cn。
