Python 3.13.5
PyTorch版本: 2.9.0+cu126
CUDA版本: 12.6

设备名称: NVIDIA GeForce RTX 3060/ 用时：3分钟/ F1值约为63%

这里一个基于Pytorch框架使用Bert完成的NER项目。
处理数据使用Pytorch架构的Dataset和DataLoader，在Dataset里主要使用了transformers里面的 BertTokenizerFast包。
这个项目主要是学习NER项目如何构建，所以在模型部分只用了Bert。在该项目的验证部分，计算的评价指标有准确率、召回率、F1值。
该项目里已经下好了预训练模型，存在了pre_model文件夹里。
配置的超参数在dataset文件夹下的my_config.json文件里，在程序中动态解析，在保存项目运行历史记录时，方便连超参数一同保存。