
#🚀项目简介:  
这里一个基于Pytorch框架使用Bert模型完成的一个命名实体识别的项目。项目包括以下模块：  
 1.数据预处理（tokenizer、动态填充、标签对齐）与数据加载  
 2.模型设计与结构配置类
 3.模型训练与模型评估，并提供可视化（Tensorboard）   
*项目里提供两种预训练模型权重，并提供了下载预训练模型的代码（downloadmodel.py），首次运行需要下载预训练模型，后续可直接运行，下载的预训练模型均存储在了pre_model文件夹里。
*项目提供两种数据集可选，所以可见配置的超参数在configs文件夹下，分别提供了.json文件以便带项目运行时动态解析。
*在保存项目运行结果时，会连同所选数据集名称、预训练模型名称、超参数配置文件一同保存，方便后续复现表现较为出色结果，结果存储在result文件夹下。
*在项目main.py文件开头已经给出不同预训练及不同数据集选择的项目执行命令，配好环境后，可直接复制命令行终端运行，运行结果的F1值约为63%。
    
#📊 数据集来源：  
 1.weibo命名实体识别数据集。下载地址：https://tianchi.aliyun.com/dataset/144312  
 2.msra命名实体识别数据集。下载地址：https://tianchi.aliyun.com/dataset/144307?spm=a2c22.12282016.0.0.432a4f03K11Mhq

#📊 预训练模型：
 预训练模型下载脚本已在pre_model文件夹下给出。
 1.bert-base-chinese  下载地址：https://huggingface.co/google-bert/bert-base-chinese
 2.chinese-bert-wwm  下载地址:https://huggingface.co/hfl/chinese-bert-wwm

#🗓项目目录:  
![image](https://github.com/Qingcheng-LLM/Bert-ner-base-project.png)  