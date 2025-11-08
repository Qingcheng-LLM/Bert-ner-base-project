import torch
import os
from model import Bert_ner
from my_config import my_Config,load_json
from transformers import BertTokenizerFast
from dataset import NERDataset
from spacy.training import biluo_tags_to_offsets

#预测函数的定义。这是完整的NER「预测流程」，结合了BERT的上下文表示能力、BiLSTM的序列建模能力和CRF的标签依赖建模
tags = [(1, 2), (3, 4), (5, 6),(7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]#标签序列
def predict(input_seq, max_length=128):
    #-------------------------------准备----------------------------------#
    #配置初始化
    config = my_Config(**load_json(json_path="E:/Desktop/BERT-NER/dataset/my_config.json"))                        #加载配置文件
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #优先选择使用GPU
    #初始化tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(config.tokenizer_path)
    #模型初始化
    dataset = NERDataset(file_path=config.train_file, label_file=config.label_file, tokenizer=tokenizer)#为了用NERDataset求tagset_size
    tagset_size = dataset.tagset_size()     #为了求标签数量用于Bert_ner模型
    model = Bert_ner(config,tagset_size).to(device)                #放到设备里
    #模型加载部分
    model_path = os.path.join("result", "final_result.pth")  # 推荐使用最佳模型
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"模型权重已加载: {model_path}")
    else:
        raise FileNotFoundError(f"权重文件不存在: {model_path}")
    #----------------------------输入序列的处理-----------------------------#
    # 预处理输入
    encoding = tokenizer(input_seq,return_tensors='pt',max_length=max_length,truncation=True,padding='max_length')
    #将输入和掩码都转换为pytorch的张量tensor并放到指定设备上, 二维, 这里mask在CRF中必须为unit8类型或者bool类型
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    #------------------------------开始预测--------------------------------#
    #输入模型前向传播，获得输出feats
    feats = model(input_ids, attention_mask)
    #最优路径out_path是一条预测路径（数字列表）, [1:-1]表示去掉一头一尾, <START>和<EOS>标志
    out_path = model.predict(feats, attention_mask)[0][1:-1]
    #将上一步输出的标签id转换为标签实体
    label2id = dataset._load_label_dict()
    id2label={v:k for k,v in label2id.items()}
    labels=[id2label.get(id,'0') for id in out_path]
    res = biluo_tags_to_offsets(feats, labels)#使用spacy里的标签 转成起始 终止位置的方法
    #res = find_all_tag_predict(out_path)
    #按类型分类存储实体
    PER = []
    LOC = []
    ORG = []
    GPE = []
    PERA = []
    LOCA = []
    ORGA = []
    GPEA = []
    for name in res:
        entities = res[name]
        for start, length in entities:
            entity_text = tokenizer.decode(input_ids[0, start:start+length])
            if name == 1:
                PER.append(entity_text)
            elif name == 2:
                LOC.append(entity_text)
            elif name == 3:
                ORG.append(entity_text)
            elif name == 4:
                GPE.append(entity_text)
            elif name == 5:
                PERA.append(entity_text)
            elif name == 6:
                LOCA.append(entity_text)
            elif name == 7:
                ORGA.append(entity_text)
            elif name == 8:
                GPEA.append(entity_text)    
    #输出识别结果
    return PER, LOC, ORG, GPE, PERA, LOCA, ORGA, GPEA # 返回实体列表

# 交互式预测入口
def interactive_predict():
    while True:
        input_seq = input("输入: ")
        if input_seq.lower() in ["exit", "quit"]:
            break
        PER, LOC, ORG, GPE, PERA, LOCA, ORGA, GPEA= predict(input_seq)
        print("预测结果:", "\n", "PER:", PER, "\n", "ORG:", ORG, "\n", "LOC:", LOC, "\n", "GPE:", GPE, "\n", "PERA:", PERA, "\n", "LOCA:", LOCA, "\n", "ORGA:", ORGA, "\n", "GPEA:", GPEA)
# 直接调用交互模式
if __name__ == "__main__":
    interactive_predict() 
