import torch
import torch.nn as nn
from transformers import BertModel #导入Hugging Face Transformers库中的BERT模型类。

#定义模型类
class Bert_ner(nn.Module):
    #模型类的初始化方法
    def __init__(self,config,tagset_size):
        super().__init__()#子类BERT调用父类nn.Module的初始化方法
        self.config = config 
        self.tagset_size = tagset_size      #标签总数
        self.embedding_dim = config.bert_embedding
        #加载预训练BERT模型（在config中已经从HuggingFace模型库中指定了想用的模型），首次运行时还会自动下载模型权重
        self.bert = BertModel.from_pretrained(config.pretrain_model_path)
        #定义dropout层
        self.dropout = config.dropout 
        self._dropout = nn.Dropout(p=self.dropout)
        #定义线性分类层
        self.Liner = nn.Linear(self.embedding_dim, self.tagset_size)#liner输入为bert的输出维度。
        #损失函数（交叉熵损失）
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)  # 忽略填充标签

    #前向传播
    def forward(self, sentence, attention_mask=None):   # sentence是形状为(batch_size, max_seq_len)的输入序列
        #通过bert获取词嵌入。embeds: [batch_size, max_seq_length, embedding_dim]
        embeds = self.bert(sentence, attention_mask=attention_mask).last_hidden_state #.last_hidden_state是Hugging Face Transformers库中BERT模型输出的一个关键属性，是bert模型提取了最高语义的最终输出。
        #应用Dropout丢弃部分神经元
        d_bert_out = self._dropout(embeds)
        #线性变换
        l_out = self.Liner(d_bert_out)
        #调整数据。将bert_out: [  , embedding_dim]数据变为(batch_size, seq_length, -1)因为embedding_dim输入线性层，输出tagset_size。
        return l_out     #返回的数据格式：(batch_size, seq_length, tagset_size)
    
    #计算损失
    def loss(self, feats, tags, mask):
        feats_flat = feats.view(-1, self.tagset_size)  # 【预测的标签】维度变换：[batch_size，seq_length, tagset_size]》》[batch_size*seq_length, tagset_size]
        tags_flat = tags.view(-1)  # 【标签】维度变换：[batch_size*seq_length] 
        loss = self.criterion(feats_flat, tags_flat)
        return loss
    
    def predict(self, feats, mask=None):
        # 在前向传播中数据已经经过了一系列的处理，每个位置有很多的预测结果并附加概率，在得到的数据里用argmax来获取每个位置概率最大的标签，这就是预测。
        predictions = torch.argmax(feats, dim=-1)  # 【初步预测结果】[batch_size, seq_length]
        return predictions.cpu().numpy().tolist()
    
