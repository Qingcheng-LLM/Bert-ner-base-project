import torch
from spacy.training.iob_utils import iob_to_biluo,tags_to_entities
from torch.utils.data import Dataset
#PyTorch的Dataset的作用：
#封装数据：将原始数据（如文本和标签）封装成一个数据集对象。
#数据预处理：在Dataset中，我们可以定义如何对每一条数据进行处理（例如使用tokenizer处理文本）。
#索引访问：通过实现__getitem__方法，我们可以通过索引直接访问数据集中任意一条数据。
#批量加载：与DataLoader配合，可以自动分批、打乱顺序、多线程加载数据，提高训练效率。
from transformers import BertTokenizerFast    
#HuggingFace的BertTokenizer的作用：
#分词：将句子分成一个个token（词片段），这些token是BERT词汇表中的元素。
#编码：将token转换成对应的ID（整数），以便BERT模型能够处理。
#添加特殊token：例如[CLS]（开头）和[SEP]（分隔/结尾），这是BERT模型要求的。
#填充和截断：将序列处理成统一的长度，以便批量处理。
#生成注意力掩码：告诉模型哪些位置是实际内容，哪些位置是填充的
from typing import List, Dict, Tuple   #python类似注解的导入语句，列表[单数]，字典{键值对，值可变}，元组（不可变，支持函数返回多个值）


#负责把数据从文件形式读入，并取成word label形式
class NERDataset(Dataset):
    #类初始化
    def __init__(self, file_path: str, label_file: str, tokenizer: BertTokenizerFast):
        self.tokenizer = tokenizer               #huggingface里BertTokenizer实例
        self.label2id = self._load_label_dict(label_file) #用来保持「标签到ID的映射字典」 
        self.samples = self._load_all_data(file_path)#根据文件路径加载数据
    #读取所有句子的word和label
    def _load_all_data(self, file_path: str) -> List[Tuple[List[str],List[str]]]:#列表里存放所有句子，每个句子用元组表示，元组包括词和标签俩列表
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content=f.read()#一次性读取整个文件
        sentence_set=file_content.strip().split('\n\n')#去除空白的首尾、按双换行符分割句子
        all_sentence=[]
        for sentence in sentence_set:
            lines=sentence.strip().split('\n')#把每个句子块内的行分开
            words, labels = [], []#初始化空列表
            parts=[line.split('\t') for line in lines if line]  #列表推导式
            for part in parts:#有效数据
                try:
                    word, label = part
                    words.append(word)
                    labels.append(label)
                except ValueError:
                    continue
            if words:#如果不是空句子
                all_sentence.append((words, labels))   #单个句子的word, label集合
        return all_sentence                            #全部句子的word, label集合
    def __len__(self):#返回数据集总长度，方便dataloader确定迭代的次数
        return len(self.samples)
    def __getitem__(self, idx):#可以通过索引直接访问数据集中任意一条数据
        words, labels = self.samples[idx]
        return {
            'words': words,
            'labels': labels}
    
    #负责对已经读入的批次数据进行预处理、动态填充
    def collate_fn(self,batch_data):#按批次动态填充
        # ----------------------------------------【tokenier】（实现分词、编码、添加特殊标记、掩码生成）-------------------------------
        batch_words=[seq['words']for seq in batch_data]
        batch_labels=[seq['labels']for seq in batch_data]
        encoding = self.tokenizer(batch_words, is_split_into_words=True, truncation=False, padding=False, return_offsets_mapping=True, return_tensors=None)
        # ----------------------------------------【标签对齐】------------------------------------------------------------------------   
        batch_labels_end = []         
        # 使用索引来分开拿出标签和词id，可以确保正确对应，但避免在不需要时使用i
        for idx in range(len(batch_labels)):
            labels = batch_labels[idx]
            word_ids = encoding.word_ids(batch_index=idx)  
            #开始标签对齐
            label_ids = []           #初始化标签索引列表
            previous_word_idx = None #记录前一个token的原始索引
            for word_idx in word_ids:#遍历每个token（每个字）
                if word_idx is None:# 如果token索引为空，这里的token可能是特殊标记([CLS], [SEP], [PAD])，对应的标签索引设为0
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:#如果当前token的索引和上个token索引不同，这是新词
                    # 当前token是新词的开始
                    label_ids.append(self.label2id[labels[word_idx]])#只有首子词保留原始标签
                    previous_word_idx = word_idx#标记上一token索引的标记后移到下一个新词元
                else:
                    # 当前token是同一个词的后续部分（同词非首），并且开头不是“I”,这输出标签ID同样为0
                    label_ids.append(self.label2id[labels[word_idx]] if labels[word_idx].startswith('I') else -100)#比如张三，这是俩词（B I)。如果是北京大学，这是一个词（B） 
            batch_labels_end.append(label_ids)
        # --------------------------------【动态填充】（取批次最长序列为最大长度）----------------------------------------------------     
        batch_input_ids=[]
        batch_attention_mask=[]
        batch_input_ids=encoding['input_ids']
        batch_attention_mask=encoding['attention_mask']
        #取每个批次的最大序列长度
        max_len = max(len(seq) for seq in batch_input_ids) 
        set_input_ids = []
        set_attention_mask = []
        set_labels = []
        #对批次内的每个序列的三类数据都进行填充
        for input_ids, attention_mask, labels in zip(batch_input_ids, batch_attention_mask, batch_labels_end):
            pad_len = max_len - len(input_ids)# 计算每个序列数据需要填充的长度
            padded_input = input_ids + [self.tokenizer.pad_token_id] * pad_len# 填充每个序列的token
            padded_attention = attention_mask + [0] * pad_len# 填充attention_mask
            padded_labels = labels + [-100] * pad_len# 填充labels（用-100忽略）
            #填充以后收集本批次填充好的数据
            set_input_ids.append(padded_input)
            set_attention_mask.append(padded_attention)
            set_labels.append(padded_labels)
        return {
            'input_ids': torch.tensor(set_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(set_attention_mask, dtype=torch.long),
            'labels': torch.tensor(set_labels, dtype=torch.long)}
    
    #加载label2id字典
    def _load_label_dict(self, label_file):
        label2id = {}
        with open(label_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                label = line.strip()
                if label:  # 跳过空行
                    label2id[label] = idx
        return label2id
    #动态获取标签数量
    def tagset_size(self):
        return len(self.label2id)
    #生成id2label字典
    def _get_id_to_label(self):
        return {v: k for k, v in self.label2id.items()}
    
    #从标签id中提取出实体标签
    def extract_entities_from_labels(self, label_ids, id2label):#从标签ID序列中提取实体
        #标签ID序列 [0, 0, 1, 2, 0, ...]》〉》〉实体列表 [(start, end, entity_type), ...]
        BIO_label= [id2label.get(id, 'O') for id in label_ids]
        #先把 BIO 转成 BILUO
        BILuo_label = iob_to_biluo(BIO_label)
        #提取实体
        entities = tags_to_entities(BILuo_label) 
        result = []
        for ent in entities:
            result.append((ent[0], ent[1] - 1, ent[2])) # ent[0]是起始字符偏移量，ent[1]是结束字符偏移量，ent[2]是实体类型
        return result
 