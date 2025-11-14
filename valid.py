import time
import torch
from tqdm import tqdm
from tool import indicator

#在验证集上 评估模型性能（返回计算损失、精准率、召回率、F1）-------------------------------------------------------------#
def valid(model,dataloader,dev_dataset):
    model.eval()                #开启评估模式（在此会关闭dropout等训练专用层）
    device = next(model.parameters()).device  #直接从模型参数获取设备
    # 获取标签映射
    id2label =  dev_dataset._get_id_to_label()
    pre_entities = []             #存储模型预测的实体
    true_entities = []            #存储真实的实体
    epoch_start = time.time()   #记录评估开始时间
    running_loss = 0.0          #累计损失值
    with torch.no_grad():       #禁用梯度计算
    #1、遍历数据
        tqdm_batch_iterator = tqdm(dataloader)
        for _, batch in enumerate(tqdm_batch_iterator):
    #2、数据移动到设备
            inputs = batch['input_ids'].to(device)#把本批次数据解包出输入、掩码、标签。从字典中获取数据
            att_masks = batch['attention_mask'].byte().to(device)
            label = batch['labels'].to(device)
            #处理真实的标签
            real_length = torch.sum(att_masks, dim=1)    #计算每个样本非填充部分的实际长度，即为「真实标签长度」
    #3、前向传播
            feats = model(inputs, att_masks)             #输入输入数据 and掩码「进行前向传播」
            loss = model.loss(feats, label, att_masks)   #输入前向传播结果 and标签 and掩码「来计算损失」
            running_loss += loss.item()                  #累计损失
    #4、预测标签
            predictions = model.predict(feats,att_masks)   #输入前向传播结果 and掩码「来预测最优标签」，即为「预测标签长度」
            batch_pre_entities = []
            batch_true_entities = []
            # 去除填充，保留有效标签
            for i in range(len(predictions)):
                # 获取有效长度内的标签
                seq_len = real_length[i].item()    
                # 预测的标签序列（去除padding）
                pred_seq = predictions[i][:seq_len]
                # 真实的标签序列（去除padding）
                true_seq = label[i][:seq_len].cpu().numpy().tolist()      #遍历批次中的每个样本。tags.numpy().tolist()是将PyTorch张量转换为Python列表
                # 提取实体
                pred_entitie = dev_dataset.extract_entities_from_labels(pred_seq, id2label)
                true_entitie = dev_dataset.extract_entities_from_labels(true_seq, id2label)          
                batch_pre_entities.append(pred_entitie)
                batch_true_entities.append(true_entitie)
            pre_entities.append(batch_pre_entities)                  #存储真实标签
            true_entities.append(batch_true_entities)              #存储预测结果    
    #（1）记录该轮次的训练时间
    epoch_time = time.time() - epoch_start  
    #（2）计算平均损失
    epoch_loss = running_loss / len(dataloader)        
    #（3）计算准确率、召回率、F1值
    Precision,Recall,F1_score=indicator(pre_entities, true_entities)
    estimator = (Precision,Recall,F1_score)#打包指标
    return epoch_time, epoch_loss, estimator
