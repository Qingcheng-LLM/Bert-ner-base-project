#python -m main
#学术加速终端开启：source /etc/network_turbo
#取消学术加速：unset http_proxy && unset https_proxy
#我现在已经有个anaconda，cuda Version: 13.0之前的尝试都太乱了，现在我要重新配置一个conda环境，需要使用gpu的pytorch、numpy、transformer，要保证他们相互之间不发生冲突
# pip uninstall numpy scipy contourpy matplotlib spacy thinc -y
# mamba install numpy=1.24.3 scipy=1.10.1 contourpy=1.2.0 -c conda-forge -y
import os
import torch
from my_config import my_Config,load_json
from Dataset import NERDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  #数据可视化工具
from torch.optim.lr_scheduler import CosineAnnealingLR #学习率衰减
from transformers import BertTokenizerFast
#​HuggingFace提供的批处理函数用于告诉DataLoader如何把多个样本合并成一个 batch。1、动态计算每 batch中最长的序列长度；2、对input_ids、attention_mask、labels做填充；3、保证 padding 部分的 label 为 -100（不参与 loss 计算）；4、最终返回可直接输入模型的 batch 张量。
from model import Bert_ner
from train import train
from valid import valid
print(f"PyTorch版本: {torch.__version__}")
print(f"设备名称: {torch.cuda.get_device_name(0)}")
print(f"CUDA版本: {torch.version.cuda}")


def main():
    config = my_Config(**load_json(json_path="E:/Desktop/BERT-NER/dataset/my_config.json")) #加载配置文件（超参数、路径）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  #初始化设备
    print('loading corpus')
    #-----------------------------------------------准备数据------------------------------------------------#
    #初始化tokenizer
    tokenizer=BertTokenizerFast.from_pretrained(config.pretrain_model_path)
    #创建数据集，读入数据
    train_dataset = NERDataset(config.train_file, config.label_file, tokenizer)
    dev_dataset = NERDataset(config.dev_file, config.label_file, tokenizer)
    #创建数据加载器，同时加载的同时处理数据
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn, num_workers=0,pin_memory=True)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=dev_dataset.collate_fn, num_workers=0,pin_memory=True)
    #----------------------------------------------训练前准备-----------------------------------------------#
    #初始化模型
    tagset_size = train_dataset.tagset_size()   #从处理完的训练数据中获取标签数量
    model = Bert_ner(config,tagset_size).to(device)
    #定义优化器（参数：学习率、学习率衰减、权重衰减）
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    #动态调整学习率（指标不提升时衰减学习率）使用自适应学习率调度器
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)
    #初始化训练状态
    best_score = -1           #记录最佳验证集F1
    start_epoch = 1           #起始epoch为1
    patience_counter = 0      #早停计数器，记录指标未提升的epoch数
    os.makedirs(config.target_dir, exist_ok=True)  # 创建保存目录
    #在训练之前评估模型的初始性能，并打印输出。会返回的参数；损失值、准确率、召回率、F1
    _, valid_loss, start_estimator = valid(model,dev_loader,dev_dataset)
    print("\t* Validation loss before training: loss = {:.4f}, accuracy: {:.4f}%, recall: {:.4f}%, F1: {:.4f}%".format(valid_loss, (start_estimator[0] * 100), (start_estimator[1] * 100), (start_estimator[2] * 100)))
    # 初始化TensorBoard Writer     记录初始指标到TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(config.target_dir, "tensorboard_logs"))
    writer.add_scalar('Loss/valid', valid_loss, 0)
    writer.add_scalar('Metrics/accuracy', start_estimator[0] * 100, 0)
    writer.add_scalar('Metrics/recall', start_estimator[1] * 100, 0)
    writer.add_scalar('Metrics/F1', start_estimator[2] * 100, 0)
    #打印分割线和设备信息
    print("\n",20 * "=","Training BERT_NER model on device: {}".format(device),20 * "=")
    #------------------------------------------按轮次训练and验证--------------------------------------------#
    #遍历每个轮次epoch
    for epoch in range(start_epoch, config.epochs+1): 
        print("* Training epoch {}:".format(epoch))
        #调用train()函数完成一个epoch的训练，返回「本轮次耗时」and「损失」。并打印
        epoch_time, epoch_loss = train(model,train_loader,optimizer,config.max_grad_norm, writer, epoch)
        print("-> Training time: {:.4f}s, loss = {:.4f}".format(epoch_time, epoch_loss))
        #调用valid()函数完成一个epoch的验证，「返回耗时」and「损失」and「准确率、召回率、F1」。并打印
        epoch_time, valid_loss, valid_estimator = valid(model,dev_loader, dev_dataset)
        print("-> Valid time: {:.4f}s, loss = {:.4f}, accuracy: {:.4f}%, recall: {:.4f}%, F1: {:.4f}%".format(epoch_time, valid_loss, (valid_estimator[0] * 100), (valid_estimator[1] * 100), (valid_estimator[2] * 100)))
        #记录验证指标到TensorBoard
        writer.add_scalar('Loss/valid', valid_loss, epoch)
        writer.add_scalar('Metrics/accuracy', valid_estimator[0] * 100, epoch)
        writer.add_scalar('Metrics/recall', valid_estimator[1] * 100, epoch)
        writer.add_scalar('Metrics/F1', valid_estimator[2] * 100, epoch)
        writer.add_scalar('Time/valid_epoch_time', epoch_time, epoch)
        #记录学习率
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning_rate', current_lr, epoch)
        #若验证集指标未提升，则按照factor=0.5降低学习率
        scheduler.step()
        #根据准确率判断是否更好。
        if valid_estimator[0] > best_score:
            best_score = valid_estimator[0]
            patience_counter = 0
        else:
            patience_counter += 1
        #早停触发
        if patience_counter >= config.patience:
            print("-> Early stopping: patience limit reached, stopping...")
            break
    #保存最后一个epoch的模型及使用的超参数
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config_dict': config.__dict__ ,
        'epoch': epoch,
        'best_score': best_score
        }
    torch.save(checkpoint, os.path.join(config.target_dir, "checkpoint.pth"))
    writer.add_scalar('Best/accuracy', best_score * 100, epoch)
    writer.add_scalar('Best/F1', valid_estimator[2] * 100, epoch)
    # 关闭TensorBoard Writer
    writer.close()
    print("训练完成！使用以下命令查看TensorBoard：python -m tensorboard.main --logdir result/tensorboard_logs")


if __name__ == "__main__":
    main()