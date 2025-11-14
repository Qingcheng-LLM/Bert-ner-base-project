
#计算【准确率】【召回率】【F1值】
def indicator(pre_entities, true_entities):
    pred_len = 0  # ！存放累加的数值必须初始化，另外.extend()和.append()方法以及字典键值都需要初始化
    truth_len = 0
    Union_len = 0
    for  batch_pre,barch_truth in zip(pre_entities, true_entities):
        for seq_pre,seq_truth in zip(batch_pre,barch_truth):
            pred = set(seq_pre)
            truth = set(seq_truth)
            Union=pred & truth #求预测标签总数、真实标签总数、预测标签和真实标签的交集总数（预测出的标签数）
#       .extend()将多层嵌套的序列合并为单层列表,允许重复,适合严格逐样本匹配；.=set()​​仅统计当前样本的标签,自动去重，适合全局累加统计
            pred_len += len(pred)
            truth_len += len(truth)
            Union_len += len(Union)
    #计算三个指标
    Precision=Union_len/pred_len if pred_len>0 else 0
    Recall=Union_len/truth_len if truth_len>0 else 0
    F1_score=2*Precision*Recall/(Precision+Recall) if (Precision+Recall)>0 else 0
    return Precision,Recall,F1_score

