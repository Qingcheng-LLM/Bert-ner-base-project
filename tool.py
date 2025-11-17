
#计算【准确率】【召回率】【F1值】
def indicator(pre_entities, true_entities):
    P_len = 0  # ！存放累加的数值必须初始化，另外.extend()和.append()方法以及字典键值都需要初始化
    R_len = 0
    U_len = 0
    for  batch_pre,barch_truth in zip(pre_entities, true_entities):
        for seq_pre,seq_truth in zip(batch_pre,barch_truth):
            P = set(seq_pre)
            R = set(seq_truth)
            U = P & R #求预测标签总数、真实标签总数、预测标签和真实标签的交集总数（预测出的标签数）
#       .extend()将多层嵌套的序列合并为单层列表,允许重复,适合严格逐样本匹配；.=set()​​仅统计当前样本的标签,自动去重，适合全局累加统计
            P_len += len(P)
            R_len += len(R)
            U_len += len(U)
    #计算三个指标
    Precision=U_len/P_len if P_len>0 else 0
    Recall=U_len/R_len if R_len>0 else 0
    F1_score=2*Precision*Recall/(Precision+Recall) if (Precision+Recall)>0 else 0
    return Precision,Recall,F1_score

