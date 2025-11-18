import os
from huggingface_hub import snapshot_download

models = ["google-bert/bert-base-chinese", "hfl/chinese-bert-wwm"]
root = os.path.dirname(__file__) # 获取当前脚本所在目录
for rid in models:
    out = os.path.join(root, rid.split("/")[-1])  # 拼接本地保存路径
    snapshot_download(rid, local_dir=out, local_dir_use_symlinks=False, resume_download=True)
print("预训练权重已下载到BERT-NER/pre_model目录下对应子文件夹中")

# https://huggingface.co/hfl/chinese-bert-wwm
# https://huggingface.co/google-bert/bert-base-chinese