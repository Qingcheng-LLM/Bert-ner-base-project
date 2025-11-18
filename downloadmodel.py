from huggingface_hub import snapshot_download

snapshot_download(repo_id="google-bert/bert-base-chinese",local_dir = "pre_model/bert-base-chinese",ignore_patterns=["*.h5","flax_model.msgpack",""], local_dir_use_symlinks=False,resume_download=True)
snapshot_download(repo_id="hfl/chinese-bert-wwm",local_dir = "pre_model/chinese-bert-wwm",ignore_patterns=["*.h5","flax_model.msgpack",""], local_dir_use_symlinks=False,resume_download=True)
print("预训练权重已下载到BERT-NER/pre_model目录下对应子文件夹中")
# https://huggingface.co/hfl/chinese-bert-wwm
# https://huggingface.co/google-bert/bert-base-chinese