from modelscope.hub.snapshot_download import snapshot_download
# python pre_model/downloadmodel.py
model_dir = snapshot_download('xd1651fhtfhtfh/bert-base-chinese', cache_dir='./pre_model')
print("模型已下载到BERT-NER/pre_model")

