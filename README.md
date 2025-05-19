# 基于 Huggingface 库从“0”实现一个大模型

本项目代码主要参考 [llm_related](https://github.com/wyf3/llm_related/tree/main)，部分参考 [minimind](https://github.com/jingyaogong/minimind/tree/master)

目前手撕实现的内容：
- 旋转位置编码 RoPE
- RMSNorm
- Group Query Attention
- GLU 实现的 FFN
- 上述模块构成的 DecoderLayer
- 上述模块构成的“大”模型（只有30+M）
- 预训练代码

下面是具体的实现步骤
## 首先，下载数据集
我们直接用的minimind整理好的数据集：https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files
点进去下载里面的 pretrain_hq.jsonl，放到data目录里

## 然后配置好相关参数
主要是 CUDA_VISIBLE_DEVICES 和 batch size。

## 直接运行预训练代码
python myllm.py
