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
- SFT代码
- RLHF PPO 伪代码（需要的模块太多，没有运行）

TODO：
- DPO 代码

下面是具体的实现步骤
## 首先，下载数据集
我们直接用的minimind整理好的数据集：https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files
点进去下载里面的 pretrain_hq.jsonl，放到data目录里

## 然后配置好相关参数
主要是 CUDA_VISIBLE_DEVICES 和 batch size。

## 直接运行预训练代码
python pretrain.py

## 测试模型
在 test_llm.ipynb 中逐个运行cell进行测试。

### pretrain 的实验结果
Question:死亡是生物进化的选择吗？
Answer:<|im_start|>请回答，什么是人工智能？人工智能是指计算机程序和机器可以模拟人类智能的技术和系统，包括机器学习、深度学习、自然语言处理、计算机视觉等技术。人工智能可以让计算机自主地学习、推理、识别和决策，以实现某些任务的能力，例如语音识别、图像识别、自然语言处理、智能推荐等。人工智能已经在许多领域得到了广泛应用，如医疗保健、金融、交通、制造业等。<|im_end|> <|im_start|>我想知道如何学好英语。学好英语需要多方面的努力。首先，你应该多听多说多读多写，�

### sft 的实验结果
Question:死亡是生物进化的选择吗？
Answer:不，生物进化的选择并不总是取决于生物的类型、体型或生活方式。生物进化是一个复杂的过程，涉及到语言、结构、生理、遗传等多个方面的多个阶段。在生物进化中，生物的进化和繁殖是非常重要的一部分，它们不仅提供了结构的基础，而且进化了现代社会。

在进化过程中，生物体的进化结构和生物进化率的变化主要受到遗传和生态系统的影响。生物体的进化速度比现代社会的变得快，从而影响其生长、繁殖、进化等生理过程。在这一过程中，生物的繁殖和繁殖受到影响，这包括了遗传、生态系统、经济等多个方面。

�