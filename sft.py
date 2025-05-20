from models.myllm import Config, MyLLM
from transformers import DefaultDataCollator, AutoTokenizer
from transformers import TrainingArguments, Trainer
from data.utils import PreTrainDataset
import sys
import os
sys.path.append(os.path.abspath('.'))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# sft 和 pretrian 的区别不大，唯一区别是数据集的构造
if __name__ == '__main__':
    config = Config()
    model = MyLLM(config)
    print(f'模型参数量为：{sum(param.numel() for param in model.parameters() if param.requires_grad)}')

    data_collator = DefaultDataCollator()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    args = TrainingArguments(output_dir='./results/', num_train_epochs=10,
                             do_train=True,
                             # 每个设备（GPU）上的 batch size：假如你有两个 GPU，总的 batch size 是 128 × 2 = 256。
                             per_device_train_batch_size=64,
                             # 梯度累积步数：等价于将 batch size 放大 8 倍（模拟大 batch）。
                             # 实际每 8 个小 batch 才执行一次 optimizer.step()。
                             # 累积期间只做 loss.backward()，第 N 步才 optimizer.step()
                             # 有效 batch size = 
                             # per_device_train_batch_size × num_devices × gradient_accumulation_steps
                             gradient_accumulation_steps=8,
                             # 如果设置了这个值，将优先按步数停止，而不是 num_train_epochs。
                             # max_steps=15000,
                             # 每训练 100 步，打印一次日志
                             logging_steps=100,
                             report_to='tensorboard',
                             # 最多保留几个 checkpoint
                             save_total_limit=5,
                             # 使用 bfloat16 精度训练
                             bf16=True,
                             learning_rate=2e-4,
                             lr_scheduler_type='cosine',
                             # 使用 8 个进程/线程预处理数据，加快数据读取速度。
                             dataloader_num_workers=8,
                             # 是否将数据固定在 CUDA 的 page-locked memory 中（加快数据传输速度）
                             dataloader_pin_memory=True,
                             # 是否保存为 .safetensors 格式：
                             # 设为 False 表示仍使用传统的 PyTorch .bin 格式。
                             save_safetensors=False
                             )
    # 注意，dataset 和 trainer 里 tokenizer 的作用不同
    # dataset 里就是用来分词，得到 token ID
    dataset = PreTrainDataset('data/pretrain_hq.jsonl', tokenizer=tokenizer, max_seq_len=512)
    # trainer 里 tokenizer 用于 日志打印、模型保存，由token ID 解码到自然语言打印出来等目的
    trainer = Trainer(model=model, args=args, train_dataset=dataset, tokenizer=tokenizer, data_collator=data_collator)

    trainer.train(resume_from_checkpoint=False)
    trainer.save_model('./results/saved_model')
    trainer.save_state()
