from transformers import TrainingArguments, Trainer, AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import torch.nn.functional as F
from data.utils import DPODataset, DPODataCollator 
from models.myllm import MyLLM, Config
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def logits_to_probs(logits, labels):
    # logits shape: (batch_size, seq_len, vocab_size)
    # labels shape: (batch_size, seq_len)
    # probs shape: (batch_size, seq_len)
    log_probs = F.log_softmax(logits, dim=2)
    # 每个label对应的概率 
    probs = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return probs

def dpo_loss(ref_probs, probs, beta):
    batch_size = ref_probs.size(0) // 2

    chosen_probs = probs[:batch_size]
    rejected_probs = probs[batch_size:]
    
    ref_chosen_probs = ref_probs[:batch_size]
    ref_rejected_probs = ref_probs[batch_size:]

    pi_logratios = chosen_probs - rejected_probs
    ref_logratios = ref_chosen_probs - ref_rejected_probs
    
    logits = pi_logratios - ref_logratios
    
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()

def mask_logits(logits, labels):
    # logits shape: (batch_size, seq_len, vocab_size)
    # labels_masks: (batch_size, seq_len)
    new_logits = []
    for logit, label in zip(logits, labels):
        new_logits.append(logit[label != -100].sum())
    return new_logits

# class DPOTrainer(Trainer):
#     def compute_loss(self, model, ref_model, inputs, return_outputs=False, num_items_in_batch=None):
#         input_ids = inputs['input_ids']
#         labels = inputs['labels']
#         with torch.no_grad():
#             ref_logits = ref_model(input_ids=input_ids, labels=labels).logits
#         ref_probs = logits_to_probs(ref_logits, labels)
#         ref_probs = mask_logits(ref_probs, labels)

#         logits = model(input_ids=input_ids, labels=labels).logits
#         probs = logits_to_probs(logits, labels)
#         probs = mask_logits(probs, labels)
#         loss = dpo_loss(ref_probs, probs, 0.1)
#         return loss

class DPOTrainer(Trainer):
    def __init__(self, ref_model=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model  # 保存参考模型
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids = inputs['input_ids']
        labels = inputs['labels']

        with torch.no_grad():
            ref_outputs = self.ref_model(input_ids=input_ids, labels=labels)
            ref_logits = ref_outputs.logits

        ref_probs = logits_to_probs(ref_logits, labels)
        ref_probs = mask_logits(ref_probs, labels)

        outputs = model(input_ids=input_ids, labels=labels)
        logits = outputs.logits

        probs = logits_to_probs(logits, labels)
        probs = mask_logits(probs, labels)

        loss = dpo_loss(ref_probs, probs, 0.1)

        return (loss, outputs) if return_outputs else loss
if __name__ == '__main__':
    AutoConfig.register('small_model', Config)
    AutoModelForCausalLM.register(Config, MyLLM)

    model = AutoModelForCausalLM.from_pretrained('./results/checkpoint-58000')
    print(f'模型可训练参数量为：{sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    ref_model = AutoModelForCausalLM.from_pretrained('./results/checkpoint-58000').eval().to('cuda')

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    # 加载的大模型旋转位置编码最大长度为1024，max_seq_len 不能超过这个值
    data_collator = DPODataCollator(tokenizer, max_seq_len=512)
    args = TrainingArguments(output_dir='./results/',
                             # 训练太多轮，模型似乎会输出很多重复内容
                             num_train_epochs=1,
                             do_train=True,
                             per_device_train_batch_size=16,
                             gradient_accumulation_steps=4,
                             # max_steps=15000,
                             logging_steps=50,
                             report_to='tensorboard',
                             save_total_limit=3,
                             bf16=True,
                             # 学习率很重要，太大会把模型训飞
                             learning_rate=0.00001,
                             lr_scheduler_type='cosine',
                             dataloader_num_workers=1,
                             dataloader_pin_memory=True,
                             save_safetensors=False,
                             save_steps=100
                             )
    dataset = DPODataset('./data/dpo.jsonl', tokenizer=tokenizer)
    trainer = DPOTrainer(model=model, ref_model=ref_model, args=args, train_dataset=dataset, 
                         tokenizer=tokenizer, data_collator=data_collator)
    
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model('./results/dpo')
    trainer.save_state()


