import torch
from torch.utils.data import Dataset
import json
import numpy as np

class PreTrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_len):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.data = f.readlines()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index:int):
        # 获取第index个json字符串
        line = self.data[index]
        # 把这一行的json字符串转换为Python字典。
        line = json.loads(line)
        text = '<s>' + line['text'] + '</s>'
        # 使用 tokenizer 对加了特殊符号的文本进行编码，转换成 token IDs
        tokens = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_seq_len,
            padding='max_length'  
        )
        # input_ids = self.tokenizer.encode(text, add_special_tokens=False)
        input_ids = tokens['input_ids']
        text_len = len(input_ids)

        if text_len > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
        else:
            # 这里要和llm的损失函数对应，ignore_index应是-100
            input_ids = input_ids + [-100] * (self.max_seq_len - text_len)
        
        # input_ids = np.array(input_ids)
        # seq = np.array(input_ids[:-1]).astype(np.int64)
        # labels = np.array(input[1:]).astype(np.int64)
        input_tensor = torch.tensor(input_ids[:-1], dtype=torch.long)
        label_tensor = torch.tensor(input_ids[1:], dtype=torch.long)
        return {
            'input_ids': input_tensor,
            'labels': label_tensor,
        }
    

class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # 确保 pad_token 存在
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 读取 .jsonl 文件
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line.strip()) for line in f if line.strip()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        conversations = self.data[idx]["conversations"]

        # 构建 prompt 和 answer
        prompt = ""
        for msg in conversations[:-1]:
            role = msg["role"]
            content = msg["content"]
            prefix = "User: " if role == "user" else "Assistant: "
            prompt += prefix + content + "\n"

        last_msg = conversations[-1]
        if last_msg["role"] != "assistant":
            raise ValueError("最后一轮必须是 assistant 的回答。")
        
        prompt += "Assistant: "
        answer = last_msg["content"] + self.tokenizer.eos_token

        # 编码
        prompt_ids = self.tokenizer.encode(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_seq_len // 2,  # 可调，避免 prompt+answer 总长超限
            padding='max_length'
        )

        answer_ids = self.tokenizer.encode(
            answer,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_seq_len // 2,
            padding='max_length'
        )
        
        vocab_size = self.tokenizer.vocab_size
        for token_id in answer_ids:
            if token_id >= vocab_size:
                raise ValueError(f"非法 token id: {token_id} 超出 vocab size {vocab_size}")

        input_ids = prompt_ids + answer_ids

        # 构建 labels：prompt 部分为 -100，answer 部分为 token id
        # 这里要和llm的损失函数对应，ignore_index应是-100
        labels = [-100] * len(prompt_ids) + answer_ids

        # # 截断或 padding
        # if len(input_ids) > self.max_seq_len:
        #     input_ids = input_ids[:self.max_seq_len]
        # else:
        #     pad_len = self.max_seq_len - len(input_ids)
        #     input_ids += [self.tokenizer.pad_token_id] * pad_len
        
        # if len(labels) > self.max_seq_len:
        #     labels = labels[:self.max_seq_len]
        # else:
        #     pad_len = self.max_seq_len - len(labels)
        #     labels += [-100] * pad_len

        input_ids = torch.tensor(input_ids[:-1], dtype=torch.long)
        labels = torch.tensor(labels[1:], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "labels": labels,
        }
