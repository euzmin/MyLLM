import torch
from torch.utils.data import Dataset
import json
import numpy as np

class LLMDataset(Dataset):
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
            padding='max_length'  # 可选：也可以用 pad 到 max_seq_len
        )
        # input_ids = self.tokenizer.encode(text, add_special_tokens=False)
        input_ids = tokens['input_ids']
        text_len = len(input_ids)

        if text_len > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
        else:
            input_ids = input_ids + [0] * (self.max_seq_len - text_len)
        
        # input_ids = np.array(input_ids)
        # seq = np.array(input_ids[:-1]).astype(np.int64)
        # labels = np.array(input[1:]).astype(np.int64)
        input_tensor = torch.tensor(input_ids[:-1], dtype=torch.long)
        label_tensor = torch.tensor(input_ids[1:], dtype=torch.long)
        return {
            'input_ids': input_tensor,
            'labels': label_tensor,
        }