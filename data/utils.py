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
    


class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_len):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f if line.strip()]

        # 确保 tokenizer 有 pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.pad_token_id = self.tokenizer.pad_token_id
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index:int):
        # 获取第index个json字符串
        line = self.data[index]
        instruction_text = line['instruction']
        input_text = line['input']
        output_text = line['output']
        history = line['history']
        query = instruction_text + input_text
        answer = output_text + self.tokenizer.eos_token
        messages = []
        if history:
            for i in history:
                messages.append({'role': 'user', 'content': i[0]})
                messages.append({'role': 'assistant', 'content': i[1]})

        messages.append({'role': 'user', 'content': query})

        # 使用 tokenizer 的 chat 模板格式化 prompt
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)

        prompt_input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        answer_input_ids = self.tokenizer.encode(answer, add_special_tokens=False)

        input_ids = prompt_input_ids + answer_input_ids
        # 在 HuggingFace 和 PyTorch 中, CrossEntropyLoss 会忽略 label == -100 的位置。
        labels = [-100] * len(prompt_input_ids) + answer_input_ids


        text_len = len(input_ids)
        if text_len > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
            labels = labels[:self.max_seq_len]
        else:
            input_ids = input_ids + [self.pad_token_id] * (self.max_seq_len - text_len)
            labels = labels + [-100] * (self.max_seq_len - text_len)
        
        # input_ids = np.array(input_ids)
        # seq = np.array(input_ids[:-1]).astype(np.int64)
        # labels = np.array(input[1:]).astype(np.int64)
        input_tensor = torch.tensor(input_ids, dtype=torch.long)
        label_tensor = torch.tensor(labels, dtype=torch.long)
        return {
            'input_ids': input_tensor,
            'labels': label_tensor,
        }