from tokenizers import decoders, models, pre_tokenizers, trainers, Tokenizer
from transformers import AutoTokenizer
import os
import json

def read_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            yield data['text']

# 使用BPE分词器
tokenizer = Tokenizer(models.BPE())
# 指定预分词方式为 ByteLevel，字符级处理,不自动在输入前加空格
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

special_tokens = ['<pad>', '<unk>', '<s>', '</s>']

trainer = trainers.BpeTrainer(
    # 训练出来的词表大小是 6400 个 token
    vocab_size=6400,
    special_tokens=special_tokens,
    # 显示训练进度条
    show_progress=True,
    # 指定了初始字符集（alphabet）是 ByteLevel 的全字符集合。
    # 它能确保所有字节都能被编码成 token，尤其对非英语字符和罕见字符很有用。
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
)

texts = read_data('./data')

# 使用 texts 中的文本，通过 trainer 来训练这个分词器（BPE）
# 这一步完成后：
# 分词器 tokenizer 会根据 texts 学习 BPE 规则；
# 构建出一个词表（vocabulary）和 merge 表（合并规则）；
# 分词器就可以开始进行 .encode() 和 .decode() 了。
tokenizer.train_from_iterator(texts, trainer)

# 前面用的ByteLevel切词，这里就用ByteLevel把token id转化为字符
tokenizer.decoder = decoders.ByteLevel()

tokenizer_dir = './tokenizer'
os.makedirs(tokenizer_dir, exist_ok=True)
tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))

tokenizer.model.save(tokenizer_dir)

config = {
    "add_bos_token":False,
    "add_eos_token":False,
    "add_prefix_space":True,
    "added_tokens_decoder":{
        # token id 是0
        "0": {
            "content": "<unk>",
            # 分词或解码时不自动去除左右空格
            "lstrip": False,
            "rstrip": False,
            # 不对 token 应用 unicode 规范化
            "normalized": False,
            # 不强制要求 token 只匹配完整单词
            "single_word": False,
            # 它是一个特殊 token（不会被普通文本直接匹配）
            "special": True
        },
        "1": {
            "content": "<s>",
            # 分词或解码时不自动去除左右空格
            "lstrip": False,
            "rstrip": False,
            # 不对 token 应用 unicode 规范化
            "normalized": False,
            # 不强制要求 token 只匹配完整单词
            "single_word": False,
            # 它是一个特殊 token（不会被普通文本直接匹配）
            "special": True
        },
        "2": {
            "content": "</s>",
            # 分词或解码时不自动去除左右空格
            "lstrip": False,
            "rstrip": False,
            # 不对 token 应用 unicode 规范化
            "normalized": False,
            # 不强制要求 token 只匹配完整单词
            "single_word": False,
            # 它是一个特殊 token（不会被普通文本直接匹配）
            "special": True
        },
    },
    # 可以继续添加额外的特殊token
    "additional_special_tokens": [],
    "bos_token": "<s>",
    # 是否在解码（decode）时清除多余的空格等标点异常。
    "clean_up_tokenization_spaces": False,
    "eos_token": "</s>",
    # 是否使用旧版 huggingface tokenizer 行为或格式
    "legacy": True,
    "model_max_length": 100000,
    # 模型不需要 pad（如 GPT 系列一般用 causal mask）
    "pad_token": None,
    # 传给 SentencePieceProcessor（如果使用 SentencePiece tokenizer）的附加参数。
    "sp_model_kwargs": {},
    # 在 decode 时，特殊 token 之间是否插入空格
    "spaces_between_special_tokens": False,
    # 使用  Hugging Face 的 Rust 实现的高速 tokenizer
    "tokenizer_class": "PreTrainedTokenizerFast",
    "unk_token": "<unk>",
    # 是否在推理时默认使用系统 prompt（主要是为聊天模型/指令模型设计的）。
    "use_default_system_prompt": False,
    # 自定义一个聊天prompt模板
    "chat_template": "{% if messages[0]['role'] == 'system' %} {% set system_message = messages[0]['content'] %} {% endif %} {% if message is defined %} {{ system_message }} {% endif %} {% for message in messages %} {% set content = message['content'] %} {% if message['role'] == 'user'%} {{'<s>user\\n' + content +'</s>\\n<s>assistant\\n'}} {% elif message['role'] == 'assistant' %} {{content + '</s>' + '\\n'}} {% endif %} {% endfor %}"
}

with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), 'w', encoding="utf-8") as config_file:
    # ensure_ascii=False	表示：保留 Unicode 字符（例如中文、emoji 等）。否则会转义为 \uXXXX
    # indent=4	设置输出的 JSON 缩进层级，使结果更易读。这里用 4 空格缩进
    json.dump(config, config_file, ensure_ascii=False, indent=4)

# 测试
tokenizer = AutoTokenizer.from_pretrained('./tokenizer')
tokenizer.encode("您好")
