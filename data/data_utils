import yaml
from collections import Counter
import jieba
def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# 分词
def tokenize(text):
    return list(jieba.cut(text))

# 过滤文档内重复字太多的文档
def has_too_many_repeated_tokens(text, max_token_repeat):
    tokens = tokenize(text)
    token_counts = Counter(tokens)
    for count in token_counts.values():
        if count > max_token_repeat:
            return True
    return False

# 过滤文档内重复行太多的文档
def repeated_line_ratio(text):
    lines = text.splitlines()
    total_lines = len(lines)
    if total_lines == 0:
        return 0
    line_counts = Counter(lines)
    repeated_lines = sum(count for line, count in line_counts.items() if count > 1)
    return repeated_lines / total_lines


def should_filter(text, config):
    if has_too_many_repeated_tokens(text, config["max_token_repeat"]):
        return True
    if repeated_line_ratio(text) > config["max_line_repeat_ratio"]:
        return True
    return False

def filter_documents(docs, config):
    return [doc for doc in docs if not should_filter(doc, config)]

# 示例运行
if __name__ == "__main__":
    config_path = 'D:\code\MyLLM\data\configs\config.yaml'
    config = load_config(config_path)
    docs = [
        "word " * 101,
        "hello world\nhello world\nhello world\nhello world",
        "this is a clean document\nwith various lines\nand no issues",
        "哈哈哈" * 101,
        "笑死" * 101,
        "笑死\n" * 101
    ]
    filtered = filter_documents(docs, config)
    print("保留的文档：", filtered)
