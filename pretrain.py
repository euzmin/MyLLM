from transformers import PretrainedConfig

# # 编写自定义配置时需要记住的2个重要事项如下：
# 1. 必须继承自 PretrainedConfig
# 2. 自定义的配置中 super().__init__() 必须能接受任意数量的参数，即**kwargs
class Config(PretrainedConfig):
    def __init__(self, 
                 hidden_size = 512,
                 num_attention_heads = 16,
                 num_key_value_heads = 8, #?
                 flash_attn = True, #?
                 attention_bias = False, #?
                 max_seq_len = 512, #?
                 intermediate_size = 2048, #?
                 mlp_bias = False,
                 vocab_size = 6400, #?
                 n_layers = 8,
                 dropout = 0.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.flash_attn = flash_attn
        self.attention_bias = attention_bias
        self.max_seq_len = max_seq_len
        self.intermediate_size = intermediate_size
        self.mlp_bias = mlp_bias
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.dropout = dropout
        super().__init__(**kwargs)

if __name__ == '__main__':
    config = Config()


