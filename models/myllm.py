import sys
import os
sys.path.append(os.path.abspath('.'))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import math
<<<<<<< HEAD
from transformers import PreTrainedModel, PretrainedConfig, DefaultDataCollator
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer
=======
from transformers import PreTrainedModel, PretrainedConfig
>>>>>>> 2da8251 (拆分了myllm的代码，把pretrain提出来了)
import torch.nn as nn
import torch
import torch.nn.functional as F
# huggingface 的输出格式
from transformers.modeling_outputs import CausalLMOutputWithPast
<<<<<<< HEAD
from data.utils import LLMDataset
=======

>>>>>>> 2da8251 (拆分了myllm的代码，把pretrain提出来了)

# 似乎 RMSNorm 和 RoPE 的优化都是把加减操作改为了单纯的乘法和除法，那残差操作是不是也可以这样改呢？意义不大，残差操作已经很快了

# 对 layer norm的优化，去掉了减去均值的步骤，能计算更快（求均值需要跨多个元素求和，慢）
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, rep):
        rep = rep.float()
        variance = rep.pow(2).mean(-1, keepdim=True)
        rep = rep * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * rep.float()


# rope，原理上，是把sinusoidal位置编码从相加式，改为相乘式
# 因为求 attention 毕竟是 q * k 的内积操作，相加式就会面临多项式的展开，相对位置信息分散在多项上的问题；
# 直接改为相乘式，则显得更加直接。
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=1024, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (10000 ** (torch.arange(0,dim,2).float() / dim))
        t = torch.arange(max_seq_len).float().unsqueeze(1)

        # 这个 freqs 和 Sinusoidal 位置编码相同
        freqs = t @ inv_freq.unsqueeze(0)
        freqs = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cached", freqs.cos())
        self.register_buffer("sin_cached", freqs.sin())

    def forward(self, q, k):
        cos = self.cos_cached[:q.shape[1], :].unsqueeze(0)
        sin = self.sin_cached[:q.shape[1], :].unsqueeze(0)

        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)

        q1, q2 = q.chunk(2, dim=-1)
        rotate_half_q = torch.cat((-q2, q1), dim=-1)

        k1, k2 = k.chunk(2, dim=-1)
        rotate_half_k = torch.cat((-k2, k1), dim=-1)

        # 模拟复数乘法， (xcos\theta - ysin\theta)+i(xsin\theta+ycos\theta)
        q_emb = (q*cos) + (rotate_half_q*sin)
        k_emb = (k*cos) + (rotate_half_k*sin)

        return q_emb, k_emb

def repeat_kv(rep, n_group):
    bs, seq_len, num_key_value_heads, head_dim = rep.shape
    
    if n_group == 1:
        return rep
    
    rep = rep[:,:,:,None,:].expand(bs, seq_len, num_key_value_heads, n_group, head_dim)
    return rep.reshape(bs, seq_len, num_key_value_heads * n_group, head_dim)


class Attention(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.dropout = config.dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size//self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.k_cache, self.v_cache = None, None
        self.is_decoder_only = True
        self.flash_attn = self.config.flash_attn


        # Group Query Attention (Key-Value head sharing), q有很多， k v 相对少一些，这种设计可以加速推理、降低显存，同时保持足够的表达力。
        # 可以对比 multi-head attention 和 multi-query attention 来看
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        # attention 计算后的线性层
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.residual_dropout = nn.Dropout(self.dropout)
        self.rope = RotaryEmbedding(self.head_dim)


    def forward(self, rep, use_kv_cache=False):
        bs, seq_len = rep.shape[:2]
        if use_kv_cache and not self.training: # 加速推理，训练时每个token都已知，不需要这一步
            if self.k_cache is None or self.k_cache.shape[0] != seq_len-1:
                q, k, v = self.q_proj(rep), self.k_proj(rep), self.v_proj(rep)
            else:
                token = rep[:, -1:, :] # q,k,v 只更新每个样本的最后一个token（前面的之前都算过了，也存下来了，self.k_cahce.shape[0] == seq_len-1）
                q = torch.cat((torch.zeros_like(rep[:, :-1, :]), self.q_proj(token)), dim=1)
                k = torch.cat((self.k_cache, self.k_proj(rep)), dim=1)
                v = torch.cat((self.v_cache, self.v_proj(rep)), dim=1)
            self.k_cache, self.v_cache = k, v

        else:
            q, k, v = self.q_proj(rep), self.k_proj(rep), self.v_proj(rep)

        # 这是为group query 做准备
        q = q.view(bs, seq_len, self.num_heads, self.head_dim)
        k = k.view(bs, seq_len, self.num_key_value_heads, self.head_dim)
        v = v.view(bs, seq_len, self.num_key_value_heads, self.head_dim)
        
        # 加入位置信息
        q, k = self.rope(q, k)

        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        #bs, self.num_heads, seq_len, self.head_dim
        # 这一步是为了做group level attention，每一个head维度求attention
        q = q.transpose(1,2) 
        k = k.transpose(1,2) 
        v = v.transpose(1,2) 

        # 判断是否启用 PyTorch 2.0+ 提供的高性能注意力实现
        if self.flash_attn:
            output = F.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                                    dropout_p=self.dropout if self.training else 0.0,
                                                    is_causal=self.is_decoder_only)
        
        # 使用我们自己实现的attention
        else:
            # 生成全是-inf值的上三角掩码，softmax后对应的attention score 为0
            # 该掩码用于确保每个sequence只看能看到当前和之前的token，用于自回归
            decode_mask = torch.full((1, 1, self.config.max_seq_len, self.config.max_seq_len), float("-inf")) 
            decode_mask = torch.triu(decode_mask, diagonal=1)
            # qk/sqrt(dk) 
            # q: bs, self.num_heads, seq_len, self.head_dim 
            # k: bs, self.num_heads, self.head_dim, seq_len
            # scores: bs, self.num_heads, seq_len, seq_len
            attn_scores = (q @ k.transpose(2,3)) / math.sqrt(self.head_dim)
            decode_mask = decode_mask.to(attn_scores.device)
            attn_scores = attn_scores + decode_mask[:,:,:seq_len,:seq_len]
            attn_scores = F.softmax(attn_scores, dim=-1)
            output = attn_scores @ v
        
        # group query attention 计算完成后，重新改为 bs, seq_len, self.hidden_size  的维度
        output = output.transpose(1,2).contiguous().view(bs, seq_len, -1)
        output = self.o_proj(output)
        # 在加入残差连接之前做 dropout，提升泛化能力、减少过拟合。
        output = self.residual_dropout(output)
        return output
    

# Transformer 结构中的前馈神经网络层 (FFN)， 这里的实现方式是LLaMa 中的 Gated Linear Unit (GLU)
# 看起来这个GLU层和attention层区别也不是特别大了
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        # FFN 中间层的维度
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)

    def forward(self, x):
        # F.silu: Swish 激活函数，x*sigmoid(x)
        down_proj = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
        

class DecoderLayer(nn.Module):
    # 这个layer idx 有什么讲究吗？这里暂时没用到
    def __init__(self, config, layer_idx, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_size = config.hidden_size
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size)
        self.post_attn_layernorm = RMSNorm(config.hidden_size)
        self.layer_idx = layer_idx
    
    def forward(self, rep, use_kv_cache):

        # self attention
        residual = rep
        rep = self.input_layernorm(rep)
        rep = self.self_attn(rep=rep, use_kv_cache=use_kv_cache)
        rep = residual + rep

        # fully connected
        residual = rep
        rep = self.post_attn_layernorm(rep)
        rep = self.mlp(rep)
        outputs = residual + rep

        return outputs


<<<<<<< HEAD
class MyLLM(PreTrainedModel):
=======
# 编写自定义配置时需要注意：
# 1. 必须继承自 PretrainedConfig
# 2. __init__ 里最后一项为**kwargs，接受任何kwargs
# 3. kwargs 传给超参
class Config(PretrainedConfig):
    model_type = 'small_model'

    def __init__(self, hidden_size=512, num_attention_heads=16, num_key_value_heads=8, flash_attn=False,
                 attention_bias=False, max_seq_len=512, intermediate_size=2048, mlp_bias=False,
                 vocab_size=50257, n_layers=8, dropout=0.0, **kwargs):
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

class MyLLM(PreTrainedModel):
    # 指明该大模型用的是什么配置格式
    config_class = Config
>>>>>>> 2da8251 (拆分了myllm的代码，把pretrain提出来了)
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.config = config
        self.vocab_size = self.config.vocab_size
        self.n_layers = self.config.n_layers
        # 将离散的词表转化为词向量
        self.token_embeddings = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.dropout = nn.Dropout(self.config.dropout)
        self.layers = nn.ModuleList()
        for layer_idx in range(self.n_layers):
            self.layers.append(DecoderLayer(self.config, layer_idx))
        self.norm = RMSNorm(self.config.hidden_size)
        self.output = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        # output 层 和 token_embeddings 层共享权重，因为可以都用一个词汇表，没必要浪费额外的资源
        # 但据说现在这样做落伍了 https://www.spaces.ac.cn/archives/9698
        # 为啥能共享权重？
        # token_embeddings.weight 的权重是 (vocab_size, hidden_size) 
        # 但是 output.weight 也是(vocab_size, hidden_size)
        # 这是因为embedding weight 和 linear weight 的实现不同
        # self.token_embeddings.weight = self.output.weight
        self.output.weight = self.token_embeddings.weight
        # 这是啥
        self.apply(self._init_weights)
        self.loss = None

        for param_name, param in self.named_parameters():
            # 这里的 w3 和 wo 是什么？wo 好像是 attention 模块的 output 层，w3好像是FFN模块中的down_proj线性层
            # 这两个层直接处理模型的主要输出路径（注意力输出和 FFN 输出），其数值稳定性对深层 Transformer 至关重要
            if param_name.endswith('w3.weight') or param_name.endswith('wo.weight'):
                # std 层数越多，标准差越小，可以防止梯度爆炸或过大激活。
                # 这里和后面的 0.02 是研究者经验得来的指标
                torch.nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2*self.config.n_layers))
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_input_embeddings(self):
        return self.token_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.token_embeddings = new_embeddings


    def forward(self, input_ids, labels, use_kv_cache=False):
        print(input_ids.max())
        rep = self.token_embeddings(input_ids)
        rep = self.dropout(rep)

        for idx, layer in enumerate(self.layers):
            rep = layer(rep, use_kv_cache=use_kv_cache)
        
        rep = self.norm(rep)

        if labels is not None:
            logits = self.output(rep)
            # PyTorch 的要求：F.cross_entropy 接受 [N=batch_size * seq_len, C] 的输入
            # （N 个样本，每个样本 C 类），labels 对应的就是[batch_size * seq_len]
            self.loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=0)
        else:
            logits = self.output(rep[:, [-1], :])
            self.loss = None
        
        return CausalLMOutputWithPast(self.loss, logits)
    
    @torch.inference_mode
    def generate(self, inputs, eos, max_new_tokens, temperature=0.7, top_k=None, stream=True,
                reptition_penalty=1., use_kv_cache=True):
        input_ids = inputs['input_ids']
        # labels = inputs['labels']
        seq_len = input_ids.shape[1]

        # -1 是为了保留尾部控制位 eos
        while input_ids.shape[1] < max_new_tokens - 1:
            # 这里删去了labels
            inference_res = self(input_ids, use_kv_cache=use_kv_cache)
            # bs, seq_len, vocab_size
            logits = inference_res.logits
            # 获取最后一个token的预测结果
            logits = logits[:, -1, :]

            # 只取 input_ids[0] 是因为generate函数 默认一次只生成一个样本（batch size = 1）
            # 让之前出现过的所有token对应的logits都除以reptition_penalty，对重复生成行为进行惩罚
            for token in set(input_ids.tolist()[0]):
                logits[:, token] /= reptition_penalty

            if temperature == 0.0:
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                logits = logits / temperature
                if top_k is not None:
                    # 从 vocab 中 提取概率最大的前k个token，其他值都赋为-inf
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('inf')
                
                probs = F.softmax(logits, dim=-1)
                # 按概率采样token
                idx_next = torch.multinomial(probs, num_samples=1, generator=None)

            if idx_next == eos:
                break

            input_ids = torch.cat((input_ids, idx_next), dim=1)
            if stream:
                yield input_ids[:, seq_len:]

        if not stream:
            yield input_ids[:, seq_len:]

<<<<<<< HEAD
# 编写自定义配置时需要注意：
# 1. 必须继承自 PretrainedConfig
# 2. __init__ 里最后一项为**kwargs，接受任何kwargs
# 3. kwargs 传给超参
class Config(PretrainedConfig):
    model_type = 'small_model'

    def __init__(self, hidden_size=512, num_attention_heads=16, num_key_value_heads=8, flash_attn=False,
                 attention_bias=False, max_seq_len=512, intermediate_size=2048, mlp_bias=False,
                 vocab_size=50257, n_layers=8, dropout=0.0, **kwargs):
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
