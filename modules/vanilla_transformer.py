from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .att_model import pack_wrapper, AttModel


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def subsequent_mask(size):
    attn_shape = (1, size, size)  # lower triangular matrix，size = [1, size, size]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) #QK/sqrt(d_k)，size=(batch,h,L,L)
    
    #padding mask
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask, past=None):
        embeddings = self.tgt_embed(tgt)
        return self.decoder(embeddings, memory, src_mask, tgt_mask, past=past)

class Retrival_transformer(Transformer): # 定义一个模型，这个模型是Transformer的子类，主要是修改encoder的部分
    def __init__(self, encoder, decoder, src_embed, tgt_embed):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed

    def forward(self, src, tgt, src_mask, tgt_mask,  retrival_rpt , retrival_mask):
        return self.decode(self.encode(src, src_mask, retrival_rpt , retrival_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask ,retrival_rpt , retrival_mask):
        # print(retrival_rpt.shape)
        return self.encoder(self.src_embed(src), src_mask,self.tgt_embed(retrival_rpt), retrival_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask, past=None):
        embeddings = self.tgt_embed(tgt)
        return self.decoder(embeddings, memory, src_mask, tgt_mask, past=past)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        _x = sublayer(self.norm(x))
        if type(_x) is tuple:
            return x + self.dropout(_x[0]), _x[1]
        return x + self.dropout(_x)

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class CrossEncoderretrivalLayer(nn.Module):
    def __init__(self, size, self_attn, retrival_attn,feed_forward, dropout): # 这里图片的两个之间共用一个attention，但是retrival report的attention是单独一个
        super(CrossEncoderretrivalLayer, self).__init__()
        self.self_attn = self_attn
        self.retrival_attn = retrival_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)  # 这里是否需要改一下留存，相当于后面的所有部分都是一个laynorm，好像也可以，好像又不太对
        self.weight1 = torch.nn.Parameter(torch.rand(size).cuda())
        self.size = size

    def forward(self, x ,mask,report,report_mask): # x 应该为异常图片,report 为检索报告， mask 是图片mask ,report_mask 是检索报告的mask
        self_attention = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask)) # 自己的attention
        report_attention = self.sublayer[0](x, lambda x: self.retrival_attn(x, report, report, report_mask)) # report的attention, 注入report的信息
        x = self_attention + report_attention.mul(self.weight1) # 加report
        #x = self_attention
        return self.sublayer[1](x, self.feed_forward)

class CrossRetrivalEncoder(nn.Module):
    def __init__(self, layer, N):
        super(CrossRetrivalEncoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask , report ,repot_mask):
        for layer in self.layers:
            x = layer(x, mask , report, repot_mask)
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask, past=None):
        if past is not None:
            present = [[], []]
            x = x[:, -1:]
            tgt_mask = tgt_mask[:, -1:] if tgt_mask is not None else None
            past = list(zip(past[0].split(2, dim=0), past[1].split(2, dim=0)))
        else:
            past = [None] * len(self.layers)
        for i, (layer, layer_past) in enumerate(zip(self.layers, past)):
            x = layer(x, memory, src_mask, tgt_mask,layer_past)
            if layer_past is not None:
                present[0].append(x[1][0])
                present[1].append(x[1][1])
                x = x[0]
        if past[0] is None:
            return self.norm(x)
        else:
            return self.norm(x), [torch.cat(present[0], 0), torch.cat(present[1], 0)]

class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask, layer_past=None):
        m = memory             #memory: encoder feature
        if layer_past is None:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
            x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
            return self.sublayer[2](x, self.feed_forward)
        else:
            present = [None, None]
            x, present[0] = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask, layer_past[0]))
            x, present[1] = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask, layer_past[1]))
            return self.sublayer[2](x, self.feed_forward), present

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0 #检测word embedding维度是否能被h整除
        self.d_k = d_model // h
        self.h = h
        #四个线性变换，前三个为QKV三个变换矩阵，最后一个用于attention后
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, layer_past=None):
        if mask is not None:
            #增加一个维度给head
            mask = mask.unsqueeze(1) # 编码器mask的size = [batch,1,1,src_L],解码器mask的size= = [batch,1,tgt_L,tgt_L]
        nbatches = query.size(0)

        #运用layer_past为了提高速度的；此代码中都使用的None，暂时不看
        if layer_past is not None and layer_past.shape[2] == key.shape[1] > 1:
            query = self.linears[0](query)
            key, value = layer_past[0], layer_past[1]
            present = torch.stack([key, value])
        else:
            query, key, value = \
                [l(x) for l, x in zip(self.linears, (query, key, value))]

        if layer_past is not None and not (layer_past.shape[2] == key.shape[1] > 1):
            past_key, past_value = layer_past[0], layer_past[1]
            key = torch.cat((past_key, key), dim=1)
            value = torch.cat((past_value, value), dim=1)
            present = torch.stack([key, value])

        #1) 利用三个全连接算出QKV向量，再维度变换 [batch,L,d_model] ---> [batch, h, L, d_model//h]
        query, key, value = \
            [x.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for x in [query, key, value]]

        # 2) 实现Scaled Dot-Product Attention。x的size = (batch,h,L,d_model//h)，attn的size = (batch,h,L,L)
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)
        
        ## 3) 实现多头的拼接
        # transpose的结果 size = (batch , L , h , d_model//h)，view的结果size = (batch , L , d_model)
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        if layer_past is not None:
            return self.linears[-1](x), present
        else:
            return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # word_embedding + positional_embedding
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class BaseCMN(AttModel):

    def make_model(self, tgt_vocab):
        c = copy.deepcopy #深拷贝
        attn = MultiHeadedAttention(self.num_heads, self.d_model)
        #attn_e = ScaledDotProductAttentionMemory(self.num_heads, self.d_model, self.m)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        position = PositionalEncoding(self.d_model, self.dropout)
        model = Retrival_transformer(
            CrossRetrivalEncoder(CrossEncoderretrivalLayer(self.d_model, c(attn),c(attn), c(ff), self.dropout), self.num_layers),
            Decoder(DecoderLayer(self.d_model, c(attn), c(attn), c(ff), self.dropout), self.num_layers),
            nn.Sequential(c(position)),
            nn.Sequential(Embeddings(self.d_model, tgt_vocab), c(position))
            )

        # randomly initiate
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, args, tokenizer):
        super(BaseCMN, self).__init__(args, tokenizer)
        self.args = args
        self.num_layers = args.num_layers
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_heads = args.num_heads
        self.dropout = args.dropout
        tgt_vocab = self.vocab_size + 1
        self.m = 40
        self.model = self.make_model(tgt_vocab)
        self.logit = nn.Linear(args.d_model, tgt_vocab)

    def init_hidden(self, bsz):
        return []

    def _prepare_feature(self, fc_feats, att_feats, att_masks, retrival_ids=None):
        # print(retrival_ids.shape)
        att_feats, seq, att_masks, seq_mask, retrival_ids, retrival_seq_mask = self._prepare_feature_forward(att_feats, att_masks, retrival_ids=retrival_ids)
        memory = self.model.encode(att_feats, att_masks, retrival_ids, retrival_seq_mask)

        return fc_feats[..., :1], att_feats[..., :1], memory, att_masks

    def _prepare_feature_forward(self, att_feats, att_masks=None, seq=None, retrival_ids=None):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks) #[bs, 98, 2048]-->[bs, 98, 512]

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        
        att_masks = att_masks.unsqueeze(-2)
    
        if seq is not None:
            seq = seq[:, :-1]         # 用于输入模型，不带末尾的<eos>
            seq_mask = (seq.data > 0)
            seq_mask[:, 0] += True    #第一个的0的含义和padding的0的含义是不一样的，所以要加上去？maybe
            seq_mask = seq_mask.unsqueeze(-2) #返回一个true/false矩阵，size = [batch , 1 , tgt_L]
            # 两个mask求和得到最终mask,[batch, 1, L]&[1, size, size]=[batch,tgt_L,tgt_L]
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
        else:
            seq_mask = None

        if retrival_ids is not None:
            retrival_seq_mask = (retrival_ids.data > 0)
            retrival_seq_mask[:, 0] += True
            retrival_seq_mask = retrival_seq_mask.unsqueeze(-2)

        else:
            retrival_seq_mask = None
        return att_feats, seq, att_masks, seq_mask, retrival_ids, retrival_seq_mask

    def _forward(self, fc_feats, att_feats, seq, att_masks=None, retrival_ids=None, retrival_seq_mask=None):
        att_feats, seq, att_masks, seq_mask, retrival_ids, retrival_seq_mask = self._prepare_feature_forward(att_feats, att_masks, seq, retrival_ids)
        out = self.model(att_feats, seq, att_masks, seq_mask, retrival_ids, retrival_seq_mask)
        outputs = F.log_softmax(self.logit(out), dim=-1)
        return outputs

    def _save_attns(self, start=False):
        if start:
            self.attention_weights = []
        self.attention_weights.append([layer.src_attn.attn.cpu().numpy() for layer in self.model.decoder.layers])

    def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask):
        if len(state) == 0:
            ys = it.unsqueeze(1)
            past = [fc_feats_ph.new_zeros(self.num_layers * 2, fc_feats_ph.shape[0], 0, self.d_model),
                    fc_feats_ph.new_zeros(self.num_layers * 2, fc_feats_ph.shape[0], 0, self.d_model)]
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
            past = state[1:]
        out, past = self.model.decode(memory, mask, ys, subsequent_mask(ys.size(1)).to(memory.device), past=past)

        if not self.training:
            self._save_attns(start=len(state) == 0)
        return out[:, -1], [ys.unsqueeze(0)] + past
