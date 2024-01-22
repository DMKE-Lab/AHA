import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import torch.fft


class MovingAvg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class SeriesDecompose(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(SeriesDecompose, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class TimeEncoding(nn.Module):
    def __init__(self, dim_t):
        super(TimeEncoding, self).__init__()
        self.w = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dim_t))).float(), requires_grad=True)

    def forward(self, dt):
        batch_size = dt.size(0)
        seq_len = dt.size(1)
        dt = dt.view(batch_size, seq_len, 1)
        t_cos = torch.cos(self.w.view(1, 1, -1) * dt)
        t_sin = torch.sin(self.w.view(1, 1, -1) * dt)
        return t_cos, t_sin




class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, moving_avg, dropout=0.1):
        super().__init__()
        self.decompose1 = SeriesDecompose(moving_avg)
        self.decompose2 = SeriesDecompose(moving_avg)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y):
        x = x + self.dropout(y)
        x, _ = self.decompose1(x)
        z = x
        z = self.dropout(F.gelu(self.layer_norm(z)))
        z = self.dropout(self.layer_norm(z))
        res, _ = self.decompose2(z + x)
        return res

class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """
    def __init__(self, temperature, attn_dropout=0.2, factor=1, output_attention=False):
        super().__init__()
        self.output_attention = output_attention
        self.factor = factor
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def time_delay_agg_training(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the training phase.
        """
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[i]), -1)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg
    def time_delay_agg_inference(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the inference phase.
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0)\
            .repeat(batch, head, channel, 1).to(values.device)
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights, delay = torch.topk(mean_value, top_k, dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg

    def time_delay_agg_full(self, values, corr):
        """
        Standard version of Autocorrelation
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0)\
            .repeat(batch, head, channel, 1).to(values.device)
        # find top k
        top_k = int(self.factor * math.log(length))
        weights, delay = torch.topk(corr, top_k, dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[..., i].unsqueeze(-1)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * (tmp_corr[..., i].unsqueeze(-1))
        return delays_agg

    def forward(self, queries, keys, values, mask=None, bias=None):
        B, H, L, E = queries.shape
        _, _, S, D = values.shape

        if L > S:
            zeros = torch.zeros_like(queries[:, :, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=2)
            keys = torch.cat([keys, zeros], dim=2)
        else:
            values = values[:, :, :L, :]
            keys = keys[:, :, :L, :]
        q_fft = torch.fft.fft(queries.contiguous(), dim=-1)
        k_fft = torch.fft.fft(keys.contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, n=E)
        # print(res.shape, corr.shape)

        V = self.time_delay_agg_full(values.contiguous(), corr)

        return V.contiguous(), corr.permute(0, 3, 1, 2)

class TempMultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """
    def __init__(self, n_head, d_model, d_k, d_v, dim_t, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_model
        self.d_v = d_model
        self.d_t = d_model

        self.time_encoding = TimeEncoding(d_model)
        self.q2tw = nn.Linear(d_model, n_head * d_model, bias=False)
        nn.init.xavier_uniform_(self.q2tw.weight)

        self.w_qs = nn.Linear(d_model, n_head * d_model, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_model, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_model, bias=False)
        nn.init.xavier_uniform_(self.w_qs.weight)
        nn.init.xavier_uniform_(self.w_ks.weight)
        nn.init.xavier_uniform_(self.w_vs.weight)

        self.fc = nn.Linear(d_model * n_head, d_model)
        nn.init.xavier_uniform_(self.fc.weight)

        self.attention = ScaledDotProductAttention(temperature=(d_model + d_model) ** 0.5, attn_dropout=0.0)

        self.rpe = nn.Embedding(512, 1)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, q_time, k_time, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        qtw = self.q2tw(q).view(sz_b, len_q, n_head, self.d_t).transpose(1, 2)

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k).transpose(1, 2)  # [batch_size, n_head, seq_len, d_k]
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k).transpose(1, 2)  # [batch_size, n_head, seq_len, d_k]
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v).transpose(1, 2)  # [batch_size, n_head, seq_len, d_v]

        q_time_cos, q_time_sin = self.time_encoding(q_time)   # [batch_size, seq_len, d_t]
        k_time_cos, k_time_sin = self.time_encoding(k_time)

        q_time_cos, q_time_sin = qtw * q_time_cos.unsqueeze(1).repeat(1, n_head, 1, 1), qtw * q_time_sin.unsqueeze(1).repeat(1, n_head, 1, 1)
        k_time_cos, k_time_sin = k_time_cos.unsqueeze(1).repeat(1, n_head, 1, 1), k_time_sin.unsqueeze(1).repeat(1, n_head, 1, 1)

        q = q + q_time_cos + q_time_sin
        k = k + k_time_cos + k_time_sin


        # rpe
        # qp = torch.arange(q.size(2), device=q.device) + k.size(2) # [q_l]
        # kp = torch.arange(k.size(2), device=k.device)  #[k_l]
        # qp = qp.unsqueeze(1).repeat(1, kp.size(0)) - kp.unsqueeze(0)
        # qp_bias = self.rpe(qp).squeeze(-1).unsqueeze(0).unsqueeze(0)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.
        # print(q.shape, k.shape, v.shape)

        # output, attn = self.attention(q, k, v, mask=mask, bias=qp_bias)
        output, attn = self.attention(q, k, v, mask=mask)
        # print(output.shape)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        # print(output.shape, residual.shape)
        output = self.dropout(self.fc(output))
        output += residual

        output = self.layer_norm(output)
        return output, attn

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, moving_avg, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.attention_layer = TempMultiHeadAttention(n_head, d_model, d_model, d_model, d_model, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_inner, moving_avg, dropout)

    def forward(self, src, src_time, tgt, tgt_time, mask=None):
        output, att = self.attention_layer(tgt, src, src, tgt_time, src_time, mask)
        # print('ATT: ', att)
        output = self.ff(tgt, output)
        return output

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, d_inner, n_layers, n_head, moving_avg, dropout):
        super(TransformerEncoder, self).__init__()
        self.n_head = n_head
        self.layer_stack = nn.ModuleList([
            TransformerEncoderLayer(d_model, d_inner, n_head, moving_avg, dropout)
            for _ in range(n_layers)])
        self.position = nn.Embedding(512, d_model)

    def forward(self, src, src_time, tgt, tgt_time, mask=None):
        src_p = torch.arange(src.size(1), device=src.device)
        tgt_p = torch.arange(tgt.size(1), device=src.device) + src.size(1)
        #
        src = src + self.position(src_p).unsqueeze(0)
        tgt = tgt + self.position(tgt_p).unsqueeze(0)
        # print(src.shape, src_time.shape, tgt.shape, tgt_time.shape)
        for enc_layer in self.layer_stack:
            tgt = enc_layer(src, src_time, tgt, tgt_time, mask)
        return tgt