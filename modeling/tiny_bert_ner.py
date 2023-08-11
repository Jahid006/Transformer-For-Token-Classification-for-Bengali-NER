import torch
import math
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda")


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, d_model, max_len=128):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        self.pe = pe.unsqueeze(0).to(device)

    def forward(self, x):
        return self.pe


class BERTEmbedding(torch.nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, seq_len=128, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """

        super().__init__()
        self.embed_size = embed_size
        self.token = torch.nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.position = PositionalEmbedding(d_model=embed_size, max_len=seq_len)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, sequence):
        x = self.token(sequence) + self.position(sequence)
        return self.dropout(x)


class MultiHeadedAttention(torch.nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()

        assert d_model % heads == 0
        self.d_k = d_model // heads
        self.heads = heads
        self.dropout = torch.nn.Dropout(dropout)

        self.query = torch.nn.Linear(d_model, d_model)
        self.key = torch.nn.Linear(d_model, d_model)
        self.value = torch.nn.Linear(d_model, d_model)
        self.output_linear = torch.nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask):
        """
        query, key, value of shape: (batch_size, max_len, d_model)
        mask of shape: (batch_size, 1, 1, max_words)
        """
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        query = query.view(query.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)
        key = key.view(key.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)
        value = value.view(value.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)

        scores = torch.matmul(query, key.permute(0, 1, 3, 2)) / math.sqrt(
            query.size(-1)
        )
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        context = torch.matmul(weights, value)

        context = (
            context.permute(0, 2, 1, 3)
            .contiguous()
            .view(context.shape[0], -1, self.heads * self.d_k)
        )

        return self.output_linear(context)


class FeedForward(torch.nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, middle_dim=2048, dropout=0.1):
        super(FeedForward, self).__init__()

        self.fc1 = torch.nn.Linear(d_model, middle_dim)
        self.fc2 = torch.nn.Linear(middle_dim, d_model)
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.GELU()

    def forward(self, x):
        out = self.activation(self.fc1(x))
        out = self.fc2(self.dropout(out))
        return out


class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model=768, heads=12, feed_forward_hidden=768 * 4, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.layernorm = torch.nn.LayerNorm(d_model)
        self.self_multihead = MultiHeadedAttention(heads, d_model)
        self.feed_forward = FeedForward(d_model, middle_dim=feed_forward_hidden)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, embeddings, mask):
        interacted = self.dropout(
            self.self_multihead(embeddings, embeddings, embeddings, mask)
        )
        interacted = self.layernorm(interacted + embeddings)
        feed_forward_out = self.dropout(self.feed_forward(interacted))
        encoded = self.layernorm(feed_forward_out + interacted)
        return encoded


class TinyBert(torch.nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, d_model=768, n_layers=12, heads=12, dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.heads = heads

        self.feed_forward_hidden = d_model * 4

        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=d_model)
        self.encoder_blocks = torch.nn.ModuleList(
            [
                EncoderLayer(d_model, heads, d_model * 4, dropout)
                for _ in range(n_layers)
            ]
        )

    def forward(self, x, *args, **kwargs):
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        x = self.embedding(x)
        for encoder in self.encoder_blocks:
            x = encoder.forward(x, mask)
        return x
    
    @staticmethod
    def from_pretrained(path):
        bert = TinyBert(
            vocab_size=4096,
            d_model=256,
            n_layers=4,
            heads=4,
        )
        checkpoint = torch.load(path)
        bert.load_state_dict(checkpoint["model"], strict=False)

        return bert


class MaskedLanguageModel(torch.nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = torch.nn.Linear(hidden, vocab_size)
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))
    
    


class TinyBertLM(torch.nn.Module):
    """
    BERT Language Model
    Next Sentence Prediction Model + Masked Language Model
    """

    def __init__(self, bert: TinyBert, vocab_size):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.bert = bert
        self.mask_lm = MaskedLanguageModel(self.bert.d_model, vocab_size)

    def forward(self, x, **kwargs):
        x = self.bert(x)
        return self.mask_lm(x)

    @staticmethod
    def from_pretrained(cfg):
        bert = TinyBert(
            vocab_size=4096,
            d_model=256,
            n_layers=4,
            heads=4,
        )
        bert_lm = TinyBertLM(bert, 4096)
        return bert_lm


class NERClassifier(nn.Module):
    def __init__(self, backbone, hidden_sizes=768, num_labels=2, dropout_rate=0.1):
        super(NERClassifier, self).__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(p=dropout_rate)
        self.logit = nn.Linear(hidden_sizes, num_labels)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs, attention_mask):
        features = self.backbone(inputs, attention_mask=attention_mask)
        features = features[-1] if isinstance(features, tuple) else features
        features = self.dropout(features)
        logits = self.logit(features)
        probs = self.softmax(logits)
        return logits, probs


def load_model(model, path):
    try:
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model"], strict=True)
    except Exception as e:
        model = load_model_unstrict(model, path)

    return model


def load_model_unstrict(model, path):
    current_model_dict = model.state_dict()
    loaded_state_dict = torch.load(path)

    new_state_dict = {
        k: v if v.size() == current_model_dict[k].size() else current_model_dict[k]
        for k, v in zip(current_model_dict.keys(), loaded_state_dict.values())
    }

    mis_matched_layers = [
        k
        for k, v in zip(current_model_dict.keys(), loaded_state_dict.values())
        if v.size() != current_model_dict[k].size()
    ]

    if mis_matched_layers:
        raise ValueError  # give appropriate value error

    model.load_state_dict(new_state_dict, strict=True)

    return model


def build_model(cfg):
    encoder = TinyBert.from_pretrained(cfg.pretrained_model_path)
    model = NERClassifier(encoder, cfg.ffn_size, cfg.n_class)
    return model  # load_model(model, cfg.pretrained_model_path)
