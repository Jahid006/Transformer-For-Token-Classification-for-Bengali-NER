import torch
from torch import nn
from transformers import AutoModel
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions


class NERClassifier(nn.Module):
    def __init__(self, backbone, hidden_sizes=768, num_labels=2, dropout_rate=0.1):
        super(NERClassifier, self).__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(p=dropout_rate)
        self.logit = nn.Linear(hidden_sizes, num_labels)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs, attention_mask):
        features = self.backbone(inputs, attention_mask=attention_mask)
        features = (
            features[-1]
            if isinstance(features, BaseModelOutputWithPastAndCrossAttentions)
            else features
        )
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
    encoder = AutoModel.from_pretrained(cfg.encoder_model)
    model = NERClassifier(encoder, cfg.ffn_size, cfg.n_class)
    return model  # load_model(model, cfg.pretrained_model_path)
