import torch
from torch import nn
from models.util import init_weights
import torch.nn.functional as F


class PredictionHead(nn.Module):
    def __init__(self, in_size, hidden_size, layer_norm_eps=1e-12):
        super().__init__()
        self.dense = nn.Linear(in_size, hidden_size)
        self.transform_act_fn = nn.functional.gelu
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class MLMHead(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, layer_norm_eps=1e-12, weight=None):
        super().__init__()
        self.transform = PredictionHead(in_size, hidden_size, layer_norm_eps)
        self.decoder = nn.Linear(hidden_size, out_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_size))
        if weight is not None:
            self.decoder.weight = weight

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x) + self.bias
        return x


class Hazard_Layer(nn.Module):
    def __init__(self, max_followup):
        super(Hazard_Layer, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        mask = torch.ones([max_followup, max_followup])
        mask = torch.tril(mask, diagonal=0)
        mask = torch.nn.Parameter(torch.t(mask), requires_grad=False)
        self.register_parameter("upper_triagular_mask", mask)

    def forward(self, x):
        B, D = x.size()  # hazards is (B, T)
        x1 = x[:, 0:-1]
        base_hazard = x[:, D - 1:D]
        T = D - 1

        expanded_hazards = x1.unsqueeze(-1).expand(
            B, T, T
        )  # expanded_hazards is (B,T, T)
        masked_hazards = (
                expanded_hazards * self.upper_triagular_mask
        )  # masked_hazards now (B,T, T)
        # base_hazard = self.base_hazard_fc(x)
        cum_prob = torch.sum(masked_hazards, dim=1) + base_hazard
        return cum_prob


class MTHead(nn.Module):
    def __init__(
            self,
            head_dim_dict,
            task_dim_dict,
            num_head_dict,
            task_head_map_dict,
            ta_embed_dim,
            loss_weight_dict,
            task_base_dict,
            task_enforce,
            max_followup=6,
    ):
        super().__init__()
        assert head_dim_dict is not None
        self.head_dim_dict = head_dim_dict
        assert task_dim_dict is not None
        self.task_dim_dict = task_dim_dict
        assert num_head_dict is not None
        self.num_head_dict = num_head_dict
        assert task_head_map_dict is not None
        self.task_head_map_dict = task_head_map_dict

        self.head_task_map_dict = self.get_head_task_map()

        self.head_task_dim_dict, self.head_out_dim_dict = self.get_head_task_param()

        for kh in self.head_dim_dict.keys():
            hidden_size_kh = self.head_dim_dict[kh]
            out_size_kh = self.head_out_dim_dict[kh]
            head_k = MLMHead(ta_embed_dim, hidden_size_kh, out_size_kh)
            head_k.apply(init_weights)
            self.__setattr__(kh, head_k)

        self.hazard_layer = Hazard_Layer(max_followup)

        assert loss_weight_dict is not None
        self.loss_weight_dict = loss_weight_dict
        assert task_base_dict is not None
        self.task_base_dict = task_base_dict
        self.task_enforce = task_enforce

    def get_head_task_map(self):
        head_task_map_dict = {}
        for kt, kh in self.task_head_map_dict.items():
            if kh not in head_task_map_dict.keys():
                head_task_map_dict[kh] = [kt]
            else:
                head_task_map_dict[kh].append(kt)
        return head_task_map_dict

    def get_head_task_param(self):
        head_task_dim_dict = {}
        head_out_dim_dict = {}

        for kh, ktl in self.head_task_map_dict.items():
            dim_current = 0
            for k in ktl:
                num_head_k = self.num_head_dict[k]
                dim_k = self.task_dim_dict[k]
                head_task_dim_dict[k] = [dim_current, dim_current + dim_k * num_head_k]
                dim_current = dim_current + dim_k * num_head_k
            head_out_dim_dict[kh] = dim_current

        return head_task_dim_dict, head_out_dim_dict

    def forward(self, embeds, data_dict=None):
        logits = self.pred_logits(embeds)
        if self.training:
            return self.loss(logits, data_dict)
        else:
            return self.inference(logits)

    def pred_logits(self, x):
        out_logits_dict = {}

        for kh, ktl in self.head_task_map_dict.items():
            head_kh = self.__getattr__(kh)
            out_kh = head_kh(x)

            for kt in ktl:
                num_head = self.num_head_dict[kt]
                dim_s, dim_e = self.head_task_dim_dict[kt]
                out_kt = out_kh[:, dim_s:dim_e]
                B = out_kt.shape[0]
                out_kt = out_kt.reshape([B, num_head, -1])
                if kt == 'cancer_risk':
                    out_kt = out_kt.reshape([B * num_head, -1])
                    out_kt = self.hazard_layer(out_kt)
                    out_kt = out_kt.reshape([B, num_head, -1])

                out_logits_dict[kt] = out_kt

        return out_logits_dict

    def get_loss_weight(self, k, loss, loss_weight):
        if self.task_base_dict[k] > 0:
            loss_base = self.task_base_dict[k]
            loss_weight = (loss / (loss_base + 1e-12)) ** self.task_enforce
        return loss_weight

    def loss(self, logits, data_dict):
        answer_dict = data_dict['answers_dict']

        loss_dict = {}
        task_weight_dict = {}
        loss_all = 0.0
        keys_task = answer_dict.keys()
        num_loss = 0
        for k in keys_task:
            if torch.all(answer_dict[k] < 0):
                targets = torch.zeros_like(answer_dict[k])
                loss_weight = 0.0
            else:
                targets = answer_dict[k]
                loss_weight = 1.0
                num_loss = num_loss + 1

            task_logit = logits[k]
            num_head = task_logit.shape[1]

            targets = targets.unsqueeze(dim=1).repeat(1, num_head, 1)

            task_logit = task_logit.reshape([task_logit.shape[0] * num_head, task_logit.shape[2]])

            if k == 'cancer_risk':
                targets = targets.reshape([targets.shape[0] * num_head, targets.shape[2]])
                label_mask = data_dict['label_mask']
                label_mask = label_mask.unsqueeze(dim=1).repeat(1, num_head, 1)
                label_mask = label_mask.reshape([task_logit.shape[0], targets.shape[-1]])
                label_mask_norm = max(1e-8, torch.sum(label_mask))
                loss_k = F.binary_cross_entropy_with_logits(task_logit, targets, weight=label_mask,
                                                            reduction='sum') / label_mask_norm
                loss_weight = self.get_loss_weight(k, loss_k.item(), loss_weight)
                loss_all = loss_all + loss_k * loss_weight

            elif k == 'clinical_number':
                task_logit = task_logit.reshape(task_logit.shape[0], 10, -1)
                targets = targets.reshape([targets.shape[0] * num_head, targets.shape[2]])
                loss_k = F.cross_entropy(
                    task_logit,
                    targets,
                    ignore_index=-100,
                )
                loss_weight = self.get_loss_weight(k, loss_k.item(), loss_weight)
                loss_all = loss_all + loss_k * loss_weight

            else:
                targets = targets.reshape([targets.shape[0] * num_head, ])

                loss_k = F.cross_entropy(
                    task_logit,
                    targets,
                    weight=torch.FloatTensor(self.loss_weight_dict[k]).to(task_logit),
                    ignore_index=-100,
                )
                loss_weight = self.get_loss_weight(k, loss_k.item(), loss_weight)
                loss_all = loss_all + loss_k * loss_weight

            loss_dict[k] = loss_k.unsqueeze(0)
            task_weight_dict[k] = loss_weight

        loss_all = loss_all / num_loss

        return loss_all, loss_dict, task_weight_dict

    def inference(self, logits):
        keys_task = logits.keys()
        output_dict = {}
        for k in keys_task:
            task_logit = logits[k]
            if k == 'cancer_risk':
                output_dict[k] = F.sigmoid(task_logit)
            elif k == 'clinical_number':
                task_logit = task_logit.reshape(task_logit.shape[0], task_logit.shape[1], 10, -1)
                task_logit = F.softmax(task_logit, dim=2)
                task_out = task_logit.argmax(dim=2)
                task_out = task_out[:, :, 0:1] * 100 + task_out[:, :, 1:2] * 10 + task_out[:, :, 2:3]
                output_dict[k] = task_out.contiguous()
            else:
                output_dict[k] = F.softmax(task_logit, dim=2)
        return output_dict
