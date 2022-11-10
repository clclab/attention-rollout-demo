import argparse
import numpy as np
import torch
import glob

from captum._utils.common import _get_module_from_name

# compute rollout between attention layers
def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration- code adapted from https://github.com/samiraabnar/attention_flow
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    matrices_aug = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
                          for i in range(len(all_layer_matrices))]
    joint_attention = matrices_aug[start_layer]
    for i in range(start_layer+1, len(matrices_aug)):
        joint_attention = matrices_aug[i].bmm(joint_attention)
    return joint_attention

class Generator:
    def __init__(self, model, key="bert.encoder.layer"):
        self.model = model
        self.key = key
        self.model.eval()

    def tokens_from_ids(self, ids):
        return list(map(lambda s: s[1:] if s[0] == "Ġ" else s, self.tokenizer.convert_ids_to_tokens(ids)))

    def _calculate_gradients(self, output, index, do_relprop=True):
        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot_vector = (torch.nn.functional
                .one_hot(
                        # one_hot requires ints
                        torch.tensor(index, dtype=torch.int64),
                        num_classes=output.size(-1)
                    )
                # but requires_grad_ needs floats
                .to(torch.float)
            ).to(output.device)

        hot_output = torch.sum(one_hot_vector.clone().requires_grad_(True) * output)
        self.model.zero_grad()
        hot_output.backward(retain_graph=True)

        if do_relprop:
            return self.model.relprop(one_hot_vector, alpha=1)

    def generate_LRP(self, input_ids, attention_mask,
                     index=None, start_layer=11):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]

        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        self._calculate_gradients(output, index)

        cams = []
        blocks = _get_module_from_name(self.model, self.key)
        for blk in blocks:
            grad = blk.attention.self.get_attn_gradients()
            cam = blk.attention.self.get_attn_cam()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
            cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            cams.append(cam.unsqueeze(0))
        rollout = compute_rollout_attention(cams, start_layer=start_layer)
        rollout[:, 0, 0] = rollout[:, 0].min()
        return rollout[:, 0]

    def generate_LRP_last_layer(self, input_ids, attention_mask,
                     index=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        self._calculate_gradients(output, index)

        cam = _get_module_from_name(self.model, self.key)[-1].attention.self.get_attn_cam()[0]
        cam = cam.clamp(min=0).mean(dim=0).unsqueeze(0)
        cam[:, 0, 0] = 0
        return cam[:, 0]

    def generate_full_lrp(self, input_ids, attention_mask,
                     index=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]

        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        cam = self._calculate_gradients(output, index)
        cam = cam.sum(dim=2)
        cam[:, 0] = 0
        return cam

    def generate_attn_last_layer(self, input_ids, attention_mask,
                     index=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        cam = _get_module_from_name(self.model, self.key)[-1].attention.self.get_attn()[0]
        cam = cam.mean(dim=0).unsqueeze(0)
        cam[:, 0, 0] = 0
        return cam[:, 0]

    def generate_rollout(self, input_ids, attention_mask, start_layer=0, index=None):
        self.model.zero_grad()
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        blocks = _get_module_from_name(self.model, self.key)
        all_layer_attentions = []
        for blk in blocks:
            attn_heads = blk.attention.self.get_attn()
            avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
            all_layer_attentions.append(avg_heads)
        rollout = compute_rollout_attention(all_layer_attentions, start_layer=start_layer)
        rollout[:, 0, 0] = 0
        return output, rollout[:, 0]

    def generate_attn_gradcam(self, input_ids, attention_mask, index=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]

        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        self._calculate_gradients(output, index)

        cam = _get_module_from_name(self.model, self.key)[-1].attention.self.get_attn()
        grad = _get_module_from_name(self.model, self.key)[-1].attention.self.get_attn_gradients()

        cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
        grad = grad.mean(dim=[1, 2], keepdim=True)
        cam = (cam * grad).mean(0).clamp(min=0).unsqueeze(0)
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        cam[:, 0, 0] = 0
        return cam[:, 0]

    def generate_rollout_attn_gradcam(self, input_ids, attention_mask, index=None, start_layer=0):
        # rule 5 from paper
        def avg_heads(cam, grad):
            return (grad * cam).clamp(min=0).mean(dim=-3)

        # rule 6 from paper
        def apply_self_attention_rules(R_ss, cam_ss):
            return torch.matmul(cam_ss, R_ss)

        output = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]

        self._calculate_gradients(output, index, do_relprop=False)

        num_tokens = input_ids.size(-1)
        R = torch.eye(num_tokens).expand(output.size(0), -1, -1).clone().to(output.device)

        blocks = _get_module_from_name(self.model, self.key)
        for i, blk in enumerate(blocks):
            if i < start_layer:
                continue
            grad = blk.attention.self.get_attn_gradients().detach()
            cam = blk.attention.self.get_attn().detach()
            cam = avg_heads(cam, grad)
            joint = apply_self_attention_rules(R, cam)
            R += joint
        # 0 because we look at the influence *on* the CLS token
        # 1:-1 because we don't want the influence *from* the CLS/SEP tokens
        return output, R[:, 0, 1:-1]

