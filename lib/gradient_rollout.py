import torch
from transformers import AutoTokenizer
from captum.attr import visualization

from roberta2 import RobertaForSequenceClassification
from util import visualize_text, PyTMinMaxScalerVectorized

classifications = ["NEGATIVE", "POSITIVE"]

class GradientRolloutExplainer:
    def __init__(self):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = RobertaForSequenceClassification.from_pretrained("textattack/roberta-base-SST-2").to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained("textattack/roberta-base-SST-2")

    def tokens_from_ids(self, ids):
        return list(map(lambda s: s[1:] if s[0] == "Ġ" else s, self.tokenizer.convert_ids_to_tokens(ids)))

    def run_attribution_model(self, input_ids, attention_mask, index=None, start_layer=0):
        def avg_heads(cam, grad):
            cam = (grad * cam).clamp(min=0).mean(dim=-3)
            # set negative values to 0, then average
            #    cam = cam.clamp(min=0).mean(dim=0)
            return cam

        def apply_self_attention_rules(R_ss, cam_ss):
            R_ss_addition = torch.matmul(cam_ss, R_ss)
            return R_ss_addition

        output = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        if index == None:
            # index = np.expand_dims(np.arange(input_ids.shape[1])
            # by default explain the class with the highest score
            index = output.argmax(axis=-1).detach().cpu().numpy()

        # create a one-hot vector selecting class we want explanations for
        one_hot = (
            torch.nn.functional.one_hot(
                torch.tensor(index, dtype=torch.int64), num_classes=output.size(-1)
            )
            .to(torch.float)
            .requires_grad_(True)
        ).to(self.device)
        one_hot = torch.sum(one_hot * output)
        self.model.zero_grad()
        # create the gradients for the class we're interested in
        one_hot.backward(retain_graph=True)

        num_tokens = self.model.roberta.encoder.layer[0].attention.self.get_attn().shape[-1]
        R = torch.eye(num_tokens).expand(output.size(0), -1, -1).clone().to(self.device)

        for i, blk in enumerate(self.model.roberta.encoder.layer):
            if i < start_layer:
                continue
            grad = blk.attention.self.get_attn_gradients()
            cam = blk.attention.self.get_attn()
            cam = avg_heads(cam, grad)
            joint = apply_self_attention_rules(R, cam)
            R += joint
        return output, R[:, 0, 1:-1]

    def build_visualization(self, input_ids, attention_mask, index=None, start_layer=8):
        # generate an explanation for the input
        vis_data_records = []

        for index in range(2):
            output, expl = self.run_attribution_model(
                input_ids, attention_mask, index=index, start_layer=start_layer
            )
            # normalize scores
            scaler = PyTMinMaxScalerVectorized()

            norm = scaler(expl)
            # get the model classification
            output = torch.nn.functional.softmax(output, dim=-1)

            for record in range(input_ids.size(0)):
                classification = output[record].argmax(dim=-1).item()
                class_name = classifications[classification]
                nrm = norm[record]

                # if the classification is negative, higher explanation scores are more negative
                # flip for visualization
                #if class_name == "NEGATIVE":
                if index == 0:
                    nrm *= -1
                tokens = self.tokens_from_ids(input_ids[record].flatten())[
                    1 : 0 - ((attention_mask[record] == 0).sum().item() + 1)
                ]
                vis_data_records.append(
                    visualization.VisualizationDataRecord(
                        nrm,
                        output[record][classification],
                        classification,
                        classification,
                        index,
                        1,
                        tokens,
                        1,
                    )
                )
        return visualize_text(vis_data_records)

    def __call__(self, input_text, start_layer=8):
        text_batch = [input_text]
        encoding = self.tokenizer(text_batch, return_tensors="pt")
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        return self.build_visualization(input_ids, attention_mask, start_layer=int(start_layer))

