import torch

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

from captum.attr import LayerIntegratedGradients
from captum.attr import visualization

from roberta2 import RobertaForSequenceClassification
from ExplanationGenerator import Generator
from util import visualize_text

classifications = ["NEGATIVE", "POSITIVE"]

class IntegratedGradientsExplainer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.device = model.device
        self.tokenizer = tokenizer
        self.baseline_map = {
                'Unknown': self.tokenizer.unk_token_id,
                'Padding': self.tokenizer.pad_token_id,
            }

    def tokens_from_ids(self, ids):
        return list(map(lambda s: s[1:] if s[0] == "Ġ" else s, self.tokenizer.convert_ids_to_tokens(ids)))

    def custom_forward(self, inputs, attention_mask=None, pos=0):
        result = self.model(inputs, attention_mask=attention_mask, return_dict=True)
        preds = result.logits
        return preds

    @staticmethod
    def summarize_attributions(attributions):
        attributions = attributions.sum(dim=-1).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        return attributions

    def run_attribution_model(self, input_ids, attention_mask, baseline=None, index=None, layer=None, steps=20):
        if baseline is None:
            baseline = self.tokenizer.unk_token_id
        else:
            baseline = self.baseline_map[baseline]

        try:
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
#            if index is None:
#                index = output.argmax(axis=-1).item()

            ablator = LayerIntegratedGradients(self.custom_forward, layer)
            input_tensor = input_ids
            attention_mask = attention_mask
            attributions = ablator.attribute(
                    inputs=input_ids,
                    baselines=baseline,
                    additional_forward_args=(attention_mask),
                    target=1,
                    n_steps=steps,
            )
            return self.summarize_attributions(attributions).unsqueeze_(0), output, index
        finally:
            pass

    def build_visualization(self, input_ids, attention_mask, **kwargs):
        vis_data_records = []
        attributions, output, index = self.run_attribution_model(input_ids, attention_mask, **kwargs)
        for record in range(input_ids.size(0)):
            classification = output[record].argmax(dim=-1).item()
            class_name = classifications[classification]
            attr = attributions[record]
            tokens = self.tokens_from_ids(input_ids[record].flatten())[
                1 : 0 - ((attention_mask[record] == 0).sum().item() + 1)
            ]
            vis_data_records.append(
                visualization.VisualizationDataRecord(
                    attr,
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

    def __call__(self, input_text, layer, baseline):
        text_batch = [input_text]
        encoding = self.tokenizer(text_batch, return_tensors="pt")
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        layer = int(layer)
        if layer == 0:
            layer = self.model.roberta.embeddings
        else:
            layer = getattr(self.model.roberta.encoder.layer, str(layer-1))

        return self.build_visualization(input_ids, attention_mask, layer=layer, baseline=baseline)
