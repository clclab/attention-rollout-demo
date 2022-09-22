import torch

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

from captum.attr import LayerIntegratedGradients
from captum.attr import visualization

from util import visualize_text

classifications = ["NEGATIVE", "POSITIVE"]

class IntegratedGradientsExplainer:
    def __init__(self):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = AutoModelForSequenceClassification.from_pretrained("textattack/roberta-base-SST-2").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("textattack/roberta-base-SST-2")
        self.ref_token_id = self.tokenizer.unk_token_id

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


    def run_attribution_model(self, input_ids, attention_mask, index=None, layer=None, steps=20):
        try:
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
#            if index is None:
#                index = output.argmax(axis=-1).item()

            ablator = LayerIntegratedGradients(self.custom_forward, layer)
            input_tensor = input_ids
            attention_mask = attention_mask
            attributions = ablator.attribute(
                    inputs=input_ids,
                    baselines=self.ref_token_id,
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

    def __call__(self, input_text, layer):
        text_batch = [input_text]
        encoding = self.tokenizer(text_batch, return_tensors="pt")
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        layer = int(layer)
        if layer == 0:
            layer = self.model.roberta.embeddings
        else:
            layer = getattr(self.model.roberta.encoder.layer, str(layer-1))

        return self.build_visualization(input_ids, attention_mask, layer=layer)
