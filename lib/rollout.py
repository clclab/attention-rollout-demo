import torch
from transformers import AutoTokenizer
from captum.attr import visualization

from roberta2 import RobertaForSequenceClassification
from ExplanationGenerator import Generator
from util import visualize_text, PyTMinMaxScalerVectorized

classifications = ["NEGATIVE", "POSITIVE"]

class RolloutExplainer(Generator):
    def __init__(self, model, tokenizer):
        super().__init__(model, key="roberta.encoder.layer")
        self.device = model.device
        self.tokenizer = tokenizer

    def build_visualization(self, input_ids, attention_mask, start_layer=8):
        # generate an explanation for the input
        vis_data_records = []

        output, expl = self.generate_rollout(
            input_ids, attention_mask, start_layer=start_layer
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
            if class_name == "NEGATIVE":
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
                    classification,
                    1,
                    tokens,
                    1,
                )
            )
        return visualize_text(vis_data_records)

    def __call__(self, input_text, start_layer=8):
        if start_layer > 0:
            start_layer -= 1

        text_batch = [input_text]
        encoding = self.tokenizer(text_batch, return_tensors="pt")
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        return self.build_visualization(input_ids, attention_mask, start_layer=int(start_layer))

