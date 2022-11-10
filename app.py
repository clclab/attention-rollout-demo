import sys
import pandas
import gradio
import pathlib

sys.path.append("lib")

import torch

from roberta2 import RobertaForSequenceClassification
from transformers import AutoTokenizer

from gradient_rollout import GradientRolloutExplainer
from rollout import RolloutExplainer
from integrated_gradients import IntegratedGradientsExplainer

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = RobertaForSequenceClassification.from_pretrained("textattack/roberta-base-SST-2").to(device)
tokenizer = AutoTokenizer.from_pretrained("textattack/roberta-base-SST-2")

ig_explainer = IntegratedGradientsExplainer(model, tokenizer)
gr_explainer = GradientRolloutExplainer(model, tokenizer)
ro_explainer = RolloutExplainer(model, tokenizer)

def run(sent, gradient, rollout, ig, ig_baseline):
    a = gr_explainer(sent, gradient)
    b = ro_explainer(sent, rollout)
    c = ig_explainer(sent, ig, ig_baseline)
    return a, b, c

examples = pandas.read_csv("examples.csv").to_numpy().tolist()

with gradio.Blocks(title="Explanations with attention rollout") as iface:
    gradio.Markdown(pathlib.Path("description.md").read_text)
    with gradio.Row(equal_height=True):
        with gradio.Column(scale=4):
            sent = gradio.Textbox(label="Input sentence")
        with gradio.Column(scale=1):
            but = gradio.Button("Submit")
    with gradio.Row(equal_height=True):
        with gradio.Column():
            rollout_layer = gradio.Slider(
                    minimum=1,
                    maximum=12,
                    value=1,
                    step=1,
                    label="Select rollout start layer"
                )
        with gradio.Column():
            gradient_layer = gradio.Slider(
                    minimum=1,
                    maximum=12,
                    value=8,
                    step=1,
                    label="Select gradient rollout start layer"
                )
        with gradio.Column():
            ig_layer = gradio.Slider(
                    minimum=0,
                    maximum=12,
                    value=0,
                    step=1,
                    label="Select IG layer"
                )
            ig_baseline = gradio.Dropdown(
                    label="Baseline token",
                    choices=['Unknown', 'Padding'], value="Unknown"
                )
    with gradio.Row(equal_height=True):
        with gradio.Column():
            gradio.Markdown("### Attention Rollout")
            rollout_result = gradio.HTML()
        with gradio.Column():
            gradio.Markdown("### Gradient-weighted Attention Rollout")
            gradient_result = gradio.HTML()
        with gradio.Column():
            gradio.Markdown("### Layer-Integrated Gradients")
            ig_result = gradio.HTML()
    gradio.Examples(examples, [sent])
    with gradio.Accordion("Some more details"):
        gradio.Markdown(pathlib.Path("notice.md").read_text)

    gradient_layer.change(gr_explainer, [sent, gradient_layer], gradient_result)
    rollout_layer.change(ro_explainer, [sent, rollout_layer], rollout_result)
    ig_layer.change(ig_explainer, [sent, ig_layer, ig_baseline], ig_result)
    but.click(run,
            inputs=[sent, gradient_layer, rollout_layer, ig_layer, ig_baseline],
            outputs=[gradient_result, rollout_result, ig_result]
        )


iface.launch()
