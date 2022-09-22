import sys
import pandas
import gradio
import pathlib

sys.path.append("lib")

import torch

from roberta2 import RobertaForSequenceClassification
from gradient_rollout import GradientRolloutExplainer
from integrated_gradients import IntegratedGradientsExplainer
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from captum.attr import LayerIntegratedGradients
from captum.attr import visualization
import util
import torch

ig_explainer = IntegratedGradientsExplainer()
gr_explainer = GradientRolloutExplainer()

def run(sent, rollout, ig, ig_baseline):
    a = gr_explainer(sent, rollout)
    b = ig_explainer(sent, ig, ig_baseline)
    return a, b

examples = pandas.read_csv("examples.csv").to_numpy().tolist()

with gradio.Blocks(title="Explanations with attention rollout") as iface:
    util.Markdown(pathlib.Path("description.md"))
    with gradio.Row(equal_height=True):
        with gradio.Column(scale=4):
            sent = gradio.Textbox(label="Input sentence")
        with gradio.Column(scale=1):
            but = gradio.Button("Submit")
    with gradio.Row(equal_height=True):
        with gradio.Column():
            rollout_layer = gradio.Slider(minimum=0, maximum=12, value=8, step=1, label="Select rollout start layer")
            rollout_result = gradio.HTML()
        with gradio.Column():
            ig_layer = gradio.Slider(minimum=0, maximum=12, value=0, step=1, label="Select IG layer")
            ig_baseline = gradio.Dropdown(label="Baseline token", choices=['Unknown', 'Padding'], value="Unknown")
            ig_result = gradio.HTML()
    gradio.Examples(examples, [sent])
    with gradio.Accordion("Some more details"):
        util.Markdown(pathlib.Path("notice.md"))

    rollout_layer.change(gr_explainer, [sent, rollout_layer], rollout_result)
    ig_layer.change(ig_explainer, [sent, ig_layer, ig_baseline], ig_result)
    but.click(run, [sent, rollout_layer, ig_layer, ig_baseline], [rollout_result, ig_result])


iface.launch()
