import sys
import gradio

sys.path.append("BERT_explainability")

import torch

from transformers import AutoModelForSequenceClassification
from BERT_explainability.ExplanationGenerator import Generator
from BERT_explainability.roberta2 import RobertaForSequenceClassification
from transformers import AutoTokenizer
from captum.attr import LayerIntegratedGradients
from captum.attr import visualization
import torch

# from https://discuss.pytorch.org/t/using-scikit-learns-scalers-for-torchvision/53455
class PyTMinMaxScalerVectorized(object):
    """
    Transforms each channel to the range [0, 1].
    """

    def __init__(self, dimension=-1):
        self.d = dimension

    def __call__(self, tensor):
        d = self.d
        scale = 1.0 / (
            tensor.max(dim=d, keepdim=True)[0] - tensor.min(dim=d, keepdim=True)[0]
        )
        tensor.mul_(scale).sub_(tensor.min(dim=d, keepdim=True)[0])
        return tensor


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = RobertaForSequenceClassification.from_pretrained(
    "textattack/roberta-base-SST-2"
).to(device)
model.eval()
model2 = AutoModelForSequenceClassification.from_pretrained("textattack/roberta-base-SST-2")
tokenizer = AutoTokenizer.from_pretrained("textattack/roberta-base-SST-2")
# initialize the explanations generator
explanations = Generator(model, "roberta")

classifications = ["NEGATIVE", "POSITIVE"]

# rule 5 from paper
def avg_heads(cam, grad):
    cam = (grad * cam).clamp(min=0).mean(dim=-3)
    # set negative values to 0, then average
    #    cam = cam.clamp(min=0).mean(dim=0)
    return cam


# rule 6 from paper
def apply_self_attention_rules(R_ss, cam_ss):
    R_ss_addition = torch.matmul(cam_ss, R_ss)
    return R_ss_addition


def generate_relevance(model, input_ids, attention_mask, index=None, start_layer=0):
    output = model(input_ids=input_ids, attention_mask=attention_mask)[0]
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
    ).to(device)
    one_hot = torch.sum(one_hot * output)
    model.zero_grad()
    # create the gradients for the class we're interested in
    one_hot.backward(retain_graph=True)

    num_tokens = model.roberta.encoder.layer[0].attention.self.get_attn().shape[-1]
    R = torch.eye(num_tokens).expand(output.size(0), -1, -1).clone().to(device)

    for i, blk in enumerate(model.roberta.encoder.layer):
        if i < start_layer:
            continue
        grad = blk.attention.self.get_attn_gradients()
        cam = blk.attention.self.get_attn()
        cam = avg_heads(cam, grad)
        joint = apply_self_attention_rules(R, cam)
        R += joint
    return output, R[:, 0, 1:-1]


def visualize_text(datarecords, legend=True):
    dom = ["<table width: 100%>"]
    rows = [
        "<tr><th>True Label</th>"
        "<th>Predicted Label</th>"
        "<th>Attribution Label</th>"
        "<th>Attribution Score</th>"
        "<th>Word Importance</th>"
    ]
    for datarecord in datarecords:
        rows.append(
            "".join(
                [
                    "<tr>",
                    visualization.format_classname(datarecord.true_class),
                    visualization.format_classname(
                        "{0} ({1:.2f})".format(
                            datarecord.pred_class, datarecord.pred_prob
                        )
                    ),
                    visualization.format_classname(datarecord.attr_class),
                    visualization.format_classname(
                        "{0:.2f}".format(datarecord.attr_score)
                    ),
                    visualization.format_word_importances(
                        datarecord.raw_input_ids, datarecord.word_attributions
                    ),
                    "<tr>",
                ]
            )
        )

    if legend:
        dom.append(
            '<div style="border-top: 1px solid; margin-top: 5px; \
            padding-top: 5px; display: inline-block">'
        )
        dom.append("<b>Legend: </b>")

        for value, label in zip([-1, 0, 1], ["Negative", "Neutral", "Positive"]):
            dom.append(
                '<span style="display: inline-block; width: 10px; height: 10px; \
                border: 1px solid; background-color: \
                {value}"></span> {label}  '.format(
                    value=visualization._get_color(value), label=label
                )
            )
        dom.append("</div>")

    dom.append("".join(rows))
    dom.append("</table>")
    html = "".join(dom)

    return html


def show_explanation(model, input_ids, attention_mask, index=None, start_layer=8):
    # generate an explanation for the input
    output, expl = generate_relevance(
        model, input_ids, attention_mask, index=index, start_layer=start_layer
    )
    # normalize scores
    scaler = PyTMinMaxScalerVectorized()

    norm = scaler(expl)
    # get the model classification
    output = torch.nn.functional.softmax(output, dim=-1)

    vis_data_records = []
    for record in range(input_ids.size(0)):
        classification = output[record].argmax(dim=-1).item()
        class_name = classifications[classification]
        nrm = norm[record]

        # if the classification is negative, higher explanation scores are more negative
        # flip for visualization
        if class_name == "NEGATIVE":
            nrm *= -1
        tokens = tokenizer.convert_ids_to_tokens(input_ids[record].flatten())[
            1 : 0 - ((attention_mask[record] == 0).sum().item() + 1)
        ]
#        vis_data_records.append(list(zip(tokens, nrm.tolist())))
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

def custom_forward(inputs, attention_mask=None, pos=0):
    result = model2(inputs, attention_mask=attention_mask, return_dict=True)
    preds = result.logits
    return preds

def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions


def run_attribution_model(input_ids, attention_mask, ref_token_id=tokenizer.unk_token_id, layer=None, steps=20):
    try:
        output = model2(input_ids=input_ids, attention_mask=attention_mask)[0]
        index = output.argmax(axis=-1).detach().cpu().numpy()

        ablator = LayerIntegratedGradients(custom_forward, layer)
        input_tensor = input_ids
        attention_mask = attention_mask
        attributions = ablator.attribute(
                inputs=input_ids,
                baselines=ref_token_id,
                additional_forward_args=(attention_mask),
                target=1,
                n_steps=steps,
        )
        attributions = summarize_attributions(attributions).unsqueeze_(0)
    finally:
        pass
    vis_data_records = []
    for record in range(input_ids.size(0)):
        classification = output[record].argmax(dim=-1).item()
        class_name = classifications[classification]
        attr = attributions[record]
        tokens = tokenizer.convert_ids_to_tokens(input_ids[record].flatten())[
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

def sentence_sentiment(input_text, layer):
    text_batch = [input_text]
    encoding = tokenizer(text_batch, return_tensors="pt")
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    layer = int(layer)
    if layer == 0:
        layer = model2.roberta.embeddings
    else:
        layer = getattr(model2.roberta.encoder.layer, str(layer-1))

    output = run_attribution_model(input_ids, attention_mask, layer=layer)
    return output

def sentiment_explanation_hila(input_text, layer):
    text_batch = [input_text]
    encoding = tokenizer(text_batch, return_tensors="pt")
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # true class is positive - 1
    true_class = 1

    return show_explanation(model, input_ids, attention_mask, start_layer=int(layer))

layer_slider = gradio.Slider(minimum=0, maximum=12, value=8, step=1, label="Select rollout layer")
hila = gradio.Interface(
    fn=sentiment_explanation_hila,
    inputs=["text", layer_slider],
    outputs="html",
)
layer_slider2 = gradio.Slider(minimum=0, maximum=12, value=0, step=1, label="Select IG layer")
lig = gradio.Interface(
    fn=sentence_sentiment,
    inputs=["text", layer_slider2],
    outputs="html",
)

iface = gradio.Parallel(hila, lig,
                           title="RoBERTa Explainability",
                        description="""
In this demo, we use the RoBERTa language model (optimized for masked language modelling and finetuned for sentiment analysis). 
The model predicts for a given sentences whether it expresses a positive, negative or neutral sentiment.
But how does it arrive at its classification? A range of so-called "attribution methods" have been developed that attempt to determine the importance of the words in the input for the final prediction.

(Note that in general, importance scores only provide a very limited form of "explanation" and that different attribution methods differ radically in how they assign importance).

Two key methods for Transformers are "attention rollout" (Abnar & Zuidema, 2020) and (layer) Integrated Gradient. Here we show:

* Gradient-weighted attention rollout, as defined by [Hila Chefer](https://github.com/hila-chefer)
  [(Transformer-MM_explainability)](https://github.com/hila-chefer/Transformer-MM-Explainability/), without rollout recursion upto selected layer
* Layer IG, as implemented in [Captum](https://captum.ai/)(LayerIntegratedGradients), based on gradient w.r.t. selected layer.
""",
    examples=[
        [
            "This movie was the best movie I have ever seen! some scenes were ridiculous, but acting was great",
            8,0
        ],
        [
            "I really didn't like this movie. Some of the actors were good, but overall the movie was boring",
            8,0
        ],
        [
            "If the acting had been better, this movie might have been pretty good.",
            8,0
        ],
        [
            "If he had hated it, he would not have said that he loved it.",
            8,3
        ],
        [
            "If he had hated it, he would not have said that he loved it.",
            8,9
        ],
        [
            "Attribution methods are very interesting, but unfortunately do not work reliably out of the box.",
            8,0
        ]
    ],
)
iface.launch()
