import sys
import gradio

sys.path.append("BERT_explainability")

import torch

from BERT_explainability.ExplanationGenerator import Generator
from BERT_explainability.roberta2 import RobertaForSequenceClassification
from transformers import AutoTokenizer

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
    print("ONE_HOT", one_hot.size(), one_hot)
    one_hot = torch.sum(one_hot * output)
    model.zero_grad()
    # create the gradients for the class we're interested in
    one_hot.backward(retain_graph=True)

    num_tokens = model.roberta.encoder.layer[0].attention.self.get_attn().shape[-1]
    print(input_ids.size(-1), num_tokens)
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


def show_explanation(model, input_ids, attention_mask, index=None, start_layer=0):
    # generate an explanation for the input
    output, expl = generate_relevance(
        model, input_ids, attention_mask, index=index, start_layer=start_layer
    )
    print(output.shape, expl.shape)
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
        print([(tokens[i], nrm[i].item()) for i in range(len(tokens))])
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


def run(input_text):
    text_batch = [input_text]
    encoding = tokenizer(text_batch, return_tensors="pt")
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # true class is positive - 1
    true_class = 1

    html = show_explanation(model, input_ids, attention_mask)
    return html


iface = gradio.Interface(
    fn=run,
    inputs="text",
    outputs="html",
    title="RoBERTa Explanability",
    description="Quick demo of a version of [Hila Chefer's](https://github.com/hila-chefer) [Transformer-Explanability](https://github.com/hila-chefer/Transformer-Explainability/) but without the layerwise relevance propagation (as in [Transformer-MM_explainability](https://github.com/hila-chefer/Transformer-MM-Explainability/)) for a RoBERTa model.",
    examples=[
        [
            "This movie was the best movie I have ever seen! some scenes were ridiculous, but acting was great"
        ],
        [
            "I really didn't like this movie. Some of the actors were good, but overall the movie was boring"
        ],
    ],
)
iface.launch()
