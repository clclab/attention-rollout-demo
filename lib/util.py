import pathlib
import gradio
from captum.attr import visualization

class Markdown(gradio.Markdown):
    def __init__(self, value, *args, **kwargs):
        if isinstance(value, pathlib.Path):
            value = value.read_text()
        elif isinstance(value, io.TextIOWrapper):
            value = value.read()
        super().__init__(value, *args, **kwargs)

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

# copied out of captum because we need raw html instead of a jupyter widget
def visualize_text(datarecords, legend=True):
    dom = ["<table width: 100%>"]
    rows = [
#        "<tr><th>True Label</th>"
        "<th>Predicted Label</th>"
        "<th>Attribution Label</th>"
#        "<th>Attribution Score</th>"
        "<th>Word Importance</th>"
    ]
    for datarecord in datarecords:
        rows.append(
            "".join(
                [
                    "<tr>",
#                    visualization.format_classname(datarecord.true_class),
#                    visualization.format_classname(
#                        "{0} ({1:.2f})".format(
#                            datarecord.pred_class#, datarecord.pred_prob
#                        )
#                    ),
                    visualization.format_classname(datarecord.pred_class),
                    visualization.format_classname(datarecord.attr_class),
#                    visualization.format_classname(
#                        "{0:.2f}".format(datarecord.attr_score)
#                    ),
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


