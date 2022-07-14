import gradio

def greet(name):
    return f"Hello {name}. Check back soon for real content"

iface = gradio.Interface(fn=greet, inputs="text", outputs="text")
iface.launch()
