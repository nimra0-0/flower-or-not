from fastai.vision.all import *
import gradio as gr

#load the model
learn = load_learner('flower_classifier.pkl')
labels = learn.dls.vocab


def predict(img):
    """
    Predict the class of the image
    Args:
        img: PIL.Image.Image
    Returns:
    dict: A dictionary of the predicted class and the probability
    """
    fastai_img = PILImage.create(img) #The pkl model expects a fastai image

    pred, pred_idx, probs = learn.predict(fastai_img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}


iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    title="Flower Classifier",
    description="Upload an image to classify if it's a flower or not"
)

if __name__ == "__main__":
    iface.launch(share=True)