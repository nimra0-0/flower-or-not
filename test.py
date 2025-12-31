from fastai.vision.all import *
from PIL import Image



def predict_with_threshold(learn, img, positive_class="flower", threshold=0.8):
    _, _, probs = learn.predict(img)

    vocab = learn.dls.vocab
    pos_idx = vocab.o2i[positive_class]
    pos_prob = probs[pos_idx].item() # get the probability of the positive class

    final_class = positive_class if pos_prob >= threshold else f"not_{positive_class}"
    final_idx = vocab.o2i[final_class]
    return (
        final_class, 
        final_idx,
        probs
    )

def test_model():
    
    test_image = PILImage.create('./example_images/not_flower.png')
    print('loading model...')
    learn = load_learner('flower_classifier.pkl')
    labels = learn.dls.vocab
    print('model loaded')
    pred, pred_idx, probs = predict_with_threshold(learn, test_image)
    print('predicting...')
    print(f'pred: {pred}, pred_idx: {pred_idx}, probs: {probs}')
    print(f'label: {labels[pred_idx]}')
    print(f'probability: {probs[pred_idx]:.4f}')

if __name__ == '__main__':
    test_model()