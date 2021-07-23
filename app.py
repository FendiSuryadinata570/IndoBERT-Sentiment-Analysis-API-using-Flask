from flask.wrappers import Response
from pkg_resources import DEVELOP_DIST
import torch
import flask
from flask import Flask
from flask import request
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig

app = Flask(__name__)

MODEL = None
DEVICE = torch.device("cpu")

model_name = "indobenchmark/indobert-base-p1"

tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
class_names = ["Negative", "Neutral", "Positive"]

def loaded_model_from_file(model_name, file_directory):
    loaded_model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels = 3,
        output_attentions = False,
        output_hidden_states = False
    )
    loaded_model = loaded_model.to(DEVICE)
    loaded_model.load_state_dict(torch.load(file_directory))

    return loaded_model

def prediksi(review_text, model,MAX_LEN=64):

    sentiment_probs = {}

    class_names = ["Negative", "Neutral", "Positive"]
    encoded_review = tokenizer.encode_plus(
        review_text,
        max_length=MAX_LEN,
        add_special_tokens=True,
        return_token_type_ids=False,
        pad_to_max_length = True,
        return_attention_mask= True,
        return_tensors='pt'
    )

    input_ids = encoded_review['input_ids'].to(DEVICE)
    attention_mask = encoded_review['attention_mask'].to(DEVICE)

    output = model(input_ids, attention_mask)
    _, prediction = torch.max(output[0], dim=1)

    np_arr = output[0].cpu().detach().numpy()

    sm = torch.nn.Softmax()
    proba = sm(output[0]).tolist()

    sentiment_probs["Negative"] = proba[0][0]*100
    sentiment_probs["Neutral"] = proba[0][1]*100
    sentiment_probs["Positive"] = proba[0][2]*100

    sentiment_probs_values = sentiment_probs.values()
    total_sentiment_values = sum(sentiment_probs_values)

    for k,v in sentiment_probs.items():
        v /= total_sentiment_values
        sentiment_probs[k] = v

    return [class_names[prediction], sentiment_probs]

@app.route("/predict")
def predict():
    sentence = request.args.get("sentence")
    response = {}
    loaded_model = loaded_model_from_file(model_name, "D:/WORK/Sentiment Analysis/IndoBERT_product.h5")
    pred = prediksi(sentence, loaded_model)[0]
    response["response"] = {
        'prediction': str(pred),
        'sentence': str(sentence)
    }
    return flask.jsonify(response)

if __name__ == "__main__":
    MODEL = loaded_model_from_file(model_name, "D:/WORK/Sentiment Analysis/IndoBERT_product.h5")
    MODEL.to(DEVICE)
    MODEL.eval()
    app.run()