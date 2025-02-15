from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

# Loading the Extractive model 
from summarizer import Summarizer
extractive_model = Summarizer()

# Loading the abstractive model
from transformers import pipeline
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

ex_length = 0.75

max_length = 200
min_length = 150

def generate_extractive_text(text):
    res = extractive_model(text, ratio = ex_length)
    res = ''.join(res)
    return res, len(res.split(" "))

def generate_abstractive_text(text):
    res = summarizer(text, max_length=len(text.split(" ")), min_length=int(len(text.split(" "))/2), do_sample=False)
    return res[0]['summary_text'], len(res[0]['summary_text'].split(" "))

@app.route('/summary', methods=["POST"])
def home():
    data = request.get_json()  
    text = data.get("text", "")
    ext_res, ext_length = generate_extractive_text(text)
    gen_res, gen_length = generate_abstractive_text(text)

    return jsonify({
        "extractive_summary": ext_res,
        "extractive_length": ext_length,
        "abstractive_summary": gen_res,
        "abstractive_length": gen_length,
        "original_length": len(text.split(" "))
    })

    
if __name__ == '__main__':
    app.run(debug=True)
    