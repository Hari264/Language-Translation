import os
import requests
from flask import Flask, render_template, request
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

import pickle
from flask import Flask,render_template

app = Flask(__name__)
hlen_mul_engmodel = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-mul-en")
hlen_mul_engtokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-mul-en")
hlen_eng_mulmodel =  AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-mul")
hlen_eng_multokenizer =  AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-mul")
@app.route('/')
def home():
      return render_template('index.html')

# Load the pre-trained model
with open('language_detection_model.pkl', 'rb') as model_file:
    clasiffier = pickle.load(model_file)

# Load the corresponding CountVectorizer
with open('count_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

def target_hlen(predicted):
  

  if(predicted=='Telugu'):
    return 'tel'
  elif (predicted=='Bengali'):
    return 'ben'
  elif(predicted=='Hindi'):
    return 'hin'
  elif(predicted=='Tamil'):
    return 'tam'
  
  elif(predicted=='Englsih'):
    return 'eng'

def predict_language_hlensk(user_input):
    # Transform user input using the loaded CountVectorizer
    user_input_vec = vectorizer.transform([user_input])

    # Predict the language using the loaded model
    predicted_language = clasiffier.predict(user_input_vec)[0]


    return predicted_language

def hlen_mul_eng(source,text):


  translator = pipeline('translation', model=hlen_mul_engmodel, tokenizer=hlen_mul_engtokenizer, src_lang=source, max_length = 400)
  return translator(text)

def hlen_eng_mul(target,text):

  hlen_eng_multranslator = pipeline('translation', model=hlen_eng_mulmodel, tokenizer=hlen_eng_multokenizer, max_length = 400)
  t = f">>{target}<<{text}"
  return hlen_eng_multranslator(t)

def hlen(source,target,text):
  if(source=='eng'):
    return hlen_eng_mul(target,text)
  elif(target=='eng'):
    return hlen_mul_eng(source,text)
  else:
    a =  hlen_mul_eng(source,text)
    return hlen_eng_mul(target,a[0]['translation_text'])
  


app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def home():
    Target = 'hello'
    if request.method == 'POST': 
        
        text = request.form.get('Text') 
        target = request.form.get('Target')
        source = predict_language_hlensk(text)
        target = target_hlen(target)
        h = hlen(source,target,text)
        trans_text = h[0]['translation_text']

        return render_template('index.html',message=trans_text) 
    else:
        return render_template('index.html') 


  

if __name__ == '__main__':
    app.run()

