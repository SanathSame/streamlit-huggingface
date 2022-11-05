import streamlit as st
from transformers import pipeline, AutoTokenizer, TFAutoModelForSequenceClassification

st.markdown("<h1 style='text-align: center; color: red;'>Sentiment Analyser for Hotel Reviews</h1>", unsafe_allow_html=True)
st.image("""https://cdn2.hubspot.net/hubfs/439788/Blog/Featured%20Images/Best%20Hotel%20Website%20Designs.jpg""")
st.write('This demo app uses DistilBERT which is a smaller general-purpose language representation model we just discussed.')


form = st.form(key='sentiment-form')
user_input = form.text_area('Enter your text')
submit = form.form_submit_button('Submit')
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

if submit:
    result = classifier(user_input)[0]
    label = result['label']
    score = result['score']
    if label == 'POSITIVE':
        st.success(f'{label} sentiment (score: {score})')
    else:
        st.error(f'{label} sentiment (score: {score})')
