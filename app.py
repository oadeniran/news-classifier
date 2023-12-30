import streamlit as st
import numpy as np
from helpers.helpers import process_req, load_vectorizer, load_model

vectorizer = load_vectorizer("helpers/vectorizer.pk")
model = load_model("model/model.h5")


st.markdown(
    """
    <style>
        .center {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True
)


st.title('News classifier')

text = st.text_area("Please paste the news content here", height = 150)


subject = st.radio(
    "What's the subject of the news",
    ["***News***", "***Politics***", "***Government News***", "***World News***", "***US-News***", "***Left-News***", "***Middle-East***"],
    index = None,
    captions = ["General news on mundane topics", "News on Politics and state affairs", "News pertaining to the government of a country", "News pertaining to other countries outside the US", "News that borders happenings in the US", "News on left wings of the US", "News on matters in the Middle East region"])

def get_preds(text, subject):
    if subject != None and not len(text) < 3:
        subject = subject.lower().strip("*")

        request = process_req(text, subject, vectorizer)

        prediction = model.predict(request)
        prediction = np.round(prediction).reshape(1, -1)
        ind = prediction[0]
        ind = int(ind)
        stat = ["Fake", "Real"][ind]
        st.subheader(stat, divider="rainbow")

st.button("Inference", type="primary", on_click=get_preds, args=[text, subject])
if st.button('Clear'):
    st.write('')





