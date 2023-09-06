import streamlit as st
import requests
import pandas as pd
from scripts.pdf import pdf
from io import StringIO
from pdfquery import PDFQuery
import time
from collections import Counter


st.header("SDG Classifier :recycle: ",divider = "rainbow")
st.title("Using a PDF :open_file_folder:")

with st.form(key='params_for_api_pdf'):
    uploaded_file = st.file_uploader("Choose a file", type= 'pdf')
    if st.form_submit_button("Upload pdf"):
        if uploaded_file is not None:
            pdf=PDFQuery(uploaded_file)
            pdf.load()
            text_elements=pdf.pq("LTTextLineHorizontal")
            text=[t.text for t in text_elements]
            #text = pdf(uploaded_file)
            text = " ".join(text)
            params = {'text':text}
            sdg_classifier_api_url = f"https://sdgclassifier-bw4yive63a-od.a.run.app/predict"
            response = requests.get(sdg_classifier_api_url,params=params)
            prediction = response.json()

            pred = prediction['The text is talking about SDG:']

            if round(pred) == 15:
                st.image("https://miro.medium.com/v2/resize:fit:1400/format:webp/1*7MDLuoSaJjS-q5tZ_vJbVA.png")

            elif round(pred) == 1:
                st.image("https://miro.medium.com/v2/resize:fit:1400/format:webp/1*9KeYUomO4E0EqXT24XUypQ.png")

            elif round(pred) == 2:
                st.image("https://www.kit.nl/wp-content/uploads/2019/02/E_SDG-goals_icons-individual-rgb-02.png")

            elif round(pred) == 3:
                st.image("https://miro.medium.com/v2/resize:fit:1400/format:webp/1*bD6Q8IDG3Ef444SAOnNiyg.png")
            elif round(pred) == 4:
                st.image("https://miro.medium.com/v2/resize:fit:1400/format:webp/1*1-A2Y3EWTX6V8ISs7zgU_Q.png")
            elif round(pred) == 5:
                st.image("https://www.kit.nl/wp-content/uploads/2019/02/E_SDG-goals_icons-individual-rgb-05.png")
            elif round(pred) == 6:
                st.image("https://www.kit.nl/wp-content/uploads/2019/02/E_SDG-goals_icons-individual-rgb-06.png")
            elif round(pred) == 7:
                st.image("https://www.kit.nl/wp-content/uploads/2019/02/E_SDG-goals_icons-individual-rgb-07.png")
            elif round(pred) == 8:
                st.image("https://www.kit.nl/wp-content/uploads/2019/02/E_SDG-goals_icons-individual-rgb-08.png")
            elif round(pred) == 9:
                st.image("https://www.kit.nl/wp-content/uploads/2019/02/E_SDG-goals_icons-individual-rgb-09.png")
            elif round(pred) == 10:
                st.image("https://www.kit.nl/wp-content/uploads/2019/02/E_SDG-goals_icons-individual-rgb-10.png")
            elif round(pred) == 11:
                st.image("https://www.kit.nl/wp-content/uploads/2019/02/E_SDG-goals_icons-individual-rgb-11.png")
            elif round(pred) == 12:
                st.image("https://www.kit.nl/wp-content/uploads/2019/02/E_SDG-goals_icons-individual-rgb-12.png")
            elif round(pred) == 13:
                st.image("https://www.kit.nl/wp-content/uploads/2019/02/E_SDG-goals_icons-individual-rgb-13.png")
            elif round(pred) == 14:
                st.image("https://www.kit.nl/wp-content/uploads/2019/02/E_SDG-goals_icons-individual-rgb-14.png")
            elif round(pred) == 16:
                st.image("https://www.kit.nl/wp-content/uploads/2019/02/E_SDG-goals_icons-individual-rgb-16.png ")

st.header("",divider="rainbow")

st.title("Paste an article :pencil:")

with st.form(key='params_for_api'):
    text = st.text_input("Article")
    if st.form_submit_button('The best 3 SDGs the article corresponds to ?'):
        params = dict(text=text)
        sdg_classifier_api_url2 = f"https://sdgclassifier-bw4yive63a-od.a.run.app/predict_proba?{text}"
        response = requests.get(sdg_classifier_api_url2,params=params)
        prediction = response.json()
        prediction={key[:-1]+str(int(key[-1])+1):round(proba,3) for key,proba in prediction.items()}
        k = Counter(prediction)
        high = k.most_common(3)
        col1, col2, col3 = st.columns(3)
        for i in high:
            st.write(i[0]," :",round(i[1],2)*100,"%")
            if "5" in i[0]:
                st.image("https://www.kit.nl/wp-content/uploads/2019/02/E_SDG-goals_icons-individual-rgb-05.png")
            if "1" in i[0]:
                st.image("https://miro.medium.com/v2/resize:fit:1400/format:webp/1*9KeYUomO4E0EqXT24XUypQ.png")
            if "2" in i[0]:
                st.image("https://www.kit.nl/wp-content/uploads/2019/02/E_SDG-goals_icons-individual-rgb-02.png")
            if "3" in i[0]:
                st.image("https://miro.medium.com/v2/resize:fit:1400/format:webp/1*bD6Q8IDG3Ef444SAOnNiyg.png")
            if "4" in i[0]:
                st.image("https://miro.medium.com/v2/resize:fit:1400/format:webp/1*1-A2Y3EWTX6V8ISs7zgU_Q.png")
            if "6" in i[0]:
                st.image("https://www.kit.nl/wp-content/uploads/2019/02/E_SDG-goals_icons-individual-rgb-06.png")
            if "7" in i[0]:
                st.image("https://www.kit.nl/wp-content/uploads/2019/02/E_SDG-goals_icons-individual-rgb-07.png")
            if "8" in i[0]:
                st.image("https://www.kit.nl/wp-content/uploads/2019/02/E_SDG-goals_icons-individual-rgb-08.png")
            if "9" in i[0]:
                st.image("https://www.kit.nl/wp-content/uploads/2019/02/E_SDG-goals_icons-individual-rgb-09.png")
            if "10" in i[0]:
                st.image("https://www.kit.nl/wp-content/uploads/2019/02/E_SDG-goals_icons-individual-rgb-10.png")
            if "11" in i[0]:
               st.image("https://www.kit.nl/wp-content/uploads/2019/02/E_SDG-goals_icons-individual-rgb-11.png")
            if "12" in i[0]:
                st.image("https://www.kit.nl/wp-content/uploads/2019/02/E_SDG-goals_icons-individual-rgb-12.png")
            if "13" in i[0]:
                 st.image("https://www.kit.nl/wp-content/uploads/2019/02/E_SDG-goals_icons-individual-rgb-13.png")
            if "14" in i[0]:
                st.image("https://www.kit.nl/wp-content/uploads/2019/02/E_SDG-goals_icons-individual-rgb-14.png")
            if "15" in i[0]:
                st.image("https://miro.medium.com/v2/resize:fit:1400/format:webp/1*7MDLuoSaJjS-q5tZ_vJbVA.png")
            if "16" in i[0]:
                st.image("https://www.kit.nl/wp-content/uploads/2019/02/E_SDG-goals_icons-individual-rgb-16.png ")


    elif st.form_submit_button('Which SDG am I ? '):

        params = dict(text=text)
        sdg_classifier_api_url = f"https://sdgclassifier-bw4yive63a-od.a.run.app/predict?{text}"

        response = requests.get(sdg_classifier_api_url,params=params)

        prediction = response.json()

        pred = prediction['The text is talking about SDG:']

        if round(pred) == 15:
            st.image("https://miro.medium.com/v2/resize:fit:1400/format:webp/1*7MDLuoSaJjS-q5tZ_vJbVA.png")

        elif round(pred) == 1:
            st.image("https://miro.medium.com/v2/resize:fit:1400/format:webp/1*9KeYUomO4E0EqXT24XUypQ.png")

        elif round(pred) == 2:
            st.image("https://www.kit.nl/wp-content/uploads/2019/02/E_SDG-goals_icons-individual-rgb-02.png")

        elif round(pred) == 3:
            st.image("https://miro.medium.com/v2/resize:fit:1400/format:webp/1*bD6Q8IDG3Ef444SAOnNiyg.png")
        elif round(pred) == 4:
            st.image("https://miro.medium.com/v2/resize:fit:1400/format:webp/1*1-A2Y3EWTX6V8ISs7zgU_Q.png")
        elif round(pred) == 5:
            st.image("https://www.kit.nl/wp-content/uploads/2019/02/E_SDG-goals_icons-individual-rgb-05.png")
        elif round(pred) == 6:
            st.image("https://www.kit.nl/wp-content/uploads/2019/02/E_SDG-goals_icons-individual-rgb-06.png")
        elif round(pred) == 7:
            st.image("https://www.kit.nl/wp-content/uploads/2019/02/E_SDG-goals_icons-individual-rgb-07.png")
        elif round(pred) == 8:
            st.image("https://www.kit.nl/wp-content/uploads/2019/02/E_SDG-goals_icons-individual-rgb-08.png")
        elif round(pred) == 9:
            st.image("https://www.kit.nl/wp-content/uploads/2019/02/E_SDG-goals_icons-individual-rgb-09.png")
        elif round(pred) == 10:
            st.image("https://www.kit.nl/wp-content/uploads/2019/02/E_SDG-goals_icons-individual-rgb-10.png")
        elif round(pred) == 11:
            st.image("https://www.kit.nl/wp-content/uploads/2019/02/E_SDG-goals_icons-individual-rgb-11.png")
        elif round(pred) == 12:
            st.image("https://www.kit.nl/wp-content/uploads/2019/02/E_SDG-goals_icons-individual-rgb-12.png")
        elif round(pred) == 13:
            st.image("https://www.kit.nl/wp-content/uploads/2019/02/E_SDG-goals_icons-individual-rgb-13.png")
        elif round(pred) == 14:
            st.image("https://www.kit.nl/wp-content/uploads/2019/02/E_SDG-goals_icons-individual-rgb-14.png")
        elif round(pred) == 16:
            st.image("https://www.kit.nl/wp-content/uploads/2019/02/E_SDG-goals_icons-individual-rgb-16.png ")

        st.header(f'This text should be classified in SDG {round(pred)}')


    elif st.form_submit_button('Which ESG I am ? '):
        params = dict(text=text)
        sdg_classifier_api_url3 = f"https://sdgclassifier-bw4yive63a-od.a.run.app/predict_category?{text}"
        response = requests.get(sdg_classifier_api_url3,params=params)
        prediction = response.json()
        pred = prediction["This text most probably belongs to the following category:"]
        if pred == 1:
            results = "Economy"
        elif pred == 2:
            results = "Environement"
        else:
            results = "Societal"

        st.header(f'This text should be classified in the category {results}')

        if pred == 3:
            st.image("https://www.pngall.com/wp-content/uploads/9/Society-PNG-300x225.png")

        elif round(pred) == 1:
            st.image("https://www.pngall.com/wp-content/uploads/1/Save-Money-300x225.png")

        elif round(pred) == 2:
            st.image("https://www.pngall.com/wp-content/uploads/2017/05/Save-Earth-PNG-Picture.png")


        #st.write()
        #max_pred = max(prediction)
        #max_keys = [key for key, value in prediction.items() if value == max(prediction.values())]
        #st.header(f'This text should be classified in SDG {max_pred}')



    #st.form_submit_button('Which SDGs am I composed of ?')
    #st.form_submit_button('Big category ?')

#st.image("https://www.ifr.sun.ac.za/wp-content/uploads/2020/07/sdgs.png")
