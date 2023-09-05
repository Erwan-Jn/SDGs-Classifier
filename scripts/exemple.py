import streamlit as st
import requests

st.title("Demo Day - SGD Classifier")
with st.form(key='params_for_api'):
    text = st.text_input("Article")
    if st.form_submit_button('Which SDG am I ? '):
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

    elif st.form_submit_button('The best 3 SDGs the article corresponds to ?'):
        params = dict(text=text)
        sdg_classifier_api_url = f"https://sdgclassifier-bw4yive63a-od.a.run.app/predict?{text}"
        response = requests.get(sdg_classifier_api_url,params=params)
        prediction = response.json()
        pred = prediction['The text refers to those 3 SDGs:']
        st.header


    #st.form_submit_button('Which SDGs am I composed of ?')
    #st.form_submit_button('Big category ?')

#st.image("https://www.ifr.sun.ac.za/wp-content/uploads/2020/07/sdgs.png")
