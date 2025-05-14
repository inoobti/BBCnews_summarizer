import streamlit as st
import requests
import json
from typing import Dict, Any
import os

class NewsSummarizerApp:

    def __init__(self, api_base_url: str = "http://0.0.0.0:8000"):
        self.api_base_url = api_base_url
        st.set_page_config(page_title='Сводка последних новостей BBCNews Russia', layout='centered')


    def run(self) -> None:
        st.title('Сводка последних новостей BBCNews Russia')
        st.markdown('Узнаём что творится в мире по версии BBCNews')
        if st.button('Получить новости'):
            with st.spinner("Получаем новости"):
                try:
                    response = requests.post(
                        f"{self.api_base_url}/summarize/"
                    )
                except Exception as e:
                    st.error(f"Ошибка при получении ответа: {str(e)}")
                result = response.json()
                st.write(f'{result['row']}) {result['news']}')
                st.link_button("Подробнее", url=result['link'])
                # for  news in result['summarize']:
                #     st.write(f'{news['row']}) {news['news']}')
                #     st.link_button("Подробнее", url=news['link'])

        prompt = st.text_area("Суммаризация своей новости", key="text")
        if prompt:
            st.write("..Минуточку...")
            data = {'text': prompt}
            with st.spinner("Отправка запроса..."):
                try:
                    response = requests.post(
                        f"{self.api_base_url}/get_users_news/", json=data
                    )
                    result: Dict[str, Any] = response.json()
                    st.write("Ваш запрос обработан:")
                    st.write(result)
                except Exception as e:
                    st.error(f"Ошибка при получении ответа: {str(e)}")



if __name__ == "__main__":
    api_base_url = os.getenv("API_BASE_URL", "http://0.0.0.0:8000")
    app = NewsSummarizerApp(
        api_base_url=api_base_url
    )
    app.run()