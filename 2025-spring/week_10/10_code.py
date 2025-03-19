#bash: 1) python -m venv venv
## 2) source venv/bin/activate
## 3) deactivate

#streamlit run venv/10_code.py

from transformers import pipeline
from langchain.chains import ConversationChain
from langchain.llms import HuggingFacePipeline
import streamlit as st

# Инициализация генератора с параметрами
# Оборачиваем pipeline в HuggingFacePipeline для использования в LangChain
llm = HuggingFacePipeline.from_model_id(
    model_id = "gpt2",
    task = "text-generation",
    pipeline_kwargs = {"max_new_tokens": 512},)

# Интерфейс чат-бота с использованием Streamlit
def chatbot_interface() :
    st.title("Чат-бот для DS-15 на базе GPT-2")
    # Создаем память для хранения истории чата
    if 'history' not in st.session_state:
        st.session_state.history = []
    # Создаем цепочку для общения с пользователем
    conversation_chain = ConversationChain(llm = llm)
    # Создаем область чата
    chat_area = st.empty()
    # Ввод от пользователя
    user_input = st.text_input("Вы: ", "")
    if st.button("Отправить"):
        if user_input:
            # Добавляем запрос пользователя в историю
            st.session_state.history.append(f"Вы: {user_input}")
            with st.spinner('Думаю над ответом...'):
                try:
                    # Получаем ответ от чат-бота
                    response = conversation_chain.run(input = user_input)
                    # Добавляем ответ бота в историю
                    st.session_state.history.append(f"Бот: {response}")
                    # Отображаем историю чата
                    chat_area.text("\n".join(st.session_state.history))
                except Exception as e:
                    st.error(f"Произошла ошибка: {str(e)}")
        else:
            st.warning("Пожалуйста, введите текст запроса.")

if __name__ == "__main__":
    chatbot_interface()

