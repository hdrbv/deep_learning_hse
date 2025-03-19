#bash: 1) python -m venv venv
## 2) source venv/bin/activate
## 3) deactivate

#streamlit run venv/10_code.py

from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
import numpy as np

# Загрузка модели для генерации текста (например, T5 для генерации)
# Оборачиваем pipeline в HuggingFacePipeline для использования в LangChain
llm = HuggingFacePipeline.from_model_id(
    model_id = "gpt2",
    task = "text-generation",
    pipeline_kwargs = {"max_new_tokens": 100},
)

# Создаем экземпляр эмбеддингов
help(HuggingFaceEmbeddings)
embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

# Создание векторного хранилища с использованием FAISS
documents = ["Документ 1: Текст", "Документ 2: Еще один текст", "Документ 3: Дополнительная информация"]
documents

document_embeddings = embedding_model.embed_documents(documents)
document_embeddings

document_embeddings = np.array(document_embeddings, dtype = np.float32)
document_embeddings

print(document_embeddings.shape)

# Индексируем документы с помощью FAISS
dimension = document_embeddings.shape[1]  # Получаем размерность эмбеддингов (например, 384)
dimension

index = faiss.IndexFlatL2(dimension)
index

# Добавляем эмбеддинги в индекс
index.add(document_embeddings)

# Создание docstore и индекса для LangChain
docstore = {}
index_to_docstore_id = {}

# Заполнение docstore и связи с индексами
for i, doc in enumerate(documents):
    docstore[i] = {"text": doc}
    index_to_docstore_id[i] = i

docstore

index_to_docstore_id

# Создаем объект FAISS для LangChain
vectorstore = FAISS(
    index = index,
    embedding_function = embedding_model,
    docstore = docstore,
    index_to_docstore_id = index_to_docstore_id
)
vectorstore


# Создаем систему RAG с извлечением и генерацией
qa_chain = RetrievalQA.from_chain_type(llm = llm, chain_type = "stuff", retriever = vectorstore.as_retriever())
qa_chain

# Вопрос от пользователя
user_input = "Какая информация содержится в документе 1?"
user_input2 = "Что за ограничения существуют?"

# Получение ответа от системы
response = qa_chain.run(user_input)
print(response)

