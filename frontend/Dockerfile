FROM python:3.11
COPY . /frontend
WORKDIR /frontend
RUN pip install -r requirements.txt
CMD ["streamlit", "run", "chat.py"]