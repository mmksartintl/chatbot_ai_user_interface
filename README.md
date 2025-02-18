# chatbot_ai_user_interface
Chatbot with a friendly user interface for user interaction

Implements:
- LangChain to prompt LLM via Groq https://python.langchain.com/docs/integrations/chat/groq/

Steps:

1) run a docker image

   $ docker container run -d -p 5000:5000 python:3.10 sleep infinity

2) pip install -r requirements.txt  
   pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu  
   pip install --no-cache-dir sentence-transformers

3) python3 init_vdb.py ( or init_vdb_trein.py )

4) python app.py
