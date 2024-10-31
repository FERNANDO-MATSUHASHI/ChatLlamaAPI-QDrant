from flask import Flask, request, jsonify
from qdrant_client import QdrantClient
import requests
import json
from dotenv import load_dotenv
import os
import uuid
from flask_cors import CORS
from langchain.memory import ConversationBufferMemory
from abc import ABC, abstractmethod 
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": ["https://saudeai.netlify.app", "https://iclinicbot.netlify.app"]}})

load_dotenv()
urlQdrant = os.getenv('DATABASE_URL')
apiQdrant = os.getenv('CHAVE_QDRANT')
apillama = os.getenv('CHAVE_LLAMA')

qdrant_client = QdrantClient(
    url=urlQdrant,
    api_key=apiQdrant
)

class EmbeddingModel(ABC):
    @abstractmethod
    def encode(self, text):
        pass

class SentenceTransformerEmbedding(EmbeddingModel):
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
        
    def encode(self, text):
        return self.model.encode(text).tolist()

class EmbeddingFactory:
    @staticmethod
    def create_embedding_model(model_type="sentence_transformer"):
        if model_type == "sentence_transformer":
            return SentenceTransformerEmbedding('paraphrase-MiniLM-L6-v2')
        raise ValueError("Unknown embedding model type")

modelo_embeddings = EmbeddingFactory.create_embedding_model()

def gerar_embedding(texto):
    return modelo_embeddings.encode(texto)

sessions_memory = {}

def get_session_memory(session_id: str) -> ConversationBufferMemory:
    if session_id not in sessions_memory:
        # Cria uma nova memória para a sessão se não existir
        sessions_memory[session_id] = ConversationBufferMemory(memory_key="chat_history")
    return sessions_memory[session_id]

def gerar_resposta_llama(query, contexto, history):
    headers = {
        "Authorization": apillama,  # Token do OpenRouter
        "Content-Type": "application/json"
    }
    data = {
        "model": "meta-llama/llama-3.1-8b-instruct:free",
        "messages": [
            {
                "role": "system",
                "content": f"Contexto: {contexto}. Você é um agente da saúde e apenas pode responder coisas da área."
            },
            {
                "role": "user",
                "content": query 
            }
        ]
    }

    if history:
        data['messages'].insert(1, {
            "role": "system",
            "content": f"Histórico da conversa: {history}"
        })

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, data=json.dumps(data))
    
    if response.status_code == 200:
        data = response.json()
        return data['choices'][0]['message']['content']
    else:
        return None

def buscar_contexto_qdrant(query_embedding):
    search_result = qdrant_client.search(
        collection_name="iClinicBot", 
        query_vector=query_embedding,
        limit=3  # Limite de documentos mais próximos
    )
    # Combine os textos dos resultados para usar como contexto
    contexto = " ".join([res.payload.get('text', '') for res in search_result])
    return contexto

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    pergunta = data.get("mensagem_usuario")
    session_id = data.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())

    if not pergunta:
        return jsonify({"error": "Pergunta não fornecida"}), 400

    memory = get_session_memory(session_id)
    
    query_embedding = gerar_embedding(pergunta)
    contexto = buscar_contexto_qdrant(query_embedding)
    
    history = memory.load_memory_variables({}).get('chat_history', '')

    resposta = gerar_resposta_llama(pergunta, contexto, history)

    if resposta:
        memory.chat_memory.add_user_message(pergunta)
        memory.chat_memory.add_ai_message(resposta)

        return jsonify({"resposta": resposta, "session_id": session_id}), 200
    else:
        return jsonify({"error": "Não foi possível gerar uma resposta."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)