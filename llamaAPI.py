from flask import Flask, request, jsonify
from qdrant_client import QdrantClient
import requests
import json
from dotenv import load_dotenv
import os
import uuid
from flask_cors import CORS
from langchain.memory import ConversationBufferMemory
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
# Permitir apenas as URLs específicas
CORS(app, resources={r"/*": {"origins": ["https://saudeai.netlify.app", "https://iclinicbot.netlify.app"]}})

# Carregar variáveis de ambiente
load_dotenv()
urlQdrant = os.getenv('DATABASE_URL')
apiQdrant = os.getenv('CHAVE_QDRANT')
apillama = os.getenv('CHAVE_LLAMA')
# portRender = os.getenv('PORT')
# hostRender = os.getenv('HOST')

# Inicializando o cliente do Qdrant
qdrant_client = QdrantClient(
    url=urlQdrant,
    api_key=apiQdrant
)

# Carregar o modelo de embeddings
modelo_embeddings = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Função para gerar o embedding da pergunta
def gerar_embedding(texto):
    return modelo_embeddings.encode(texto).tolist()

# Dicionário para armazenar a memória de conversas por sessão
sessions_memory = {}

# Função para obter ou criar o histórico de uma sessão
def get_session_memory(session_id: str) -> ConversationBufferMemory:
    if session_id not in sessions_memory:
        # Cria uma nova memória para a sessão se não existir
        sessions_memory[session_id] = ConversationBufferMemory(memory_key="chat_history")
    return sessions_memory[session_id]

# Função para gerar resposta do LLaMA diretamente
def gerar_resposta_llama(query, contexto, history):
    headers = {
        "Authorization": apillama,  # Token do OpenRouter
        "Content-Type": "application/json"
    }
    data = {
        "model": "meta-llama/llama-3.1-8b-instruct:free",  # Modelo a ser utilizado
        "messages": [
            {
                "role": "system",
                "content": f"Contexto: {contexto}. Você é um agente da saúde e apenas pode responder coisas da área."  # Contexto extraído do Qdrant
            },
            {
                "role": "user",
                "content": query  # Pergunta do usuário
            }
        ]
    }

    # Adicionando histórico da conversa ao prompt
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

# Função para buscar o contexto no Qdrant baseado nos embeddings
def buscar_contexto_qdrant(query_embedding):
    search_result = qdrant_client.search(
        collection_name="iClinicBot",  # Substitua pelo nome da sua coleção no Qdrant
        query_vector=query_embedding,
        limit=3  # Limite de documentos mais próximos
    )
    # Combine os textos dos resultados para usar como contexto
    contexto = " ".join([res.payload.get('text', '') for res in search_result])
    return contexto

# Endpoint para o chat
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    pergunta = data.get("mensagem_usuario")
    session_id = data.get("session_id")  # Identificador único para a sessão

    # Verificar se o session_id foi fornecido, caso contrário, gerar um novo
    if not session_id:
        session_id = str(uuid.uuid4())  # Gera um UUID para a nova sessão

    if not pergunta:
        return jsonify({"error": "Pergunta não fornecida"}), 400

    # Obter a memória da sessão atual
    memory = get_session_memory(session_id)
    
    # # Para esta versão ajustada, vamos usar um contexto genérico por enquanto
    # contexto = "Você é um agente da área da saúde"

    query_embedding = gerar_embedding(pergunta)
    contexto = buscar_contexto_qdrant(query_embedding)
    
    # Carregar o histórico da memória
    history = memory.load_memory_variables({}).get('chat_history', '')

    # Gerar resposta usando o LLaMA com o contexto adicional e histórico
    resposta = gerar_resposta_llama(pergunta, contexto, history)

    if resposta:
        # Adiciona a pergunta e resposta ao histórico de memória
        memory.chat_memory.add_user_message(pergunta)
        memory.chat_memory.add_ai_message(resposta)

        return jsonify({"resposta": resposta, "session_id": session_id}), 200
    else:
        return jsonify({"error": "Não foi possível gerar uma resposta."}), 500

# Executa a aplicação Flask
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
    # app.run(host=hostRender, port=portRender, debug=True)
