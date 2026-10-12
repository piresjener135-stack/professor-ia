import streamlit as st
import google.generativeai as genai
from gtts import gTTS
from io import BytesIO
import pypdf
from PIL import Image

# ------------------ CONFIGURAÇÃO DA API ------------------
API_KEY = "SUA_CHAVE_GEMINI_AQUI" 
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# ------------------ EXTRAÇÃO DE PDF ------------------
def extrair_texto_pdf(arquivo):
    texto = ""
    try:
        with BytesIO(arquivo.read()) as pdf_bytes:
            reader = pypdf.PdfReader(pdf_bytes)
            for page in reader.pages:
                texto += page.extract_text() or ""
    except Exception as e:
        st.error(f"Erro ao ler PDF: {e}")
    return texto

# ------------------ GERADOR DE ÁUDIO ------------------
def gerar_audio_em_memoria(texto):
    try:
        texto_limpo = texto.replace("$$", "").replace("$", "").replace("#", "")
        fp = BytesIO()
        tts = gTTS(text=texto_limpo, lang="pt", tld="com.br", slow=False)
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp
    except Exception as e:
        st.error(f"Falha ao gerar áudio: {e}")
        return None

# ------------------ PROFESSOR VIRTUAL ------------------
def professor_virtual(pergunta, contexto_texto="", imagem=None):
    prompt_sistema = (
        "Você é um Professor de Engenharia especialista.\n"
        "Regras estritas:\n"
        "1. Mantenha frações como 1/2, nunca converta em decimal.\n"
        "2. Mostre passo a passo dos cálculos.\n"
        "3. Explique termos técnicos.\n"
        "4. Crie um exemplo novo para fixação.\n"
        "5. Use LaTeX para fórmulas: $$E = mc^2$$\n"
        "6. Responda de forma clara e objetiva."
    )
    if imagem:
        resposta = model.generate_content([prompt_sistema, pergunta, imagem])
    else:
        mensagem = f"{prompt_sistema}\n\nConteúdo do arquivo:\n{contexto_texto}\n\nPergunta: {pergunta}"
        resposta = model.generate_content(mensagem)
    return resposta.text

# ------------------ INTERFACE STREAMLIT ------------------
st.set_page_config(page_title="Professor IA", layout="wide")

st.title("🧑‍🏫 Professor IA - Engenharia")
st.caption("Envie seus arquivos/exercícios e tire suas dúvidas com explicação passo a passo.")

with st.sidebar:
    st.header("📁 Material do Aluno")
    arquivo = st.file_uploader("Envie PDF ou Imagem", type=["pdf", "png", "jpg", "jpeg"])
    chave_user = st.text_input("Sua API Key do Gemini (opcional)", type="password")
    if chave_user:
        genai.configure(api_key=chave_user)

if "historico" not in st.session_state:
    st.session_state.historico = []
if "contexto" not in st.session_state:
    st.session_state.contexto = ""
if "imagem" not in st.session_state:
    st.session_state.imagem = None

if arquivo is not None:
    if arquivo.type == "application/pdf":
        st.session_state.contexto = extrair_texto_pdf(arquivo)
        st.session_state.imagem = None
        st.sidebar.success("✅ PDF carregado com sucesso!")
    else:
        st.session_state.imagem = Image.open(arquivo)
        st.session_state.contexto = ""
        st.sidebar.success("✅ Imagem carregada com sucesso!")

for msg in st.session_state.historico:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])

pergunta = st.chat_input("Digite sua dúvida...")

if pergunta:
    st.session_state.historico.append({"role": "user", "content": pergunta})
    st.chat_message("user").write(pergunta)
    
    with st.spinner("Professor pensando..."):
        if st.session_state.imagem:
            resposta = professor_virtual(pergunta, imagem=st.session_state.imagem)
        else:
            resposta = professor_virtual(pergunta, contexto_texto=st.session_state.contexto)
        
        st.session_state.historico.append({"role": "assistant", "content": resposta})
        st.chat_message("assistant").write(resposta)
        
        audio_fp = gerar_audio_em_memoria(resposta)
        if audio_fp:
            st.audio(audio_fp, format="audio/mp3")
