"""
Professor IA - Engenharia e Matematica (v14-streamlit)
- Compativel com Streamlit Cloud (interface Streamlit, sem Gradio)
- Chave mestra (st.secrets) so inicializa a sessao; o processamento exige a chave PROPRIA do usuario, validada
- Modelo Gemini escolhido dinamicamente ao testar a chave (st.session_state['modelo_ativo']) com fallback automatico
- Leitura VISUAL do PDF: somente a pagina em foco e renderizada (cache LRU)
- PDF dividido em 4 aulas progressivas e dependentes
- TTS didatico (edge-tts): roteiro falado como professor na lousa
- Pasta de trabalho em /tmp (disco do Streamlit Cloud e efemero)
"""
from __future__ import annotations

import asyncio
import base64
import io
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

import requests
import streamlit as st
from pypdf import PdfReader, PdfWriter

try:
    import edge_tts
except ImportError:
    edge_tts = None

try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz
    except ImportError:
        fitz = None

VERSAO = "v14-streamlit"

st.set_page_config(
    page_title=f"Professor IA {VERSAO}",
    page_icon="🎓",
    layout="wide",
)

# ----------------------------------------------------------------------------
# Diretorios (sempre em /tmp: o disco do app nao e confiavel para escrita)
# ----------------------------------------------------------------------------
LIB = Path(tempfile.gettempdir()) / "biblioteca_professor_ia"
LIB.mkdir(parents=True, exist_ok=True)

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"

N_AULAS = 4
LIMITE_AULA_GEMINI = 100_000
LIMITE_RESUMO = 2_500
MAX_MENSAGENS_HISTORICO = 8

# --- Leitura visual da pagina em foco ---
DPI_VISUAL = 110
QUALIDADE_JPEG = 70
MAX_LADO_PX = 1400
USAR_CINZA = True
RENDER_TIMEOUT_SEG = 10
CACHE_PAGINAS_MAX = 8
MAX_BYTES_PAGINA_VISUAL = 2 * 1024 * 1024
MAX_BYTES_FALLBACK_PDF = 1_500_000

# --- TTS ---
VOZ_TTS = "pt-BR-AntonioNeural"
TAXA_TTS = "-10%"
PITCH_TTS = "+0Hz"
LIMITE_BLOCO_TTS = 3_000
CONCORRENCIA_TTS = 3
IDADE_MAX_ARQUIVOS_SEG = 3600

PROMPT_SISTEMA = (
    "Você é um Professor Particular Universal de Ciências Exatas (Matemática, Física, "
    "Química, Engenharia, Circuitos Elétricos, Mecânica, Estatística, Cálculo, Álgebra etc.). "
    "Fale sempre em português do Brasil, com didática, precisão e estrutura clara.\n\n"
    "REGRA INVIOLÁVEL — MÉTODO PASSO A PASSO:\n"
    "Para QUALQUER problema, cálculo, expressão numérica, equação, circuito, diagrama de forças, "
    "derivada, integral, limite, sistema linear, conversão de unidades, etc., você NUNCA deve "
    "entregar apenas a resposta final, o resultado numérico ou um resumo. É OBRIGATÓRIO:\n"
    "  1) Identificar o que é dado e o que é pedido.\n"
    "  2) Escolher e enunciar o método / lei / fórmula (Lei de Ohm, Kirchhoff, 2ª Lei de Newton, "
    "regra da cadeia, Bhaskara...).\n"
    "  3) MONTAR AS EQUAÇÕES explicitamente, com variáveis e unidades (LaTeX $...$ ou $$...$$).\n"
    "  4) Resolver PASSO A PASSO, mostrando cada manipulação algébrica e substituição numérica.\n"
    "  5) Conferir dimensionalmente.\n"
    "  6) Só então apresentar a resposta final destacada.\n"
    "  7) Fechar com 'como você resolveria sozinho na próxima vez'.\n\n"
    "Se o aluno pedir apenas 'a resposta', conduza o raciocínio antes de dá-la. "
    "Se enviar a própria resolução, corrija item por item, aponte onde errou e mostre o "
    "caminho correto.\n\n"
    "Ao analisar circuitos em imagens, PRIMEIRO liste componentes e valores legíveis e descreva "
    "a topologia (série, paralelo, mista, malhas, nós) ANTES de calcular. Nunca invente valores "
    "ilegíveis: diga qual não ficou claro e peça confirmação.\n\n"
    "Se houver contexto de um documento (PDF/aula), baseie-se nele e diga quando algo não estiver "
    "disponível no texto."
)

PROMPT_RESUMO = (
    "Gere um RESUMO DE MEMÓRIA desta aula (máx. 250 palavras) para as próximas aulas: "
    "definições, fórmulas (LaTeX), métodos, circuitos e valores relevantes, resultados numéricos, "
    "o que o aluno resolveu ou errou e o ponto onde o texto da aula terminou. "
    "Responda apenas com o resumo."
)


@st.cache_resource
def _caches() -> dict:
    """Caches globais que sobrevivem aos reruns do Streamlit."""
    return {"paginas": OrderedDict()}


_cache_paginas_visual: "OrderedDict[tuple, list]" = _caches()["paginas"]


def chave_padrao() -> str:
    """Chave dos Secrets do Streamlit Cloud ou da variavel de ambiente."""
    try:
        valor = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        valor = ""
    return str(valor or os.environ.get("GEMINI_API_KEY") or "").strip()


# ============================================================================
# MOTOR DE IA - Gemini
# ============================================================================
# Lista prioritaria de modelos. O primeiro que responder na validacao da chave vira o
# st.session_state["modelo_ativo"]; os demais servem de fallback automatico durante as aulas.
MODELOS_PRIORITARIOS = [
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-2.0-flash-exp",
    "gemini-1.0-pro",
]
MAX_MODELOS_TENTADOS = 6


def _classificar_http(r) -> str:
    """Traduz a resposta HTTP em um tipo de erro: chave | cota | modelo | http."""
    try:
        corpo = r.text[:500]
    except Exception:
        corpo = ""
    if r.status_code == 401:
        return "chave"
    if r.status_code in (400, 403) and ("API_KEY_INVALID" in corpo or "API key" in corpo):
        return "chave"
    if r.status_code == 429:
        return "cota"
    if r.status_code == 404:
        return "modelo"
    return "http"


def _pontuar_modelo(nome):
    """Pontuacao para modelos extras descobertos na API (None = descartar)."""
    if not nome.startswith("gemini"):
        return None
    if any(x in nome for x in ("embedding", "tts", "image", "vision", "live", "audio", "robotics", "aqa")):
        return None
    pontos = 0.0
    if "flash" in nome:
        pontos += 10
    if "lite" in nome:
        pontos -= 3
    if any(x in nome for x in ("preview", "exp", "thinking")):
        pontos -= 5
    if "latest" in nome:
        pontos += 2
    v = re.search(r"gemini-(\d+(?:\.\d+)?)", nome)
    if v:
        pontos += float(v.group(1))
    return pontos


def listar_modelos_api(api_key):
    """Retorna (nomes_com_generateContent | None, erro). None = nao foi possivel listar."""
    try:
        r = requests.get(
            f"{GEMINI_BASE}/models",
            headers={"x-goog-api-key": api_key},
            params={"pageSize": 100},
            timeout=15,
        )
    except requests.RequestException as e:
        print(f"[Gemini] não foi possível listar modelos: {e}")
        return None, "rede"
    if r.status_code != 200:
        erro = _classificar_http(r)
        print(f"[Gemini] listagem de modelos falhou: HTTP {r.status_code} ({erro})")
        return None, erro
    nomes = []
    for m in r.json().get("models", []):
        nome = m.get("name", "").replace("models/", "")
        if "generateContent" in m.get("supportedGenerationMethods", []):
            nomes.append(nome)
    return nomes, ""


def montar_lista_modelos(api_key):
    """
    Ordem de tentativa: modelos prioritarios que a chave realmente enxerga, depois outros
    modelos de texto descobertos na API. Se a listagem falhar, usa a lista prioritaria.
    Retorna (lista, erro).
    """
    disponiveis, erro = listar_modelos_api(api_key)
    if erro == "chave":
        return [], "chave"
    if disponiveis is None:
        return list(MODELOS_PRIORITARIOS), ""
    prioridade = [m for m in MODELOS_PRIORITARIOS if m in disponiveis]
    extras = [m for m in disponiveis if m not in prioridade and _pontuar_modelo(m) is not None]
    extras.sort(key=_pontuar_modelo, reverse=True)
    lista = (prioridade + extras)[:MAX_MODELOS_TENTADOS]
    return lista or list(MODELOS_PRIORITARIOS), ""


def _chamar_modelo(modelo, payload, api_key, timeout):
    """Uma chamada generateContent. Retorna (texto, erro): ('...', '') em sucesso."""
    url = f"{GEMINI_BASE}/models/{modelo}:generateContent"
    t0 = time.perf_counter()
    try:
        r = requests.post(
            url,
            headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
            json=payload,
            timeout=timeout,
        )
        if r.status_code != 200:
            erro = _classificar_http(r)
            print(f"[Gemini] {modelo}: HTTP {r.status_code} ({erro}) {r.text[:200]}")
            return None, erro
        print(f"[TEMPO] {modelo} respondeu em {time.perf_counter() - t0:.1f}s")
        res_json = r.json()
        if not (res_json.get("candidates") and res_json["candidates"][0].get("content")):
            return None, "vazio"
        partes = res_json["candidates"][0]["content"].get("parts", [])
        texto = "".join(p.get("text", "") for p in partes).strip()
        return (texto, "") if texto else (None, "vazio")
    except requests.Timeout as e:
        print(f"[Gemini] {modelo}: timeout: {e}")
        return None, "timeout"
    except requests.RequestException as e:
        print(f"[Gemini] {modelo}: erro de rede: {e}")
        return None, "rede"
    except (KeyError, IndexError, ValueError) as e:
        print(f"[Gemini] {modelo}: resposta inesperada/bloqueada: {e}")
        return None, "vazio"


def chamar_gemini(pergunta, contexto, historico, api_key, anexos=None):
    """
    Usa st.session_state['modelo_ativo'] e, se ele falhar, tenta automaticamente os proximos
    modelos da lista de fallback antes de desistir.
    Retorna SEMPRE (texto, tipo_de_erro):
      (str, "")  -> sucesso
      (None, e)  -> e em: chave | cota | modelo | http | timeout | rede | vazio
    """
    ss = st.session_state
    ativo = ss.get("modelo_ativo")
    lista = list(ss.get("modelos_fallback") or MODELOS_PRIORITARIOS)
    ordem = ([ativo] if ativo else []) + [m for m in lista if m != ativo]
    ordem = ordem[:MAX_MODELOS_TENTADOS]

    contents = []
    for msg in historico[-MAX_MENSAGENS_HISTORICO:]:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})
    texto_usuario = f"{contexto}Pergunta do aluno: {pergunta}" if contexto else pergunta
    partes_usuario = []
    for mime, dados, rotulo in anexos or []:
        if rotulo:
            partes_usuario.append({"text": rotulo})
        partes_usuario.append({
            "inlineData": {
                "mimeType": mime,
                "data": base64.b64encode(dados).decode("ascii"),
            }
        })
    partes_usuario.append({"text": texto_usuario})
    contents.append({"role": "user", "parts": partes_usuario})
    payload = {
        "systemInstruction": {"parts": [{"text": PROMPT_SISTEMA}]},
        "contents": contents,
    }

    ultimo_erro = "vazio"
    for modelo in ordem:
        texto, erro = _chamar_modelo(modelo, payload, api_key, 60 if anexos else 45)
        if texto:
            if modelo != ativo:
                print(f"[Gemini] fallback: '{modelo}' passou a ser o modelo ativo.")
                ss["modelo_ativo"] = modelo
            return texto, ""
        ultimo_erro = erro
        if erro == "chave":
            return None, "chave"  # chave invalida: trocar de modelo nao adianta
        print(f"[Gemini] '{modelo}' falhou ({erro}); tentando o próximo modelo...")
    return None, ultimo_erro


def motor_hibrido(pergunta, estado, historico, api_key, usar_visual=True):
    """Usa SOMENTE Gemini. Retorna (texto, ok, nota)."""
    notas = []
    chave = (api_key or "").strip()
    if not chave:
        return (
            "🔒 **Informe e valide a sua própria chave do Gemini.**\n\n"
            "Cole a chave no campo da barra lateral e clique em **Testar Chave**. "
            "Obtenha uma chave grátis em: https://aistudio.google.com/app/apikey",
            False,
            "",
        )

    tem_pdf = bool(estado.get("aulas")) and bool(estado.get("caminho_pdf"))
    anexos, desc_visual = [], ""
    if usar_visual and tem_pdf:
        anexos, desc_visual, nota_pdf = preparar_anexo_pagina_atual(estado)
        if nota_pdf:
            notas.append(nota_pdf)

    ctx = montar_contexto(estado, LIMITE_AULA_GEMINI, desc_visual if anexos else "")
    resultado, erro = chamar_gemini(pergunta, ctx, historico, chave, anexos or None)

    if not resultado and anexos and erro in ("http", "vazio"):
        print("[motor_hibrido] Repetindo chamada sem anexo visual...")
        ctx = montar_contexto(estado, LIMITE_AULA_GEMINI, "")
        resultado, erro = chamar_gemini(pergunta, ctx, historico, chave, None)
        if resultado:
            notas.append("⚠️ Não consegui enviar a página como imagem; respondi só com o texto.")

    if resultado:
        return resultado, True, " ".join(notas)

    mensagens = {
        "chave": "⚠️ A API recusou a chave informada. Gere/valide a chave novamente com **Testar Chave**.",
        "cota": ("⚠️ A cota da sua chave foi excedida em todos os modelos testados. "
                 "Aguarde um pouco ou use outra chave."),
        "modelo": ("⚠️ Nenhum dos modelos da lista está disponível para esta chave. "
                   "Clique em **Testar Chave** para reavaliar os modelos."),
        "timeout": ("⚠️ O Gemini demorou demais para responder. Tente novamente — ou desmarque "
                    "'leitura visual' para acelerar."),
        "http": "⚠️ A API do Gemini respondeu com erro em todos os modelos tentados. Tente novamente.",
        "rede": "⚠️ Não consegui falar com a API do Gemini. Verifique a conexão.",
    }
    return (
        mensagens.get(erro, "⚠️ Não consegui obter a resposta do Gemini. Tente novamente em alguns segundos."),
        False,
        "",
    )


# ============================================================================
# TESTE DA CHAVE GEMINI
# ============================================================================
def _indicador_html(cor, texto, detalhe=""):
    cores = {
        "verde": ("#22c55e", "rgba(34,197,94,0.12)"),
        "vermelho": ("#ef4444", "rgba(239,68,68,0.12)"),
        "cinza": ("#9ca3af", "rgba(156,163,175,0.10)"),
    }
    fg, bg = cores[cor]
    extra = (
        f'<div style="font-weight:400;font-size:0.8em;opacity:0.85;margin-top:4px">{detalhe}</div>'
        if detalhe
        else ""
    )
    return (
        f'<div style="padding:10px 14px;border-radius:10px;border:1px solid {fg};'
        f'background:{bg};color:{fg};font-weight:600">{texto}{extra}</div>'
    )


def indicador_neutro() -> str:
    if st.session_state.get("chave_mestra"):
        return _indicador_html(
            "cinza",
            "🔒 Chave mestra carregada (uso restrito)",
            "Ela só mantém o app aberto. Para usar o professor, insira e valide a sua própria chave.",
        )
    return _indicador_html(
        "cinza",
        "⚪ Nenhuma chave informada",
        "Cole a sua chave acima e clique em Testar Chave.",
    )


def _gerar_teste(modelo, api_key):
    """Teste real de geracao num modelo. Retorna '' se respondeu; senao o tipo de erro."""
    payload = {
        "contents": [{"role": "user", "parts": [{"text": "Responda apenas com a palavra: ok"}]}],
        "generationConfig": {"maxOutputTokens": 32},
    }
    try:
        r = requests.post(
            f"{GEMINI_BASE}/models/{modelo}:generateContent",
            headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
            json=payload,
            timeout=20,
        )
    except requests.Timeout:
        return "timeout"
    except requests.RequestException:
        return "rede"
    if r.status_code == 200:
        return ""
    return _classificar_http(r)


_ROTULO_ERRO = {
    "chave": "a API recusou a chave",
    "cota": "cota excedida",
    "modelo": "modelo indisponível",
    "timeout": "tempo esgotado",
    "rede": "sem conexão",
    "http": "erro da API",
}


def testar_chave_gemini(api_key):
    """
    Valida a chave do USUARIO testando a geracao de conteudo em cada modelo da lista.
    O primeiro que responder vira st.session_state['modelo_ativo'].
    Retorna (html_indicador, valida: bool).
    """
    ss = st.session_state
    ss["modelo_ativo"] = None
    ss["modelos_fallback"] = []
    api_key = (api_key or "").strip()
    if not api_key:
        return _indicador_html("cinza", "⚪ Cole uma chave para testar"), False
    mestra = chave_padrao()
    if mestra and api_key == mestra:
        return _indicador_html(
            "vermelho",
            "🔴 Esta é a chave mestra do app",
            "Use a sua própria chave para consumir o processamento.",
        ), False

    candidatos, erro = montar_lista_modelos(api_key)
    if erro == "chave":
        return _indicador_html("vermelho", "🔴 Chave inválida", "A API do Google recusou esta chave."), False

    falhas = []
    for modelo in candidatos:
        erro = _gerar_teste(modelo, api_key)
        if not erro:
            ss["modelo_ativo"] = modelo
            ss["modelos_fallback"] = candidatos
            print(f"[Gemini] modelo ativo: {modelo}")
            return _indicador_html(
                "verde",
                "🟢 Chave Gemini VÁLIDA e conectada!",
                f"Modelo ativo: {modelo}",
            ), True
        if erro == "chave":
            return _indicador_html("vermelho", "🔴 Chave inválida", "A API do Google recusou esta chave."), False
        falhas.append(f"{modelo}: {_ROTULO_ERRO.get(erro, erro)}")

    detalhe = "Nenhum modelo respondeu. " + "; ".join(falhas[:4]) if falhas else "Nenhum modelo disponível."
    return _indicador_html("vermelho", "🔴 Nenhum modelo respondeu com esta chave", detalhe), False


# ============================================================================
# PDF - extracao de texto e divisao em aulas
# ============================================================================
def extrair_paginas_pdf(caminho_pdf):
    """Somente TEXTO (nunca imagem). PyMuPDF e bem mais rapido que o pypdf."""
    t0 = time.perf_counter()
    if fitz is not None:
        try:
            with fitz.open(caminho_pdf) as doc:
                if doc.needs_pass and not doc.authenticate(""):
                    return []
                paginas = [(pg.get_text("text") or "") for pg in doc]
            print(f"[TEMPO] texto de {len(paginas)} páginas (PyMuPDF): {time.perf_counter() - t0:.2f}s")
            return paginas
        except Exception as e:
            print(f"[PDF] PyMuPDF falhou no texto ({e}); usando pypdf.")
    try:
        leitor = PdfReader(caminho_pdf)
        if leitor.is_encrypted:
            leitor.decrypt("")
    except Exception as e:
        print(f"[PDF] Erro ao ler: {e}")
        return []
    paginas = []
    for pagina in leitor.pages:
        try:
            paginas.append(pagina.extract_text() or "")
        except Exception:
            paginas.append("")
    return paginas


def _rotulo_paginas(ini, fim):
    return f"página {ini}" if ini == fim else f"páginas {ini}–{fim}"


def dividir_em_aulas(paginas, n=N_AULAS):
    paginas = [p.strip() for p in paginas]
    total = sum(len(p) for p in paginas)
    n = max(1, min(n, total // 400))
    if len(paginas) >= n:
        alvo = total / n
        grupos = []
        inicio = 0
        acumulado = 0
        for i, p in enumerate(paginas):
            acumulado += len(p)
            restantes = len(paginas) - (i + 1)
            faltando = n - len(grupos) - 1
            if len(grupos) < n - 1 and (
                restantes == faltando
                or (acumulado >= alvo * (len(grupos) + 1) and restantes >= faltando)
            ):
                grupos.append((inicio, i))
                inicio = i + 1
        grupos.append((inicio, len(paginas) - 1))
        return [
            {
                "texto": "\n\n".join(paginas[a:b + 1]).strip(),
                "pag_ini": a + 1,
                "pag_fim": b + 1,
                "rotulo": _rotulo_paginas(a + 1, b + 1),
                "modo": "paginas",
            }
            for a, b in grupos
        ]
    texto = "\n".join(paginas)
    cortes = [0]
    for k in range(1, n):
        pos = int(len(texto) * k / n)
        nl = texto.find("\n", pos)
        corte = nl if (nl != -1 and nl - pos <= 2000) else pos
        cortes.append(max(corte, cortes[-1]))
    cortes.append(len(texto))
    return [
        {
            "texto": texto[cortes[i]:cortes[i + 1]].strip(),
            "pag_ini": 1,
            "pag_fim": len(paginas),
            "rotulo": f"parte {i + 1} de {n} do texto",
            "modo": "texto",
        }
        for i in range(n)
    ]


# ============================================================================
# RENDERIZACAO DE UMA PAGINA (leitura visual)
# ============================================================================
def _renderizar_uma_pagina(caminho, numero_pagina):
    if fitz is None:
        return None
    if not Path(caminho).exists():
        print(f"[PyMuPDF] arquivo não existe: {caminho}")
        return None
    resultado = {"bytes": None, "erro": None}

    def _trabalho():
        t0 = time.perf_counter()
        try:
            with fitz.open(caminho) as doc:
                if doc.needs_pass:
                    doc.authenticate("")
                idx = numero_pagina - 1
                if idx < 0 or idx >= doc.page_count:
                    raise IndexError(f"página {numero_pagina} fora de [1..{doc.page_count}]")
                pagina = doc.load_page(idx)
                zoom = DPI_VISUAL / 72
                maior_lado = max(pagina.rect.width, pagina.rect.height) * zoom
                if maior_lado > MAX_LADO_PX:
                    zoom *= MAX_LADO_PX / maior_lado
                matriz = fitz.Matrix(zoom, zoom)
                pix = pagina.get_pixmap(
                    matrix=matriz,
                    colorspace=(fitz.csGRAY if USAR_CINZA else fitz.csRGB),
                    alpha=False,
                )
                try:
                    dados = pix.tobytes("jpeg", jpg_quality=QUALIDADE_JPEG)
                except TypeError:
                    dados = pix.tobytes("jpeg")
                pix = None
                if len(dados) > MAX_BYTES_PAGINA_VISUAL:
                    raise ValueError(f"JPEG grande demais: {len(dados) // 1024} KB")
                resultado["bytes"] = dados
                print(f"[PyMuPDF] página {numero_pagina} renderizada em "
                      f"{time.perf_counter() - t0:.2f}s ({len(dados) // 1024} KB)")
        except Exception as e:
            resultado["erro"] = e
            print(f"[PyMuPDF] erro ao renderizar página {numero_pagina}: {e}")

    t = threading.Thread(target=_trabalho, daemon=True)
    t.start()
    t.join(timeout=RENDER_TIMEOUT_SEG)
    if t.is_alive():
        print(f"[PyMuPDF] TIMEOUT ({RENDER_TIMEOUT_SEG}s) na página {numero_pagina}; abortando.")
        return None
    if resultado["erro"]:
        return None
    return resultado["bytes"]


def _anexos_da_pagina(caminho, numero_pagina):
    """Lazy: renderiza SO esta pagina, apenas uma vez (cache LRU, inclusive de falhas)."""
    chave = (caminho, numero_pagina)
    if chave in _cache_paginas_visual:
        _cache_paginas_visual.move_to_end(chave)
        return _cache_paginas_visual[chave]
    dados = _renderizar_uma_pagina(caminho, numero_pagina)
    anexos = []
    if dados is not None:
        anexos = [("image/jpeg", dados, f"[Página {numero_pagina} do material]")]
    elif fitz is None:
        try:
            leitor = PdfReader(caminho)
            if leitor.is_encrypted:
                leitor.decrypt("")
            escritor = PdfWriter()
            escritor.add_page(leitor.pages[numero_pagina - 1])
            buf = io.BytesIO()
            escritor.write(buf)
            dados_pdf = buf.getvalue()
            if len(dados_pdf) <= MAX_BYTES_FALLBACK_PDF:
                anexos = [("application/pdf", dados_pdf, f"[Página {numero_pagina} do material]")]
                print(f"[PDF] fallback: página {numero_pagina} como PDF ({len(dados_pdf) // 1024} KB).")
            else:
                print(f"[PDF] fallback pesado ({len(dados_pdf) // 1024} KB): sem imagem.")
        except Exception as e:
            print(f"[PDF] falha no fallback da página {numero_pagina}: {e}")
    _cache_paginas_visual[chave] = anexos
    while len(_cache_paginas_visual) > CACHE_PAGINAS_MAX:
        _cache_paginas_visual.popitem(last=False)
    return anexos


def preparar_anexo_pagina_atual(estado):
    aulas = estado.get("aulas") or []
    caminho = estado.get("caminho_pdf")
    if not aulas or not caminho or not Path(caminho).exists():
        return [], "", ""
    aula = aulas[estado["atual"]]
    pag = estado.get("pagina_atual") or aula["pag_ini"]
    pag = max(aula["pag_ini"], min(pag, aula["pag_fim"]))
    anexos = _anexos_da_pagina(caminho, pag)
    if not anexos:
        return [], "", "⚠️ Não consegui renderizar esta página para leitura visual; usei só o texto."
    desc = f"a página {pag} do material original (página em foco nesta aula)"
    return anexos, desc, ""


def estado_vazio():
    return {
        "aulas": [],
        "resumos": [],
        "atual": 0,
        "fim": False,
        "nome": "",
        "caminho_pdf": "",
        "hist_inicio": 0,
        "pagina_atual": 1,
    }


def _garantir_estado(estado):
    base = estado_vazio()
    if isinstance(estado, dict):
        base.update(estado)
    return base


def _descartar_pdf_do_curso(estado):
    caminho = (estado or {}).get("caminho_pdf")
    if not caminho:
        return
    for chave in [k for k in _cache_paginas_visual if k[0] == caminho]:
        _cache_paginas_visual.pop(chave, None)
    try:
        Path(caminho).unlink(missing_ok=True)
    except OSError:
        pass


def montar_contexto(estado, limite_texto, visual_desc=""):
    aulas = estado.get("aulas") or []
    if not aulas:
        return ""
    i = estado["atual"]
    n = len(aulas)
    partes = [
        f"Você está conduzindo um curso em {n} aulas progressivas a partir do documento "
        f"'{estado.get('nome', '')}'. Aula atual: {i + 1} de {n}. "
        "Cada aula continua exatamente de onde a anterior parou."
    ]
    if i > 0:
        resumos = "\n\n".join(f"Aula {k + 1}: {r}" for k, r in enumerate(estado["resumos"][:i]))
        partes.append(f"Resumo acumulado das aulas anteriores:\n{resumos}")
        partes.append(f"Final do texto da aula anterior:\n{aulas[i - 1]['texto'][-800:]}")
    if visual_desc:
        partes.append(
            f"ANEXO VISUAL: o anexo (imagem) contém {visual_desc}. Analise o diagrama, esquema de "
            "circuito, gráficos ou tabela diretamente nessa imagem e leia os valores dos componentes "
            "(tensões, correntes, resistências etc.). O texto extraído abaixo pode estar incompleto; "
            "se um valor estiver ilegível, diga qual e peça confirmação em vez de inventar."
        )
    texto = aulas[i]["texto"]
    if len(texto) > limite_texto:
        texto = texto[:limite_texto] + "\n[... texto da aula truncado por limite de tamanho ...]"
    partes.append(f"Texto extraído da Aula {i + 1}:\n{texto}")
    return "\n\n".join(partes) + "\n\n"


def render_status(estado):
    aulas = estado.get("aulas") or []
    if not aulas:
        return "📭 **Nenhum PDF carregado.** Envie um PDF para dividi-lo em aulas, ou converse normalmente."
    n = len(aulas)
    atual = estado["atual"]
    linhas = []
    for i, a in enumerate(aulas):
        if i < atual or (i == atual and estado["fim"]):
            icone = "✅"
        elif i == atual:
            icone = "▶️"
        else:
            icone = "🔒"
        linhas.append(f"{icone} **Aula {i + 1}** — {a['rotulo']}")
    if estado["fim"]:
        titulo = f"### 🎉 Curso concluído ({n} de {n} aulas)"
    else:
        titulo = f"### 📚 Aula {atual + 1} de {n}"
    aula = aulas[atual]
    pag_foco = estado.get("pagina_atual") or aula["pag_ini"]
    pag_foco = max(aula["pag_ini"], min(pag_foco, aula["pag_fim"]))
    foco = f"👁️ **Página em foco:** {pag_foco} (de {aula['pag_ini']}–{aula['pag_fim']} nesta aula)"
    notas = [foco]
    tam = len(aula["texto"])
    if tam > LIMITE_AULA_GEMINI:
        notas.append(
            f"⚠️ Esta aula tem {tam:,} caracteres: a IA verá só os primeiros "
            f"{LIMITE_AULA_GEMINI:,}."
        )
    return f"{titulo}\n\n" + "\n\n".join(linhas) + "\n\n" + "\n\n".join(notas)


# ============================================================================
# TTS DIDATICO - roteiro falado como professor na lousa
# ============================================================================
_NUM_EXTENSO = {
    "0": "zero", "1": "um", "2": "dois", "3": "três", "4": "quatro",
    "5": "cinco", "6": "seis", "7": "sete", "8": "oito", "9": "nove",
    "10": "dez", "11": "onze", "12": "doze",
    "-1": "menos um", "-2": "menos dois", "-3": "menos três", "-4": "menos quatro",
    "-5": "menos cinco", "-6": "menos seis", "-7": "menos sete",
    "-8": "menos oito", "-9": "menos nove", "-10": "menos dez",
}

_LATEX_PALAVRAS = {
    "cdot": "vezes", "times": "vezes", "div": "dividido por",
    "Omega": "ohms", "omega": "ômega", "pi": "pi", "mu": "micro",
    "Delta": "delta", "delta": "delta", "theta": "teta", "alpha": "alfa",
    "beta": "beta", "phi": "fi", "lambda": "lambda", "sigma": "sigma", "tau": "tau",
    "approx": " aproximadamente ", "leq": " menor ou igual a ", "geq": " maior ou igual a ",
    "neq": " diferente de ", "pm": " mais ou menos ", "parallel": " em paralelo com ",
    "sum": " somatório de ", "int": " integral de ", "infty": " infinito ",
    "to": " tende a ", "rightarrow": " leva a ", "angle": " ângulo ", "circ": " graus ",
    "sin": " seno de ", "cos": " cosseno de ", "tan": " tangente de ",
    "log": " log de ", "ln": " logaritmo natural de ",
    "frac": " sobre ", "sqrt": " raiz quadrada de ",
    "qquad": " , ", "quad": " , ", "text": " ", "mathrm": " ",
}


def _num_falado(valor):
    v = (valor or "").strip()
    if v in _NUM_EXTENSO:
        return _NUM_EXTENSO[v]
    if v.startswith("-"):
        return f"menos {v[1:]}"
    return v


def _extrair_coeficientes_edo(texto_md):
    """
    Procura por A*y'' + B*y' + C*y = 0 e devolve os coeficientes.
    Termos ausentes (B ou C) viram zero. Aceita LaTeX (d²y/dx²).
    """
    t = texto_md.replace("$", " ").replace("\\,", " ").replace("\\!", " ")
    t = re.sub(r"\s+", " ", t).strip()
    d2 = r"(?:y\s*''|y\s*¨|y\s*′′|\\frac\{d\^?2\s*y\}\{d\s*x\^?2\})"
    d1 = r"(?:y\s*'|y\s*˙|y\s*′|\\frac\{d\s*y\}\{d\s*x\})"
    termo_a = (
        r"(?:(?P<a_neg>-\s*" + d2 + r")"
        r"|(?P<a_num>[+-]?\s*(?:\d+(?:[.,]\d+)?|\\?frac\{\d+\}\{\d+\}))?\s*" + d2 + r")"
    )
    termo_b = (
        r"(?:\s*(?P<b_sinal>[+-])\s*"
        r"(?:(?P<b_num>\d+(?:[.,]\d+)?|\\?frac\{\d+\}\{\d+\})?\s*" + d1 + r")"
        r")?"
    )
    termo_c = (
        r"(?:\s*(?P<c_sinal>[+-])\s*"
        r"(?:(?P<c_num>\d+(?:[.,]\d+)?|\\?frac\{\d+\}\{\d+\})?\s*[yY])"
        r")?"
    )
    padrao = re.compile(termo_a + termo_b + termo_c + r"\s*=\s*0", re.IGNORECASE)
    m = padrao.search(t)
    if not m:
        return None
    equacao_original = t[m.start():m.end()].strip()

    def _norm(num, sinal, forca_um=False):
        if num is None or num.strip() == "":
            if forca_um:
                return "-1" if sinal == "-" else "1"
            return "0"
        v = num.strip().replace(" ", "").replace("\\frac", "frac")
        if sinal == "-" and not v.startswith("-"):
            v = "-" + v
        return v

    if m.group("a_neg"):
        a = "-1"
    else:
        a = _norm(m.group("a_num"), None, forca_um=True)
    b = "0" if m.group("b_sinal") is None else _norm(m.group("b_num"), m.group("b_sinal"), forca_um=True)
    c = "0" if m.group("c_sinal") is None else _norm(m.group("c_num"), m.group("c_sinal"), forca_um=True)
    return {"A": a, "B": b, "C": c, "equacao": equacao_original}


def _equacao_falada(equacao):
    t = equacao
    t = re.sub(r"y\s*''|y\s*¨|y\s*′′", " ípsilon duas linhas ", t)
    t = re.sub(r"y\s*'|y\s*˙|y\s*′", " ípsilon linha ", t)
    t = re.sub(r"(?<![A-Za-z])y(?![A-Za-z])", " ípsilon ", t)
    t = t.replace("+", " mais ").replace("-", " menos ")
    t = t.replace("=", " igual a ")
    return re.sub(r"\s+", " ", t).strip()


def _rotulo_item(texto_md):
    m = re.search(r"\b(?:item|letra|exerc[ií]cio)\s+([A-Za-z])\b", texto_md, re.IGNORECASE)
    return f"item {m.group(1).upper()}" if m else ""


def gerar_roteiro_edo(texto_md, coefs):
    equacao_falada = _equacao_falada(coefs["equacao"])
    rotulo = _rotulo_item(texto_md)
    abertura = f"Vamos resolver agora o {rotulo}. " if rotulo else "Vamos resolver agora. "
    a_fala = _num_falado(coefs["A"])
    b_fala = _num_falado(coefs["B"])
    c_fala = _num_falado(coefs["C"])
    return (
        f"{abertura}Temos aqui a equação... {equacao_falada}. "
        "Primeiro, vamos achar os coeficientes A, B e C. "
        "O coeficiente A é o termo que acompanha a derivada segunda, ípsilon duas linhas. "
        "O coeficiente B é o que acompanha a derivada primeira, ípsilon linha. "
        "E o coeficiente C é o termo que acompanha a função ípsilon, sem derivada. "
        "Lembrando que, se algum desses termos não aparecer na equação, "
        "o valor dele será zero. "
        "Observando a nossa equação, temos: "
        f"o A é igual a... {a_fala}, "
        f"o B é igual a... {b_fala}, "
        f"e o C é igual a... {c_fala}. "
        "Guardando esses três valores, já podemos seguir para o próximo passo da resolução."
    )


def _latex_para_fala(f):
    f = re.sub(r"\\(?:text|mathrm|mathbf)\{([^{}]*)\}", r" \1 ", f)
    for _ in range(3):
        f = re.sub(r"\\d?frac\{([^{}]+)\}\{([^{}]+)\}", r" \1 sobre \2 ", f)
    f = re.sub(r"\\sqrt\{([^{}]*)\}", r" raiz quadrada de \1 ", f)
    f = re.sub(r"\^\{?2\}?(?!\w)", " ao quadrado ", f)
    f = re.sub(r"\^\{?3\}?(?!\w)", " ao cubo ", f)
    f = re.sub(r"\^\{([^{}]*)\}|\^(\w)", lambda m: f" elevado a {m.group(1) or m.group(2)} ", f)
    f = re.sub(r"\{([^{}]*)\}", r" \1 ", f)
    f = re.sub(r"\\([a-zA-Z]+)", lambda m: _LATEX_PALAVRAS.get(m.group(1), " , "), f)
    f = f.replace("=", " igual a ").replace("+", " mais ").replace("−", " menos ")
    f = re.sub(r"\s-\s", " menos ", f)
    f = re.sub(r"[{}\\]", " ", f)
    return f


def _limpar_para_fala_generica(texto):
    """Fallback: limpa markdown, converte LaTeX e insere pausas para soar natural."""
    t = re.sub(r"```.*?```", " ... trecho de código omitido ... ", texto, flags=re.S)
    t = re.sub(r"\$\$(.*?)\$\$", lambda m: " ... " + _latex_para_fala(m.group(1)) + " ... ", t, flags=re.S)
    t = re.sub(r"\$([^$\n]+)\$", lambda m: " " + _latex_para_fala(m.group(1)) + ", ", t)
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)
    t = re.sub(r"[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F]", "", t)
    t = re.sub(r"(?m)^\s*[-•]\s+", "", t)
    t = re.sub(r"\n\s*\n", " ... ", t)
    t = re.sub(r"(?<![.!?:;,])[ \t]*\n", ". ", t)
    t = re.sub(r"(?m)^\s*(\d+)[.)]\s+", r" passo \1, ", t)
    t = re.sub(r"\b(portanto|logo|então|assim|agora)\b", r" ... \1,", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+igual a\s+", " ... igual a ", t)
    t = re.sub(r"([;:])\s*", r"\1 ... ", t)
    t = re.sub(r"\b(\w+)(\s+\1\b)+", r"\1", t, flags=re.IGNORECASE)
    t = re.sub(r"[|~`*_#>]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"(,\s*){2,}", ", ", t)
    t = re.sub(r"(\s*\.\s*\.\s*\.\s*)+", " ... ", t)
    return t


def gerar_roteiro_falado(texto_md):
    """Roteiro falado COMPLETO. Em EDOs, abre com os coeficientes A, B e C."""
    geral = _limpar_para_fala_generica(texto_md)
    coefs = _extrair_coeficientes_edo(texto_md)
    if coefs:
        try:
            intro = gerar_roteiro_edo(texto_md, coefs)
            return f"{intro} ... ... Agora, vamos acompanhar a explicação completa, do começo. ... {geral}"
        except Exception as e:
            print(f"[TTS] falha ao gerar roteiro de EDO ({e}); usando apenas o texto completo.")
    return geral


def _dividir_texto_fala(texto, limite=LIMITE_BLOCO_TTS):
    frases = re.split(r"(?<=[.!?;:])\s+", texto)
    blocos = []
    atual = ""
    for f in frases:
        while len(f) > limite:
            corte = f.rfind(" ", 0, limite)
            corte = corte if corte > 0 else limite
            parte, f = f[:corte], f[corte:].lstrip()
            if atual:
                blocos.append(atual)
                atual = ""
            blocos.append(parte)
        if len(atual) + len(f) + 1 <= limite:
            atual = f"{atual} {f}".strip()
        else:
            if atual:
                blocos.append(atual)
            atual = f
    if atual:
        blocos.append(atual)
    return [b for b in blocos if b.strip()]


async def _sintetizar_bloco(texto):
    audio = bytearray()
    async for chunk in edge_tts.Communicate(texto, VOZ_TTS, rate=TAXA_TTS, pitch=PITCH_TTS).stream():
        if chunk["type"] == "audio":
            audio.extend(chunk["data"])
    return bytes(audio)


async def _sintetizar_blocos(blocos):
    sem = asyncio.Semaphore(CONCORRENCIA_TTS)

    async def um(bloco):
        async with sem:
            ultimo_erro = RuntimeError("não iniciado")
            for tentativa in range(3):
                try:
                    dados = await _sintetizar_bloco(bloco)
                    if dados:
                        return dados
                    ultimo_erro = RuntimeError("áudio vazio")
                except Exception as e:
                    ultimo_erro = e
                await asyncio.sleep(1.5 * (tentativa + 1))
            raise ultimo_erro

    partes = await asyncio.gather(*(um(b) for b in blocos))
    return b"".join(partes)


def limpar_arquivos_antigos():
    agora = time.time()
    for padrao, idade in (("resposta_*.mp3", IDADE_MAX_ARQUIVOS_SEG), ("curso_*.pdf", 24 * 3600)):
        for f in LIB.glob(padrao):
            try:
                if agora - f.stat().st_mtime > idade:
                    f.unlink()
            except OSError:
                pass


def gerar_audio_resposta(texto):
    """Gera um unico MP3 com a resposta COMPLETA. Retorna (caminho_mp3, mensagem_de_erro)."""
    if edge_tts is None:
        return None, "Áudio indisponível: instale o pacote edge-tts."
    roteiro = gerar_roteiro_falado(texto)
    if not roteiro:
        return None, ""
    blocos = _dividir_texto_fala(roteiro)
    if not blocos:
        return None, ""
    limpar_arquivos_antigos()
    caminho = LIB / f"resposta_{uuid.uuid4().hex}.mp3"
    resultado = {}

    def rodar():
        try:
            resultado["audio"] = asyncio.run(_sintetizar_blocos(blocos))
        except Exception as e:
            resultado["erro"] = e

    t = threading.Thread(target=rodar, daemon=True)
    t.start()
    t.join(timeout=300 + 90 * len(blocos))
    if t.is_alive():
        return None, "O áudio demorou demais. A resposta está no texto."
    if "erro" in resultado or not resultado.get("audio"):
        print(f"[TTS] erro: {resultado.get('erro')}")
        return None, "Falha ao gerar o áudio. A resposta está no texto."
    caminho.write_bytes(resultado["audio"])
    return str(caminho), ""


# ============================================================================
# ACOES (logica pura; a interface Streamlit so chama estas funcoes)
# ============================================================================
def _finalizar_resposta(resposta, ok, nota, gerar_audio):
    caminho_audio = None
    aviso_audio = ""
    if ok and gerar_audio:
        t0 = time.perf_counter()
        caminho_audio, erro = gerar_audio_resposta(resposta)
        print(f"[TEMPO] áudio (TTS) gerado em {time.perf_counter() - t0:.1f}s")
        if erro:
            aviso_audio = f"🔇 {erro}"
    extras = " ".join(x for x in (nota, aviso_audio) if x)
    texto_exibido = resposta + (f"\n\n_{extras}_" if extras else "")
    return texto_exibido, caminho_audio


def processar_pdf(caminho, estado):
    """Le o PDF, divide em aulas e devolve (novo_estado, aviso_ou_vazio)."""
    _descartar_pdf_do_curso(estado)
    novo = estado_vazio()
    if not caminho:
        return novo, ""
    paginas = extrair_paginas_pdf(caminho)
    if not paginas:
        return novo, "⚠️ Não consegui abrir este PDF (arquivo corrompido ou protegido por senha)."
    total = sum(len(p.strip()) for p in paginas)
    if total < 50:
        return novo, (
            "⚠️ Quase não há texto neste PDF (provavelmente escaneado ou só imagens). "
            "A divisão em aulas precisa de algum texto."
        )
    destino = LIB / f"curso_{uuid.uuid4().hex}.pdf"
    try:
        shutil.copyfile(caminho, destino)
        novo["caminho_pdf"] = str(destino)
    except OSError as e:
        print(f"[PDF] não consegui copiar: {e}")
    novo["aulas"] = dividir_em_aulas(paginas, N_AULAS)
    novo["nome"] = Path(caminho).name
    novo["pagina_atual"] = novo["aulas"][0]["pag_ini"] if novo["aulas"] else 1
    return novo, ""


def navegar_pagina(direcao, estado):
    aulas = estado.get("aulas") or []
    if not aulas:
        return estado
    aula = aulas[estado["atual"]]
    pag = estado.get("pagina_atual") or aula["pag_ini"]
    estado["pagina_atual"] = max(aula["pag_ini"], min(pag + direcao, aula["pag_fim"]))
    return estado


def responder_usuario(mensagem, historico, estado, api_key, gerar_audio=True, usar_visual=True):
    """Retorna (historico_atualizado, caminho_audio)."""
    mensagem = (mensagem or "").strip()
    if not mensagem:
        return historico, None
    hist_modelo = historico[estado.get("hist_inicio", 0):]
    resposta, ok, nota = motor_hibrido(mensagem, estado, hist_modelo, api_key, usar_visual)
    texto_exibido, caminho_audio = _finalizar_resposta(resposta, ok, nota, gerar_audio)
    historico.append({"role": "user", "content": mensagem})
    historico.append({"role": "assistant", "content": texto_exibido})
    return historico, caminho_audio


def acao_iniciar_aula(historico, estado, api_key, gerar_audio=True, usar_visual=True):
    if not estado.get("aulas"):
        historico.append({"role": "assistant", "content": "📭 Envie um PDF primeiro para eu montar as aulas."})
        return historico, None
    if estado["fim"]:
        historico.append({
            "role": "assistant",
            "content": "🎉 Você já concluiu todas as aulas. Clique em **🔄 Reiniciar curso** para recomeçar.",
        })
        return historico, None
    i = estado["atual"]
    n = len(estado["aulas"])
    pag = estado.get("pagina_atual") or estado["aulas"][i]["pag_ini"]
    pedido = f"Inicie a Aula {i + 1} de {n}, com foco principal na página {pag} do material. "
    if i > 0:
        pedido += ("Esta aula CONTINUA exatamente de onde a anterior parou (use o resumo e o final "
                   "da aula anterior fornecidos no contexto), sem repetir o que já foi ensinado. ")
    pedido += (
        "Estruture assim: 1) objetivo da aula; 2) conceitos e fórmulas principais desta aula, "
        "descrevendo os esquemas e circuitos das páginas (componentes e valores lidos); "
        "3) um exemplo resolvido PASSO A PASSO do zero (dados → método → equações → álgebra → "
        "resultado), demonstrando o raciocínio completo; "
        "4) proponha 3 exercícios de cálculo para eu resolver agora; "
        "5) termine pedindo que eu envie minhas resoluções. "
        "Explique apenas o conteúdo desta aula, sem adiantar as próximas."
    )
    hist_modelo = historico[estado["hist_inicio"]:]
    resposta, ok, nota = motor_hibrido(pedido, estado, hist_modelo, api_key, usar_visual)
    texto_exibido, caminho_audio = _finalizar_resposta(resposta, ok, nota, gerar_audio)
    historico.append({"role": "user", "content": f"▶️ Explicar a Aula {i + 1} de {n} (foco na página {pag})"})
    historico.append({"role": "assistant", "content": texto_exibido})
    return historico, caminho_audio


def acao_concluir_aula(historico, estado, api_key, usar_visual=True):
    if not estado.get("aulas"):
        historico.append({"role": "assistant", "content": "📭 Envie um PDF primeiro para eu montar as aulas."})
        return historico
    if estado["fim"]:
        historico.append({"role": "assistant", "content": "🎉 O curso já está concluído."})
        return historico
    i = estado["atual"]
    n = len(estado["aulas"])
    hist_modelo = historico[estado["hist_inicio"]:]
    resumo, ok, _nota = motor_hibrido(PROMPT_RESUMO, estado, hist_modelo, api_key, False)
    if not ok:
        t = estado["aulas"][i]["texto"]
        resumo = f"{t[:700]} [...] {t[-500:]}" if len(t) > 1200 else t
    estado["resumos"] = estado["resumos"][:i] + [resumo[:LIMITE_RESUMO]]
    if i + 1 < n:
        estado["atual"] = i + 1
        estado["pagina_atual"] = estado["aulas"][i + 1]["pag_ini"]
        msg = (
            f"✅ **Aula {i + 1} concluída!**\n\nResumo salvo para as próximas aulas:\n{resumo}\n\n"
            f"➡️ Agora você está na **Aula {i + 2} de {n}**. "
            "Clique em **▶️ Explicar aula atual** quando quiser continuar."
        )
    else:
        estado["fim"] = True
        msg = (
            f"🎉 **Aula {n} concluída — você terminou todo o material!**\n\n"
            f"Resumo da última aula:\n{resumo}"
        )
    historico.append({"role": "user", "content": f"✅ Concluir a Aula {i + 1}"})
    historico.append({"role": "assistant", "content": msg})
    estado["hist_inicio"] = len(historico)
    return historico


def acao_reiniciar_curso(historico, estado):
    if not estado.get("aulas"):
        return historico
    estado["atual"] = 0
    estado["resumos"] = []
    estado["fim"] = False
    estado["pagina_atual"] = estado["aulas"][0]["pag_ini"]
    historico.append({"role": "assistant", "content": "🔄 Curso reiniciado. Você está de volta à **Aula 1**."})
    estado["hist_inicio"] = len(historico)
    return historico


# ============================================================================
# INTERFACE STREAMLIT
# ============================================================================
CSS = """
<style>
:root {
    color-scheme: dark;
    --gelo-1: #1e2530;
    --gelo-2: #263238;
    --gelo-borda: #4a5a66;
    --texto: #e6eef3;
    --texto-suave: #b8c7d1;
    --vinho: #7a2441;
    --vinho-hover: #a8325a;
    --vinho-ativo: #5c1229;
}
html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background-color: #0b1210 !important;
    color: var(--texto);
}
[data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child {
    background-color: #121a20 !important;
    border-right: 1px solid #2a3640;
}
footer { visibility: hidden; }

/* ---------- Rotulos e textos legiveis ---------- */
label, label p, label span,
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p,
[data-testid="stCheckbox"] label, [data-testid="stCheckbox"] p,
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li,
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
[data-testid="stSpinner"] * {
    color: var(--texto) !important;
}
[data-testid="stCaptionContainer"], small, [data-testid="stSidebar"] small {
    color: var(--texto-suave) !important;
}
[data-testid="stTooltipIcon"] svg { color: var(--texto-suave) !important; }

/* ---------- Inputs, selects e textareas (gelo escuro) ---------- */
[data-baseweb="input"], [data-baseweb="textarea"], [data-baseweb="select"] > div {
    background-color: var(--gelo-1) !important;
    border: 1px solid var(--gelo-borda) !important;
    border-radius: 8px !important;
}
[data-baseweb="base-input"], [data-baseweb="base-input"] input,
[data-testid="stSidebar"] input, [data-testid="stSidebar"] textarea {
    background-color: var(--gelo-1) !important;
    color: var(--texto) !important;
    -webkit-text-fill-color: var(--texto) !important;
    caret-color: #ffffff;
}
input::placeholder, textarea::placeholder {
    color: #93a4af !important;
    -webkit-text-fill-color: #93a4af !important;
    opacity: 1 !important;
}
[data-baseweb="input"]:focus-within, [data-baseweb="textarea"]:focus-within {
    border-color: var(--vinho-hover) !important;
    box-shadow: 0 0 0 1px var(--vinho-hover) !important;
}

/* ---------- Upload de arquivo ---------- */
[data-testid="stFileUploaderDropzone"] {
    background-color: var(--gelo-1) !important;
    border: 1px dashed var(--gelo-borda) !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploaderDropzone"] *, [data-testid="stFileUploaderFile"] * {
    color: #c9d6de !important;
}

/* ---------- Caixas de selecao ---------- */
[data-testid="stCheckbox"] label { color: var(--texto) !important; }

/* ---------- Campo de chat ---------- */
[data-testid="stBottom"], [data-testid="stBottom"] > div { background-color: #0b1210 !important; }
[data-testid="stChatInput"], [data-testid="stChatInput"] > div {
    background-color: var(--gelo-1) !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"] textarea {
    background-color: var(--gelo-1) !important;
    color: var(--texto) !important;
    -webkit-text-fill-color: var(--texto) !important;
}

/* ---------- Botoes: gelo escuro; vinho ao passar o mouse / clicar ---------- */
.stButton > button, [data-testid="stFileUploaderDropzone"] button {
    background-color: var(--gelo-2) !important;
    color: var(--texto) !important;
    border: 1px solid var(--gelo-borda) !important;
    border-radius: 8px !important;
    transition: background-color .15s ease, border-color .15s ease, transform .05s ease;
}
.stButton > button p, .stButton > button span { color: inherit !important; }
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] {
    background-color: var(--vinho) !important;
    border-color: var(--vinho-hover) !important;
    color: #ffffff !important;
}
.stButton > button:hover, [data-testid="stFileUploaderDropzone"] button:hover {
    background-color: var(--vinho-hover) !important;
    border-color: #d65b84 !important;
    color: #ffffff !important;
}
.stButton > button:active, .stButton > button:focus:not(:active),
[data-testid="stFileUploaderDropzone"] button:active {
    background-color: var(--vinho-ativo) !important;
    border-color: #d65b84 !important;
    color: #ffffff !important;
    transform: scale(0.98);
}
/* botao do "olho" do campo de senha: sem fundo de botao */
[data-baseweb="input"] button, [data-baseweb="input"] button:hover {
    background-color: transparent !important;
    border: none !important;
    color: var(--texto-suave) !important;
}
[data-baseweb="input"] svg { fill: var(--texto-suave) !important; color: var(--texto-suave) !important; }

/* ---------- Alertas ---------- */
[data-testid="stAlert"] { border-radius: 10px !important; }
[data-testid="stAlert"] * { color: #f6eef1 !important; }

/* ---------- Quadro negro: mensagens do chat ---------- */
[data-testid="stChatMessage"] {
    background: #0f3b25 !important;
    border: 3px solid #8b5a2b;
    border-radius: 12px;
    box-shadow: inset 0 0 40px rgba(0,0,0,.45);
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] .katex { color: #f6faf4 !important; font-size: 1.05rem; line-height: 1.7; }
[data-testid="stChatMessage"] h1,
[data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3 { color: #ffe9a8 !important; }
[data-testid="stChatMessage"] strong { color: #ffffff !important; }
[data-testid="stChatMessage"] code { color: #ffe9a8 !important; background: rgba(255,255,255,.10) !important; }
</style>
"""


def inicializar_sessao():
    st.session_state.setdefault("estado", estado_vazio())
    st.session_state.setdefault("historico", [])
    st.session_state.setdefault("audio_atual", None)
    st.session_state.setdefault("audio_tocar", False)
    st.session_state.setdefault("indicador", None)
    st.session_state.setdefault("pdf_id", None)
    st.session_state.setdefault("aviso_pdf", "")
    # Chave mestra: carregada em silencio so para a sessao abrir; NUNCA usada no processamento.
    st.session_state.setdefault("chave_mestra", chave_padrao())
    st.session_state.setdefault("chave_validada", "")
    st.session_state.setdefault("modelo_ativo", None)
    st.session_state.setdefault("modelos_fallback", [])


def _guardar_audio(caminho):
    if caminho:
        st.session_state["audio_atual"] = caminho
        st.session_state["audio_tocar"] = True


MSG_CHAVE_OBRIGATORIA = (
    "🔒 Para consumir o processamento, insira **a sua própria chave** do Gemini na barra lateral "
    "e clique em **Testar Chave** até ela ser validada. A chave mestra do app não é usada "
    "nesta operação."
)


def _invalidar_chave():
    """Ao editar a caixinha, a validacao anterior deixa de valer."""
    st.session_state["indicador"] = None
    st.session_state["chave_validada"] = ""
    st.session_state["modelo_ativo"] = None
    st.session_state["modelos_fallback"] = []


def chave_propria_valida(chave: str) -> bool:
    """True somente se o usuario digitou uma chave E ela foi validada pelo botao Testar Chave."""
    chave = (chave or "").strip()
    return bool(chave) and chave == st.session_state.get("chave_validada", "")


def main():
    inicializar_sessao()
    ss = st.session_state
    st.markdown(CSS, unsafe_allow_html=True)

    st.title(f"🎓 Professor IA — Tutor Universal de Exatas {VERSAO}")
    st.caption(
        "Assistente (Gemini) que ensina passo a passo, nunca só a resposta. "
        "Envie um PDF e estude em 4 aulas progressivas, com leitura visual da página em foco."
    )

    # ---------------- Barra lateral ----------------
    with st.sidebar:
        chave = st.text_input(
            "🔒 Chave API do Gemini (sua chave pessoal)",
            type="password",
            placeholder="Cole sua API Key aqui...",
            key="chave_usuario",
            on_change=_invalidar_chave,
        )
        if st.button("🔑 Testar Chave", use_container_width=True):
            with st.spinner("Testando chave..."):
                html_ind, valida = testar_chave_gemini(chave)
            ss["indicador"] = html_ind
            ss["chave_validada"] = chave.strip() if valida else ""
        st.markdown(ss["indicador"] or indicador_neutro(), unsafe_allow_html=True)

        arquivo = st.file_uploader("Biblioteca (upload de PDF de aula/livro)", type=["pdf"])
        if arquivo is not None:
            id_arquivo = f"{arquivo.name}:{arquivo.size}"
            if ss["pdf_id"] != id_arquivo:
                tmp = LIB / f"upload_{uuid.uuid4().hex}.pdf"
                tmp.write_bytes(arquivo.getvalue())
                try:
                    with st.spinner("Lendo o PDF e montando as aulas..."):
                        novo, aviso = processar_pdf(str(tmp), ss["estado"])
                finally:
                    tmp.unlink(missing_ok=True)
                novo["nome"] = arquivo.name
                novo["hist_inicio"] = len(ss["historico"])
                ss["estado"] = novo
                ss["aviso_pdf"] = aviso
                ss["pdf_id"] = id_arquivo
        elif ss["pdf_id"] is not None:
            _descartar_pdf_do_curso(ss["estado"])
            ss["estado"] = estado_vazio()
            ss["pdf_id"] = None
            ss["aviso_pdf"] = ""

        if ss["aviso_pdf"]:
            st.warning(ss["aviso_pdf"])

        slot_status = st.empty()

        c1, c2 = st.columns(2)
        if c1.button("◀️ Página", use_container_width=True):
            navegar_pagina(-1, ss["estado"])
        if c2.button("Página ▶️", use_container_width=True):
            navegar_pagina(+1, ss["estado"])

        usar_visual = st.checkbox(
            "👁️ Leitura visual da página em foco",
            value=True,
            help="Envia SOMENTE a página em foco como imagem ao Gemini (rápido e econômico).",
        )
        gerar_audio = st.checkbox("🔊 Gerar áudio didático das respostas", value=True)

        # --- Operacoes pesadas: exigem chave PROPRIA validada ---
        if st.button("▶️ Explicar aula atual", type="primary", use_container_width=True):
            if not chave_propria_valida(chave):
                st.error(MSG_CHAVE_OBRIGATORIA)
            else:
                with st.spinner("O professor está preparando a aula..."):
                    ss["historico"], audio = acao_iniciar_aula(
                        ss["historico"], ss["estado"], chave.strip(), gerar_audio, usar_visual
                    )
                _guardar_audio(audio)

        if st.button("✅ Concluir aula e avançar", use_container_width=True):
            if not chave_propria_valida(chave):
                st.error(MSG_CHAVE_OBRIGATORIA)
            else:
                with st.spinner("Gerando resumo da aula..."):
                    ss["historico"] = acao_concluir_aula(
                        ss["historico"], ss["estado"], chave.strip(), usar_visual
                    )

        if st.button("🔄 Reiniciar curso", use_container_width=True):
            ss["historico"] = acao_reiniciar_curso(ss["historico"], ss["estado"])

        slot_audio = st.empty()

    # ---------------- Area principal: quadro negro ----------------
    for msg in ss["historico"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Digite sua dúvida ou sua resolução (Cálculo, Física, Circuitos, Estruturas...)")
    if prompt:
        if not chave_propria_valida(chave):
            st.error(MSG_CHAVE_OBRIGATORIA)
        else:
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("O professor está pensando..."):
                    ss["historico"], audio = responder_usuario(
                        prompt, ss["historico"], ss["estado"], chave.strip(), gerar_audio, usar_visual
                    )
                st.markdown(ss["historico"][-1]["content"])
            _guardar_audio(audio)

    # ---------------- Slots preenchidos por ultimo (refletem as acoes acima) ----------------
    slot_status.markdown(render_status(ss["estado"]))

    caminho_audio = ss["audio_atual"]
    if caminho_audio and Path(caminho_audio).exists():
        tocar = ss["audio_tocar"]
        ss["audio_tocar"] = False
        with slot_audio.container():
            st.caption("Explicação em áudio (narração de professor)")
            st.audio(Path(caminho_audio).read_bytes(), format="audio/mp3", autoplay=tocar)


main()
