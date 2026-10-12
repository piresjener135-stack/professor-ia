"""
Professor IA - Engenharia e Matematica (v14-st)
- Versao STREAMLIT: mesma logica, layout e estrutura do v14 (Gradio).
- Leitura VISUAL do PDF: SOMENTE a pagina em foco e renderizada.
- Navegacao de paginas com cache LRU.
- Chave do Gemini lida de st.secrets (Streamlit Cloud) ou do campo.
- TTS DIDATICO: EDOs com coeficientes A, B e C; voz pt-BR-ThalitaNeural.
- Controle de velocidade da voz (0.75x a 1.5x).
"""
from __future__ import annotations

import asyncio
import base64
import hashlib as _hashlib
import io
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from pathlib import Path

import requests
import streamlit as st
from pypdf import PdfReader, PdfWriter

try:
    from gtts import gTTS
except ImportError:
    gTTS = None

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

VERSAO = "v14-st"

# ----------------------------------------------------------------------------
# Diretorios e configuracoes (mesma logica do Gradio, adaptada a /tmp)
# ----------------------------------------------------------------------------
BASE = Path(tempfile.gettempdir()) / "professor_ia"
LIB = BASE / "biblioteca"
LIB.mkdir(parents=True, exist_ok=True)

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"

N_AULAS = 4
LIMITE_AULA_GEMINI = 100_000
LIMITE_RESUMO = 2_500
MAX_MENSAGENS_HISTORICO = 8

DPI_VISUAL = 110
QUALIDADE_JPEG = 70
MAX_LADO_PX = 1400
USAR_CINZA = True
RENDER_TIMEOUT_SEG = 10
CACHE_PAGINAS_MAX = 8
MAX_BYTES_PAGINA_VISUAL = 2 * 1024 * 1024
MAX_BYTES_FALLBACK_PDF = 1_500_000

IDADE_MAX_ARQUIVOS_SEG = 3600

# ---- Audio ----------------------------------------------------------------
VOZ_EDGE = "pt-BR-ThalitaNeural"
LIMITE_CARACTERES_AUDIO = 9000
TAMANHO_PEDACO_GTTS = 350

MAPA_VELOCIDADE = {
    "0.75x (Lenta)":       "-20%",
    "1.0x (Normal)":       "+0%",
    "1.25x (Rápida)":      "+25%",
    "1.5x (Muito Rápida)": "+50%",
}
VELOCIDADE_PADRAO = "1.0x (Normal)"


# ----------------------------------------------------------------------------
# Chave do Gemini: tenta st.secrets, depois variável de ambiente
# ----------------------------------------------------------------------------
def _chave_dos_secrets() -> str:
    """Le GEMINI_API_KEY dos Secrets do Streamlit Cloud (fallback: env var)."""
    try:
        v = st.secrets.get("GEMINI_API_KEY", "")
        if v:
            return str(v).strip()
    except Exception:
        pass
    return (os.environ.get("GEMINI_API_KEY") or "").strip()


CHAVE_ENV = _chave_dos_secrets()


PROMPT_SISTEMA = (
    "Você é um Professor Particular Universal de Ciências Exatas (Matemática, Física, "
    "Química, Engenharia, Circuitos Elétricos, Mecânica, Estatística, Cálculo, Álgebra etc.). "
    "Fale sempre em português do Brasil, com didática, precisão e estrutura clara.\n\n"
    "REGRA INVIOLÁVEL — MÉTODO PASSO A PASSO:\n"
    "Para QUALQUER problema, cálculo, expressão numérica, equação, circuito, diagrama de forças, "
    "derivada, integral, limite, sistema linear, conversão de unidades, etc., você NUNCA deve "
    "entregar apenas a resposta final, o resultado numérico ou um resumo. É OBRIGATÓRIO:\n"
    "  1) Identificar o que é dado e o que é pedido.\n"
    "  2) Escolher e justificar o método / lei / fórmula (Lei de Ohm, Kirchhoff, 2ª Lei de Newton, "
    "     regra da cadeia, Bhaskara...).\n"
    "  3) MONTAR AS EQUAÇÕES explicitamente, com variáveis e unidades (LaTeX $...$ ou $$...$$).\n"
    "  4) Resolver PASSO A PASSO, mostrando cada manipulação algébrica e substituição numérica.\n"
    "  5) Conferir dimensionalmente.\n"
    "  6) Só então apresentar a resposta final destacada.\n"
    "  7) Fechar com 'como você resolveria sozinho da próxima vez'.\n\n"
    "Se o aluno pedir apenas 'a resposta', conduza o raciocínio antes de dá-la. "
    "Se enviar a própria resolução, corrija-a item por item, aponte onde errou e mostre o "
    "caminho correto.\n\n"
    "Ao analisar circuitos nas imagens, PRIMEIRO liste componentes e valores legíveis e descreva "
    "a topologia (série, paralelo, misto, malhas, nós) ANTES de calcular. Nunca invente valores "
    "ilegíveis: diga qual não ficou claro e peça confirmação.\n\n"
    "Se houver contexto de um documento (PDF/aula), baseie-se nele e diga quando algo não estiver "
    "no texto."
)

PROMPT_RESUMO = (
    "Gere um RESUMO DE MEMÓRIA desta aula (máx. 250 palavras) para as próximas aulas: "
    "definições, fórmulas (LaTeX), métodos, circuitos e valores relevantes, resultados numéricos, "
    "o que o aluno resolveu ou errou e o ponto onde o texto da aula terminou. "
    "Responda apenas com o resumo."
)

# ---- Caches em memoria (iguais ao v14) -------------------------------------
_cache_modelo_gemini: dict = {}
_cache_modelo_gemini_expira: dict = {}
_cache_paginas_visual: "OrderedDict[tuple, list]" = OrderedDict()
_CACHE_MODELOS_GEMINI_TTL_SEG = 300


def _id_cache_chave(api_key: str) -> str:
    return _hashlib.sha256((api_key or "").encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------
# Motores de IA (LOGICA IDENTICA AO v14 - so mudou o wrapper)
# ----------------------------------------------------------------------------
def _prioridade_modelo_gemini(nome: str) -> tuple:
    baixo = nome.lower()
    versao = re.search(r"gemini-(\d+)(?:\.(\d+))?", baixo)
    maior = int(versao.group(1)) if versao else 0
    menor = int(versao.group(2) or 0) if versao else 0
    experimental = any(x in baixo for x in ("preview", "exp", "experimental"))
    flash = "flash" in baixo
    lite = "lite" in baixo
    return (not experimental, maior, menor, flash and not lite, flash, nome)


def listar_modelos_gemini(api_key: str) -> list:
    id_chave = _id_cache_chave(api_key)
    agora = time.monotonic()
    cache = _cache_modelo_gemini.get(id_chave)
    if cache and agora < _cache_modelo_gemini_expira.get(id_chave, 0):
        return list(cache)
    candidatos = set()
    page_token = None
    tokens_vistos = set()
    try:
        while True:
            params = {"pageSize": 100}
            if page_token:
                params["pageToken"] = page_token
            r = requests.get(
                f"{GEMINI_BASE}/models",
                headers={"x-goog-api-key": api_key},
                params=params,
                timeout=15,
            )
            r.raise_for_status()
            dados = r.json()
            for modelo in dados.get("models", []):
                nome = str(modelo.get("name", "")).removeprefix("models/")
                metodos = modelo.get("supportedGenerationMethods", [])
                if "generateContent" not in metodos or not nome.startswith("gemini"):
                    continue
                if any(x in nome.lower() for x in (
                    "embedding", "tts", "image", "vision", "live", "audio",
                    "robotics", "transcribe",
                )):
                    continue
                candidatos.add(nome)
            page_token = dados.get("nextPageToken")
            if not page_token or page_token in tokens_vistos:
                break
            tokens_vistos.add(page_token)
    except (requests.RequestException, ValueError, TypeError, AttributeError) as e:
        print(f"[Gemini] Falha ao listar modelos: {type(e).__name__}: {e}")
    lista = sorted(candidatos, key=_prioridade_modelo_gemini, reverse=True)
    if cache:
        ultimo = cache[0]
        if ultimo in lista:
            lista.remove(ultimo)
            lista.insert(0, ultimo)
    if lista:
        _cache_modelo_gemini[id_chave] = lista
        _cache_modelo_gemini_expira[id_chave] = time.monotonic() + _CACHE_MODELOS_GEMINI_TTL_SEG
    else:
        _cache_modelo_gemini.pop(id_chave, None)
        _cache_modelo_gemini_expira.pop(id_chave, None)
    return lista


def detectar_modelo_gemini(api_key: str) -> str:
    modelos = listar_modelos_gemini(api_key)
    return modelos[0] if modelos else ""


def _pausa_apos_429(response) -> None:
    try:
        segundos = float(response.headers.get("Retry-After", "0.8"))
    except (TypeError, ValueError):
        segundos = 0.8
    segundos = min(max(segundos, 0.4), 2.0)
    time.sleep(segundos)


def chamar_gemini(pergunta, contexto, historico, api_key, anexos=None):
    modelos = listar_modelos_gemini(api_key)
    if not modelos:
        return None, "models"
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
    erros = []
    for modelo in modelos:
        try:
            r = requests.post(
                f"{GEMINI_BASE}/models/{modelo}:generateContent",
                headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
                json=payload,
                timeout=60 if anexos else 45,
            )
            if r.status_code == 404:
                erros.append("not_found")
                continue
            if r.status_code == 429:
                erros.append("quota")
                _pausa_apos_429(r)
                continue
            if r.status_code == 503:
                erros.append("unavailable")
                time.sleep(0.4)
                continue
            if r.status_code in (401, 403):
                return None, "auth"
            if r.status_code >= 500:
                erros.append("unavailable")
                time.sleep(0.4)
                continue
            if r.status_code >= 400:
                erros.append("http")
                continue
            r.raise_for_status()
            partes = r.json().get("candidates", [{}])[0].get("content", {}).get("parts", [])
            texto = "".join(p.get("text", "") for p in partes).strip()
            if not texto:
                erros.append("vazio")
                continue
            _cache_modelo_gemini[_id_cache_chave(api_key)] = [modelo] + [m for m in modelos if m != modelo]
            _cache_modelo_gemini_expira[_id_cache_chave(api_key)] = (
                time.monotonic() + _CACHE_MODELOS_GEMINI_TTL_SEG
            )
            return texto, ""
        except requests.Timeout:
            erros.append("timeout")
        except requests.HTTPError:
            erros.append("http")
        except requests.RequestException:
            erros.append("rede")
        except (KeyError, IndexError, ValueError, TypeError):
            erros.append("vazio")

    if erros and all(e == "not_found" for e in erros):
        return None, "not_found"
    for k in ("unavailable", "quota", "timeout", "rede"):
        if k in erros:
            return None, k
    return None, erros[-1] if erros else "http"


def motor_hibrido(pergunta, estado, historico, api_key, usar_visual=True):
    if not (api_key and api_key.strip()):
        return ("📌 Chave do Gemini necessária. Cole sua chave no campo ou configure "
                "GEMINI_API_KEY nos Secrets.", False, "")
    notas = []
    tem_pdf = bool(estado.get("aulas")) and bool(estado.get("caminho_pdf"))
    chave = api_key.strip()
    anexos, desc_visual = [], ""
    if usar_visual and tem_pdf:
        anexos, desc_visual, nota_pdf = preparar_anexo_pagina_atual(estado)
        if nota_pdf:
            notas.append(nota_pdf)
    ctx = montar_contexto(estado, LIMITE_AULA_GEMINI, desc_visual if anexos else "")
    resultado, erro = chamar_gemini(pergunta, ctx, historico, chave, anexos or None)
    if not resultado and anexos and erro in ("http", "vazio"):
        ctx = montar_contexto(estado, LIMITE_AULA_GEMINI, "")
        resultado, erro = chamar_gemini(pergunta, ctx, historico, chave, None)
        if resultado:
            notas.append("⚠️ Não consegui enviar a página como imagem; respondi só com o texto.")
    if resultado:
        return resultado, True, " ".join(notas)
    motivos = {
        "quota": "O Gemini limitou todas as tentativas (erro 429). Aguarde e confira a cota do projeto.",
        "unavailable": "Os modelos estão temporariamente indisponíveis (erro 503). Tente novamente.",
        "not_found": "Os modelos listados não estão disponíveis (erro 404). Teste a chave para atualizar.",
        "models": "Não consegui descobrir modelos Gemini com suporte a generateContent.",
        "auth": "A chave foi recusada pela API (401/403). Confira a chave e as permissões.",
        "timeout": "O Gemini demorou demais para responder. Tente de novo.",
        "rede": "Sem conexão com o Gemini. Verifique a internet.",
    }
    return (f"⚠️ Não consegui obter resposta. {motivos.get(erro, 'Veja o terminal para detalhes.')}",
            False, "")


# ----------------------------------------------------------------------------
# PDF (LOGICA IDENTICA AO v14)
# ----------------------------------------------------------------------------
def extrair_paginas_pdf(caminho_pdf):
    if fitz is not None:
        try:
            with fitz.open(caminho_pdf) as doc:
                if doc.needs_pass and not doc.authenticate(""):
                    return []
                return [(pg.get_text("text") or "") for pg in doc]
        except Exception:
            pass
    try:
        reader = PdfReader(caminho_pdf)
        if reader.is_encrypted:
            reader.decrypt("")
    except Exception:
        return []
    paginas = []
    for pagina in reader.pages:
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
        inicio, acumulado = 0, 0
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
                "texto": "\n\n".join(paginas[a: b + 1]).strip(),
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
            "texto": texto[cortes[i]: cortes[i + 1]].strip(),
            "pag_ini": 1,
            "pag_fim": len(paginas),
            "rotulo": f"parte {i + 1} de {n} do texto",
            "modo": "texto",
        }
        for i in range(n)
    ]


def _renderizar_uma_pagina(caminho, numero_pagina):
    if fitz is None or not Path(caminho).exists():
        return None
    resultado = {"bytes": None, "erro": None}

    def _trabalho():
        try:
            with fitz.open(caminho) as doc:
                if doc.needs_pass:
                    doc.authenticate("")
                idx = numero_pagina - 1
                if idx < 0 or idx >= doc.page_count:
                    raise IndexError("página fora do intervalo")
                pagina = doc.load_page(idx)
                zoom = DPI_VISUAL / 72
                maior = max(pagina.rect.width, pagina.rect.height) * zoom
                if maior > MAX_LADO_PX:
                    zoom *= MAX_LADO_PX / maior
                pix = pagina.get_pixmap(
                    matrix=fitz.Matrix(zoom, zoom),
                    colorspace=(fitz.csGRAY if USAR_CINZA else fitz.csRGB),
                    alpha=False,
                )
                buf = io.BytesIO()
                try:
                    buf.write(pix.tobytes("jpeg", jpg_quality=QUALIDADE_JPEG))
                except TypeError:
                    buf.write(pix.tobytes("jpeg"))
                dados = buf.getvalue()
                pix = None
                if len(dados) > MAX_BYTES_PAGINA_VISUAL:
                    raise ValueError(f"JPEG grande demais: {len(dados) // 1024} KB")
                resultado["bytes"] = dados
        except Exception as e:
            resultado["erro"] = e

    t = threading.Thread(target=_trabalho, daemon=True)
    t.start()
    t.join(timeout=RENDER_TIMEOUT_SEG)
    if t.is_alive() or resultado["erro"]:
        return None
    return resultado["bytes"]


def _anexos_da_pagina(caminho, numero_pagina):
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
            reader = PdfReader(caminho)
            if reader.is_encrypted:
                reader.decrypt("")
            writer = PdfWriter()
            writer.add_page(reader.pages[numero_pagina - 1])
            buf = io.BytesIO()
            writer.write(buf)
            dados_pdf = buf.getvalue()
            if len(dados_pdf) <= MAX_BYTES_FALLBACK_PDF:
                anexos = [("application/pdf", dados_pdf, f"[Página {numero_pagina} do material]")]
        except Exception:
            pass
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
        return [], "", "⚠️ Não consegui renderizar esta página; usei só o texto."
    return anexos, f"a página {pag} do material original (página em foco nesta aula)", ""


def estado_vazio():
    return {
        "aulas": [], "resumos": [], "atual": 0, "fim": False,
        "nome": "", "caminho_pdf": "", "hist_inicio": 0, "pagina_atual": 1,
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
            "circuito, gráfico ou tabela diretamente nessa imagem. Se um valor estiver ilegível, "
            "diga qual e peça confirmação em vez de inventar."
        )
    texto = aulas[i]["texto"]
    if len(texto) > limite_texto:
        texto = texto[:limite_texto] + "\n[... texto truncado ...]"
    partes.append(f"Texto extraído da Aula {i + 1}:\n{texto}")
    return "\n\n".join(partes) + "\n\n"


# ============================================================================
# TTS DIDATICO (LOGICA IDENTICA AO v14)
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
    "cdot": " vezes ", "times": " vezes ", "div": " dividido por ",
    "Omega": " ohms ", "omega": " ômega ", "pi": " pi ", "mu": " micro ",
    "Delta": " delta ", "delta": " delta ", "theta": " teta ", "alpha": " alfa ",
    "beta": " beta ", "phi": " fi ", "lambda": " lambda ", "sigma": " sigma ", "tau": " tau ",
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
    v = (valor or "").strip().lstrip("+")
    if v in _NUM_EXTENSO:
        return _NUM_EXTENSO[v]
    if v.startswith("-"):
        return f"menos {v[1:]}"
    return v


def _extrair_coeficientes_edo(texto_md):
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
        A = "-1"
    else:
        A = _norm(m.group("a_num"), None, forca_um=True)
    B = "0" if m.group("b_sinal") is None else _norm(m.group("b_num"), m.group("b_sinal"), forca_um=True)
    C = "0" if m.group("c_sinal") is None else _norm(m.group("c_num"), m.group("c_sinal"), forca_um=True)
    return {"A": A, "B": B, "C": C, "equacao": equacao_original}


def _equacao_falada(equacao):
    t = equacao
    t = re.sub(r"y\s*''|y\s*¨|y\s*′′", " ípsilon duas linhas ", t)
    t = re.sub(r"y\s*'|y\s*˙|y\s*′", " ípsilon linha ", t)
    t = re.sub(r"(?<![A-Za-z])y(?![A-Za-z])", " ípsilon ", t)
    t = t.replace("+", " mais ").replace("-", " menos ").replace("=", " igual a ")
    return re.sub(r"\s+", " ", t).strip()


def _rotulo_item(texto_md):
    m = re.search(r"\b(?:item|letra|exerc[ií]cio)\s+([A-Za-z])\b", texto_md, re.IGNORECASE)
    return f"item {m.group(1).upper()}" if m else ""


def gerar_roteiro_edo(texto_md, coefs):
    A, B, C = coefs["A"], coefs["B"], coefs["C"]
    eq_falada = _equacao_falada(coefs["equacao"])
    rotulo = _rotulo_item(texto_md)
    abertura = f"Vamos resolver agora o {rotulo}. " if rotulo else "Vamos resolver agora. "
    return (
        f"{abertura}Temos aqui a equação... {eq_falada}. "
        "Primeiro, vamos achar os coeficientes A, B e C. "
        "O coeficiente A é o termo que acompanha a derivada segunda, ípsilon duas linhas. "
        "O coeficiente B é o que acompanha a derivada primeira, ípsilon linha. "
        "E o coeficiente C é o termo que acompanha a função ípsilon, sem derivada. "
        "Lembrando que, se qualquer um desses termos não aparecer na equação, "
        "o valor dele será zero. "
        "Observando a nossa equação, temos: "
        f"o A é igual a... {_num_falado(A)}, "
        f"o B é igual a... {_num_falado(B)}, "
        f"e o C é igual a... {_num_falado(C)}. "
        "Guardando esses três valores, já podemos seguir para o próximo passo da resolução."
    )


def _latex_para_fala(f):
    f = re.sub(r"\\(?:text|mathrm|mathbf)\{([^{}]*)\}", r" \1 ", f)
    for _ in range(3):
        f = re.sub(r"\\d?frac\{([^{}]*)\}\{([^{}]*)\}", r" \1 sobre \2 ", f)
    f = re.sub(r"\\sqrt\{([^{}]*)\}", r" raiz quadrada de \1 ", f)
    f = re.sub(r"\^\{?2\}?(?!\w)", " ao quadrado ", f)
    f = re.sub(r"\^\{?3\}?(?!\w)", " ao cubo ", f)
    f = re.sub(r"\^\{([^{}]*)\}|\^(\w)", lambda m: f" elevado a {m.group(1) or m.group(2)} ", f)
    f = re.sub(r"_\{([^{}]*)\}|_(\w)", lambda m: f" {m.group(1) or m.group(2)} ", f)
    f = re.sub(r"\\([a-zA-Z]+)", lambda m: _LATEX_PALAVRAS.get(m.group(1), " , "), f)
    f = f.replace("=", " igual a ").replace("+", " mais ").replace("−", " menos ")
    f = re.sub(r"\s-\s", " menos ", f)
    f = re.sub(r"[{}\\]", " ", f)
    return f


def _limpar_para_fala_generica(texto):
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
    geral = _limpar_para_fala_generica(texto_md)
    coefs = _extrair_coeficientes_edo(texto_md)
    if coefs:
        try:
            intro = gerar_roteiro_edo(texto_md, coefs)
            return f"{intro} ... ... Agora, vamos acompanhar a explicação completa, do começo. ... {geral}"
        except Exception:
            pass
    return geral


def limpar_arquivos_antigos():
    agora = time.time()
    for padrao, idade in (("resposta_*.mp3", IDADE_MAX_ARQUIVOS_SEG), ("curso_*.pdf", 24 * 3600)):
        for f in LIB.glob(padrao):
            try:
                if agora - f.stat().st_mtime > idade:
                    f.unlink()
            except OSError:
                pass


def _cortar_roteiro(roteiro, limite):
    if len(roteiro) <= limite:
        return roteiro
    corte = roteiro[:limite]
    pos = max(corte.rfind(". "), corte.rfind("... "))
    if pos > limite * 0.6:
        corte = corte[: pos + 1]
    return corte + " ... O restante da explicação está escrito no quadro."


def _dividir_em_pedacos(texto, tamanho):
    pedacos = []
    resto = texto.strip()
    while resto:
        if len(resto) <= tamanho:
            pedacos.append(resto)
            break
        janela = resto[:tamanho]
        pos = max(janela.rfind(". "), janela.rfind("? "), janela.rfind("! "),
                  janela.rfind(", "), janela.rfind("; "))
        if pos < tamanho * 0.4:
            pos = janela.rfind(" ")
        if pos <= 0:
            pos = tamanho - 1
        pedacos.append(resto[: pos + 1].strip())
        resto = resto[pos + 1:].strip()
    return [p for p in pedacos if p]


def _tts_edge(roteiro, caminho, taxa_velocidade="+0%"):
    if edge_tts is None:
        raise RuntimeError("edge-tts não instalado")
    if not isinstance(taxa_velocidade, str) or not re.fullmatch(r"[+-]\d+%", taxa_velocidade):
        taxa_velocidade = "+0%"
    erro = {"e": None}

    def _trabalho():
        try:
            async def _run():
                await edge_tts.Communicate(roteiro, VOZ_EDGE, rate=taxa_velocidade).save(str(caminho))
            asyncio.run(_run())
        except Exception as e:
            erro["e"] = e

    t = threading.Thread(target=_trabalho, daemon=True)
    t.start()
    t.join(timeout=120)
    if t.is_alive():
        raise TimeoutError("edge-tts demorou mais de 120s")
    if erro["e"]:
        raise erro["e"]
    if not caminho.exists() or caminho.stat().st_size == 0:
        raise RuntimeError("edge-tts gerou arquivo vazio")


def _tts_gtts(roteiro, caminho):
    if gTTS is None:
        raise RuntimeError("gTTS não instalado")
    saida = io.BytesIO()
    for pedaco in _dividir_em_pedacos(roteiro, TAMANHO_PEDACO_GTTS):
        ultimo = None
        for tentativa in range(3):
            parcial = io.BytesIO()
            try:
                gTTS(text=pedaco, lang="pt", tld="com.br", slow=False, timeout=(5, 30)).write_to_fp(parcial)
                dados = parcial.getvalue()
                if not dados:
                    raise RuntimeError("pedaço de áudio vazio")
                saida.write(dados)
                ultimo = None
                break
            except Exception as e:
                ultimo = e
                time.sleep(1.0 * (tentativa + 1))
        if ultimo is not None:
            raise ultimo
        time.sleep(0.15)
    dados = saida.getvalue()
    if not dados:
        raise RuntimeError("gTTS retornou áudio vazio")
    caminho.write_bytes(dados)


def _resolver_taxa_velocidade(velocidade):
    if isinstance(velocidade, str):
        v = velocidade.strip()
        if v in MAPA_VELOCIDADE:
            return MAPA_VELOCIDADE[v]
        if re.fullmatch(r"[+-]\d+%", v):
            return v
    return MAPA_VELOCIDADE[VELOCIDADE_PADRAO]


def gerar_audio_resposta(texto, velocidade=VELOCIDADE_PADRAO):
    roteiro = gerar_roteiro_falado(texto)
    if not roteiro:
        return None, ""
    roteiro = _cortar_roteiro(roteiro, LIMITE_CARACTERES_AUDIO)
    limpar_arquivos_antigos()
    caminho = LIB / f"resposta_{uuid.uuid4().hex}.mp3"
    taxa = _resolver_taxa_velocidade(velocidade)
    erros = []
    tentativas = (
        ("edge-tts", lambda r, c: _tts_edge(r, c, taxa)),
        ("gTTS",     _tts_gtts),
    )
    for nome, funcao in tentativas:
        try:
            funcao(roteiro, caminho)
            return str(caminho), ""
        except Exception as e:
            erros.append(f"{nome}: {type(e).__name__}: {str(e)[:120]}")
            caminho.unlink(missing_ok=True)
    return None, "Falha no áudio; a resposta está no texto. (" + " | ".join(erros) + ")"


# ----------------------------------------------------------------------------
# Historico e acoes (logica identica ao v14, adaptada ao fluxo do Streamlit)
# ----------------------------------------------------------------------------
def _normalizar_historico(historico):
    resultado = []
    for item in historico or []:
        if isinstance(item, dict) and item.get("content"):
            resultado.append({"role": item["role"], "content": str(item["content"])})
    return resultado


def _finalizar_resposta(resposta, ok, nota, gerar_audio, velocidade=VELOCIDADE_PADRAO):
    caminho_audio = None
    aviso_audio = ""
    if ok and gerar_audio:
        caminho_audio, erro = gerar_audio_resposta(resposta, velocidade=velocidade)
        if erro:
            aviso_audio = f"🔇 {erro}"
    extras = " ".join(x for x in (nota, aviso_audio) if x)
    texto_exibido = resposta + (f"\n\n_{extras}_" if extras else "")
    return texto_exibido, caminho_audio


def _processar_upload(arquivo, estado_atual):
    """Faz o papel do antigo processar_pdf (mesma logica)."""
    if arquivo is None:
        return estado_atual, ""
    _descartar_pdf_do_curso(estado_atual)
    novo = estado_vazio()
    novo["hist_inicio"] = len(st.session_state.get("historico", []))
    tmp = LIB / f"upload_{uuid.uuid4().hex}.pdf"
    tmp.write_bytes(arquivo.getbuffer())
    paginas = extrair_paginas_pdf(str(tmp))
    if not paginas:
        tmp.unlink(missing_ok=True)
        return novo, "⚠️ Não consegui abrir este PDF."
    total = sum(len(p.strip()) for p in paginas)
    if total < 50:
        tmp.unlink(missing_ok=True)
        return novo, ("⚠️ Quase não há texto neste PDF. Ele pode ser lido visualmente, "
                      "mas a divisão em aulas precisa de algum texto.")
    destino = LIB / f"curso_{uuid.uuid4().hex}.pdf"
    shutil.copyfile(tmp, destino)
    tmp.unlink(missing_ok=True)
    novo["caminho_pdf"] = str(destino)
    novo["aulas"] = dividir_em_aulas(paginas, N_AULAS)
    novo["nome"] = arquivo.name
    novo["pagina_atual"] = novo["aulas"][0]["pag_ini"] if novo["aulas"] else 1
    return novo, ""


# ============================================================================
# INTERFACE STREAMLIT — mesmo layout e estrutura do Gradio
# ============================================================================
st.set_page_config(
    page_title=f"Professor IA {VERSAO}",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS — o mesmo quadro negro, tema escuro e notranslate do v14
st.markdown("""
<meta name="google" content="notranslate">
<style>
:root, .stApp {
    background-color: #0b1210 !important;
    color: #e5f5ec !important;
    color-scheme: dark;
}
.stApp > header { background: #0b1210 !important; }
section.main > div { padding-left: 24px; padding-right: 24px; }
footer { visibility: hidden; }

/* ===== QUADRO NEGRO ===== */
.quadro-negro {
    background: #0f3b25 !important;
    border: 8px solid #8b5a2b !important;
    border-radius: 12px !important;
    box-shadow: inset 0 0 60px rgba(0,0,0,.55), 0 4px 18px rgba(0,0,0,.6) !important;
    padding: 16px 20px;
    color: #f6faf4;
    margin-bottom: 12px;
}
.quadro-negro .prose,
.quadro-negro .prose *,
.quadro-negro .katex,
.quadro-negro .katex * { color: #f6faf4 !important; opacity: 1 !important; }
.quadro-negro h1, .quadro-negro h2, .quadro-negro h3, .quadro-negro h4 {
    color: #ffe9a8 !important;
    border-bottom: 2px dashed rgba(255,233,168,.6);
    padding-bottom: 4px;
}
.quadro-negro strong { color: #ffffff !important; }
.quadro-negro hr { border: 0; border-top: 2px solid #e8efe6 !important; }
.quadro-negro table { border-collapse: collapse; }
.quadro-negro th, .quadro-negro td { border: 1px solid #cfe3d2 !important; padding: 6px 10px; }
.quadro-negro code, .quadro-negro pre {
    background: rgba(255,255,255,.10) !important;
    border: 1px solid rgba(255,255,255,.35) !important;
    color: #ffe9a8 !important;
}
.quadro-negro a { color: #9fe3b0 !important; }

/* bloco do professor: giz branco com borda definida */
.professor-bloco {
    background: #0f3b25 !important;
    border: 2px solid #e8efe6 !important;
    border-radius: 10px !important;
    padding: 12px 16px;
    margin: 8px 0;
    color: #f6faf4 !important;
}
/* bloco do aluno: giz amarelo tracejado */
.aluno-bloco {
    background: #0b2d1c !important;
    border: 2px dashed #ffe9a8 !important;
    border-radius: 10px !important;
    padding: 12px 16px;
    margin: 8px 0;
    color: #ffe9a8 !important;
}
</style>
""", unsafe_allow_html=True)


# ---- Estados de sessão (substituem o gr.State) ------------------------------
if "estado" not in st.session_state:
    st.session_state.estado = estado_vazio()
if "historico" not in st.session_state:
    st.session_state.historico = []      # [{"role","content"}]
if "audio_path" not in st.session_state:
    st.session_state.audio_path = None
if "status_msg" not in st.session_state:
    st.session_state.status_msg = ""


st.markdown(f"# 🎓 Professor IA — Tutor Universal de Exatas  `{VERSAO}`")
st.markdown(
    "Assistente que ensina **passo a passo**, nunca só a resposta. "
    "Envie um PDF e estude em 4 aulas progressivas, com leitura visual da **página em foco**. "
    "O áudio narra como uma professora de verdade, com pausas de respiração."
)

# ============================================================================
# PAINEL LATERAL (equivalente ao gr.Column(scale=1, min_width=340))
# ============================================================================
with st.sidebar:
    st.header("🔒 Configurações")

    # Chave do Gemini: pré-preenchida pelo Secret (mas editável)
    if CHAVE_ENV:
        st.success("🟢 Chave carregada das Secrets do Streamlit", icon="🔑")
        chave_gemini = st.text_input(
            "Chave de API do Gemini (opcional)",
            value=CHAVE_ENV,
            type="password",
            help="Pré-preenchida com o Secret GEMINI_API_KEY. Você pode substituir aqui.",
        )
    else:
        st.info("⚪ Nenhuma chave configurada", icon="🔑")
        chave_gemini = st.text_input(
            "Chave de API do Gemini",
            value="",
            type="password",
            help="Cole sua chave. No Streamlit Cloud, prefira definir o Secret GEMINI_API_KEY.",
        )

    # Testar chave
    if st.button("🔑 Testar Chave", use_container_width=True):
        with st.spinner("Testando..."):
            _ = listar_modelos_gemini(chave_gemini.strip())
            modelo = detectar_modelo_gemini(chave_gemini.strip())
        if modelo:
            st.success(f"🟢 Chave VÁLIDA! Modelo: {modelo}")
        else:
            st.error("🔴 Chave inválida ou sem modelos disponíveis.")

    st.divider()

    # Upload de PDF
    upload_pdf = st.file_uploader(
        "Biblioteca (Upload de PDF de aula/livro)",
        type=["pdf"],
        key="upload_pdf",
    )
    if upload_pdf is not None:
        novo, msg = _processar_upload(upload_pdf, st.session_state.estado)
        if msg:
            st.warning(msg)
        else:
            st.session_state.estado = novo
            st.success(f"📚 PDF '{upload_pdf.name}' carregado em {len(novo['aulas'])} aulas.")

    # Status
    _estado = _garantir_estado(st.session_state.estado)
    if not _estado["aulas"]:
        st.caption("📭 Nenhum PDF carregado.")
    else:
        n = len(_estado["aulas"])
        atual = _estado["atual"]
        linhas = []
        for i, a in enumerate(_estado["aulas"]):
            if i < atual or (i == atual and _estado["fim"]):
                linhas.append(f"✅ Aula {i+1} — {a['rotulo']}")
            elif i == atual:
                linhas.append(f"▶️ **Aula {i+1}** — {a['rotulo']}")
            else:
                linhas.append(f"🔒 Aula {i+1} — {a['rotulo']}")
        aula = _estado["aulas"][atual]
        pag_foco = _estado.get("pagina_atual") or aula["pag_ini"]
        pag_foco = max(aula["pag_ini"], min(pag_foco, aula["pag_fim"]))
        st.markdown("\n\n".join(linhas))
        st.caption(f"👁️ Página em foco: {pag_foco} (de {aula['pag_ini']}–{aula['pag_fim']} nesta aula)")

    st.divider()

    # Navegação de páginas
    col_prev, col_next = st.columns(2)
    with col_prev:
        if st.button("◀ Página", use_container_width=True):
            e = _garantir_estado(st.session_state.estado)
            if e["aulas"]:
                aula = e["aulas"][e["atual"]]
                pag = e.get("pagina_atual") or aula["pag_ini"]
                e["pagina_atual"] = max(aula["pag_ini"], min(pag - 1, aula["pag_fim"]))
                st.session_state.estado = e
    with col_next:
        if st.button("Página ▶", use_container_width=True):
            e = _garantir_estado(st.session_state.estado)
            if e["aulas"]:
                aula = e["aulas"][e["atual"]]
                pag = e.get("pagina_atual") or aula["pag_ini"]
                e["pagina_atual"] = max(aula["pag_ini"], min(pag + 1, aula["pag_fim"]))
                st.session_state.estado = e

    # Ações de aula
    col_i, col_c = st.columns(2)
    with col_i:
        if st.button("▶️ Explicar aula", variant="primary", use_container_width=True):
            st.session_state["_acao"] = "iniciar"
    with col_c:
        if st.button("✅ Concluir aula", use_container_width=True):
            st.session_state["_acao"] = "concluir"
    if st.button("🔄 Reiniciar curso", use_container_width=True):
        st.session_state["_acao"] = "reiniciar"

    st.divider()

    # Opções
    usar_visual = st.checkbox(
        "👁️ Leitura visual da página em foco",
        value=True,
        help="Envia SOMENTE a página em foco como imagem ao Gemini.",
    )
    gerar_audio = st.checkbox("🔊 Gerar áudio didático das respostas", value=True)
    velocidade_audio = st.radio(
        "⚡ Velocidade da Voz",
        options=["0.75x (Lenta)", "1.0x (Normal)", "1.25x (Rápida)", "1.5x (Muito Rápida)"],
        index=1,
        help="Ajuste o ritmo da explicação do professor.",
    )


# ============================================================================
# ÁREA CENTRAL: QUADRO NEGRO + CHAT + ÁUDIO
# ============================================================================
col_chat, col_audio = st.columns([4, 1])

with col_chat:
    st.markdown("**🧑‍🏫 Quadro Negro — Resposta do Professor**")

    # Renderiza o histórico com a estética do quadro negro
    for msg in st.session_state.historico:
        classe = "aluno-bloco" if msg["role"] == "user" else "professor-bloco"
        icone = "🧑‍🎓" if msg["role"] == "user" else "🧑‍🏫"
        st.markdown(
            f'<div class="quadro-negro"><div class="{classe}">'
            f'<b>{icone}</b><br>{msg["content"]}</div></div>',
            unsafe_allow_html=True,
        )

with col_audio:
    st.markdown("**🔊 Áudio**")
    if st.session_state.audio_path and Path(st.session_state.audio_path).exists():
        st.audio(st.session_state.audio_path, format="audio/mp3", autoplay=True)
    else:
        st.caption("Nenhum áudio gerado ainda.")

    if st.button("🔊 Testar áudio", use_container_width=True):
        with st.spinner("Gerando áudio de teste..."):
            caminho, erro = gerar_audio_resposta(
                "Olá! Eu sou a sua professora. Se você está me ouvindo, o áudio está funcionando.",
                velocidade=velocidade_audio,
            )
        if caminho:
            st.session_state.audio_path = caminho
            st.success(f"🟢 Áudio gerado (velocidade: {velocidade_audio}).")
        else:
            st.error(f"🔴 {erro}")


# ---- Ações disparadas pelo painel lateral -----------------------------------
acao = st.session_state.pop("_acao", None)

if acao == "reiniciar":
    e = _garantir_estado(st.session_state.estado)
    if e["aulas"]:
        e["atual"] = 0
        e["resumos"] = []
        e["fim"] = False
        e["pagina_atual"] = e["aulas"][0]["pag_ini"]
        st.session_state.estado = e
        st.session_state.historico.append({
            "role": "assistant",
            "content": "🔄 Curso reiniciado. Você está de volta à **Aula 1**.",
        })
    st.rerun()

elif acao == "concluir":
    e = _garantir_estado(st.session_state.estado)
    if e["aulas"] and not e["fim"]:
        i, n = e["atual"], len(e["aulas"])
        hist_modelo = st.session_state.historico[e.get("hist_inicio", 0):]
        with st.spinner("Gerando resumo da aula..."):
            resumo, ok, _nota = motor_hibrido(
                PROMPT_RESUMO, e, hist_modelo, chave_gemini, usar_visual
            )
        if not ok:
            t = e["aulas"][i]["texto"]
            resumo = f"{t[:700]} [...] {t[-500:]}" if len(t) > 1200 else t
        e["resumos"] = e["resumos"][:i] + [resumo[:LIMITE_RESUMO]]
        if i + 1 < n:
            e["atual"] = i + 1
            e["pagina_atual"] = e["aulas"][i + 1]["pag_ini"]
            msg = (f"✅ **Aula {i+1} concluída!**\n\n**Resumo guardado:**\n{resumo}\n\n"
                   f"➡️ Agora você está na **Aula {i+2} de {n}**.")
        else:
            e["fim"] = True
            msg = (f"🎉 **Aula {n} concluída — você terminou todo o material!**\n\n"
                   f"**Resumo da última aula:**\n{resumo}")
        st.session_state.historico.append({"role": "user", "content": f"✅ Concluir a Aula {i+1}"})
        st.session_state.historico.append({"role": "assistant", "content": msg})
        e["hist_inicio"] = len(st.session_state.historico)
        st.session_state.estado = e
    st.rerun()

elif acao == "iniciar":
    e = _garantir_estado(st.session_state.estado)
    if not e["aulas"]:
        st.session_state.historico.append({
            "role": "assistant",
            "content": "📭 Envie um PDF primeiro para eu montar as aulas.",
        })
        st.rerun()
    elif e["fim"]:
        st.session_state.historico.append({
            "role": "assistant",
            "content": "🎉 Você já concluiu todas as aulas. Clique em **🔄 Reiniciar curso**.",
        })
        st.rerun()
    else:
        i, n = e["atual"], len(e["aulas"])
        pag = e.get("pagina_atual") or e["aulas"][i]["pag_ini"]
        pedido = f"Inicie a Aula {i+1} de {n}, com foco principal na página {pag} do material. "
        if i > 0:
            pedido += ("Esta aula CONTINUA exatamente de onde a anterior parou, "
                       "sem repetir o que já foi ensinado. ")
        pedido += (
            "Estruture assim: 1) objetivo da aula; 2) conceitos e fórmulas principais, "
            "descrevendo os esquemas e circuitos das páginas; "
            "3) um exemplo resolvido PASSO A PASSO do zero; "
            "4) proponha 3 exercícios de cálculo para eu resolver agora; "
            "5) termine pedindo que eu envie minhas resoluções."
        )
        hist_modelo = st.session_state.historico[e.get("hist_inicio", 0):]
        with st.spinner("Preparando a aula..."):
            resposta, ok, nota = motor_hibrido(pedido, e, hist_modelo, chave_gemini, usar_visual)
            texto_exibido, caminho_audio = _finalizar_resposta(
                resposta, ok, nota, gerar_audio, velocidade=velocidade_audio
            )
        st.session_state.historico.append({
            "role": "user",
            "content": f"▶️ Explicar a Aula {i+1} de {n} (foco na página {pag})",
        })
        st.session_state.historico.append({"role": "assistant", "content": texto_exibido})
        if caminho_audio:
            st.session_state.audio_path = caminho_audio
        st.rerun()


# ---- Entrada do chat --------------------------------------------------------
pergunta = st.chat_input("Digite sua dúvida ou cole sua resolução...")

if pergunta:
    e = _garantir_estado(st.session_state.estado)
    st.session_state.historico.append({"role": "user", "content": pergunta})
    hist_modelo = st.session_state.historico[e.get("hist_inicio", 0):]
    with st.spinner("O professor está pensando..."):
        resposta, ok, nota = motor_hibrido(pergunta.strip(), e, hist_modelo, chave_gemini, usar_visual)
        texto_exibido, caminho_audio = _finalizar_resposta(
            resposta, ok, nota, gerar_audio, velocidade=velocidade_audio
        )
    st.session_state.historico.append({"role": "assistant", "content": texto_exibido})
    if caminho_audio:
        st.session_state.audio_path = caminho_audio
    st.rerun()
