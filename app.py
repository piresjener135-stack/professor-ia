"""
Professor IA — Engenharia e Matemática (v13-cloud)
- Hugging Face Spaces: sem Ollama (localhost não existe), só Gemini na nuvem
- Chave lida de GEMINI_API_KEY (Secrets do HF); campo de texto como fallback
- Tema escuro + quadro negro (giz branco/amarelo)
- Leitura VISUAL do PDF: SOMENTE a página em foco é renderizada
- Navegação de páginas (◀️ / ▶️) com cache LRU
- PDF dividido em 4 aulas progressivas e dependentes
- TTS DIDÁTICO: roteiro falado como professor na lousa (EDOs com coeficientes A, B e C)
- Handlers com parâmetros finais OPCIONAIS
- Página marcada como notranslate (Chrome não traduz mais a interface)
- Pasta de trabalho em /tmp (HF: fora de /tmp o disco é somente leitura)
"""
from _future_ import annotations

import asyncio
import base64
import io
import os
import re
import shutil
import sys
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from pathlib import Path

import gradio as gr
import requests
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

VERSAO = "v13-nuvem"

# ----------------------------------------------------------------------------
# Diretórios — no HF Spaces, o disco é somente leitura fora de /tmp
# ----------------------------------------------------------------------------
if os.environ.get("SPACE_ID") or not os.access(Path(_file_).resolve().parent, os.W_OK):
    BASE = Path("/tmp")
else:
    BASE = Path(_file_).resolve().parent
LIB = BASE / "biblioteca"
try:
    LIB.mkdir(parents=True, exist_ok=True)
except Exception:
    BASE = Path("/tmp")
    LIB = BASE / "biblioteca"
    LIB.mkdir(parents=True, exist_ok=True)

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"

# Chave do Gemini: lê a variável de ambiente (Secrets do HF)
CHAVE_ENV = (os.environ.get("GEMINI_API_KEY") or "").strip()

N_AULAS = 4
LIMITE_AULA_GEMINI = 100_000
LIMITE_RESUMO = 2_500
MAX_MENSAGENS_HISTORICO = 8

# --- Leitura visual da página em foco ---
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
    "Se houver contexto de um documento (PDF/aula), baseie-se nele e diga quando algo não estiver disponível "
    "no texto."
)

PROMPT_RESUMO = (
    "Gere um RESUMO DE MEMÓRIA desta aula (máx. 250 palavras) para as próximas aulas: "
    "definições, fórmulas (LaTeX), métodos, circuitos e valores relevantes, resultados numéricos, "
    "o que o aluno resolveu ou errou e o ponto onde o texto da aula terminou. "
    "Responda apenas com o resumo."
)

_cache_modelo_gemini: dict = {}
_cache_paginas_visual: "OrderedDict[tuple, list]" = OrderedDict()


# ============================================================================
# MOTOR DE IA — Gemini (Ollama removido: não existe no HF Spaces)
# ============================================================================
def detectar_modelo_gemini(api_key: str) -> str:
    """Lista os modelos e escolhe o melhor 'flash' estável para a chave."""
    if api_key in _cache_modelo_gemini:
        return _cache_modelo_gemini[api_key]
    escolhido = "gemini-2.0-flash"
    try:
        r = requests.get(
            f"{GEMINI_BASE}/models",
            headers={"x-goog-api-key": api_key},
            params={"pageSize": 100},
            timeout=15,
        )
        r.raise_for_status()
        candidatos = []
        for m in r.json().get("models", []):
            nome = m.get("name", "").replace("models/", "")
            if "generateContent" not in m.get("supportedGenerationMethods", []):
                continue
            if not nome.startswith("gemini"):
                continue
            if any(x in nome for x in ("embedding", "tts", "image", "vision", "live", "audio", "robotics")):
                continue
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
            candidatos.append((pontos, nome))
        if candidatos:
            escolhido = max(candidatos)[1]
    except Exception as e:
        print(f"[Gemini] Não foi possível listar modelos ({e}); usando {escolhido}.")
    _cache_modelo_gemini[api_key] = escolhido
    print(f"[Gemini] Modelo em uso: {escolhido}")
    return escolhido


def chamar_gemini(pergunta, contexto, historico, api_key, anexos=None):
    """
    Chama o Gemini, opcionalmente com anexos visuais.
    Retorna SEMPRE uma tupla (texto, tipo_de_erro):
      - (str, "")       -> sucesso
      - (None, "vazio") -> resposta bloqueada ou sem texto
      - (None, "http")  -> erro HTTP (401, 403, 429, 500...)
      - (None, "timeout") -> demorou demais
      - (None, "rede")  -> erro de conexão
    """
    modelo = detectar_modelo_gemini(api_key)
    url = f"{GEMINI_BASE}/models/{modelo}:generateContent"
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
    _t0 = time.perf_counter()
    try:
        r = requests.post(
            url,
            headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
            json=payload,
            timeout=60 if anexos else 45,
        )
        if r.status_code == 404:
            _cache_modelo_gemini.pop(api_key, None)
        r.raise_for_status()
        print(f"[TEMPO] Gemini respondeu em {time.perf_counter() - _t0:.1f}s")
        res_json = r.json()
        if not (res_json.get("candidates") and res_json["candidates"][0].get("content")):
            return None, "vazio"
        partes = res_json["candidates"][0]["content"].get("parts", [])
        texto = "".join(p.get("text", "") for p in partes).strip()
        return (texto, "") if texto else (None, "vazio")
    except requests.Timeout as e:
        print(f"[Gemini] timeout: {e}")
        return None, "timeout"
    except requests.HTTPError as e:
        corpo = ""
        try:
            corpo = e.response.text[:300]
        except Exception:
            pass
        print(f"[Gemini] erro HTTP: {e} {corpo}")
        return None, "http"
    except requests.RequestException as e:
        print(f"[Gemini] erro de rede: {e}")
        return None, "rede"
    except (KeyError, IndexError, ValueError) as e:
        print(f"[Gemini] resposta inesperada/bloqueada: {e}")
        return None, "vazio"


def motor_hibrido(pergunta, estado, historico, api_key, usar_visual=True):
    """
    Versão nuvem: usa SOMENTE Gemini.
    Se a chave estiver vazia, devolve uma mensagem amigável e não tenta nada.
    """
    notas = []
    chave = (api_key or CHAVE_ENV or "").strip()
    if not chave:
        return (
            "⚠️ *Nenhuma chave do Gemini configurada.*\n\n"
            "*Como resolver:*\n"
            "• No Space do Hugging Face: vá em *Settings → Variables and secrets* e crie um "
            "secret chamado GEMINI_API_KEY com a sua chave.\n"
            "• Ou cole a chave no campo à esquerda e clique em *Testar Chave*.\n\n"
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

    if erro == "timeout":
        return ("⚠️ O Gemini demorou demais para responder. Tente novamente — ou desmarque "
                "'leitura visual' para acelerar.", False, "")
    if erro == "http":
        return ("⚠️ A API do Gemini respondeu com erro. Verifique se a chave está correta e se a "
                "cota não foi excedida.", False, "")
    if erro == "rede":
        return ("⚠️ Não consegui falar com a API do Gemini. Verifique a conexão do Space.",
                False, "")
    return ("⚠️ Não consegui obter a resposta do Gemini. Tente novamente em alguns segundos.",
            False, "")


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


if CHAVE_ENV:
    INDICADOR_NEUTRO = _indicador_html(
        "verde",
        "🟢 Chave compartilhada dos segredos do Hugging Face",
        "Você também pode colar outra chave no campo acima para substituir temporariamente.",
    )
else:
    INDICADOR_NEUTRO = _indicador_html(
        "cinza",
        "⚪ Nenhuma chave configurada",
        "Cole a chave no campo acima ou defina o secret GEMINI_API_KEY no Space.",
    )


def testar_chave_gemini(api_key):
    api_key = (api_key or CHAVE_ENV or "").strip()
    if not api_key:
        return _indicador_html("cinza", "⚪ Cole uma chave para testar")
    try:
        r = requests.get(
            f"{GEMINI_BASE}/models",
            headers={"x-goog-api-key": api_key},
            params={"pageSize": 1},
            timeout=10,
        )
    except requests.RequestException as e:
        return _indicador_html("vermelho", "🔴 Chave inválida / erro de conexão",
                               f"Sem conexão: {type(e)._name_}")
    if r.status_code == 200:
        _cache_modelo_gemini.pop(api_key, None)
        modelo = detectar_modelo_gemini(api_key)
        return _indicador_html("verde", "🟢 Chave Gemini VÁLIDA e conectada!",
                               f"Modelo que será usado: {modelo}")
    try:
        msg = r.json().get("error", {}).get("message", "")
    except ValueError:
        msg = ""
    detalhe = f"HTTP {r.status_code}" + (f" — {msg[:120]}" if msg else "")
    return _indicador_html("vermelho", "🔴 Chave inválida / erro de conexão", detalhe)


# ============================================================================
# PDF — extração de texto e divisão em aulas
# ============================================================================
def extrair_paginas_pdf(caminho_pdf):
    """Somente TEXTO (nunca imagem). PyMuPDF é bem mais rápido que o pypdf."""
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
# RENDERIZAÇÃO DE UMA PÁGINA (leitura visual)
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
    """Lazy: renderiza SÓ esta página, apenas uma vez (cache LRU, inclusive de falhas)."""
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
                print(f"[PDF] fallback pesado ({len(dados_pdf) // 1024} KB): sem imagem. "
                      "Instale o PyMuPDF (pip install pymupdf).")
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
        return "📭 *Nenhum PDF carregado.* Envie um PDF para dividi-lo em aulas, ou converse normalmente."
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
        linhas.append(f"{icone} *Aula {i + 1}* — {a['rotulo']}")
    if estado["fim"]:
        titulo = f"### 🎉 Curso concluído ({n} de {n} aulas)"
    else:
        titulo = f"### 📚 Aula {atual + 1} de {n}"
    aula = aulas[atual]
    pag_foco = estado.get("pagina_atual") or aula["pag_ini"]
    pag_foco = max(aula["pag_ini"], min(pag_foco, aula["pag_fim"]))
    foco = f"👁️ *Página em foco:* {pag_foco} (de {aula['pag_ini']}–{aula['pag_fim']} nesta aula)"
    notas = [foco]
    tam = len(aula["texto"])
    if tam > LIMITE_AULA_GEMINI:
        notas.append(
            f"⚠️ Esta aula tem {tam:,} caracteres: a IA verá só os primeiros "
            f"{LIMITE_AULA_GEMINI:,}."
        )
    return f"{titulo}\n\n" + "\n\n".join(linhas) + "\n\n" + "\n\n".join(notas)


def _caminho_do_arquivo(arquivo):
    if arquivo is None:
        return None
    if isinstance(arquivo, str):
        return arquivo
    if isinstance(arquivo, dict):
        return arquivo.get("path") or arquivo.get("name")
    return getattr(arquivo, "name", None) or getattr(arquivo, "path", None)


# ============================================================================
# TTS DIDÁTICO — roteiro falado como professor na lousa
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
    Procura por A·y'' + B·y' + C·y = 0 e devolve os coeficientes.
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
    t = t.replace("+", " mais ").replace("-", " menos ")
    t = t.replace("=", " igual a ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _rotulo_item(texto_md):
    m = re.search(r"\b(?:item|letra|exerc[ií]cio)\s+([A-Za-z])\b", texto_md, re.IGNORECASE)
    return f"item {m.group(1).upper()}" if m else ""


def gerar_roteiro_edo(texto_md, coefs):
    A, B, C = coefs["A"], coefs["B"], coefs["C"]
    equacao_falada = _equacao_falada(coefs["equacao"])
    rotulo = _rotulo_item(texto_md)
    abertura = f"Vamos resolver agora o {rotulo}. " if rotulo else "Vamos resolver agora. "
    A_fala = _num_falado(A)
    B_fala = _num_falado(B)
    C_fala = _num_falado(C)
    return (
        f"{abertura}Temos aqui a equação... {equacao_falada}. "
        "Primeiro, vamos achar os coeficientes A, B e C. "
        "O coeficiente A é o termo que acompanha a derivada segunda, ípsilon duas linhas. "
        "O coeficiente B é o que acompanha a derivada primeira, ípsilon linha. "
        "E o coeficiente C é o termo que acompanha a função ípsilon, sem derivada. "
        "Lembrando que, se algum desses termos não aparecer na equação, "
        "o valor dele será zero. "
        "Observando a nossa equação, temos: "
        f"o A é igual a... {A_fala}, "
        f"o B é igual a... {B_fala}, "
        f"e o C é igual a... {C_fala}. "
        "Guardando esses três valores, já podemos seguir para o próximo passo da resolução."
    )


def _latex_para_fala(f):
    f = re.sub(r"\\(?:text|mathrm|mathbf)\{([^{}]*)\}", r" \1 ", f)
    for _ in range(3):
        f = re.sub(r"\\d?frac\{([^{}])\}\{([^{}])\}", r" \1 sobre \2 ", f)
    f = re.sub(r"\\sqrt\{([^{}]*)\}", r" raiz quadrada de \1 ", f)
    f = re.sub(r"\^\{?2\}?(?!\w)", " ao quadrado ", f)
    f = re.sub(r"\^\{?3\}?(?!\w)", " ao cubo ", f)
    f = re.sub(r"\^\{([^{}]*)\}|\^(\w)", lambda m: f" elevado a {m.group(1) or m.group(2)} ", f)
    f = re.sub(r"\{([^{}]*)\}|(\w)", lambda m: f" {m.group(1) or m.group(2)} ", f)
    f = re.sub(r"\\([a-zA-Z]+)", lambda m: _LATEX_PALAVRAS.get(m.group(1), " , "), f)
    f = f.replace("=", " igual a ").replace("+", " mais ").replace("−", " menos ")
    f = re.sub(r"\s-\s", " menos ", f)
    f = re.sub(r"[{}\\]", " ", f)
    return f


def _limpar_para_fala_generica(texto):
    """Fallback: limpa markdown, converte LaTeX e insere pausas para soar natural."""
    t = re.sub(r".*?", " ... trecho de código omitido ... ", texto, flags=re.S)
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
    frases = re.split(r"(?<=[.!?;:])\s+|(?<=\.\.\.)\s+", texto)
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
    for padrao, idade in (("resposta_.mp3", IDADE_MAX_ARQUIVOS_SEG), ("curso_.pdf", 24 * 3600)):
        for f in LIB.glob(padrao):
            try:
                if agora - f.stat().st_mtime > idade:
                    f.unlink()
            except OSError:
                pass


def gerar_audio_resposta(texto):
    """Gera um único MP3 com a resposta COMPLETA. Retorna (caminho_mp3, mensagem_de_erro)."""
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
# HISTÓRICO
# ============================================================================
def _texto_do_conteudo(c):
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if isinstance(c, dict):
        return str(c.get("text") or c.get("content") or "")
    if isinstance(c, (list, tuple)):
        return " ".join(_texto_do_conteudo(x) for x in c).strip()
    return str(c)


def _normalizar_historico(historico):
    resultado = []
    for item in historico or []:
        if isinstance(item, dict):
            texto = _texto_do_conteudo(item.get("content"))
            if texto:
                resultado.append({"role": item["role"], "content": texto})
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            usuario, assistente = item
            if usuario:
                resultado.append({"role": "user", "content": _texto_do_conteudo(usuario)})
            if assistente:
                resultado.append({"role": "assistant", "content": _texto_do_conteudo(assistente)})
    return resultado


def _finalizar_resposta(resposta, ok, nota, gerar_audio):
    caminho_audio = None
    aviso_audio = ""
    if ok and gerar_audio:
        _t0 = time.perf_counter()
        caminho_audio, erro = gerar_audio_resposta(resposta)
        print(f"[TEMPO] áudio (TTS) gerado em {time.perf_counter() - _t0:.1f}s")
        if erro:
            aviso_audio = f"🔇 {erro}"
    extras = " ".join(x for x in (nota, aviso_audio) if x)
    texto_exibido = resposta + (f"\n\n_{extras}_" if extras else "")
    return texto_exibido, caminho_audio


# ============================================================================
# AÇÕES DA INTERFACE
# (parâmetros finais com valor padrão => tolerantes a diferença de entradas)
# ============================================================================
def processar_pdf(arquivo_pdf, estado=None, historico=None):
    historico = _normalizar_historico(historico)
    estado = _garantir_estado(estado)
    _descartar_pdf_do_curso(estado)
    caminho = _caminho_do_arquivo(arquivo_pdf)
    novo = estado_vazio()
    novo["hist_inicio"] = len(historico)
    if not caminho:
        return novo, render_status(novo)
    paginas = extrair_paginas_pdf(caminho)
    if not paginas:
        return novo, "⚠️ Não consegui abrir este PDF (arquivo corrompido ou protegido por senha)."
    total = sum(len(p.strip()) for p in paginas)
    if total < 50:
        return novo, (
            "⚠️ Quase não há texto neste PDF (provavelmente escaneado ou só imagens). "
            "Ele ainda pode ser lido visualmente, mas a divisão em aulas precisa de algum texto."
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
    return novo, render_status(novo)


def navegar_pagina(direcao, estado=None):
    estado = _garantir_estado(estado)
    aulas = estado.get("aulas") or []
    if not aulas:
        return estado, render_status(estado)
    aula = aulas[estado["atual"]]
    pag = estado.get("pagina_atual") or aula["pag_ini"]
    pag = max(aula["pag_ini"], min(pag + direcao, aula["pag_fim"]))
    estado["pagina_atual"] = pag
    return estado, render_status(estado)


def responder_usuario(mensagem, historico=None, estado=None, api_key="", gerar_audio=True, usar_visual=True):
    historico = _normalizar_historico(historico)
    estado = _garantir_estado(estado)
    if not mensagem or not mensagem.strip():
        return historico, None, ""
    hist_modelo = historico[estado.get("hist_inicio", 0):]
    resposta, ok, nota = motor_hibrido(mensagem.strip(), estado, hist_modelo, api_key, usar_visual)
    texto_exibido, caminho_audio = _finalizar_resposta(resposta, ok, nota, gerar_audio)
    historico.append({"role": "user", "content": mensagem})
    historico.append({"role": "assistant", "content": texto_exibido})
    return historico, caminho_audio, ""


def acao_iniciar_aula(historico=None, estado=None, api_key="", gerar_audio=True, usar_visual=True):
    historico = _normalizar_historico(historico)
    estado = _garantir_estado(estado)
    if not estado.get("aulas"):
        historico.append({"role": "assistant", "content": "📭 Envie um PDF primeiro para eu montar as aulas."})
        return historico, None, estado, render_status(estado)
    if estado["fim"]:
        historico.append({
            "role": "assistant",
            "content": "🎉 Você já concluiu todas as aulas. Clique em *🔄 Reiniciar curso* para recomeçar.",
        })
        return historico, None, estado, render_status(estado)
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
    return historico, caminho_audio, estado, render_status(estado)


def acao_concluir_aula(historico=None, estado=None, api_key="", usar_visual=True):
    historico = _normalizar_historico(historico)
    estado = _garantir_estado(estado)
    if not estado.get("aulas"):
        historico.append({"role": "assistant", "content": "📭 Envie um PDF primeiro para eu montar as aulas."})
        return historico, None, estado, render_status(estado)
    if estado["fim"]:
        historico.append({"role": "assistant", "content": "🎉 O curso já está concluído."})
        return historico, None, estado, render_status(estado)
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
            f"✅ *Aula {i + 1} concluída!\n\nResumo salvo para as próximas aulas:*\n{resumo}\n\n"
            f"➡️ Agora você está na *Aula {i + 2} de {n}*. "
            "Clique em *▶️ Explicar aula atual* quando quiser continuar."
        )
    else:
        estado["fim"] = True
        msg = (
            f"🎉 *Aula {n} concluída — você terminou todo o material!*\n\n"
            f"*Resumo da última aula:*\n{resumo}"
        )
    historico.append({"role": "user", "content": f"✅ Concluir a Aula {i + 1}"})
    historico.append({"role": "assistant", "content": msg})
    estado["hist_inicio"] = len(historico)
    return historico, None, estado, render_status(estado)


def acao_reiniciar_curso(historico=None, estado=None):
    historico = _normalizar_historico(historico)
    estado = _garantir_estado(estado)
    if not estado.get("aulas"):
        return historico, None, estado, render_status(estado)
    estado["atual"] = 0
    estado["resumos"] = []
    estado["fim"] = False
    estado["pagina_atual"] = estado["aulas"][0]["pag_ini"]
    historico.append({"role": "assistant", "content": "🔄 Curso reiniciado. Você está de volta à *Aula 1*."})
    estado["hist_inicio"] = len(historico)
    return historico, None, estado, render_status(estado)


# ============================================================================
# INTERFACE GRADIO
# ============================================================================
TEMA = gr.themes.Soft(
    primary_hue="green",
    secondary_hue="emerald",
    neutral_hue="slate",
).set(
    body_background_fill="#0b1210",
    body_background_fill_dark="#0b1210",
    body_text_color="#e5f5ec",
    body_text_color_dark="#e5f5ec",
    block_background_fill="#111c18",
    block_background_fill_dark="#111c18",
    input_background_fill="#0e1714",
    input_background_fill_dark="#0e1714",
)

CSS = """
:root, .dark, body, gradio-app, .gradio-container {
    color-scheme: dark;
    background-color: #0b1210 !important;
}
.gradio-container, .gradio-container .main, .gradio-container > .main, .app {
    max-width: none !important;
    width: 100% !important;
}
.gradio-container { padding-left: 24px !important; padding-right: 24px !important; }
footer { visibility: hidden; }
/* ===== QUADRO NEGRO (área de resposta do professor) ===== */
#quadro-negro {
    background: #0f3b25 !important;
    border: 8px solid #8b5a2b !important;
    border-radius: 12px !important;
    box-shadow: inset 0 0 60px rgba(0,0,0,.55), 0 4px 18px rgba(0,0,0,.6) !important;
    --body-text-color: #f6faf4;
    --background-fill-primary: #0f3b25;
    --background-fill-secondary: #0f3b25;
    --block-background-fill: #0f3b25;
    --block-label-background-fill: #0b2d1c;
    --block-label-text-color: #ffe9a8;
    --border-color-primary: #e8efe6;
    --color-accent-soft: #0b2d1c;
}
#quadro-negro .bubble-wrap,
#quadro-negro .wrapper,
#quadro-negro .message-wrap { background: #0f3b25 !important; }
#quadro-negro .message-content,
#quadro-negro .prose,
#quadro-negro .prose *,
#quadro-negro .katex,
#quadro-negro .katex * { color: #f6faf4 !important; opacity: 1 !important; }
#quadro-negro .prose { font-size: 1.08rem !important; line-height: 1.75 !important; }
#quadro-negro .bot-row {
    background: #0f3b25 !important;
    border: 2px solid #e8efe6 !important;
    border-radius: 10px !important;
}
#quadro-negro .user-row {
    background: #0b2d1c !important;
    border: 2px dashed #ffe9a8 !important;
    border-radius: 10px !important;
}
#quadro-negro .user-row .prose,
#quadro-negro .user-row .prose * { color: #ffe9a8 !important; }
#quadro-negro h1, #quadro-negro h2, #quadro-negro h3, #quadro-negro h4 {
    color: #ffe9a8 !important;
    border-bottom: 2px dashed rgba(255,233,168,.6);
    padding-bottom: 4px;
}
#quadro-negro strong { color: #ffffff !important; }
#quadro-negro hr { border: 0; border-top: 2px solid #e8efe6 !important; }
#quadro-negro table { border-collapse: collapse; }
#quadro-negro th, #quadro-negro td { border: 1px solid #cfe3d2 !important; padding: 6px 10px; }
#quadro-negro code, #quadro-negro pre {
    background: rgba(255,255,255,.10) !important;
    border: 1px solid rgba(255,255,255,.35) !important;
    color: #ffe9a8 !important;
}
#quadro-negro a { color: #9fe3b0 !important; }
"""

JS_FORCAR_ESCURO = """
() => {
    document.documentElement.classList.add('dark');
    document.body.classList.add('dark');
    document.documentElement.setAttribute('translate', 'no');
    document.documentElement.setAttribute('lang', 'pt-BR');
    document.body.classList.add('notranslate');
    if (!document.querySelector('meta[name="google"][content="notranslate"]')) {
        const m = document.createElement('meta');
        m.name = 'google';
        m.content = 'notranslate';
        document.head.appendChild(m);
    }
}
"""

try:
    GRADIO_MAJOR = int(gr._version_.split(".")[0])
except (ValueError, AttributeError):
    GRADIO_MAJOR = 5

if GRADIO_MAJOR >= 6:
    BLOCKS_KWARGS = {}
    LAUNCH_KWARGS = dict(theme=TEMA, css=CSS, js=JS_FORCAR_ESCURO, allowed_paths=[str(LIB)])
else:
    BLOCKS_KWARGS = dict(theme=TEMA, css=CSS, js=JS_FORCAR_ESCURO)
    LAUNCH_KWARGS = dict(allowed_paths=[str(LIB)])


def criar_chatbot():
    opcoes = dict(
        label="🎓 Quadro Negro — Resposta do Professor",
        elem_id="quadro-negro",
        height="78vh",
        latex_delimiters=[
            {"left": "$$", "right": "$$", "display": True},
            {"left": "$", "right": "$", "display": False},
        ],
    )
    try:
        return gr.Chatbot(type="messages", **opcoes)
    except TypeError:
        return gr.Chatbot(**opcoes)


def criar_blocos():
    try:
        return gr.Blocks(title=f"Professor IA {VERSAO}", fill_width=True, **BLOCKS_KWARGS)
    except TypeError:
        return gr.Blocks(title=f"Professor IA {VERSAO}", **BLOCKS_KWARGS)


app = criar_blocos()

with app:
    estado = gr.State(estado_vazio())

    gr.Markdown(f"# 🎓 Professor IA — Tutor Universal de Exatas {VERSAO}")
    gr.Markdown(
        "Assistente híbrido (Gemini) que ensina *passo a passo*, nunca só a resposta. "
        "Envie um PDF e estude em 4 aulas progressivas, com leitura visual da *página em foco*. "
        "O áudio narra como um professor de verdade, com pausas de respiração."
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=340):
            chave_gemini = gr.Textbox(
                label="🔒 Chave de API do Gemini (opcional se definida nos Secrets)",
                placeholder="Cole sua API Key aqui...",
                type="password",
            )
            indicador_chave = gr.HTML(INDICADOR_NEUTRO)
            btn_testar = gr.Button("🔑 Testar Chave")

            upload_pdf = gr.File(
                label="Biblioteca (Upload de PDF de aula/livro)",
                file_types=[".pdf"],
            )
            status_aulas = gr.Markdown(render_status(estado_vazio()))

            with gr.Row():
                btn_pag_prev = gr.Button("◀️ Página", size="sm")
                btn_pag_next = gr.Button("Página ▶️", size="sm")

            with gr.Row():
                btn_iniciar = gr.Button("▶️ Explicar aula atual", variant="primary")
                btn_concluir = gr.Button("✅ Concluir aula e avançar")
            btn_reiniciar = gr.Button("🔄 Reiniciar curso", size="sm")

            usar_visual = gr.Checkbox(
                label="👁️ Leitura visual da página em foco",
                value=True,
                info="Envia SOMENTE a página em foco como imagem ao Gemini (rápido e econômico).",
            )
            gerar_audio = gr.Checkbox(label="🔊 Gerar áudio didático das respostas", value=True)
            audio_resposta = gr.Audio(
                label="Explicação em Áudio (narração de professor)",
                interactive=False,
                autoplay=True,
                type="filepath",
                format="mp3",
            )

        with gr.Column(scale=4, min_width=600):
            chatbot = criar_chatbot()
            txt_mensagem = gr.Textbox(
                label="Sua dúvida ou sua resolução (Cálculo, Física, Circuitos, Estruturas...)",
                placeholder="Digite sua pergunta aqui e aperte Enter...",
            )
            btn_enviar = gr.Button("Enviar", variant="primary")

    # --- Testar chave Gemini ---
    btn_testar.click(fn=testar_chave_gemini, inputs=[chave_gemini], outputs=[indicador_chave])
    chave_gemini.change(fn=lambda: INDICADOR_NEUTRO, inputs=None, outputs=[indicador_chave])

    # --- PDF -> aulas ---
    upload_pdf.change(fn=processar_pdf, inputs=[upload_pdf, estado, chatbot], outputs=[estado, status_aulas])

    # --- Navegação de páginas ---
    btn_pag_prev.click(fn=lambda e: navegar_pagina(-1, e), inputs=[estado], outputs=[estado, status_aulas])
    btn_pag_next.click(fn=lambda e: navegar_pagina(+1, e), inputs=[estado], outputs=[estado, status_aulas])

    # --- Chat: 6 entradas / 3 saídas ---
    entradas_chat = [txt_mensagem, chatbot, estado, chave_gemini, gerar_audio, usar_visual]
    saidas_chat = [chatbot, audio_resposta, txt_mensagem]
    btn_enviar.click(fn=responder_usuario, inputs=entradas_chat, outputs=saidas_chat)
    txt_mensagem.submit(fn=responder_usuario, inputs=entradas_chat, outputs=saidas_chat)

    # --- Ações de aula ---
    saidas_aula = [chatbot, audio_resposta, estado, status_aulas]
    btn_iniciar.click(
        fn=acao_iniciar_aula,
        inputs=[chatbot, estado, chave_gemini, gerar_audio, usar_visual],
        outputs=saidas_aula,
    )
    btn_concluir.click(
        fn=acao_concluir_aula,
        inputs=[chatbot, estado, chave_gemini, usar_visual],
        outputs=saidas_aula,
    )
    btn_reiniciar.click(
        fn=acao_reiniciar_curso,
        inputs=[chatbot, estado],
        outputs=saidas_aula,
    )


# ============================================================================
# ARRANQUE — compatível com Hugging Face Spaces
# (sem server_port fixo, sem share; o HF cuida da exposição)
# ============================================================================
print("=" * 72)
print(f"🎓 Professor IA — Tutor Universal de Exatas ({VERSAO})")
print("=" * 72)
print(f"  Python     : {sys.version.split()[0]}")
print(f"  Gradio     : {getattr(gr, '_version_', '?')}")
print(f"  PyMuPDF    : {'OK' if fitz else 'AUSENTE (pip install pymupdf)'}")
print(f"  Edge-TTS   : {'OK' if edge_tts else 'AUSENTE (pip install edge-tts)'}")
print(f"  Biblioteca : {LIB}")
print(f"  Chave Gemini: {'OK (env)' if CHAVE_ENV else 'ausente — configure no campo ou nos Secrets'}")
print("=" * 72)

try:
    app.launch(**LAUNCH_KWARGS)
except Exception as _e:
    print("\n----------------------------------------")
    print(f"[X] ERRO FATAL: {type(e).name_}: {_e}")
    print("----------------------------------------\n")
    traceback.print_exc()
