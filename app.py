"""
Professor IA — Engenharia e Matemática (v13-cloud)
- Hugging Face Spaces: sem Ollama (localhost não existe), só Gemini na nuvem
- Chave lida de GEMINI_API_KEY (Segredos do HF); campo de texto como fallback
- Tema escuro + quadro negro (giz branco/amarelo)
- Leitura VISUAL do PDF: SOMENTE a página em foco é renderizada
- Navegação de páginas (◀ / ▶) com cache LRU
- PDF dividido em 4 aulas progressivas e dependentes
- TTS DIDÁTICO: roteiro falado como professor na lousa (EDOs com coeficientes A, B e C)
- Handlers com parâmetros finais OPCIONAIS (nunca mais "necessário: 6, recebidos: 5")
- Página marcada como notranslate (Chrome não traduz mais a interface)
- Pasta de trabalho em /tmp (HF: fora de /tmp o disco é somente leitura)
"""
from __future__ import annotations

import asyncio
importar base64
importar cópia
importar io
importar os
importar re
importar shutil
importar sys
importar threading
tempo de importação
rastreamento de importação
importar uuid
from collections import OrderedDict
from pathlib import Path

importar gradio como gr
solicitações de importação
from pypdf import PdfReader, PdfWriter

tentar :
    importar edge_tts
exceto ImportError:
    edge_tts = Nenhum

tentar :
    import pymupdf as fitz
exceto ImportError:
    tentar :
        importar fitz
    exceto ImportError:
        fitz = Nenhum

VERSAO = "v13-nuvem"

# ----------------------------------------------------------------------------
# Diretórios — no HF Spaces, o disco é somente leitura fora de /tmp
# ----------------------------------------------------------------------------
se os.environ.get( "SPACE_ID" ) ou  não os.access(Path(__file__).resolve().parent, os.W_OK):
    BASE = Caminho( "/tmp" )
outro :
    BASE = Path(__file__).resolve().parent
LIB = BASE / "biblioteca"
tentar :
    LIB.mkdir(parents= True , exist_ok= True )
exceto Exceção:
    BASE = Caminho( "/tmp" )
    LIB = BASE / "biblioteca"
    LIB.mkdir(parents= True , exist_ok= True )

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"

# Chave do Gemini: lê a primeira variável de ambiente (Secrets do HF)
CHAVE_ENV = (os.environ.get( "GEMINI_API_KEY" ) or  "" ).strip()

N_AULAS = 4
LIMITE_AULA_GEMINI = 100_000
LIMITE_RESUMO = 2_500
MAX_MENSAGENS_HISTORICO = 8

# --- Leitura visual da página em foco (memória e custo equilibrado) ---
DPI_VISUAL = 110
QUALIDADE_JPEG = 70
MAX_LADO_PX = 1400
USAR_CINZA = Verdadeiro
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
    "Química, Engenharia, Circuitos Elétricos, Mecânica, Estatística, Cálculo, Álgebra etc.)"
    "Fale sempre em português do Brasil, com didática, precisão e estrutura clara.\n\n"
    "REGRA INVIOLÁVEL — MÉTODO PASSO A PASSO:\n"
    "Para QUALQUER problema, cálculo, expressão numérica, descoberta, circuito, diagrama de forças, "
    "derivada, integral, limite, sistema linear, conversão de unidades, etc., você NUNCA deve"
    "entregar apenas a resposta final, o resultado numérico ou um resumo. É OBRIGATÓRIO:\n"
    " 1) identificar o que é dado e o que é pedido.\n"
    " 2) Escolher e prescrever o método / lei / fórmula (Lei de Ohm, Kirchhoff, 2ª Lei de Newton, "
    "regra da cadeia, Bhaskara...).\n"
    " 3) MONTAR AS EQUAÇÕES explicitamente, com variáveis ​​e unidades (LaTeX $...$ ou $$...$$).\n"
    " 4) Resolver PASSO A PASSO, mostrando cada manipulação algébrica e substituição numérica.\n"
    " 5) Conferir dimensionalmente.\n"
    " 6) Só então apresente a resposta final destacada.\n"
    " 7) Fechar com 'como você resolveria sozinho na próxima vez'.\n\n"
    "Se o aluno pedir apenas 'a resposta', conduza o julgamento antes de dar-la."
    "Se enviar a própria resolução, corrija item por item, aponte onde errou e mostre o "
    "caminho correto.\n\n"
    "Ao analisar circuitos nas imagens, PRIMEIRO liste componentes e valores legíveis e descreva "
    "a topologia (série, paralela, mista, malhas, nós) ANTES de calcular. Nunca invente valores "
    "ilegíveis: diga qual não ficou claro e peça de confirmação.\n\n"
    "Se houver contexto de um documento (PDF/aula), baseie-se nele e diga quando algo não estiver disponível "
    "sem texto."
)

PROMPT_RESUMO = (
    "Gere um RESUMO DE MEMÓRIA desta aula (máx. 250 palavras) para as próximas aulas: "
    "definições, fórmulas (LaTeX), métodos, circuitos e valores relevantes, resultados numéricos, "
    "o que o aluno resolveu ou errou e o ponto onde o texto da aula terminou."
    "Responda apenas com o resumo."
)

_cache_modelo_gemini: dict = {}
_cache_paginas_visual: "OrderedDict[tuple, list]" = OrderedDict()


# ============================================================================
# MOTOR DE IA — Gemini (Ollama removido: não existe no HF Spaces)
# ============================================================================
def  detectar_modelo_gemini ( api_key: str ) -> str :
    """Lista os modelos e escolhe o melhor 'flash' estável para a chave."""
    se api_key estiver em _cache_modelo_gemini:
        retornar _cache_modelo_gemini[api_key]
    escolhido = "gemini-2.0-flash"
    tentar :
        r = solicitações.get(
            f" {GEMINI_BASE} /modelos" ,
            cabeçalhos={ "x-goog-api-key" : api_key},
            params={ "pageSize" : 100 },
            tempo limite = 15 ,
        )
        r.raise_for_status()
        candidatos = []
        para m em r.json().get( "models" , []):
            nome = m.get( "name" , "" ).replace( "models/" , "" )
            se  "generateContent"  não estiver  em m.get( "supportedGenerationMethods" , []):
                continuar
            se  não nome.startswith( "gemini" ):
                continuar
            se  houver algum (x em nome para x em ( "embedding" , "tts" , "image" , "vision" , "live" , "audio" , "robotics" )):
                continuar
            pontos = 0,0
            se  "flash"  em nome:
                pontos += 10
            se  "lite"  em nome:
                pontos == 3
            se  houver (x em nome para x em ( "preview" , "exp" , "thinking" )):
                pontos == 5
            se  "mais recente"  em nome:
                pontos += 2
            v = re.search( r"gemini-(\d+(?:\.\d+)?)" , nome)
            se v:
                pontos += float (v.group( 1 ))
            candidatos.append((pontos, nome))
        se os candidatos:
            escolhido = max (candidatos)[ 1 ]
    exceto Exception como e:
        print ( f"[Gemini] Não foi possível listar modelos ( {e} ); usando {escolhido} ." )
    _cache_modelo_gemini[api_key] = escolhido
    print ( f"[Gemini] Modelo em uso: {escolhido} " )
    retorno​


def  chamar_gemini ( pergunta, contexto, histórico, api_key, anexos= None ):
    """
    Chama o Gemini, opcionalmente com anexos visuais.
    Retorna SEMPRE uma tupla (texto, tipo_de_erro):
      - (str, "") → sucesso
      - (Nenhum, "vazio") → resposta bloqueada ou sem texto
      - (Nenhum, "http") → erro HTTP (401, 403, 429, 500...)
      - (Nenhum, "timeout") → demorou demais
      - (Nenhum, "rede") → erro de conexão
    """
    modelo = detectar_modelo_gemini(api_key)
    url = f" {GEMINI_BASE} /models/ {modelo} :generateContent"
    conteúdo = []
    para mensagem no histórico[-MAX_MENSAGENS_HISTORICO:]:
        role = "user"  se msg[ "role" ] == "user"  senão  "model"
        contents.append({ "role" : role, "parts" : [{ "text" : msg[ "content" ]}]})
    texto_usuario = f" {contexto} Pergunta do aluno: {pergunta} "  if contexto else pergunta
    partes_usuario = []
    para mime, dados, rotulo em anexos ou []:
        se rotulo:
            partes_usuario.append({ "texto" : rotulo})
        partes_usuario.append({
            "inlineData" : {
                "mimeType" : mime,
                "dados" : base64.b64encode(dados).decode( "ascii" ),
            }
        })
    partes_usuario.append({ "texto" : texto_usuario})
    contents.append({ "role" : "user" , "parts" : partes_usuario})
    carga útil = {
        "systemInstruction" : { "parts" : [{ "text" : PROMPT_SISTEMA}]},
        "conteúdo" : conteúdo,
    }
    _t0 = time.perf_counter()
    tentar :
        r = solicitações.post(
            URL,
            headers={ "Content-Type" : "application/json" , "x-goog-api-key" : api_key},
            json=carga útil,
            tempo limite = 60  se anexos senão  45 ,
        )
        se r.status_code == 404 :
            _cache_modelo_gemini.pop(api_key, None )
        r.raise_for_status()
        print ( f"[TEMPO] Gemini respondeu em {time.perf_counter() - _t0: .1 f} s" )
        res_json = r.json()
        se  não (res_json.get( "candidates" ) e res_json[ "candidates" ][ 0 ].get( "content" )):
            retornar  Nenhum , "vazio"
        partes = res_json[ "candidates" ][ 0 ][ "content" ].get( "parts" , [])
        texto = "" .join(p.get( "texto" , "" ) for p in partes).strip()
        return (texto, "" ) if texto else ( None , "vazio" )
    exceto requests.Timeout como e:
        print ( f"[Gemini] tempo limite: {e} " )
        retornar  None , "tempo limite"
    exceto requests.HTTPError como e:
        corpo = ""
        tentar :
            corpo = e.response.text[: 300 ]
        exceto Exceção:
            passar
        imprimir ( f"[Gemini] Erro HTTP: {e}  {corpo} " )
        retornar  Nenhum , "http"
    exceto requests.RequestException como e:
        print ( f"[Gemini] erro de rede: {e} " )
        retornar  Nenhum , "rede"
    exceto (KeyError, IndexError, ValueError) como e:
        print ( f"[Gêmeos] resposta inesperada/bloqueada: {e} " )
        retornar  Nenhum , "vazio"


def  motor_hibrido ( pergunta, estado, histórico, api_key, usar_visual= True ):
    """
    Versão nuvem: usa SOMENTE Gemini.
    Se a chave estiver vazia, devolva uma mensagem amigável e não tente nada.
    """
    vago = []
    chave = (api_key ou CHAVE_ENV ou  "" ).strip()
    se  não for chave:
        retornar (
            "⚠️ **Nenhuma chave do Gemini configurada.**\n\n"
            **Como resolver:**\n
            "• Espaços sem abraços no rosto: vá em **Configurações → Variáveis ​​e segredos** e chore um "
            "secret chamado `GEMINI_API_KEY` com sua chave.\n"
            "• Ou cole a chave no campo à esquerda e clique em **Testar Chave**.\n\n"
            "Obtenha uma chave grátis em: https://aistudio.google.com/app/apikey" ,
            Falso ,
            "" ,
        )

    tem_pdf = bool (estado.get( "aulas" )) e  bool (estado.get( "caminho_pdf" ))
    anexos, desc_visual = [], ""
    se usar_visual e tem_pdf:
        anexos, desc_visual, nota_pdf = preparar_anexo_pagina_atual(estado)
        se nota_pdf:
            notas.append(nota_pdf)

    ctx = montar_contexto(estado, LIMITE_AULA_GEMINI, desc_visual if anexos else  "" )
    resultado, erro = chamar_gemini(pergunta, ctx, histórico, chave, anexos ou  None )

    se  não resultado e anexos e erro em ( "http" , "vazio" ):
        print ( "[motor_hibrido] Repetindo chamada sem anexo visual..." )
        ctx = montar_contexto(estado, LIMITE_AULA_GEMINI, "" )
        resultado, erro = chamar_gemini(pergunta, ctx, histórico, chave, None )
        se resultado:
            notas.append( "⚠️ Não consegui enviar a página como imagem; respondi só com o texto." )

    se resultado:
        retornar resultado, True , " " .join(notas)

    se erro == "tempo limite" :
        return ( "⚠️ O Gêmeos demorou demais para responder. Tente novamente — ou desmarque "
                "'leitura visual' para acelerar." , Falso , "" )
    se erro == "http" :
        return ( "⚠️ A API do Gemini respondeu com erro. Verifique se a chave está correta e se a "
                "cota não foi excedida." , Falso , "" )
    se erro == "rede" :
        return ( "⚠️ Não consegui falar com a API do Gemini. Verifique a conexão do Space." ,
                Falso , "" )
    return ( "⚠️ Não consegui obter a resposta do Gemini. Tente novamente em alguns segundos." ,
            Falso , "" )


# ============================================================================
# TESTE DA CHAVE GEMINI
# ============================================================================
def  _indicador_html ( cor, texto, detalhe= "" ):
    núcleos = {
        "verde" : ( "#22c55e" , "rgba(34,197,94,0.12)" ),
        "vermelho" : ( "#ef4444" , "rgba(239,68,68,0.12)" ),
        "cinza" : ( "#9ca3af" , "rgba(156.163.175.0.10)" ),
    }
    fg, bg = cores[cor]
    extra = (
        f'<div style="font-weight:400;font-size:0.8em;opacity:0.85;margin-top:4px"> {detalhe} </div>'
        se detalhe
        outro  ""
    )
    retornar (
        f'<div style="padding:10px 14px;border-radius:10px;border:1px solid {fg} ;'
        f'background: {bg} ;color: {fg} ;font-weight:600"> {texto} {extra} </div>'
    )


se CHAVE_ENV:
    INDICADOR_NEUTRO = _indicador_html(
        "verde" ,
        "🟢 Chave compartilhada dos segredos do Hugging Face" ,
        "Você também pode colar outra chave no campo acima para substituir temporariamente." ,
    )
outro :
    INDICADOR_NEUTRO = _indicador_html(
        "cinza" ,
        "⚪ Nenhuma chave configurada" ,
        "Cole a chave no campo acima ou defina o secret GEMINI_API_KEY no Space." ,
    )


def  testar_chave_gemini ( api_key ):
    api_key = (api_key ou CHAVE_ENV ou  "" ).strip()
    se  não for uma chave de API:
        return _indicador_html( "cinza" , "⚪ Cole uma chave para testar" )
    tentar :
        r = solicitações.get(
            f" {GEMINI_BASE} /modelos" ,
            cabeçalhos={ "x-goog-api-key" : api_key},
            params={ "pageSize" : 1 },
            tempo limite = 10 ,
        )
    exceto requests.RequestException como e:
        return _indicador_html( "vermelho" , "🔴 Chave Inválida / Erro de Conexão" ,
                               f"Sem conexão: { type (e).__name__} " )
    se r.status_code == 200 :
        _cache_modelo_gemini.pop(api_key, None )
        modelo = detectar_modelo_gemini(api_key)
        return _indicador_html( "verde" , "🟢 Chave Gemini VALIDA e Conectada!" ,
                               f"Modelo que será usado: {modelo} " )
    tentar :
        msg = r.json().get( "error" , {}).get( "message" , "" )
    exceto ValueError:
        msg = ""
    detalhe = f"HTTP {r.status_code} " + ( f" — {msg[: 120 ]} "  if msg else  "" )
    return _indicador_html( "vermelho" , "🔴 Chave Inválida / Erro de Conexão" , detalhe)


# ============================================================================
# PDF — remoção de texto e divisão em aulas
# ============================================================================
def  extrair_paginas_pdf ( caminho_pdf ):
    """Somente TEXTO (nunca imagem). PyMuPDF é ~10-50x mais rápido que o pypdf."""
    t0 = tempo.perf_counter()
    se fitz não  for  None :
        tentar :
            com fitz. abra (caminho_pdf) como documento:
                se doc.needs_pass e  não doc.authenticate( "" ):
                    retornar []
                paginas = [(pg.get_text( "texto" ) ou  "" ) para pg em doc]
            imprimir ( f"[TEMPO] texto de { len (páginas)} páginas (PyMuPDF): {time.perf_counter() - t0: .2 f} s" )
            retornar páginas
        exceto Exception como e:
            print ( f"[PDF] PyMuPDF falhou no texto ( {e} ); usando pypdf." )
    tentar :
        leitor = PDFReader(caminho_pdf)
        se reader.is_encrypted:
            leitor.descriptografar( "" )
    exceto Exception como e:
        print ( f"[PDF] Erro ao ler: {e} " )
        retornar []
    paginas = []
    para pagina em reader.pages:
        tentar :
            paginas.append(pagina.extract_text() ou  "" )
        exceto Exceção:
            paginas.append( "" )
    retornar páginas


def  _rotulo_paginas ( ini, fim ):
    return  f"página {ini} "  if ini == fim else  f"páginas {ini} – {fim} "


def  dividir_em_aulas ( paginas, n=N_AULAS ):
    paginas = [p.strip() para p em paginas]
    total = soma ( len (p) para p em páginas)
    n = máx ( 1 , min (n, total // 400 ))
    se  len (páginas) >= n:
        alvo = total / n
        grupos = []
        início = 0
        acumulado = 0
        para i, p em  enumerate (páginas):
            acumulado += len (p)
            restante = len (páginas) - (i + 1 )
            faltando = n - len (grupos) - 1
            se  len (grupos) < n - 1  e (
                restante == faltando
                ou (acumulado >= alvo * ( len (grupos) + 1 ) e restante >= faltando)
            ):
                grupos.append((inicio, i))
                início = i + 1
        grupos.append((inicio, len (páginas) - 1 ))
        retornar [
            {
                "texto" : "\n\n" .join(páginas[a: b + 1 ]).strip(),
                "pag_ini" : a + 1 ,
                "pag_fim" : b + 1 ,
                "rotulo" : _rotulo_paginas(a + 1 , b + 1 ),
                "modo" : "páginas" ,
            }
            para a, b em grupos
        ]
    texto = "\n" .join(paginas)
    cortes = [ 0 ]
    para k no  intervalo ( 1 , n):
        pos = int ( len (texto) * k / n)
        nl = texto.find( "\n" , pos)
        corte = nl se (nl != -1  e nl - pos <= 2000 ) senão pos
        cortes.append( max (corte, cortes[- 1 ]))
    cortes.append( len (texto))
    retornar [
        {
            "texto" : texto[cortes[i]: cortes[i + 1 ]].strip(),
            "pag_ini" : 1 ,
            "pag_fim" : len (páginas),
            "rotulo" : f"parte {i + 1 } de {n} do texto" ,
            "modo" : "texto" ,
        }
        para i no  intervalo (n)
    ]


# ============================================================================
# RENDERIZAÇÃO DE UMA PÁGINA (leitura visual)
# ============================================================================
def  _renderizar_uma_pagina ( caminho, numero_pagina ):
    Se Fitz for  None :
        retornar  Nenhum
    se  não Path(caminho).exists():
        print ( f"[PyMuPDF] arquivo não existe: {caminho} " )
        retornar  Nenhum
    resultado = { "bytes" : Nenhum , "erro" : Nenhum }

    def  __ ():
        t0 = tempo.perf_counter()
        tentar :
            com fitz. abra (caminho) conforme doc:
                se doc.needs_pass:
                    doc.autenticar( "" )
                idx = numero_pagina - 1
                se idx < 0  ou idx >= doc.page_count:
                    raise IndexError( f"página {numero_pagina} fora de [1.. {doc.page_count} ]" )
                pagina = doc.load_page(idx)
                zoom = DPI_VISUAL / 72
                maior_lado = max (pagina.rect.width, pagina.rect.height) * zoom
                se maior_lado > MAX_LADO_PX:
                    zoom *= MAX_LADO_PX / maior_lado
                matriz = fitz.Matrix(zoom, zoom)
                pix = pagina.get_pixmap(
                    matriz=matriz,
                    colorspace=(fitz.csGRAY se USAR_CINZA senão fitz.csRGB),
                    alfa = Falso ,
                )
                buf = io.BytesIO()
                tentar :
                    buf.write(pix.tobytes( "jpeg" , jpg_quality=QUALIDADE_JPEG))
                exceto TypeError:
                    buf.write(pix.tobytes( "jpeg" ))
                dados = buf.getvalue()
                pix = Nenhum
                if  len (dados) > MAX_BYTES_PAGINA_VISUAL:
                    raise ValueError( f"JPEG grande demais: { len (dados) // 1024 } KB" )
                resultado[ "bytes" ] = dados
                print ( f"[PyMuPDF] página {numero_pagina} renderizada em "
                      f" {time.perf_counter() - t0: .2 f} s ( { len (dados) // 1024 } KB)" )
        exceto Exception como e:
            resultado[ "erro" ] = e
            print ( f"[PyMuPDF] erro ao renderizar página {numero_pagina} : {e} " )

    t = threading.Thread(target=_trabalho, daemon= True )
    t.iniciar()
    t.join(timeout=RENDER_TIMEOUT_SEG)
    se t.is_alive():
        print ( f"[PyMuPDF] TIMEOUT ( {RENDER_TIMEOUT_SEG} s) na página {numero_pagina} ; abortando." )
        retornar  Nenhum
    se resultado[ "erro" ]:
        retornar  Nenhum
    retornar resultado[ "bytes" ]


def  _anexos_da_pagina ( caminho, numero_pagina ):
    """Lazy: renderiza SÓ esta página, apenas uma vez (cache LRU, inclusive de falhas)."""
    chave = (caminho, número_pagina)
    se chave em _cache_paginas_visual:
        _cache_paginas_visual.move_to_end(chave)
        return _cache_paginas_visual[chave]
    dados = _renderizar_uma_pagina(caminho, numero_pagina)
    anexos = []
    Se dados não  for  None :
        anexos = [( "image/jpeg" , dados, f"[Página {numero_pagina} do material]" )]
    elif fitz é  Nenhum :
        tentar :
            leitor = PdfReader(caminho)
            se reader.is_encrypted:
                leitor.descriptografar( "" )
            escritor = PdfWriter()
            escritor.adicionar_página(leitor.páginas[número_página - 1 ])
            buf = io.BytesIO()
            escritor.escrever(buf)
            dados_pdf = buf.getvalue()
            se  len (dados_pdf) <= MAX_BYTES_FALLBACK_PDF:
                anexos = [( "aplicativo/pdf" , dados_pdf, f"[Página {numero_pagina} do material]" )]
                print ( f"[PDF] substituto: página {numero_pagina} como PDF ( { len (dados_pdf) // 1024 } KB)." )
            outro :
                print ( f"[PDF] fallback pesado ( { len (dados_pdf) // 1024 } KB): sem imagem. "
                      "Instale o PyMuPDF (pip install pymupdf)." )
        exceto Exception como e:
            print ( f"[PDF] falha no fallback da página {numero_pagina} : {e} " )
    _cache_paginas_visual[chave] = anexos
    enquanto  len (_cache_paginas_visual) > CACHE_PAGINAS_MAX:
        _cache_paginas_visual.popitem(último= Falso )
    retornar anexos


def  preparar_anexo_pagina_atual ( estado ):
    aulas = estado.get( "aulas" ) ou []
    caminho = estado.get( "caminho_pdf" )
    se  não aulas ou  não caminho ou  não Path(caminho).exists():
        retornar [], "" , ""
    aula = aulas[estado[ "atual" ]]
    pag = estado.get( "pagina_atual" ) ou aula[ "pag_ini" ]
    pag = max (aula[ "pag_ini" ], min (pag, aula[ "pag_fim" ]))
    anexos = _anexos_da_pagina(caminho, pag)
    se  não houver anexos:
        return [], "" , "⚠️ Não consegui renderizar esta página para leitura visual; usei só o texto."
    desc = f"a página {pag} do material original (página em foco nesta aula)"
    retornar anexos, desc, ""


def  estado_vazio ():
    retornar {
        "aulas" : [],
        "resumos" : [],
        "atual" : 0 ,
        "fim" : Falso ,
        "nome" : "" ,
        "cameno_pdf" : "" ,
        "hist_inicio" : 0 ,
        "pagina_atual" : 1 ,
    }


def  _garantir_estado ( estado ):
    base = estado_vazio()
    if  isinstance (estado, dict ):
        base.atualizar(estado)
    retornar base


def  _descartar_pdf_do_curso ( estado ):
    caminho = (estado ou {}).get( "caminho_pdf" )
    se  não for caminho:
        retornar
    for chave in [k for k in _cache_paginas_visual if k[ 0 ] == caminho]:
        _cache_paginas_visual.pop(chave, Nenhum )
    tentar :
        Path(caminho).unlink(missing_ok= True )
    exceto OSError:
        passar


def  montar_contexto ( estado, limite_texto, visual_desc= "" ):
    aulas = estado.get( "aulas" ) ou []
    se  não forem aulas:
        retornar  ""
    i = estado[ "atual" ]
    n = len (aulas)
    partes = [
        f"Você está conduzindo um curso em {n} aulas progressivas a partir do documento "
        f"' {estado.get( 'nome' , '' )} '. Aula atual: {i + 1 } de {n} . "
        "Cada aula continua exatamente de onde a anterior parou."
    ]
    se i > 0 :
        resumos = "\n\n" .join( f"Aula {k + 1 } : {r} "  para k, r em  enumerar (estado[ "resumos" ][:i]))
        partes.append( f"Resumo acumulado das aulas anteriores:\n {resumos} " )
        partes.append( f"Final do texto da aula anterior:\n {aulas[i - 1 ][ 'texto' ][- 800 :]} " )
    se visual_desc:
        partes.append(
            f"ANEXO VISUAL: o anexo (imagem) contém {visual_desc} . Analise o diagrama, esquema de "
            "circuito, gráficos ou tabela diretamente nessa imagem e leia os valores dos componentes "
            "(tensões, correntes, resistências etc.). O texto extraído abaixo pode estar incompleto; "
            "se um valor estiver ilegível, diga qual e peça confirmada em vez de inventar."
        )
    texto = aulas[i][ "texto" ]
    if  len (texto) > limite_texto:
        texto = texto[:limite_texto] + "\n[... texto da aula truncado por limite de tamanho ...]"
    partes.append( f"Texto extraído da Aula {i + 1 } :\n {texto} " )
    retornar  "\n\n" .join(partes) + "\n\n"


def  render_status ( estado ):
    aulas = estado.get( "aulas" ) ou []
    se  não forem aulas:
        return  "📭 **Nenhum PDF carregado.** Envie um PDF para dividir-lo em aulas, ou conversar normalmente."
    n = len (aulas)
    atual = estado[ "atual" ]
    linhas = []
    para i, a em  enumerate (aulas):
        if i < atual ou (i == atual e estado[ "fim" ]):
            ícone = "✅"
        senão se i == atual:
            ícone = "▶️"
        outro :
            ícone = "🔒"
        linhas.append( f" {icone} **Aula {i + 1 } ** — {a[ 'rotulo' ]} " )
    se estado[ "fim" ]:
        titulo = f"### 🎉 Curso concluído ( {n} de {n} aulas)"
    outro :
        titulo = f"### 📚 Aula {atual + 1 } de {n} "
    futuro = aulas[atual]
    pag_foco = estado.get( "pagina_atual" ) ou aula[ "pag_ini" ]
    pag_foco = max (aula[ "pag_ini" ], min (pag_foco, aula[ "pag_fim" ]))
    foco = f"👁️ **Página em foco:** {pag_foco} (de {aula[ 'pag_ini' ]} – {aula[ 'pag_fim' ]} nesta aula)"
    cajado = [foco]
    tam = len (aula[ "texto" ])
    se tam > LIMITE_AULA_GEMINI:
        notas.append(
            f"⚠️ Esta aula tem {tam:,} caracteres: a IA ver só os primeiros "
            f" {LIMITE_AULA_GEMINI:,} ."
        )
    return  f" {título} \n\n" + "\n\n" .join(linhas) + "\n\n" + "\n\n" .join(notas)


def  _caminho_do_arquivo ( arquivo ):
    Se o arquivo for  None :
        retornar  Nenhum
    if  isinstance (arquivo, str ):
        retornar arquivo
    se  isinstance (arquivo, dict ):
        return arquivo.get( "caminho" ) ou arquivo.get( "nome" )
    retornar  getattr (arquivo, "name" , None ) ou  getattr (arquivo, "path" , None )


# ============================================================================
#TTS DIDÁTICO — Roteiro falou como professor na lousa
# ============================================================================
_NUM_EXTENSO = {
    "0" : "zero" , "1" : "um" , "2" : "dois" , "3" : "três" , "4" : "quatro" ,
    "5" : "cinco" , "6" : "seis" , "7" : "sete" , "8" : "oito" , "9" : "nove" ,
    "10" : "dez" , "11" : "onze" , "12" : "cochilar" ,
    "-1" : "menos um" , "-2" : "menos dois" , "-3" : "menos três" , "-4" : "menos quatro" ,
    "-5" : "menos cinco" , "-6" : "menos seis" , "-7" : "menos sete" ,
    "-8" : "menos oito" , "-9" : "menos nove" , "-10" : "menos dez" ,
}

_LATEX_PALAVRAS = {
    "cdot" : "vezes" , "vezes" : "vezes" , "div" : "dividido por" ,
    "Ômega" : "ohms" , "ômega" : "ômega" , "pi" : "pi" , "mu" : "micro" ,
    "Delta" : "delta" , "delta" : "delta" , "teta" : "teta" , "alfa" : "alfa" ,
    "beta" : "beta" , "phi" : "fi" , "lambda" : "lambda" , "sigma" : "sigma" , "tau" : "tau" ,
    "approx" : " aproximadamente " , "leq" : "menor ou igual a " , "geq" : "maior ou igual a " ,
    "neq" : "diferente de" , "pm" : "mais ou menos" , "parallel" : "em paralelo com" ,
    "sum" : " somatório de " , "int" : "integral de " , "infty" : "infinito " ,
    "to" : "tende a " , "rightarrow" : "leva a " , "angle" : "ângulo" , "circ" : "graus" ,
    "sin" : "seno de " , "cos" : " cosseno de " , "tan" : "tangente de " ,
    "log" : "log de" , "ln" : "logaritmo natural de" ,
    "frac" : " sobre " , "sqrt" : " raiz quadrada de " ,
    "qquad" : " , " , "quad" : " , " , "text" : " " , "mathrm" : " " ,
}


def  _num_falado ( valor ):
    v = (valor ou  "" ).strip()
    se v estiver em _NUM_EXTENSO:
        retornar _NUM_EXTENSO[v]
    se v.startswith( "-" ) e v[ 1 :] em _NUM_EXTENSO:
        retornar _NUM_EXTENSO[v]
    se v.startswith( "-" ):
        retornar  f"menos {v[ 1 :]} "
    retornar v


def  _extrair_coeficientes_edo ( texto_md ):
    """
    Procura por A·y'' + B·y' + C·y = 0 e devolve os coeficientes.
    Termos ausentes (B ou C) viram zero. Aceita LaTeX (d²y/dx²).
    """
    t = texto_md.replace( "$" , " " ).replace( "\\," , " " ).replace( "\\!" , " " )
    t = re.sub( r"\s+" , " " , t).strip()
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
    padrao = re. compilar (termo_a + termo_b + termo_c + r"\s*=\s*0" , re.IGNORECASE)
    m = padrao.search(t)
    se  não m:
        retornar  Nenhum
    equacao_original = t[m.start():m.end()].strip()

    def  _norm ( num, sinal, forca_um= False ):
        Se num for  None  ou num.strip() == "" :
            se forca_um:
                retorna  "-1"  se sinal == "-"  senão  "1"
            retornar  "0"
        v = num.strip().replace( " " , "" ).replace( "\\frac" , "frac" )
        se sinal == "-"  e  não v.startswith( "-" ):
            v = "-" + v
        retornar v

    se m.group( "a_neg" ):
        A = "-1"
    outro :
        A = _norm(m.group( "a_num" ), None , forca_um= True )
    B = "0"  se m.group( "b_sinal" ) for  None  senão _norm(m.group( "b_num" ), m.group( "b_sinal" ), forca_um= True )
    C = "0"  se m.group( "c_sinal" ) for  None  senão _norm(m.group( "c_num" ), m.group( "c_sinal" ), forca_um= True )
    return { "A" : A, "B" : B, "C" : C, "equacao" : equacao_original}


def  _equacao_falada ( equação ):
    t = equação
    t = re.sub( r"y\s*''|y\s*¨|y\s*′′" , " ípsilon duas linhas " , t)
    t = re.sub( r"y\s*'|y\s*˙|y\s*′" , " ípsilon linha " , t)
    t = re.sub( r"(?<![A-Za-z])y(?![A-Za-z])" , " ípsilon " , t)
    t = t.replace( "+" , "mais " ).replace( "-" , "menos " )
    t = t.replace( "=" , " igual a " )
    t = re.sub( r"\s+" , " " , t).strip()
    retornar t


def  _rotulo_item ( texto_md ):
    m = re.search( r"\b(?:item|letra|exerc[ií]cio)\s+([A-Za-z])\b" , texto_md, re.IGNORECASE)
    retornar  f"item {m.group( 1 ).upper()} "  se m senão  ""


def  gerar_roteiro_edo ( texto_md, coefs ):
    A, B, C = coefs[ "A" ], coefs[ "B" ], coefs[ "C" ]
    equacao_falada = _equacao_falada(coefs[ "equacao" ])
    rotulo = _rotulo_item(texto_md)
    abertura = f"Vamos resolver agora o {rotulo} . "  if rotulo else  "Vamos resolver agora. "
    A_fala = _num_falado(A)
    B_fala = _num_falado(B)
    C_fala = _num_falado(C)
    retornar (
        f" {abertura} Temos aqui a descoberta... {equacao_falada} . "
        "Primeiro, vamos achar os coeficientes A, B e C. "
        "O coeficiente A é o termo que acompanha a derivada segunda, ípsilon duas linhas."
        "O coeficiente B é o que acompanha a derivada primeira, ípsilon linha."
        "E o coeficiente C é o termo que acompanha a função ípsilon, sem derivado."
        "Lembrando que, se algum desses termos não aparecer na descoberta, "
        "o valor dele será zero."
        "Observando a nossa pesquisa, temos: "
        f"o A é igual a... {A_fala} , "
        f"o B é igual a... {B_fala} , "
        f"eo C é igual a... {C_fala} . "
        “Guardando esses três valores, já podemos seguir para o próximo passo da resolução.”
    )


def  _latex_para_fala ( f ):
    f = re.sub( r"\\(?:text|mathrm|mathbf)\{([^{}]*)\}" , r" \1 " , f)
    para _ no  intervalo ( 3 ):
        f = re.sub( r"\\d?frac\{([^{}]*)\}\{([^{}]*)\}" , r" \1 sobre \2 " , f)
    f = re.sub( r"\\sqrt\{([^{}]*)\}" , r" raiz quadrada de \1 " , f)
    f = re.sub( r"\^\{?2\}?(?!\w)" , " ao quadrado " , f)
    f = re.sub( r"\^\{?3\}?(?!\w)" , " ao cubo " , f)
    f = re.sub( r"\^\{([^{}]*)\}|\^(\w)" , lambda m: f" elevado a {m.group( 1 ) or m.group( 2 )} " , f)
    f = re.sub( r"_\{([^{}]*)\}|_(\w)" , lambda m: f" {m.group( 1 ) or m.group( 2 )} " , f)
    f = re.sub( r"\\([a-zA-Z]+)" , lambda m: _LATEX_PALAVRAS.get(m.group( 1 ), " , " ), f)
    f = f.replace( "=" , "igual a " ).replace( "+" , " mais " ).replace( "−" , "menos " )
    f = re.sub( r"\s-\s" , " menos " , f)
    f = re.sub( r"[{}\\]" , " " , f)
    retornar f


def  _limpar_para_fala_generica ( texto ):
    """Fallback: limpar markdown, converter LaTeX e inserir pausas sem subir engessado."""
    t = re.sub( r"```.*?```" , " ... trecho de código omitido ... " , texto, flags=re.S)
    t = re.sub( r"\$\$(.*?)\$\$" , lambda m: " ... " + _latex_para_fala(m.group( 1 )) + " ... " , t, flags=re.S)
    t = re.sub( r"\$([^$\n]+)\$" , lambda m: " " + _latex_para_fala(m.group( 1 )) + ", " , t)
    t = re.sub( r"\[([^\]]+)\]\([^)]+\)" , r"\1" , t)
    t = re.sub( r"[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F]" , "" , t)
    t = re.sub( r"(?m)^\s*[-•]\s+" , "" , t)
    t = re.sub( r"\n\s*\n" , " ... " , t)
    t = re.sub( r"(?<![.!?:;,])[ \t]*\n" , ". " , t)
    t = re.sub( r"(?m)^\s*(\d+)[.)]\s+" , r" passo \1, " , t)
    t = re.sub( r"\b(portanto|logo|então|assim|agora)\b" , r" ... \1," , t, flags=re.IGNORECASE)
    t = re.sub( r"\s+igual a\s+" , " ... igual a " , t)
    t = re.sub( r"([;:])\s*" , r"\1 ... " , t)
    t = re.sub( r"\b(\w+)(\s+\1\b)+" , r"\1" , t, flags=re.IGNORECASE)
    t = re.sub( r"[|~`*_#>]" , " " , t)
    t = re.sub( r"\s+" , " " , t).strip()
    t = re.sub( r"(,\s*){2,}" , ", " , t)
    t = re.sub( r"(\s*\.\s*\.\s*\.\s*)+" , " ... " , t)
    retornar t


def  gerar_roteiro_falado ( texto_md ):
    """Roteiro falou COMPLETO. Em EDOs, abre com os coeficientes A, B e C."""
    geral = _limpar_para_fala_generica(texto_md)
    coefs = _extrair_coeficientes_edo(texto_md)
    se coefs:
        tentar :
            intro = gerar_roteiro_edo(texto_md, coefs)
            return  f" {introdução} ... ... Agora, vamos acompanhar a explicação completa, do começo. ... {geral} "
        exceto Exception como e:
            print ( f"[TTS] falha ao gerar roteiro de EDO ( {e} ); usando apenas o texto completo." )
    retornar geral


def  _dividir_texto_fala ( texto, limite=LIMITE_BLOCO_TTS ):
    frases = re.split( r"(?<=[.!?;:])\s+|(?<=\.\.\.)\s+" , texto)
    blocos = []
    atual = ""
    para f em frases:
        enquanto  len (f) > limite:
            corte = f.rfind( " " , 0 , limite)
            corte = corte se corte > 0  senão limite
            parte, f = f[:corte], f[corte:].lstrip()
            se atual:
                blocos.append(atual)
                atual = ""
            blocos.append(parte)
        se  len (atual) + len (f) + 1 <= limite:
            atual = f" {atual}  {f} " .strip()
        outro :
            blocos.append(atual)
            atual = f
    se atual:
        blocos.append(atual)
    retornar [b para b em blocos se b.strip()]


assíncrono  def  _sintetizar_bloco ( texto ):
    áudio = bytearray ()
    async  for chunk in edge_tts.Communicate(texto, VOZ_TTS, rate=RATE_TTS, pitch=PITCH_TTS).stream():
        se chunk[ "type" ] == "audio" :
            áudio.extend(chunk[ "dados" ])
    retornar  bytes (áudio)


async  def  _sintetizar_blocos ( blocos ):
    sem = assíncio.Semaphore(CONCORRENCIA_TTS)

    async  def  um ( bloco ):
        assíncrono  com sem:
            ultimo_erro = RuntimeError( "não iniciado" )
            para tentado no  intervalo ( 3 ):
                tentar :
                    dados = aguardar _sintetizar_bloco(bloco)
                    se dados:
                        retornar dados
                    ultimo_erro = RuntimeError( "áudio vazio" )
                exceto Exception como e:
                    último_erro = e
                aguarde asyncio.sleep( 1.5 * (tentativa + 1 ))
            levantar último_erro

    partes = await asyncio.gather(*(um(b) for b em blocos))
    retornar  b"" .join(partes)


def  limpar_arquivos_antigos ():
    agora = tempo.tempo()
    for padrao, idade in (( "resposta_*.mp3" , IDADE_MAX_ARQUIVOS_SEG), ( "curso_*.pdf" , 24 * 3600 )):
        para f em LIB.glob(padrao):
            tentar :
                if agora - f.stat().st_mtime > idade:
                    f.unlink()
            exceto OSError:
                passar


def  gerar_audio_resposta ( texto ):
    """Gera um único MP3 com a resposta COMPLETA. Retorna (caminho_mp3, mensagem_de_erro)."""
    Se edge_tts for  None :
        return  None , "Áudio indisponível: instale o pacote edge-tts."
    roteiro = gerar_roteiro_falado(texto)
    se  não for:
        retornar  Nenhum , ""
    blocos = _dividir_texto_fala(roteiro)
    se  não blocos:
        retornar  Nenhum , ""
    limpar_arquivos_antigos()
    caminho = LIB / f"resposta_ {uuid.uuid4(). hex } .mp3"
    resultado = {}

    def  rodar ():
        tentar :
            resultado[ "áudio" ] = asyncio.run(_sintetizar_blocos(blocos))
        exceto Exception como e:
            resultado[ "erro" ] = e

    t = threading.Thread(target=rodar, daemon= True )
    t.iniciar()
    t.join(timeout= 300 + 90 * len (blocos))
    se t.is_alive():
        return  None , "O áudio demorou demais. A resposta está no texto."
    if  "erro"  em resultado ou  não resultado.get( "audio" ):
        imprimir ( f"[TTS] erro: {resultado.get( 'erro' )} " )
        return  None , "Falha ao gerar o áudio. A resposta está no texto."
    caminho.write_bytes(resultado[ "áudio" ])
    retornar  str (cadinho), ""


# ============================================================================
# HISTÓRICO
# ============================================================================
def  _texto_do_conteudo ( c ):
    Se c for  None :
        retornar  ""
    se  isinstance (c, str ):
        retornar c
    se  isinstance (c, dict ):
        retornar  str (c.get( "texto" ) ou c.get( "conteúdo" ) ou  "" )
    se  isinstance (c, ( lista , tupla )):
        retornar  " " .join(_texto_do_conteudo(x) para x em c).strip()
    retornar  str (c)


def  _normalizar_histórico ( histórico ):
    resultado = []
    para item em histórico ou []:
        se  isinstance (item, dict ):
            texto = _texto_do_conteudo(item.get( "conteúdo" ))
            se texto:
                resultado.append({ "role" : item[ "role" ], "content" : texto})
        elif  isinstance (item, ( lista , tupla )) and  len (item) == 2 :
            usuário, assistente = item
            se usuário:
                resultado.append({ "papel" : "usuário" , "conteúdo" : _texto_do_conteudo(usuário)})
            se assistente:
                resultado.append({ "papel" : "assistente" , "conteúdo" : _texto_do_conteudo(assistente)})
    retornar resultado


def  _finalizar_resposta ( resposta, ok, nota, gerar_audio ):
    caminho_audio = Nenhum
    aviso_audio = ""
    se estiver tudo bem e gerar_audio:
        _t0 = time.perf_counter()
        caminho_audio, erro = gerar_audio_resposta(resposta)
        print ( f"[TEMPO] áudio (TTS) gerado em {time.perf_counter() - _t0: .1 f} s" )
        se erro:
            aviso_audio = f"🔇 {erro} "
    extras = " " .join(x para x em (nota, aviso_audio) se x)
    texto_exibido = resposta + ( f"\n\n_ {extras} _"  se extras else  "" )
    retornar texto_exibido, caminho_audio


# ============================================================================
# AÇÕES DA INTERFACE
# (parâmetros finais com valor padrão => tolerantes a diferença de insumos)
# ============================================================================
def  processar_pdf ( arquivo_pdf, estado= None , histórico= None ):
    histórico = _normalizar_histórico(histórico)
    estado = _garantir_estado(estado)
    _descartar_pdf_do_curso(estado)
    caminho = _caminho_do_arquivo(arquivo_pdf)
    novo = estado_vazio()
    novo[ "hist_inicio" ] = len (histórico)
    se  não for caminho:
        retornar novo, render_status(novo)
    paginas = extrair_paginas_pdf(caminho)
    se  não houver páginas:
        return novo, "⚠️ Não consegui abrir este PDF (arquivo corrompido ou protegido por senha)."
    total = soma ( len (p.strip()) para p em páginas)
    Se o total for menor que 50 :
        retornar novo, (
            "⚠️ Quase não há texto neste PDF (provavelmente escaneado ou apenas imagens). "
            "Ele ainda pode ser lido visualmente, mas a divisão em aulas precisa de algum texto."
        )
    destino = LIB / f"curso_ {uuid.uuid4(). hex } .pdf"
    tentar :
        shutil.copyfile(caminho, destino)
        novo[ "caminho_pdf" ] = str (destino)
    exceto OSError como e:
        print ( f"[PDF] não consegui copiar: {e} " )
    novo[ "aulas" ] = dividir_em_aulas(paginas, N_AULAS)
    novo[ "nome" ] = Caminho(caminho).nome
    novo[ "pagina_atual" ] = novo[ "aulas" ][ 0 ][ "pag_ini" ] if novo[ "aulas" ] else  1
    retornar novo, render_status(novo)


def  navegar_pagina ( direção, estado= None ):
    estado = _garantir_estado(estado)
    aulas = estado.get( "aulas" ) ou []
    se  não forem aulas:
        retornar estado, render_status(estado)
    aula = aulas[estado[ "atual" ]]
    pag = estado.get( "pagina_atual" ) ou aula[ "pag_ini" ]
    pag = max (aula[ "pag_ini" ], min (pag + direção, aula[ "pag_fim" ]))
    estado[ "pagina_atual" ] = página
    retornar estado, render_status(estado)


def  responder_usuario ( mensagem, histórico= None , estado= None , api_key= "" , gerar_audio= True , usar_visual= True ):
    histórico = _normalizar_histórico(histórico)
    estado = _garantir_estado(estado)
    se  não for mensagem ou  não for mensagem.strip():
        retornar histórico, Nenhum , ""
    hist_modelo = histórico[estado.get( "hist_inicio" , 0 ):]
    resposta, ok, nota = motor_hibrido(mensagem.strip(), estado, hist_modelo, api_key, usar_visual)
    texto_exibido, caminho_audio = _finalizar_resposta(resposta, ok, nota, gerar_audio)
    histórico.append({ "role" : "usuário" , "conteúdo" : mensagem})
    historico.append({ "papel" : "assistente" , "conteúdo" : texto_exibido})
    retornar histórico, caminho_audio, ""


def  acao_iniciar_aula ( histórico= None , estado= None , api_key= "" , gerar_audio= True , usar_visual= True ):
    histórico = _normalizar_histórico(histórico)
    estado = _garantir_estado(estado)
    se  não estado.get( "aulas" ):
        historico.append({ "role" : "assistente" , "content" : "📭 Envie um PDF primeiro para eu montar as aulas." })
        return histórico, None , estado, render_status(estado)
    se estado[ "fim" ]:
        histórico.append({
            "papel" : "assistente" ,
            "content" : "🎉 Você já concluiu todas as aulas. Clique em **🔄 Reiniciar curso** para recomeçar." ,
        })
        return histórico, None , estado, render_status(estado)
    i = estado[ "atual" ]
    n = len (estado[ "aulas" ])
    pag = estado.get( "pagina_atual" ) ou estado[ "aulas" ][i][ "pag_ini" ]
    pedido = f"Inicie a Aula {i + 1 } de {n} , com foco principal na página {pag} do material. "
    se i > 0 :
        pedido += ( "Esta aula CONTINUA exatamente de onde a anterior parou (use o resumo e o final "
                   "da aula anterior fornecida no contexto), sem repetir o que já foi ensinado. " )
    pedido += (
        "Estrutura assim: 1) objetivo da aula; 2) conceitos e fórmulas principais desta aula, "
        "descrevendo os esquemas e circuitos das páginas (componentes e valores lidos); "
        "3) um exemplo resolvido PASSO A PASSO do zero (dados → método → equações → álgebra → "
        "resultado), demonstrando o julgamento completo; "
        "4) proponha 3 exercícios de cálculo para eu resolver agora; "
        "5) termine pedindo que eu inveje minhas resoluções."
        "Explique apenas o conteúdo desta aula, sem adiantar as próximas."
    )
    hist_modelo = histórico[estado[ "hist_inicio" ]:]
    resposta, ok, nota = motor_hibrido(pedido, estado, hist_modelo, api_key, usar_visual)
    texto_exibido, caminho_audio = _finalizar_resposta(resposta, ok, nota, gerar_audio)
    historico.append({ "role" : "user" , "content" : f"▶️ Explicar a Aula {i + 1 } de {n} (foco na página {pag} )" })
    historico.append({ "papel" : "assistente" , "conteúdo" : texto_exibido})
    return histórico, caminho_audio, estado, render_status(estado)


def  acao_concluir_aula ( histórico= None , estado= None , api_key= "" , usar_visual= True ):
    histórico = _normalizar_histórico(histórico)
    estado = _garantir_estado(estado)
    se  não estado.get( "aulas" ):
        historico.append({ "role" : "assistente" , "content" : "📭 Envie um PDF primeiro para eu montar as aulas." })
        return histórico, None , estado, render_status(estado)
    se estado[ "fim" ]:
        histórico.append({ "role" : "assistente" , "content" : "🎉 O curso já está concluído." })
        return histórico, None , estado, render_status(estado)
    i = estado[ "atual" ]
    n = len (estado[ "aulas" ])
    hist_modelo = histórico[estado[ "hist_inicio" ]:]
    resumo, ok, _nota = motor_hibrido(PROMPT_RESUMO, estado, hist_modelo, api_key, False )
    Se  não estiver tudo bem:
        t = estado[ "aulas" ][i][ "texto" ]
        resumo = f" {t[: 700 ]} [...] {t[- 500 :]} "  se  len (t) > 1200  senão t
    estado[ "resumos" ] = estado[ "resumos" ][:i] + [resumo[:LIMITE_RESUMO]]
    se i + 1 < n:
        estado[ "atual" ] = i + 1
        estado[ "pagina_atual" ] = estado[ "aulas" ][i + 1 ][ "pag_ini" ]
        msg = (
            f"✅ **Aula {i + 1 } concluída!**\n\n**Resumo salvo para as próximas aulas:**\n {resumo} \n\n"
            f"➡️ Agora você está na **Aula {i + 2 } de {n} **. "
            "Clique em **▶️ Explicar aula atual** quando quiser continuar."
        )
    outro :
        estado[ "fim" ] = Verdadeiro
        msg = (
            f"🎉 **Aula {n} concluída — você terminou todo o material!**\n\n"
            f"**Resumo da última aula:**\n {resumo} "
        )
    historico.append({ "role" : "user" , "content" : f"✅ Concluir a Aula {i + 1 } " })
    historico.append({ "role" : "assistant" , "content" : msg})
    estado[ "hist_inicio" ] = len (histórico)
    return histórico, None , estado, render_status(estado)


def  acao_reiniciar_curso ( histórico= Nenhum , estado= Nenhum ):
    histórico = _normalizar_histórico(histórico)
    estado = _garantir_estado(estado)
    se  não estado.get( "aulas" ):
        return histórico, None , estado, render_status(estado)
    estado[ "atual" ] = 0
    estado[ "resumos" ] = []
    estado[ "fim" ] = Falso
    estado[ "pagina_atual" ] = estado[ "aulas" ][ 0 ][ "pag_ini" ]
    histórico.append({ "role" : "assistant" , "content" : "🔄 Curso reiniciado. Você está de volta à **Aula 1**." })
    estado[ "hist_inicio" ] = len (histórico)
    return histórico, None , estado, render_status(estado)


# ============================================================================
# INTERFACE GRADIO
# ============================================================================
TEMA = gr.themes.Soft(
    matiz_primário = "verde" ,
    matiz_secundário = "esmeralda" ,
    neutral_hue= "ardósia" ,
). definir (
    body_background_fill= "#0b1210" ,
    body_background_fill_dark= "#0b1210" ,
    body_text_color= "#e5f5ec" ,
    body_text_color_dark= "#e5f5ec" ,
    block_background_fill= "#111c18" ,
    block_background_fill_dark= "#111c18" ,
    input_background_fill= "#0e1714" ,
    input_background_fill_dark= "#0e1714" ,
)

CSS = """
:root, .dark, body, gradio-app, .gradio-container {
    esquema de cores: escuro;
    cor de fundo: #0b1210 !importante;
}
.gradio-container, .gradio-container .main, .gradio-container > .main, .app {
    largura-máxima: nenhuma !importante;
    largura: 100% !importante;
}
.gradio-container { padding-left: 24px !important; padding-right: 24px !important; }
rodapé { visibilidade: oculta; }
/* ===== QUADRO NEGRO (área de resposta do professor) ===== */
#quadro-negro {
    fundo: #0f3b25 !importante;
    borda: 8px sólida #8b5a2b !importante;
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
#quadro-negro .plástico-bolha,
#quadro-negro .wrapper,
#quadro-negro .message-wrap { background: #0f3b25 !important; }
#quadro-negro .mensagem-conteúdo,
#quadro-negro .prosa,
#quadro-negro .prosa *,
#quadro-negro .katex,
#quadro-negro .katex * { color: #f6faf4 !important; opacity: 1 !important; }
#quadro-negro .prose { font-size: 1.08rem !important; line-height: 1.75 !important; }
#quadro-negro .bot-row {
    fundo: #0f3b25 !importante;
    borda: 2px sólida #e8efe6 !importante;
    border-radius: 10px !important;
}
#quadro-negro .user-row {
    fundo: #0b2d1c !importante;
    borda: 2px tracejada #ffe9a8 !importante;
    border-radius: 10px !important;
}
#quadro-negro .user-row .prosa,
#quadro-negro .user-row .prose * { color: #ffe9a8 !important; }
#quadro-negro h1, #quadro-negro h2, #quadro-negro h3, #quadro-negro h4 {
    cor: #ffe9a8 !importante;
    borda-inferior: 2px tracejada rgba(255,233,168,.6);
    preenchimento-inferior: 4px;
}
#quadro-negro forte { cor: #ffffff !importante; }
#quadro-negro hr { border: 0; border-top: 2px solid #e8efe6 !important; }
#quadro-negro tabela { border-collapse: collapse; }
#quadro-negro th, #quadro-negro td { border: 1px solid #cfe3d2 !important; padding: 6px 10px; }
#código quadro-negro, #quadro-negro pré {
    fundo: rgba(255,255,255,.10) !importante;
    borda: 1px sólida rgba(255,255,255,.35) !importante;
    cor: #ffe9a8 !importante;
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
        documento.cabeçalho.appendChild(m);
    }
}
"""

tentar :
    GRADIO_MAJOR = int (gr.__version__.split( "." )[ 0 ])
exceto (ValueError, AttributeError):
    GRADIO_MAJOR = 5

se GRADIO_MAJOR >= 6 :
    BLOCKS_KWARGS = {}
    LAUNCH_KWARGS = dict (theme=TEMA, css=CSS, js=JS_FORCAR_ESCURO, allowed_paths=[ str (LIB)])
outro :
    BLOCKS_KWARGS = dict (theme=TEMA, css=CSS, js=JS_FORCAR_ESCURO)
    LAUNCH_KWARGS = dict (allowed_paths=[ str (LIB)])


def  criar_chatbot ():
    opcoes = dict (
        label= "🧑‍🏫 Quadro Negro — Resposta do Professor" ,
        elem_id= "quadro-negro" ,
        altura= "78vh" ,
        latex_delimiters=[
            { "esquerda" : "$$" , "direita" : "$$" , "exibir" : True },
            { "esquerda" : "$" , "direita" : "$" , "exibir" : False },
        ],
    )
    tentar :
        retornar gr.Chatbot( type = "messages" , **opcoes)
    exceto TypeError:
        retornar gr.Chatbot(**opcoes)


def  criar_blocos ():
    tentar :
        retornar gr.Blocks(title= f"Professor IA {VERSAO} " , fill_width= True , **BLOCKS_KWARGS)
    exceto TypeError:
        retornar gr.Blocks(title= f"Professor IA {VERSAO} " , **BLOCKS_KWARGS)


app = criar_blocos()

com aplicativo:
    estado = gr.Estado(estado_vazio())

    gr.Markdown( f"# 🎓 Professor IA — Tutor Universal de Exatas ` {VERSAO} `" )
    gr.Markdown(
        "Assistente híbrido (Gemini) que ensina **passo a passo**, nunca só a resposta."
        "Envie um PDF e estude em 4 aulas progressivas, com leitura visual da **página em foco**. "
        "O áudio narra como um professor de verdade, com pausas de respiração."
    )

    com gr.Row(equal_height= False ):
        com gr.Column(scale= 1 , min_width= 340 ):
            chave_gemini = gr.Textbox(
                label= "🔒 Chave de API do Gemini (opcional se definida nos Secrets)" ,
                placeholder= "Cole sua API Key aqui..." ,
                tipo = "senha" ,
            )
            indicador_chave = gr.HTML(INDICADOR_NEUTRO)
            btn_testar = gr.Button( "🔑 Testar Chave" )

            upload_pdf = gr.File(
                label= "Biblioteca (Upload de PDF de aula/livro)" ,
                file_types=[ ".pdf" ],
            )
            status_aulas = gr.Markdown(render_status(estado_vazio()))

            com gr.Row():
                btn_pag_prev = gr.Button( "◀ Página" , size= "sm" )
                btn_pag_next = gr.Button( "Página ▶" , size= "sm" )

            com gr.Row():
                btn_iniciar = gr.Button( "▶️ Explicar aula atual" , variante= "primary" )
                btn_concluir = gr.Button( "✅ Concluir aula e avançar" )
            btn_reiniciar = gr.Button( "🔄 Reiniciar curso" , size= "sm" )

            usar_visual = gr.Checkbox(
                label= "👁️ Leitura visual da página em foco" ,
                valor = Verdadeiro ,
                info= "Envia SOMENTE a página em foco como imagem ao Gemini (rápido e econômico)." ,
            )
            gerar_audio = gr.Checkbox(label= "🔊 Gerar áudio didático das respostas" , value= True )
            resposta_áudio = gr.Áudio(
                label= "Explicação em Áudio (narração de professor)" ,
                interativo = Falso ,
                autoplay= Verdadeiro ,
                tipo = "caminho do arquivo" ,
                formato = "mp3" ,
            )

        com gr.Column(scale= 4 , min_width= 600 ):
            chatbot = criar_chatbot()
            txt_mensagem = gr.Textbox(
                label= "Sua dúvida ou sua resolução (Cálculo, Física, Circuitos, Estruturas...)" ,
                placeholder= "Digite sua pergunta aqui e aperte Enter..." ,
            )
            btn_enviar = gr.Button( "Enviar" , variante= "primário" )

    # --- Testar chave Gemini ---
    btn_testar.click(fn=testar_chave_gemini, inputs=[chave_gemini], outputs=[indicador_chave])
    chave_gemini.change(fn= lambda : INDICADOR_NEUTRO, inputs= None , outputs=[indicador_chave])

    # --- PDF -> aulas ---
    upload_pdf.change(fn=processar_pdf, inputs=[upload_pdf, estado, chatbot], outputs=[estado, status_aulas])

    # --- Navegação de páginas ---
    btn_pag_prev.click(fn= lambda e: navegar_pagina(- 1 , e), inputs=[estado], outputs=[estado, status_aulas])
    btn_pag_next.click(fn= lambda e: navegar_pagina(+ 1 , e), inputs=[estado], outputs=[estado, status_aulas])

    # --- Chat: 6 entradas / 3 saídas ---
    entradas_chat = [txt_mensagem, chatbot, estado, chave_gemini, gerar_audio, usar_visual]
    saidas_chat = [chatbot, audio_resposta, txt_mensagem]
    btn_enviar.click(fn=responder_usuario, inputs=entradas_chat, outputs=saidas_chat)
    txt_mensagem.submit(fn=responder_usuario, inputs=entradas_chat, outputs=saidas_chat)

    # --- Ações de ---
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
        entradas=[chatbot, estado],
        outputs=saidas_aula,
    )


# ============================================================================
# ARRANQUE — compatível com Hugging Face Spaces
# (sem server_port fixo, sem share, sem socket; o HF cuida da exposição)
# ============================================================================
imprimir ( "=" * 72 )
print ( f"🎓 Professor IA — Tutor Universal de Exatas ( {VERSAO} )" )
imprimir ( "=" * 72 )
print ( f" Python : {sys.version.split()[ 0 ]} " )
print ( f" Gradio : { getattr (gr, '__version__' , '?' )} " )
print ( f" PyMuPDF : { 'OK'  if fitz else  'AUSENTE (pip install pymupdf)' } " )
print ( f" Edge-TTS : { 'OK'  if edge_tts else  'AUSENTE (pip install edge-tts)' } " )
imprimir ( f" Biblioteca : {LIB} " )
print ( f" Chave Gemini: { 'OK (env)'  if CHAVE_ENV else  'ausente — configure no campo ou nos Secrets' } " )
imprimir ( "=" * 72 )

tentar :
    app.launch(**LAUNCH_KWARGS)
exceto Exception como _e:
    imprimir ( "\n----------------------------------------" )
    print ( f"[X] ERRO FATAL: { type (_e).__name__} : {_e} " )
    imprimir ( "----------------------------------------\n" )
    traceback.print_exc()
