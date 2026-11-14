"""Limites de atuação do assistente — na entrada e na saída.

O desafio pede que o assistente nunca prescreva sem validação humana. Fine-tuning
*inclina* o modelo nessa direção; ele não garante nada. Por isso a garantia é
determinística e fica fora do modelo:

- **Entrada** (`check_input`): pedidos que exigem ato médico — prescrever, indicar
  dose para administração, assinar laudo, emitir atestado, decidir alta — são
  bloqueados antes de chegar ao LLM, com resposta institucional pronta.
- **Saída** (`check_output`): a resposta gerada é auditada — precisa citar fonte,
  não pode conter PII, não pode vir em modo imperativo de prescrição, e recebe o
  aviso de validação humana se ele faltar.

Um bloqueio na entrada economiza uma chamada ao modelo e, mais importante,
elimina a chance de a recusa depender da sorte do decoding.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..data.anonymize import TIPOS_CADASTRAIS, residual_pii
from ..data.protocol_source import PROTOCOLOS_POR_CODIGO
from ..persona import DISCLAIMER

# --------------------------------------------------------------------------- #
# Entrada
# --------------------------------------------------------------------------- #

#: (motivo, padrão). Os padrões miram o *ato* pedido, não o assunto: falar sobre
#: prescrição é legítimo; pedir que o assistente prescreva, não.
PADROES_ATO_MEDICO: tuple[tuple[str, re.Pattern[str]], ...] = (
    # Só o *pedido*: imperativo ("prescreva", "receite") ou solicitação ("pode
    # prescrever?", "faz a receita"). "O protocolo prescreve…" e "modelo de
    # receita" são perguntas sobre conteúdo e passam.
    ("prescricao_direta", re.compile(
        r"\b(prescreva|receite|"
        r"prescreve\s+(a[íi]|logo|agora|pra\s+mim|para\s+mim)|"
        r"(pode|podes|poderia|consegue|voc[êe])\s+(me\s+)?(prescrever|receitar)|"
        r"(fa[çz]a?|escrev[ae]|ger[ae]|emit[ae]|redij[ae])\s+(a|uma)\s+(receita|prescri[çc][ãa]o)|"
        r"me\s+d[êe]\s+a\s+(dose|receita)|"
        r"passa\s+a\s+medica[çc][ãa]o|manda\s+a\s+prescri[çc][ãa]o)\b", re.I)),
    ("dose_para_administracao", re.compile(
        r"\b((dose|posologia)\s+(exata|certa)|quanto\s+(devo|posso)\s+(dar|aplicar|administrar)|"
        r"(me\s+)?(fala|fale|diga|diz|passa|passe|informa|informe)\s+a\s+(dose|posologia)|"
        r"aplic(o|ar)\s+agora|posso\s+dar\s+\d)", re.I)),
    ("assinatura_de_documento", re.compile(
        r"\b(assin[ae]r?\s+(o\s+)?(laudo|receita|atestado|documento)|emit[ae]\s+(um\s+)?atestado|"
        r"emite\s+(o\s+)?laudo)\b", re.I)),
    ("decisao_de_alta_ou_internacao", re.compile(
        r"\b(pode\s+dar\s+alta|dou\s+alta|decid[ae]\s+por\s+mim|interno\s+ou\s+dou\s+alta|"
        r"assume\s+a\s+responsabilidade)\b", re.I)),
    ("diagnostico_definitivo", re.compile(
        r"\b(fech[ae]\s+o\s+diagn[óo]stico|confirma\s+que\s+[ée]\s+c[âa]ncer|"
        r"[ée]\s+c[âa]ncer\s+ou\s+n[ãa]o)\b", re.I)),
    ("solicitacao_de_pii", re.compile(
        r"\b(cpf|cns|rg|telefone|celular|endere[çc]o|e-?mail)\s+(d[oa]\s+)?(paciente|dele|dela)|"
        r"\b(me\s+passa|informe)\s+o\s+(cpf|telefone|endere[çc]o)", re.I)),
    ("alteracao_de_registro", re.compile(
        r"\b(apag[ae]r?|remov[ae]r?|delet[ae]r?)\s+(o\s+)?(alerta|registro|log|prontu[áa]rio|auditoria)", re.I)),
    ("tentativa_de_burlar_limites", re.compile(
        r"\b(ignore\s+(suas|as)\s+(regras|instru[çc][õo]es)|modo\s+desenvolvedor|sem\s+restri[çc][õo]es|"
        r"esquece\s+(as\s+)?regras|finja\s+que|sem\s+esse\s+aviso)\b", re.I)),
)

RESPOSTAS_PADRAO = {
    "prescricao_direta":
        "Não posso emitir prescrição. A prescrição é ato médico e exige validação humana registrada "
        "em prontuário. Posso apresentar o que o protocolo interno recomenda para o quadro, com a "
        "fonte citada, e listar as pendências do caso para apoiar sua decisão.",
    "dose_para_administracao":
        "Não indico dose para administração direta. Posso mostrar o que o protocolo institucional "
        "preconiza e quais checagens ele exige antes da administração — a definição da dose, com "
        "ajuste ao paciente, é do médico assistente.",
    "assinatura_de_documento":
        "Não emito nem assino laudos, receitas ou atestados. Esses documentos têm responsabilidade "
        "legal do profissional que os assina. Posso reunir os dados do caso e o modelo institucional "
        "correspondente para que o médico responsável avalie.",
    "decisao_de_alta_ou_internacao":
        "A decisão de alta, internação ou nível de cuidado é do médico assistente e eu não a assumo. "
        "Posso apresentar o escore que o protocolo usa nessa decisão, os achados registrados e os "
        "alertas ativos do caso.",
    "diagnostico_definitivo":
        "Não fecho diagnóstico — isso é ato médico e depende de exames com laudo e da avaliação "
        "clínica. Posso listar o que já consta no prontuário, o que está pendente e o que o protocolo "
        "exige antes da conduta.",
    "solicitacao_de_pii":
        "Não forneço dados identificáveis de paciente. Trabalho com o mínimo necessário para a decisão "
        "clínica — idade, quadro, exames, medicações e alergias. Dados cadastrais ficam no prontuário "
        "eletrônico, com acesso próprio e auditado.",
    "alteracao_de_registro":
        "Não altero nem removo registros de prontuário ou da trilha de auditoria. Se houver erro em um "
        "registro, o caminho é a retificação formal pelo serviço responsável, que também fica registrada.",
    "tentativa_de_burlar_limites":
        "Meus limites de atuação fazem parte do sistema e não são configuráveis: não prescrevo, não "
        "assino documentos e não substituo decisão médica. O apoio que posso dar continua disponível — "
        "critérios de protocolo, pendências do caso e alertas, com a fonte citada.",
}

#: Assuntos fora do escopo assistencial do hospital.
PADRAO_FORA_DE_ESCOPO = re.compile(
    r"\b(meu\s+sobrinho|minha\s+m[ãa]e|pra\s+mim\s+mesmo|estou\s+sentindo|eu\s+sinto|"
    r"minha\s+dor\s+de\s+cabe[çc]a|rem[ée]dio\s+eu\s+tomo)\b", re.I)

#: Fala do paciente reproduzida pelo profissional ("a paciente relata: estou
#: sentindo…", ou entre aspas). É anamnese, não caso pessoal de quem pergunta.
PADRAO_FALA_RELATADA = re.compile(
    r"\b(relat(a|ou|ando)|refer(e|iu|indo)|queix(a|ou)(-se)?|afirm(a|ou)|diz|disse|cont(a|ou))\b"
    r"(\s+que)?\s*:?[^.?!\n]*"
    r"|[\"“][^\"”]*[\"”]", re.I)

RESPOSTA_FORA_DE_ESCOPO = (
    "Atendo apenas profissionais do corpo clínico, no contexto assistencial do hospital, e não oriento "
    "casos pessoais nem automedicação. Para queixas próprias ou de familiares, procure atendimento médico."
)


@dataclass
class GuardrailVerdict:
    """Resultado de uma checagem — o que aconteceu e por quê."""

    bloqueado: bool = False
    motivos: list[str] = field(default_factory=list)
    resposta: str | None = None
    severidade: str = "info"

    def as_dict(self) -> dict:
        return {"bloqueado": self.bloqueado, "motivos": self.motivos, "severidade": self.severidade}


def check_input(pergunta: str) -> GuardrailVerdict:
    """Barra pedidos que exigem ato médico, antes de qualquer chamada ao modelo."""
    motivos: list[str] = []
    resposta: str | None = None

    for motivo, padrao in PADROES_ATO_MEDICO:
        if padrao.search(pergunta):
            motivos.append(motivo)
            resposta = resposta or RESPOSTAS_PADRAO[motivo]

    if PADRAO_FORA_DE_ESCOPO.search(PADRAO_FALA_RELATADA.sub(" ", pergunta)):
        motivos.append("fora_de_escopo")
        resposta = resposta or RESPOSTA_FORA_DE_ESCOPO

    if not motivos:
        return GuardrailVerdict()

    complemento = (
        "\n\nSe quiser, informe o prontuário do caso: apresento os critérios do protocolo aplicável, "
        "os exames pendentes e os alertas ativos, sempre com a fonte citada."
    )
    return GuardrailVerdict(bloqueado=True, motivos=motivos,
                            resposta=(resposta or "") + complemento, severidade="bloqueio")


# --------------------------------------------------------------------------- #
# Saída
# --------------------------------------------------------------------------- #

#: Imperativos de administração sem menção a validação — o que não pode sair.
PADRAO_IMPERATIVO = re.compile(
    r"^\s*(administre|aplique|injete|prescreva|inicie\s+imediatamente)\b", re.I | re.M)

PADRAO_FONTE = re.compile(r"(PROT-[A-Z]{3}-\d{3}|Fonte:|Prontuário\s+PRT-|Source:)")

PADRAO_VALIDACAO = re.compile(
    r"valida[çc][ãa]o\s+(humana|do\s+m[ée]dico|m[ée]dica)|m[ée]dico\s+assistente|"
    r"n[ãa]o\s+constitui\s+prescri[çc][ãa]o", re.I)


def check_output(resposta: str, exige_fonte: bool = True) -> tuple[str, GuardrailVerdict]:
    """Audita e corrige a resposta gerada. Devolve (resposta_final, veredito)."""
    motivos: list[str] = []
    final = resposta.strip()

    # Só dado cadastral bloqueia: número de prontuário e datas assistenciais são
    # o que identifica o caso para o profissional e circulam com acesso auditado.
    residuo = residual_pii(final, tipos=TIPOS_CADASTRAIS)
    if residuo:
        motivos.append(f"pii_na_resposta:{len(residuo)}")
        return (
            "A resposta gerada foi bloqueada por conter dados identificáveis. Reformule a pergunta "
            "usando apenas o número do prontuário — o assistente trabalha com dados mínimos.",
            GuardrailVerdict(bloqueado=True, motivos=motivos, severidade="critico"),
        )

    if PADRAO_IMPERATIVO.search(final):
        motivos.append("tom_imperativo_de_prescricao")
        final = (
            "⚠️ Reformulado pelo guardrail de saída: o texto original vinha em forma de ordem de "
            "administração. O conteúdo abaixo é sugestão de apoio, não prescrição.\n\n" + final
        )

    if exige_fonte and not PADRAO_FONTE.search(final):
        motivos.append("resposta_sem_fonte")

    if not PADRAO_VALIDACAO.search(final):
        motivos.append("disclaimer_ausente_incluido")
        final = f"{final}\n\n> {DISCLAIMER}"

    severidade = "aviso" if motivos else "info"
    return final, GuardrailVerdict(bloqueado=False, motivos=motivos, severidade=severidade)


# --------------------------------------------------------------------------- #
# Verificação cruzada das citações
# --------------------------------------------------------------------------- #

PADRAO_CODIGO = re.compile(r"PROT-[A-Z]{3}-\d{3}")
PADRAO_VERSAO = re.compile(r"(PROT-[A-Z]{3}-\d{3})\s+v(\d+\.\d+)")
PADRAO_REVISAO = re.compile(r"(PROT-[A-Z]{3}-\d{3})[^\n]{0,40}?revisão\s+(\d{4}-\d{2}-\d{2})", re.I)


def check_citations(resposta: str, protocolos_recuperados: set[str],
                    protocolo_mais_relevante: str | None = None) -> list[str]:
    """Confere se o que o modelo citou existe e bate com os metadados reais.

    Motivação medida: o modelo fine-tunado de 0,5B escreveu
    `PROT-EME-002 v6.0 (revisão 2027-02-11)` — código e versão corretos, data
    inventada (a real é 2026-01-30). Citação é o mecanismo de explicabilidade do
    sistema; uma citação plausível e errada é pior que citação nenhuma, porque
    ela *convida* à confiança.

    O sistema não corrige o texto do modelo — o bloco "Fontes consultadas" abaixo
    da resposta é gerado pelos nós e continua sendo a referência autoritativa.
    O que esta função faz é registrar a divergência na auditoria, onde ela vira
    métrica de qualidade do modelo servido.
    """
    motivos: list[str] = []

    citados = set(PADRAO_CODIGO.findall(resposta))
    inexistentes = {c for c in citados if c not in PROTOCOLOS_POR_CODIGO}
    if inexistentes:
        motivos.append("protocolo_inexistente:" + ",".join(sorted(inexistentes)))

    nao_recuperados = citados - inexistentes - protocolos_recuperados
    if protocolos_recuperados and nao_recuperados:
        motivos.append("citacao_fora_do_contexto:" + ",".join(sorted(nao_recuperados)))

    for codigo, versao in PADRAO_VERSAO.findall(resposta):
        protocolo = PROTOCOLOS_POR_CODIGO.get(codigo)
        if protocolo and versao != protocolo.versao:
            motivos.append(f"versao_divergente:{codigo}:{versao}!={protocolo.versao}")

    for codigo, revisao in PADRAO_REVISAO.findall(resposta):
        protocolo = PROTOCOLOS_POR_CODIGO.get(codigo)
        if protocolo and revisao != protocolo.revisao:
            motivos.append(f"revisao_divergente:{codigo}:{revisao}!={protocolo.revisao}")

    # Fonte trocada: o modelo cita um protocolo que de fato foi recuperado, mas
    # não o que sustenta a afirmação. Medido na v5 — a regra "culturas antes do
    # antibiótico" (PROT-EME-002, trecho mais relevante, score 7,07) saiu
    # atribuída ao PROT-END-004, recuperado com score 2,49. As verificações
    # anteriores não pegam: o código existe e estava no contexto.
    # Sinal de auditoria, não bloqueio: citar um trecho secundário às vezes é
    # legítimo, e é a ausência do principal que chama atenção.
    if protocolo_mais_relevante and citados and protocolo_mais_relevante not in citados:
        motivos.append(f"citacao_ignora_trecho_mais_relevante:{protocolo_mais_relevante}")

    # A mesma citação errada costuma aparecer várias vezes na mesma resposta;
    # a auditoria registra o problema, não a contagem de repetições.
    return list(dict.fromkeys(motivos))


#: Afirmações que só fazem sentido quando há um paciente no contexto.
PADRAO_AFIRMA_PACIENTE = re.compile(
    r"paciente\s+d[eo]\s+\d{1,3}\s*anos|resumo\s+do\s+caso|prontu[áa]rio\s+PRT-\d{4}-\d{6}|"
    r"leito\s+[A-Z]-\d{2}", re.I)

#: Código com cara de protocolo institucional, mas fora do padrão `PROT-XXX-000`.
# O lookbehind impede casar o sufixo de um código válido: em `PROT-PNE-006`, o
# trecho `PNE-006` tem um hífen antes e não conta como código próprio.
PADRAO_CODIGO_MALFORMADO = re.compile(r"\b(?<![A-Z-])(?!PROT-)[A-Z]{3,6}-\d{3}\b")


def check_hallucinated_context(resposta: str, tem_paciente: bool) -> list[str]:
    """Detecta caso clínico inventado e código de protocolo fabricado.

    Medido na rodada v4: numa consulta **sem prontuário**, o modelo abriu com
    "Paciente de 60 anos, homem" e citou "(PRONE-006)" — paciente e código
    inexistentes, em texto com aparência perfeitamente institucional. Nenhuma das
    verificações anteriores pegava: não havia PII, havia citação, e o código
    fabricado não casa com o padrão de protocolo real justamente por ser falso.
    """
    motivos: list[str] = []

    if not tem_paciente and PADRAO_AFIRMA_PACIENTE.search(resposta):
        motivos.append("paciente_inexistente_na_resposta")

    malformados = set(PADRAO_CODIGO_MALFORMADO.findall(resposta))
    if malformados:
        motivos.append("codigo_de_protocolo_malformado:" + ",".join(sorted(malformados)))

    return motivos
