"""Alertas clínicos para a equipe médica — regras determinísticas.

O enunciado pede que o fluxo "emita alertas para a equipe médica". Alerta é
justamente o tipo de saída que **não pode** depender de geração: um alerta
alucinado gera ruído, e um alerta omitido gera dano. Então cada alerta aqui é
uma regra explícita sobre os dados estruturados do prontuário, com o protocolo
que a fundamenta e a severidade que ela merece.

O LLM entra depois, para *redigir* o caso — nunca para decidir se há alerta.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

SEVERIDADES = ("critico", "alto", "moderado", "informativo")


@dataclass(frozen=True)
class Alerta:
    codigo: str
    severidade: str
    mensagem: str
    protocolo: str
    destinatario: str = "equipe assistencial"

    def as_dict(self) -> dict:
        return asdict(self)


def _tem_pendente(contexto: dict, termo: str) -> bool:
    return any(termo.lower() in e["exame"].lower() and e["status"] != "concluído"
               for e in contexto.get("exames", []))


def _usa_medicamento(contexto: dict, *termos: str) -> bool:
    nomes = " ".join(m["medicamento"].lower() for m in contexto.get("medicacoes", []))
    return any(t.lower() in nomes for t in termos)


def _escore(contexto: dict, nome: str) -> float | None:
    valor = (contexto.get("escores") or {}).get(nome)
    return float(valor) if isinstance(valor, (int, float)) else None


def _vital(contexto: dict, nome: str) -> float | None:
    valor = (contexto.get("sinais_vitais") or {}).get(nome)
    return float(valor) if isinstance(valor, (int, float)) else None


# Cada regra recebe o contexto clínico minimizado e devolve um alerta ou None.
Regra = Callable[[dict], Alerta | None]


def _r_sepse_hemocultura(c: dict) -> Alerta | None:
    """Antibiótico correndo com hemocultura ainda pendente: inverte o bundle."""
    if c.get("protocolo") != "PROT-EME-002":
        return None
    if _tem_pendente(c, "hemocultura") and _usa_medicamento(
            c, "piperacilina", "meropenem", "ceftriaxona", "vancomicina", "cefepime"):
        return Alerta(
            "SEPSE_HEMOCULTURA_APOS_ATB", "critico",
            "Antimicrobiano ativo com hemoculturas ainda pendentes. O PROT-EME-002 exige duas "
            "hemoculturas de sítios distintos ANTES da primeira dose — coleta posterior reduz a "
            "positividade e inviabiliza o descalonamento. Registrar a justificativa.",
            "PROT-EME-002")
    return None


def _r_sepse_lactato(c: dict) -> Alerta | None:
    if c.get("protocolo") != "PROT-EME-002":
        return None
    lactato = _escore(c, "lactato")
    qsofa = _escore(c, "qSOFA")
    if lactato is not None and lactato > 2 and (qsofa or 0) >= 2:
        severidade = "critico" if lactato >= 4 else "alto"
        return Alerta(
            "SEPSE_LACTATO_ELEVADO", severidade,
            f"Lactato {lactato} mmol/L com qSOFA {qsofa:.0f}: critério de sepse pelo PROT-EME-002. "
            "Reavaliar lactato em 2 a 4 horas e garantir cristaloide 30 mL/kg se hipotensão ou lactato acima de 4.",
            "PROT-EME-002")
    return None


def _r_cetoacidose_potassio(c: dict) -> Alerta | None:
    if c.get("protocolo") != "PROT-END-004":
        return None
    potassio = _escore(c, "K")
    if potassio is not None and potassio < 3.3 and _usa_medicamento(c, "insulina"):
        return Alerta(
            "CAD_INSULINA_COM_HIPOCALEMIA", "critico",
            f"Insulina em curso com potássio {potassio} mEq/L. O PROT-END-004 determina repor potássio "
            "antes da insulina quando K é menor que 3,3 mEq/L — risco de arritmia fatal. "
            "Suspender a infusão e comunicar o plantão imediatamente.",
            "PROT-END-004")
    return None


def _r_avc_janela(c: dict) -> Alerta | None:
    if c.get("protocolo") != "PROT-NEU-005":
        return None
    tempo = _escore(c, "tempo_sintomas_h")
    nihss = _escore(c, "NIHSS")
    if tempo is not None and tempo < 4.5:
        restante = 4.5 - tempo
        return Alerta(
            "AVC_JANELA_ATIVA", "critico",
            f"Paciente dentro da janela de trombólise: {tempo:.1f}h do início dos sintomas "
            f"(NIHSS {nihss:.0f} — restam ~{restante:.1f}h). PROT-NEU-005 exige tomografia sem hemorragia, "
            "PA abaixo de 185/110 mmHg e avaliação do neurologista antes da alteplase.",
            "PROT-NEU-005")
    return None


def _r_avc_pressao(c: dict) -> Alerta | None:
    if c.get("protocolo") != "PROT-NEU-005":
        return None
    pa = (c.get("sinais_vitais") or {}).get("PA", "")
    try:
        sistolica, diastolica = (int(v) for v in pa.split(" ")[0].split("/"))
    except (ValueError, AttributeError):
        return None
    if sistolica > 185 or diastolica > 110:
        return Alerta(
            "AVC_PA_ACIMA_DO_LIMITE", "alto",
            f"PA {pa} acima do limite de 185/110 mmHg exigido pelo PROT-NEU-005 antes da trombólise. "
            "Controle pressórico é pré-requisito, não etapa paralela.",
            "PROT-NEU-005")
    return None


def _r_sca_troponina(c: dict) -> Alerta | None:
    if c.get("protocolo") != "PROT-CAR-003":
        return None
    if _tem_pendente(c, "troponina"):
        return Alerta(
            "SCA_TROPONINA_PENDENTE", "alto",
            "Curva de troponina incompleta em investigação de síndrome coronariana. O PROT-CAR-003 "
            "não autoriza liberação com troponina em ascensão ou não avaliada — coletar 0h e 3h.",
            "PROT-CAR-003")
    return None


def _r_pac_hipoxemia(c: dict) -> Alerta | None:
    if c.get("protocolo") != "PROT-PNE-006":
        return None
    spo2 = _vital(c, "SpO2")
    curb = _escore(c, "CURB-65")
    if spo2 is not None and spo2 < 90:
        return Alerta(
            "PAC_HIPOXEMIA", "alto",
            f"SpO2 {spo2:.0f}% em ar ambiente indica internação independentemente do CURB-65 "
            f"(atual: {curb:.0f}), conforme PROT-PNE-006.",
            "PROT-PNE-006")
    return None


def _r_tev_profilaxia(c: dict) -> Alerta | None:
    if c.get("protocolo") != "PROT-HEM-007":
        return None
    padua = _escore(c, "Padua")
    if padua is not None and padua >= 4 and not _usa_medicamento(c, "enoxaparina", "heparina"):
        return Alerta(
            "TEV_SEM_PROFILAXIA", "alto",
            f"Escore de Pádua {padua:.0f} (maior ou igual a 4) sem profilaxia farmacológica registrada. "
            "O PROT-HEM-007 indica profilaxia ou, se contraindicada, compressão pneumática intermitente.",
            "PROT-HEM-007")
    return None


def _r_antibiotico_prolongado(c: dict) -> Alerta | None:
    if c.get("protocolo") != "PROT-INF-008":
        return None
    dias = _escore(c, "dias_antibiotico")
    if dias is not None and dias > 7 and _usa_medicamento(c, "meropenem", "polimixina", "imipenem"):
        return Alerta(
            "STEWARDSHIP_PARECER_PENDENTE", "moderado",
            f"Carbapenêmico no {dias:.0f}º dia de uso. O PROT-INF-008 exige parecer da infectologia "
            "acima de 7 dias e descalonamento assim que a cultura permitir.",
            "PROT-INF-008")
    return None


def _r_biopsia_prazo(c: dict) -> Alerta | None:
    if c.get("protocolo") != "PROT-ONC-001":
        return None
    if _tem_pendente(c, "core biopsy"):
        return Alerta(
            "ONC_BIOPSIA_PENDENTE", "alto",
            "Core biopsy pendente em caso BI-RADS 4. O PROT-ONC-001 estabelece prazo de até 10 dias "
            "corridos e discussão no comitê de mama antes de qualquer proposta terapêutica.",
            "PROT-ONC-001")
    return None


def _r_alergia_prescricao(c: dict) -> Alerta | None:
    """Regra transversal: medicação ativa que colide com alergia registrada."""
    alergias = [a.lower() for a in (c.get("alergias") or [])]
    equivalentes = {
        "penicilina": ("piperacilina", "amoxicilina", "oxacilina", "ampicilina"),
        "sulfa": ("sulfametoxazol", "trimetoprima"),
        "dipirona": ("dipirona",),
        "iodo": ("contraste iodado",),
    }
    for alergia in alergias:
        for chave, familia in equivalentes.items():
            if chave in alergia and _usa_medicamento(c, *familia):
                return Alerta(
                    "ALERGIA_X_PRESCRICAO", "critico",
                    f"Alergia registrada a {chave} com medicação ativa da mesma classe. "
                    "Revisar a prescrição com o médico assistente antes da próxima dose.",
                    "política institucional de segurança do paciente")
    return None


REGRAS: tuple[Regra, ...] = (
    _r_sepse_hemocultura,
    _r_sepse_lactato,
    _r_cetoacidose_potassio,
    _r_avc_janela,
    _r_avc_pressao,
    _r_sca_troponina,
    _r_pac_hipoxemia,
    _r_tev_profilaxia,
    _r_antibiotico_prolongado,
    _r_biopsia_prazo,
    _r_alergia_prescricao,
)


def avaliar(contexto: dict, pendencias: dict | None = None) -> list[Alerta]:
    """Aplica todas as regras e ordena por severidade."""
    alertas = [alerta for regra in REGRAS if (alerta := regra(contexto)) is not None]

    if pendencias and pendencias.get("nao_solicitados"):
        faltantes = ", ".join(pendencias["nao_solicitados"])
        alertas.append(Alerta(
            "EXAME_OBRIGATORIO_NAO_SOLICITADO", "alto",
            f"Exames obrigatórios do {pendencias['protocolo']} sem solicitação registrada: {faltantes}.",
            pendencias["protocolo"]))

    return sorted(alertas, key=lambda a: SEVERIDADES.index(a.severidade))
