"""Fonte dos protocolos internos sintéticos do Hospital Santa Helena.

Este módulo é a *matéria-prima* do projeto: um hospital real teria estes
documentos em um repositório interno. Aqui eles são sintéticos e autorais —
nenhum dado de paciente real é usado em nenhuma etapa.

Cada protocolo carrega, de propósito, dados identificáveis do responsável
(nome, CRM, e-mail, ramal). Isso não é descuido: o pipeline de anonimização
(`med_assist.data.anonymize`) precisa de PII real-ista para provar que funciona,
e os relatórios em `results/` mostram quantas ocorrências foram removidas.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Protocolo:
    codigo: str
    titulo: str
    especialidade: str
    versao: str
    revisao: str
    responsavel: str
    crm: str
    email: str
    ramal: str
    indicacao: str
    criterios: tuple[str, ...]
    conduta: tuple[str, ...]
    exames_obrigatorios: tuple[str, ...]
    alertas: tuple[str, ...]
    referencias: tuple[str, ...] = field(default=())


PROTOCOLOS: tuple[Protocolo, ...] = (
    Protocolo(
        codigo="PROT-ONC-001",
        titulo="Nódulo mamário suspeito: rastreamento, biópsia e conduta",
        especialidade="Oncologia Mamária",
        versao="4.2",
        revisao="2026-03-14",
        responsavel="Dra. Helena Vasconcelos Prado",
        crm="CRM-SP 142.887",
        email="helena.prado@santahelena.hosp.br",
        ramal="(11) 3255-4180",
        indicacao=(
            "Aplica-se a pacientes com nódulo mamário palpável ou achado de imagem "
            "classificado como BI-RADS 4 ou 5, encaminhadas pelo ambulatório de mastologia."
        ),
        criterios=(
            "BI-RADS 4 ou 5 em mamografia ou ultrassonografia mamária.",
            "Nódulo palpável com bordas irregulares, endurecido ou aderido a planos profundos.",
            "Descarga papilar sanguinolenta unilateral espontânea.",
            "Linfonodo axilar palpável ipsilateral maior que 1 cm.",
        ),
        conduta=(
            "Solicitar core biopsy guiada por ultrassom em até 10 dias corridos.",
            "Discutir o caso no comitê de mama (quintas, 7h30) antes de qualquer proposta terapêutica.",
            "Confirmado carcinoma invasivo, solicitar estadiamento e perfil imuno-histoquímico "
            "(RE, RP, HER2, Ki-67) antes da definição de tratamento sistêmico.",
            "Toda proposta de quimioterapia neoadjuvante exige validação do oncologista responsável "
            "e registro em prontuário eletrônico.",
        ),
        exames_obrigatorios=(
            "Mamografia digital bilateral (até 12 meses)",
            "Ultrassonografia mamária e axilar",
            "Core biopsy com laudo anatomopatológico",
            "Imuno-histoquímica (RE, RP, HER2, Ki-67)",
            "Hemograma completo, função hepática e renal",
        ),
        alertas=(
            "Biópsia com resultado discordante da imagem exige nova avaliação do comitê — não liberar a paciente.",
            "Atraso maior que 30 dias entre diagnóstico e início do tratamento deve gerar alerta à coordenação.",
        ),
        referencias=("Diretriz interna HSH 2026", "Consenso Brasileiro de Mastologia"),
    ),
    Protocolo(
        codigo="PROT-EME-002",
        titulo="Sepse e choque séptico: pacote da primeira hora",
        especialidade="Emergência",
        versao="6.0",
        revisao="2026-01-30",
        responsavel="Dr. Ricardo Menezes Tavares",
        crm="CRM-SP 98.442",
        email="ricardo.tavares@santahelena.hosp.br",
        ramal="(11) 3255-4102",
        indicacao=(
            "Todo paciente adulto com foco infeccioso suspeito e sinais de disfunção orgânica, "
            "em qualquer ponto de atendimento do hospital."
        ),
        criterios=(
            "qSOFA maior ou igual a 2 (FR maior ou igual a 22, PAS menor ou igual a 100 mmHg, alteração do nível de consciência).",
            "Lactato arterial acima de 2 mmol/L.",
            "Hipotensão persistente após reposição volêmica adequada indica choque séptico.",
        ),
        conduta=(
            "Coletar duas hemoculturas de sítios distintos ANTES da primeira dose de antibiótico.",
            "Administrar antimicrobiano de amplo espectro em até 60 minutos do reconhecimento.",
            "Iniciar cristaloide 30 mL/kg em até 3 horas nos casos de hipotensão ou lactato acima de 4 mmol/L.",
            "Manter PAM maior ou igual a 65 mmHg; se refratário, iniciar noradrenalina com acesso central.",
            "Reavaliar lactato em 2 a 4 horas.",
        ),
        exames_obrigatorios=(
            "Duas hemoculturas (pré-antibiótico)",
            "Lactato arterial seriado",
            "Hemograma, PCR, creatinina, bilirrubinas",
            "Gasometria arterial",
            "Cultura do foco suspeito (urina, secreção, líquor)",
        ),
        alertas=(
            "Antibiótico administrado após 60 minutos é indicador de não conformidade e gera notificação automática.",
            "Lactato acima de 4 mmol/L em paciente normotenso ainda caracteriza sepse grave — não subestimar.",
        ),
        referencias=("Surviving Sepsis Campaign (adaptado)", "Protocolo institucional HSH"),
    ),
    Protocolo(
        codigo="PROT-CAR-003",
        titulo="Dor torácica e síndrome coronariana aguda",
        especialidade="Cardiologia",
        versao="5.1",
        revisao="2026-02-11",
        responsavel="Dra. Cláudia Ferreira Bittencourt",
        crm="CRM-SP 110.365",
        email="claudia.bittencourt@santahelena.hosp.br",
        ramal="(11) 3255-4207",
        indicacao="Paciente adulto com dor torácica aguda não traumática atendido no pronto-socorro.",
        criterios=(
            "Dor torácica típica com duração maior que 20 minutos.",
            "Supradesnivelamento de ST em duas derivações contíguas caracteriza IAMCSST.",
            "Troponina ultrassensível acima do percentil 99 com curva ascendente caracteriza IAMSSST.",
        ),
        conduta=(
            "Realizar ECG de 12 derivações em até 10 minutos da chegada.",
            "IAMCSST: acionar hemodinâmica; meta porta-balão menor que 90 minutos.",
            "Sem indicação de reperfusão imediata: dupla antiagregação conforme prescrição do cardiologista de plantão.",
            "Estratificar risco por escore HEART antes da alta.",
        ),
        exames_obrigatorios=(
            "ECG de 12 derivações seriado",
            "Troponina ultrassensível (0h e 3h)",
            "Radiografia de tórax",
            "Eletrólitos, função renal e coagulograma",
        ),
        alertas=(
            "ECG normal não exclui SCA — repetir em 20 minutos se a dor persistir.",
            "Nunca liberar paciente com troponina em ascensão, mesmo assintomático.",
        ),
        referencias=("Diretriz institucional HSH", "Sociedade Brasileira de Cardiologia"),
    ),
    Protocolo(
        codigo="PROT-END-004",
        titulo="Cetoacidose diabética no adulto",
        especialidade="Endocrinologia",
        versao="3.4",
        revisao="2025-11-22",
        responsavel="Dr. Paulo Sérgio Andrade Lima",
        crm="CRM-SP 87.209",
        email="paulo.lima@santahelena.hosp.br",
        ramal="(11) 3255-4311",
        indicacao="Paciente diabético com hiperglicemia e sinais de descompensação metabólica aguda.",
        criterios=(
            "Glicemia acima de 250 mg/dL.",
            "pH arterial abaixo de 7,3 ou bicarbonato abaixo de 18 mEq/L.",
            "Cetonemia ou cetonúria positiva.",
        ),
        conduta=(
            "Hidratação com salina 0,9% 15 a 20 mL/kg na primeira hora.",
            "Insulina regular em bomba de infusão contínua 0,1 U/kg/h após confirmar potássio maior que 3,3 mEq/L.",
            "Repor potássio antes da insulina se K menor que 3,3 mEq/L.",
            "Monitorar glicemia capilar de hora em hora e eletrólitos a cada 4 horas.",
            "Transicionar para insulina subcutânea somente após resolução da acidose e com prescrição validada.",
        ),
        exames_obrigatorios=(
            "Gasometria arterial ou venosa",
            "Glicemia capilar horária",
            "Eletrólitos (Na, K, Cl) a cada 4 horas",
            "Cetonemia ou cetonúria",
            "Função renal e hemograma",
        ),
        alertas=(
            "Iniciar insulina com potássio abaixo de 3,3 mEq/L pode causar arritmia fatal.",
            "Queda de glicemia maior que 100 mg/dL/h aumenta risco de edema cerebral — reduzir infusão.",
        ),
        referencias=("Protocolo institucional HSH", "Sociedade Brasileira de Diabetes"),
    ),
    Protocolo(
        codigo="PROT-NEU-005",
        titulo="AVC isquêmico agudo: janela terapêutica e trombólise",
        especialidade="Neurologia",
        versao="4.0",
        revisao="2026-02-27",
        responsavel="Dra. Marina Okamoto Ribeiro",
        crm="CRM-SP 123.774",
        email="marina.ribeiro@santahelena.hosp.br",
        ramal="(11) 3255-4425",
        indicacao="Paciente com déficit neurológico focal súbito e início dos sintomas conhecido.",
        criterios=(
            "Déficit neurológico focal com NIHSS maior ou igual a 4, ou afasia isolada incapacitante.",
            "Tempo de início dos sintomas menor que 4,5 horas para trombólise endovenosa.",
            "Tomografia de crânio sem hemorragia.",
        ),
        conduta=(
            "Acionar o código AVC na chegada; meta porta-tomografia menor que 20 minutos.",
            "Trombólise com alteplase somente após avaliação do neurologista e checklist de contraindicações completo.",
            "Controlar PA abaixo de 185/110 mmHg antes da trombólise.",
            "Manter o paciente em unidade de AVC por no mínimo 24 horas.",
        ),
        exames_obrigatorios=(
            "Tomografia de crânio sem contraste",
            "Glicemia capilar imediata",
            "Coagulograma e plaquetas",
            "ECG e troponina",
            "Angiotomografia de crânio e vasos cervicais quando indicada trombectomia",
        ),
        alertas=(
            "Hora de início desconhecida contraindica trombólise fora de protocolo de imagem avançada.",
            "Glicemia abaixo de 50 mg/dL pode mimetizar AVC — corrigir antes de qualquer decisão.",
        ),
        referencias=("Protocolo Código AVC HSH", "Diretriz da Academia Brasileira de Neurologia"),
    ),
    Protocolo(
        codigo="PROT-PNE-006",
        titulo="Pneumonia adquirida na comunidade em adultos",
        especialidade="Pneumologia",
        versao="3.1",
        revisao="2025-12-09",
        responsavel="Dr. Anderson Luiz Carvalho",
        crm="CRM-SP 76.518",
        email="anderson.carvalho@santahelena.hosp.br",
        ramal="(11) 3255-4506",
        indicacao="Paciente adulto com tosse, febre e infiltrado pulmonar novo à radiografia.",
        criterios=(
            "Infiltrado pulmonar novo associado a febre, tosse ou dispneia.",
            "CURB-65 orienta o local de tratamento: 0 a 1 ambulatorial, 2 enfermaria, 3 ou mais considerar UTI.",
        ),
        conduta=(
            "Calcular CURB-65 antes de decidir internação.",
            "Iniciar antibiótico em até 4 horas do diagnóstico, conforme padronização institucional.",
            "Coletar culturas antes do antibiótico nos casos internados.",
            "Reavaliar em 48 a 72 horas e descalonar conforme cultura.",
        ),
        exames_obrigatorios=(
            "Radiografia de tórax PA e perfil",
            "Hemograma, ureia e creatinina",
            "Saturação de oxigênio ou gasometria",
            "Hemocultura e cultura de escarro nos internados",
        ),
        alertas=(
            "Saturação abaixo de 90% em ar ambiente indica internação independentemente do CURB-65.",
            "Ausência de melhora em 72 horas exige reavaliação diagnóstica, não apenas troca de antibiótico.",
        ),
        referencias=("Padronização de antimicrobianos HSH", "Sociedade Brasileira de Pneumologia e Tisiologia"),
    ),
    Protocolo(
        codigo="PROT-HEM-007",
        titulo="Tromboembolismo venoso: profilaxia e tratamento",
        especialidade="Hematologia",
        versao="2.8",
        revisao="2026-01-15",
        responsavel="Dra. Beatriz Nogueira Salles",
        crm="CRM-SP 134.902",
        email="beatriz.salles@santahelena.hosp.br",
        ramal="(11) 3255-4614",
        indicacao="Pacientes internados com risco de TEV e pacientes com suspeita de TVP ou embolia pulmonar.",
        criterios=(
            "Escore de Pádua maior ou igual a 4 indica profilaxia farmacológica em paciente clínico.",
            "Escore de Wells orienta a probabilidade pré-teste de TVP e de embolia pulmonar.",
            "D-dímero negativo com Wells baixo exclui TEV com segurança.",
        ),
        conduta=(
            "Profilaxia com enoxaparina conforme função renal e peso, prescrita pelo médico assistente.",
            "Contraindicação a anticoagulante: usar compressão pneumática intermitente.",
            "TEP com instabilidade hemodinâmica: acionar time de resposta rápida e avaliar trombólise.",
            "Anticoagulação plena exige registro de indicação e tempo previsto em prontuário.",
        ),
        exames_obrigatorios=(
            "D-dímero quando probabilidade pré-teste baixa ou intermediária",
            "Ultrassom Doppler de membros inferiores",
            "Angiotomografia de tórax na suspeita de embolia pulmonar",
            "Hemograma, creatinina e coagulograma",
        ),
        alertas=(
            "Clearance de creatinina abaixo de 30 mL/min exige ajuste de dose da enoxaparina.",
            "Paciente oncológico tem risco aumentado — reavaliar profilaxia a cada mudança de status clínico.",
        ),
        referencias=("Protocolo institucional HSH", "Diretrizes de tromboprofilaxia"),
    ),
    Protocolo(
        codigo="PROT-INF-008",
        titulo="Uso racional de antimicrobianos (stewardship)",
        especialidade="Infectologia",
        versao="7.3",
        revisao="2026-03-02",
        responsavel="Dr. Fernando Kazuo Watanabe",
        crm="CRM-SP 65.130",
        email="fernando.watanabe@santahelena.hosp.br",
        ramal="(11) 3255-4720",
        indicacao="Toda prescrição de antimicrobiano de uso restrito no hospital.",
        criterios=(
            "Antimicrobianos de uso restrito exigem justificativa e aprovação da Comissão de Controle de Infecção.",
            "Terapia empírica deve ser reavaliada em até 72 horas com o resultado das culturas.",
        ),
        conduta=(
            "Registrar indicação, dose, via e duração prevista no momento da prescrição.",
            "Descalonar para o espectro mais estreito assim que a cultura permitir.",
            "Solicitar parecer da infectologia para uso maior que 7 dias de carbapenêmico ou polimixina.",
            "Nenhuma liberação de antimicrobiano restrito ocorre sem validação de um médico da comissão.",
        ),
        exames_obrigatorios=(
            "Culturas do sítio suspeito antes do início da terapia",
            "Hemograma e PCR seriados",
            "Função renal para ajuste de dose",
        ),
        alertas=(
            "Prescrição de restrito sem justificativa é bloqueada pelo sistema e notificada à CCIH.",
            "Uso prolongado sem descalonamento aumenta risco de resistência e é auditado mensalmente.",
        ),
        referencias=("Programa de Stewardship HSH", "ANVISA — diretrizes de uso racional"),
    ),
)


PROTOCOLOS_POR_CODIGO = {p.codigo: p for p in PROTOCOLOS}
