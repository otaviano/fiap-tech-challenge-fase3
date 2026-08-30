"""Curadoria: o segundo gate, que decide o que *deveria* virar dado de treino."""

from med_assist.data.curation import curate, split

BOM = {"categoria": "protocolo_qa", "pergunta": "Qual a conduta no PROT-EME-002?",
       "resposta": "Coletar duas hemoculturas antes do antibiótico e administrar em até 60 minutos. "
                   "Fonte: PROT-EME-002 v6.0, seção 3 — Conduta."}


def test_aprova_exemplo_bem_formado():
    aprovados, relatorio = curate([BOM])
    assert len(aprovados) == 1
    assert relatorio.descartes == {}


def test_descarta_exemplo_com_pii_residual():
    ruim = {**BOM, "resposta": BOM["resposta"] + " Contato do médico: (11) 3255-4102."}
    aprovados, relatorio = curate([ruim])
    assert aprovados == []
    assert relatorio.descartes["pii_residual"] == 1


def test_exige_citacao_de_fonte_em_exemplo_interno():
    sem_fonte = {**BOM, "resposta": "Coletar hemoculturas antes do antibiótico e infundir volume "
                                    "conforme a resposta hemodinâmica do paciente internado."}
    aprovados, relatorio = curate([sem_fonte])
    assert aprovados == []
    assert relatorio.descartes["sem_citacao_de_fonte"] == 1


def test_remove_duplicata_exata_e_quase_duplicata():
    quase = {**BOM, "pergunta": "qual a conduta no prot-eme-002?"}
    aprovados, relatorio = curate([BOM, dict(BOM), quase])
    assert len(aprovados) == 1
    assert relatorio.descartes["duplicata_exata"] >= 1


def test_split_e_estratificado_por_categoria():
    exemplos = ([{**BOM, "categoria": "protocolo_qa", "pergunta": f"Pergunta A número {i} do PROT-EME-002?"}
                 for i in range(20)]
                + [{**BOM, "categoria": "limite_de_atuacao", "pergunta": f"Pergunta B número {i} do PROT-EME-002?"}
                   for i in range(8)])
    treino, validacao = split(exemplos, val_ratio=0.25)
    assert {e["categoria"] for e in validacao} == {"protocolo_qa", "limite_de_atuacao"}
    assert len(treino) + len(validacao) == len(exemplos)
