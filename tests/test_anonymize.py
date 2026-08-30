"""Anonimização: o gate que decide o que pode virar dado de treino."""

from med_assist.data.anonymize import anonymize, pseudonym, residual_pii

TEXTO = (
    "Paciente Maria Aparecida Souza, CPF 123.456.789-00, CNS 898 0012 3456 7890, "
    "RG 12.345.678-9, prontuário PRT-2026-000123, nascida em 14/03/1979, "
    "telefone (11) 98765-4321, e-mail maria.souza@exemplo.com.br, "
    "residente à Rua das Acácias, nº 220, CEP 04567-000. "
    "Avaliada pela Dra. Helena Vasconcelos Prado (CRM-SP 142.887)."
)


def test_remove_todos_os_tipos_de_pii():
    limpo, relatorio = anonymize(TEXTO)
    for tipo in ("CPF", "CNS", "RG", "PRONTUARIO", "DATA", "TELEFONE", "EMAIL",
                 "ENDERECO", "CEP", "CRM", "NOME"):
        assert relatorio.counts[tipo] >= 1, f"{tipo} não foi detectado"
    assert residual_pii(limpo) == []


def test_pseudonimo_e_estavel_e_nao_reversivel():
    assert pseudonym("CPF", "123.456.789-00") == pseudonym("CPF", "123.456.789-00")
    assert pseudonym("CPF", "123.456.789-00") != pseudonym("CPF", "987.654.321-00")
    assert "123" not in pseudonym("CPF", "123.456.789-00")


def test_mantem_o_titulo_clinico_fora_do_pseudonimo():
    limpo, _ = anonymize("Avaliada pela Dra. Helena Vasconcelos Prado.")
    assert limpo.startswith("Avaliada pela Dra. [NOME_")


def test_nao_confunde_dose_com_identificador():
    """Regressão: doses e volumes clínicos não podem virar PII."""
    texto = ("Administrar cristaloide 30 mL/kg em 3 horas, insulina 0,1 U/kg/h, "
             "meta de PAM 65 mmHg e lactato abaixo de 2 mmol/L.")
    limpo, relatorio = anonymize(texto)
    assert limpo == texto
    assert relatorio.total == 0


def test_idade_acima_de_89_e_generalizada():
    limpo, relatorio = anonymize("Acompanhante com 92 anos relata o caso.")
    assert "[IDADE_90+]" in limpo
    assert relatorio.counts["IDADE_90+"] == 1
