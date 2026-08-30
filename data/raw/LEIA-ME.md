# `data/raw/` — corpus bruto (com PII sintética, de propósito)

Todo o conteúdo desta pasta é **gerado por código** (`med_assist.data.synth`,
semente fixa) e é **inteiramente fictício**. Nenhum dado de paciente real foi
usado em qualquer etapa deste projeto.

Os nomes, CPFs, CNS, RGs, prontuários, telefones, e-mails e endereços aqui
presentes são fabricados e mantidos **de propósito**: sem PII na entrada não há
como demonstrar — nem testar — o pipeline de anonimização exigido pela Fase 3.

| Arquivo | Conteúdo |
|---|---|
| `protocolos/*.md` | 8 protocolos clínicos internos, com responsável técnico identificado |
| `pacientes.json` | 24 pacientes sintéticos com cadastro completo, exames, medicações e escores |
| `faq_medicos.jsonl` | Perguntas frequentes do corpo clínico, incluindo casos com PII e pedidos que devem ser recusados |
| `documentos/*.md` | Modelos institucionais de laudo, receituário e descrição de procedimento, preenchidos |

**Nada daqui vai para o treino.** O que alimenta o fine-tuning e a base de
conhecimento do assistente é `data/processed/`, produzido por
`med-assist data build`, que aplica anonimização e curadoria e rejeita qualquer
exemplo com PII residual.

Para regenerar:

```bash
med-assist data synth    # reescreve esta pasta
med-assist data build    # produz data/processed/ e os relatórios em results/eval/
```
