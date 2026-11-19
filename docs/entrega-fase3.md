# Documento de Entrega — Fase 3

Este documento reúne os entregáveis do Tech Challenge da Fase 3: um **assistente
médico virtual** para o Hospital Santa Helena (fictício), com uma **LLM
fine-tunada por LoRA** sobre dados internos do hospital, orquestrada por
**LangChain e LangGraph**, consultando prontuários, verificando exames pendentes e
emitindo alertas para a equipe — com limites de atuação, trilha de auditoria e
fonte em toda resposta.

## Links do projeto

| Recurso | URL |
|---|---|
| **Repositório GitHub** | https://github.com/otaviano/fiap-tech-challenge-fase3 |
| **Vídeo de demonstração** | https://youtu.be/clQDJsVoSDc |
| **Relatório técnico** | https://github.com/otaviano/fiap-tech-challenge-fase3/blob/main/RELATORIO_TECH_CHALLENGE_FASE3.md |
| **Notebook de demonstração** | https://github.com/otaviano/fiap-tech-challenge-fase3/blob/main/notebooks/demo.ipynb |
| **Adapter LoRA treinado (release)** | https://github.com/otaviano/fiap-tech-challenge-fase3/releases/tag/v1.0-0.5b |
| **Arquitetura e fluxo LangGraph** | https://github.com/otaviano/fiap-tech-challenge-fase3/blob/main/docs/arquitetura.md |
| **Dataset: anonimização e curadoria** | https://github.com/otaviano/fiap-tech-challenge-fase3/blob/main/docs/dataset.md |
| **Fine-tuning e avaliação** | https://github.com/otaviano/fiap-tech-challenge-fase3/blob/main/docs/fine-tuning.md |
| **Segurança e validação** | https://github.com/otaviano/fiap-tech-challenge-fase3/blob/main/docs/seguranca.md |
| **Fases anteriores (base)** | https://github.com/otaviano/fiap-tech-challenge-fase1 · https://github.com/otaviano/fiap-tech-challenge-fase2 |

## Entregáveis do repositório

| Item exigido | Onde encontrar |
|---|---|
| Pipeline de fine-tuning | `src/med_assist/finetune/` — dataset, treino LoRA, merge/GGUF e avaliação; CLI `med-assist finetune` |
| Integração com LangChain | `src/med_assist/assistant/` — `chain.py`, `llm.py`, `prompts.py`, `tools.py` |
| Fluxos do LangGraph | `src/med_assist/assistant/graph.py` — oito nós e três arestas condicionais |
| Dataset anonimizado / dados sintéticos | `data/raw/` (corpus sintético), `data/processed/train.jsonl` e `val.jsonl`; relatórios em `results/eval/` |
| Relatório técnico | `RELATORIO_TECH_CHALLENGE_FASE3.md` (reproduzido a seguir) |
| Vídeo de demonstração (≤ 15 min) | https://youtu.be/clQDJsVoSDc |

O relatório cobre os quatro pontos pedidos: **processo de fine-tuning** (§4),
**descrição do assistente** (§3), **diagrama do fluxo** (§3.1) e **avaliação do
modelo e análise dos resultados** (§6).

## Mapeamento das entregas técnicas exigidas

### 1. Fine-tuning de LLM com dados médicos internos

| Requisito do desafio | Atendido em |
|---|---|
| Fine-tuning de um modelo LLM | Qwen2.5-0.5B-Instruct com LoRA (r=16, α=32; 1,75% dos parâmetros), `finetune/train.py`; perplexidade de validação 9,60 → 3,19; relatório §4 e §6.1 |
| Protocolos médicos do hospital | 8 protocolos, 40 seções, em `data/raw/protocolos/`; categoria `protocolo_qa`; relatório §2.1 |
| Perguntas frequentes de médicos | 47 itens em `data/raw/faq_medicos.jsonl` (32 dúvidas de plantão e 15 pedidos de ato médico, base da recusa); categorias `faq`, `faq_contextualizada` e `limite_de_atuacao` |
| Modelos de laudos, receitas e procedimentos | 72 modelos em `data/raw/documentos/`; categoria `documento` |
| Preprocessing, anonimização e curadoria | `data/anonymize.py` (pseudônimos HMAC, gate de PII residual) e `data/curation.py` (deduplicação, qualidade, split estratificado); relatório §2.2 a §2.4 |

### 2. Criação de assistente médico com LangChain

| Requisito do desafio | Atendido em |
|---|---|
| Pipeline que integra a LLM customizada | `assistant/llm.py` — `ChatOpenAI` sobre o llama-server com o modelo `med-assist` (GGUF), com fallback determinístico; relatório §3 |
| Consultas em base de dados estruturada | `knowledge/ehr.py` — prontuários em SQLite, expostos como ferramentas LangChain em `assistant/tools.py`; relatório §3.2 |
| Contextualizar com informações atualizadas do paciente | nó `contexto_paciente` — contexto clínico mínimo do prontuário, sem dados cadastrais; relatório §3.2 |
| Fluxo automatizado: exames pendentes, condutas e alertas | `assistant/graph.py` (LangGraph) e `assistant/alerts.py` — pendência por diferença de conjuntos, alertas por regra e notificação da equipe; relatório §3.1 e §3.4 |

### 3. Segurança e validação

| Requisito do desafio | Atendido em |
|---|---|
| Limites de atuação (nunca prescrever sem validação humana) | `safety/guardrails.py` — bloqueio de ato médico antes do modelo, revisão da saída e aviso de validação obrigatório; relatório §5.1 |
| Logging detalhado para rastreamento e auditoria | `safety/audit.py` — trilha JSONL append-only por `trace_id`, um evento por nó; `med-assist audit`; relatório §5.3 |
| Explainability (fonte da informação) | `safety/explainability.py` — bloco de fontes com protocolo, seção e score, e verificação cruzada das citações; relatório §5.4 e §5.5 |

### 4. Organização do código

| Requisito do desafio | Atendido em |
|---|---|
| Projeto modularizado em Python | pacote `med_assist` (`data`, `knowledge`, `assistant`, `safety`, `finetune`, `serving`) instalável, com CLI `med-assist`; relatório §7 |
| Instruções completas no README | `README.md` — instalação, dados, fine-tuning, servidor de LLM, API e Docker |
| Testes automatizados | `tests/` — 106 testes, 75% de cobertura; rodam sem GPU, sem rede e sem servidor de LLM |

## Conteúdo do vídeo (≤ 15 minutos)

| Item exigido | Onde aparece |
|---|---|
| Treinamento e funcionamento da LLM personalizada | Pipeline de dados, LoRA, curva de convergência e avaliação das rodadas v1 a v5 |
| Execução de um fluxo automatizado | `med-assist ask` no caso de sepse: prontuário → exames pendentes → alertas → protocolo → resposta → notificação |
| Resposta a perguntas clínicas contextualizadas | Notebook `demo.ipynb`, com o modelo `med-assist` servido localmente |
| Logs e validação das respostas | Trilha de auditoria por `trace_id`, guardrail de entrada e de saída, bloco de fontes e `/health` da API |

## Como reproduzir

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

med-assist ehr seed                                  # prontuário eletrônico
med-assist ask "O que priorizar neste caso?" --paciente PRT-2026-100137
med-assist audit --limite 15                         # trilha de auditoria
med-assist serve --port 8000                         # API REST (Swagger em /docs)
pytest -q                                            # 106 testes
```

O assistente funciona com ou sem servidor de LLM no ar. Para usar o modelo
fine-tunado, o README descreve como baixar o adapter da release, fazer o merge,
converter para GGUF e servir com o llama.cpp.
