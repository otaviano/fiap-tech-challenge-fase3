# Tech Challenge — Fase 3: Assistente Médico com LLM Customizada

> **FIAP Postech — IA para Devs**
> LLM fine-tunada com dados internos + LangChain + LangGraph

Terceira fase de uma linha contínua: a [Fase 1](https://github.com/otaviano/fiap-tech-challenge-fase1)
construiu o classificador de diagnóstico (câncer de mama, dataset Wisconsin); a
[Fase 2](https://github.com/otaviano/fiap-tech-challenge-fase2) otimizou os
hiperparâmetros por algoritmo genético e trouxe um **LLM local** para interpretar
os resultados. Aqui o LLM deixa de ser intérprete e passa a ser **assistente
clínico**: fine-tunado com os protocolos do hospital, orquestrado por
LangChain/LangGraph, consultando prontuários e emitindo alertas — com limites de
atuação, auditoria e citação de fonte.

O hospital deste projeto é o **Hospital Santa Helena**, fictício. Todos os dados
são sintéticos e autorais.

---

## O que foi construído

| Requisito do enunciado | Onde está |
|---|---|
| Fine-tuning de LLM com dados médicos internos | [`finetune/train.py`](src/med_assist/finetune/train.py) — LoRA sobre Qwen2.5 |
| Protocolos, FAQ de médicos, modelos de laudo/receita/procedimento | [`data/protocol_source.py`](src/med_assist/data/protocol_source.py) + [`data/synth.py`](src/med_assist/data/synth.py) |
| Preprocessing, anonimização e curadoria | [`data/anonymize.py`](src/med_assist/data/anonymize.py), [`data/curation.py`](src/med_assist/data/curation.py) |
| Pipeline LangChain integrando a LLM customizada | [`assistant/chain.py`](src/med_assist/assistant/chain.py), [`assistant/llm.py`](src/med_assist/assistant/llm.py) |
| Consulta a base de dados estruturada | [`knowledge/ehr.py`](src/med_assist/knowledge/ehr.py) — prontuários em SQLite |
| Contextualização com dados atualizados do paciente | nós `contexto_paciente` e `exames_pendentes` do fluxo |
| Fluxos do LangGraph | [`assistant/graph.py`](src/med_assist/assistant/graph.py) — 8 nós, 3 arestas condicionais |
| Limites de atuação (nunca prescrever sem validação) | [`safety/guardrails.py`](src/med_assist/safety/guardrails.py) |
| Logging detalhado para rastreamento e auditoria | [`safety/audit.py`](src/med_assist/safety/audit.py) — trilha JSONL por `trace_id` |
| Explainability (fonte da informação) | [`safety/explainability.py`](src/med_assist/safety/explainability.py) + verificação cruzada da citação em [`guardrails.py`](src/med_assist/safety/guardrails.py) |
| Alertas para a equipe médica | [`assistant/alerts.py`](src/med_assist/assistant/alerts.py) — 11 regras determinísticas |
| Dataset anonimizado / sintético | [`data/processed/`](data/processed/) + relatórios em [`results/eval/`](results/eval/) |
| Projeto modularizado em Python | [`src/med_assist/`](src/med_assist/) |
| Relatório técnico | [RELATORIO_TECH_CHALLENGE_FASE3.md](RELATORIO_TECH_CHALLENGE_FASE3.md) |
| Diagrama do fluxo | [`docs/arquitetura.md`](docs/arquitetura.md), [`docs/fluxo-langgraph.mmd`](docs/fluxo-langgraph.mmd) |
| Avaliação do modelo e análise | [relatório §6](RELATORIO_TECH_CHALLENGE_FASE3.md) + [`results/eval/`](results/eval/) |
| Testes automatizados | [`tests/`](tests/) — 84 testes |

---

## Arquitetura em uma tela

```
OFFLINE   corpus sintético (com PII) ─▶ anonimização ─▶ curadoria ─▶ train/val.jsonl
                                             │                            │
                                             ▼                            ▼
                                   base de conhecimento          LoRA ─▶ merge ─▶ GGUF
                                                                              │
ONLINE    médico ─▶ fluxo LangGraph ─────────────────────────────▶ llama-server (local)
                        │
                        ├─ prontuário SQLite ────┐
                        ├─ protocolos (BM25) ────┼─▶ resposta + fontes citadas + alertas
                        ├─ regras de alerta ─────┘
                        └─ guardrails + trilha de auditoria
```

Sete dos oito nós do fluxo são determinísticos. O modelo redige; ele não decide o
que é verdade sobre o paciente. Detalhes em [`docs/arquitetura.md`](docs/arquitetura.md).

---

## Como executar

### 1. Ambiente

Requer **Python ≥ 3.10** (validado no 3.14).

```bash
python3 -m venv .venv
source .venv/bin/activate            # Linux/Mac  (.venv\Scripts\activate no Windows)
pip install -r requirements.txt
pip install -e .                     # habilita o comando `med-assist`
```

Para treinar (opcional — o assistente roda sem isso):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # ou torch com CUDA
pip install -r requirements-finetune.txt
```

### 2. Preparar dados

O repositório já traz `data/raw/` e `data/processed/` versionados. Para
regenerar do zero:

```bash
./scripts/fetch_medquad.sh     # baixe ANTES do build: ver o aviso abaixo
med-assist data synth          # corpus bruto sintético (com PII, de propósito)
med-assist data build          # anonimiza, cura, faz o split e escreve os relatórios
med-assist ehr seed            # popula o prontuário eletrônico (SQLite)
```

> **Ordem importa.** `data build` reescreve `data/processed/`. Sem o MedQuAD
> baixado, o dataset sai com ~195 exemplos a menos que o versionado — o pipeline
> funciona, mas deixa de bater com os relatórios em `results/eval/`. Ou baixe o
> corpus antes, ou não rode o `build` (o dataset já vem pronto no repositório).

### 3. Servir a LLM

O assistente fala com qualquer endpoint OpenAI-compatível (llama.cpp, Ollama,
LM Studio, vLLM). Com o modelo desta fase:

```bash
llama-server -m models/med-assist-f16.gguf --alias med-assist --port 8080
export LLM_BASE_URL=http://localhost:8080/v1   # padrão
export LLM_MODEL=med-assist                    # padrão
```

**Como obter o modelo**, em ordem de esforço:

| Caminho | O que fazer | Tempo |
|---|---|---|
| **Baixar o adapter treinado** | [release v1.0](https://github.com/otaviano/fiap-tech-challenge-fase3/releases/tag/v1.0-0.5b) → `unzip` em `models/` → `merge` → `gguf` | ~5 min |
| Nenhum | não fazer nada — o assistente roda em modo contingência, com resposta citada e auditável | 0 |
| Treinar em CPU | `med-assist finetune train` → `merge` → `gguf` | ~60 min |
| Treinar em GPU | [`notebooks/finetune_colab.ipynb`](notebooks/finetune_colab.ipynb) na T4 gratuita | ~20 min |

O pipeline gera `med-assist-f16.gguf` (~950 MB). A quantização para Q4_K_M é
**opcional** e exige o binário `llama-quantize` compilado no seu llama.cpp
(`cmake --build build --target llama-quantize`); sem ele, o comando avisa e
mantém o f16, que roda sem problema para um modelo de 0,5B.

> Os artefatos de treino (adapter e GGUF) não são versionados por peso — o que o
> repositório versiona é o **pipeline que os produz** e as **evidências de que ele
> rodou** (`results/eval/`, `results/logs/`).

O modelo desta fase foi treinado em CPU nesta máquina (Qwen2.5-0.5B + LoRA,
59 min): perplexidade de validação **9,60 → 3,19**, citação de fonte **0% → 100%**
e recusa indevida **25% → 0%**. Cinco rodadas, e o que cada uma ensinou está na
[seção 6 do relatório](RELATORIO_TECH_CHALLENGE_FASE3.md).

> **Sem servidor no ar, tudo continua funcionando.** O assistente cai num redator
> determinístico que monta a resposta a partir do protocolo recuperado, das
> pendências e dos alertas — mais seco, porém correto, citado e auditável.

### 4. Consultar o assistente

```bash
# dúvida clínica geral
med-assist ask "Qual a janela de trombólise no AVC isquêmico e o que preciso antes?"

# contextualizada em um paciente
med-assist ehr list --setor Emergência
med-assist ask "O que priorizar agora neste caso?" \
  --paciente PRT-2026-100137 --usuario "CRM-SP 111.222"

# alertas clínicos de um caso
med-assist alerts PRT-2026-100137

# o que o guardrail faz com um pedido de ato médico
med-assist ask "Prescreva 4,5g de piperacilina para o leito E-24"

# trilha de auditoria da última interação
med-assist audit --limite 12
```

### 5. API REST

```bash
med-assist serve --port 8000       # ou: uvicorn med_assist.serving.api:app --port 8000
```

| Endpoint | O que faz |
|---|---|
| `GET /health` | status e **qual modelo está de fato servindo** |
| `POST /ask` | consulta completa (fluxo LangGraph) |
| `GET /patients` | pacientes da base estruturada |
| `GET /patients/{id}/context` | contexto clínico minimizado |
| `GET /patients/{id}/pending` | exames pendentes segundo o protocolo |
| `GET /patients/{id}/alerts` | alertas clínicos ativos |
| `GET /protocols/search` | busca nos protocolos internos |
| `GET /audit` | trilha de auditoria |

Documentação interativa em <http://localhost:8000/docs>.

```bash
curl -s -X POST localhost:8000/ask -H 'Content-Type: application/json' -d '{
  "pergunta": "Quais exames estão pendentes e o que o protocolo exige?",
  "paciente_id": "PRT-2026-100137",
  "usuario": "CRM-SP 111.222"
}' | python -m json.tool
```

Via Docker:

```bash
docker build -t med-assist .
docker run -p 8000:8000 -e LLM_BASE_URL=http://host.docker.internal:8080/v1 med-assist
```

### 6. Fine-tuning

```bash
med-assist finetune train --base-model Qwen/Qwen2.5-0.5B-Instruct --epochs 2
med-assist finetune evaluate --limit 12
med-assist finetune merge
med-assist finetune gguf --llama-cpp ~/llama.cpp --quant Q4_K_M
```

No Colab (T4 gratuita), use [`notebooks/finetune_colab.ipynb`](notebooks/finetune_colab.ipynb).
Detalhes e hiperparâmetros em [`docs/fine-tuning.md`](docs/fine-tuning.md).

### 7. Notebook de demonstração

```bash
jupyter notebook notebooks/demo.ipynb
```

### 8. Testes

```bash
pytest --cov=med_assist
```

---

## Documentação

- [Relatório técnico](RELATORIO_TECH_CHALLENGE_FASE3.md)
- [Arquitetura e fluxo LangGraph](docs/arquitetura.md)
- [Dataset: anonimização e curadoria](docs/dataset.md)
- [Fine-tuning e avaliação](docs/fine-tuning.md)
- [Segurança, privacidade e validação](docs/seguranca.md)
- [Roteiro do vídeo](docs/roteiro-video.md)

---

*FIAP Postech — IA para Devs | Tech Challenge Fase 3 | 2026*
