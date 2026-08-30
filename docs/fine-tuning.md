# Fine-tuning da LLM com os dados internos

## O que o treino ensina (e o que não ensina)

O objetivo do fine-tuning aqui é **comportamental**: registro clínico em
português, formato dos documentos institucionais, citação obrigatória da fonte e
recusa de ato médico. Conhecimento de protocolo **não** é objetivo do treino — ele
entra por recuperação, em tempo de consulta.

A razão é operacional: protocolo é documento vivo. O bundle de sepse na versão
6.0 vira 6.1 no mês seguinte, e um modelo treinado com a versão antiga continuaria
repetindo a antiga com toda a confiança do mundo. Trocar o arquivo em
`data/processed/protocolos/` corrige a resposta imediatamente, sem retreino.

## Escolha do modelo base

A Fase 2 já servia um **Qwen3 4B Instruct quantizado (Q4_K_M)** em `llama.cpp`.
Duas restrições decidiram o modelo desta fase:

1. **GGUF quantizado não é treinável.** Pesos em 4 bits são formato de
   inferência; gradiente exige os pesos originais.
2. **Sem GPU CUDA na máquina de desenvolvimento** (Intel Arc integrada, 30 GB de
   RAM, 8 núcleos). LoRA num modelo de 4B em CPU levaria muitas horas por época.

A saída mantém a continuidade sem fingir capacidade que não existe: treinar um
**Qwen menor da mesma família** — mesmo chat template, mesmo tokenizer — e servir
o resultado no mesmo `llama-server`. O que muda entre as fases é o modelo servido.

| Cenário | Modelo base | Onde roda |
|---|---|---|
| Desenvolvimento e demonstração | `Qwen/Qwen2.5-0.5B-Instruct` | CPU local |
| Treino de maior capacidade | `Qwen/Qwen2.5-1.5B-Instruct` | Colab T4 (`--device cuda`) |

O pipeline é o mesmo nos dois; muda um parâmetro.

## LoRA

| Hiperparâmetro | Valor | Por quê |
|---|---|---|
| `r` | 16 | posto suficiente para ajuste de estilo e formato |
| `lora_alpha` | 32 | escala 2× o posto, razão usual |
| `lora_dropout` | 0,05 | regularização leve para dataset pequeno |
| Módulos-alvo | `q,k,v,o,gate,up,down_proj` | atenção + MLP, cobertura padrão Qwen/Llama |
| Learning rate | 2e-4 | faixa típica de LoRA (10–100× o de fine-tuning completo) |
| Épocas | 2 | dataset pequeno; além disso o modelo decora |
| Batch efetivo | 8 (1 × 8 de acumulação) | cabe em CPU sem estourar memória |
| `max_len` | 2048 tokens | p95 do dataset é 1.596 (os exemplos contextualizados são longos) |

Treináveis: **8.798.208 de 502.830.976 parâmetros (1,75%)**.

## Dois detalhes que definem a qualidade

**1. Chat template do próprio modelo.** As mensagens são renderizadas com
`tokenizer.apply_chat_template`, então o formato de treino é exatamente o que o
servidor usa na inferência. Formato divergente entre treino e inferência é a
causa mais comum de "treinei e piorou".

**2. Loss só na resposta.** Os tokens do system e do turno do usuário recebem
label `-100`. Sem essa máscara, boa parte do sinal de treino é gasta decorando o
enunciado da pergunta — e o system prompt, que é longo e se repete em todos os
exemplos, dominaria a loss.

```python
labels = [IGNORE_INDEX] * len(tokens_do_prompt) + tokens_da_resposta
```

## Laço de treino

Explícito (AdamW + warmup linear de 6% + cosine decay + clipping em 1,0) em vez do
`Trainer` da `transformers`: são ~60 linhas, o comportamento fica visível e não há
dependência de mudanças de API entre versões da biblioteca. A avaliação roda antes
do treino e ao fim de cada época, medindo loss e perplexidade só sobre os tokens
supervisionados.

## Execução

```bash
pip install -r requirements-finetune.txt

# CPU (validado nesta máquina)
med-assist finetune train --base-model Qwen/Qwen2.5-0.5B-Instruct --epochs 2

# GPU (Colab T4) — ver notebooks/finetune_colab.ipynb
med-assist finetune train --base-model Qwen/Qwen2.5-1.5B-Instruct --device cuda --epochs 3

# Avaliação comparando base e fine-tunado nos mesmos prompts
med-assist finetune evaluate --limit 12
```

## Publicação: do adapter ao servidor

```bash
med-assist finetune merge                                    # funde o LoRA nos pesos base
med-assist finetune gguf --llama-cpp ~/llama.cpp            # gera med-assist-f16.gguf
llama-server -m models/med-assist-f16.gguf --alias med-assist --port 8080
```

A quantização (`--quant Q4_K_M`) é opcional e depende do binário `llama-quantize`
compilado no llama.cpp; sem ele o comando avisa e mantém o f16 — que para 0,5B
roda folgado em CPU.

O `GET /health` da API mostra qual modelo está de fato servindo — auditar isso
importa, porque a mesma pergunta responde diferente conforme o servidor esteja
com o modelo base ou com o do hospital.

## Avaliação

Perplexidade sozinha não responde ao que interessa. A avaliação gera respostas
com o modelo base e com o fine-tunado para o **mesmo** conjunto de perguntas e
compara critérios objetivos, verificáveis por regex — e é por serem simples que
servem de teste de regressão:

| Métrica | O que mede |
|---|---|
| `cita_fonte` | resposta referencia `PROT-XXX-000` ou "Fonte:" |
| `mantem_disclaimer` | menciona validação humana / médico assistente |
| `recusa_ato_medico` | recusa pedidos de prescrição **sem** o guardrail atuar |
| `responde_em_portugues` | registro em pt-BR |
| `sem_pii` | nenhum dado cadastral na resposta |

Os resultados ficam em `results/eval/avaliacao_modelo.json`. A análise completa
está na [seção 6 do relatório](../RELATORIO_TECH_CHALLENGE_FASE3.md).

## Resultados medidos (CPU, Qwen2.5-0.5B)

| Rodada | Dataset | Duração | val_loss | Perplexidade |
|---|---|---|---|---|
| v1 | 328 exemplos, `max_len` 1024 | 56,2 min | 2,451 → 1,418 | 11,60 → 4,13 |
| **v2** | 365 exemplos, `max_len` 2048 | 61,4 min | 2,451 → **1,183** | 11,60 → **3,27** |

Comportamento na v2, base vs. fine-tunado nos mesmos 14 prompts:

| Métrica | Base | Fine-tunado |
|---|---|---|
| Cita a fonte | 0,000 | **1,000** |
| Recusa ato médico (sem guardrail) | 0,667 | **1,000** |
| Sem PII na resposta | 1,000 | 1,000 |
| Tamanho médio | 599 caracteres | 274 caracteres |

## A lição da v1: formato de treino = formato de inferência

A v1 tinha boas métricas e **falhou no fluxo**: diante do prompt de produção, com
suas quatro seções de contexto, o modelo ecoou as seções em vez de responder.
Nenhuma métrica agregada capturou isso — só a execução ponta a ponta.

A causa era estrutural: 328 exemplos de "pergunta curta → resposta" não ensinam a
responder um prompt com contexto. A correção foi
[`data/contextualized.py`](../src/med_assist/data/contextualized.py), que gera
exemplos com o mesmo prompt do runtime. Na v2 o eco desapareceu, e a recusa de ato
médico — que na v1 não tinha se movido — subiu de 66,7% para 100%.

## Uma limitação que fica

O modelo de 0,5B acerta código e versão do protocolo e **inventa a data de
revisão** (3 de 3 respostas clínicas). Está mitigado por verificação cruzada
determinística (`check_citations`), que detectou 100% dos casos e registra a
divergência na auditoria. Modelos maiores tendem a errar menos; a verificação
continua necessária de qualquer forma.

> A métrica `recusa_ato_medico` mede o **modelo sozinho**, sem o guardrail de
> entrada. Em produção esses pedidos nem chegam ao LLM: a medida existe para
> isolar o efeito do fine-tuning, não para justificar remover a barreira.
