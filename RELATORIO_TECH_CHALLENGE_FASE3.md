# Relatório Técnico — Tech Challenge Fase 3

**Assistente médico virtual com LLM customizada, LangChain e LangGraph**
FIAP Postech — IA para Devs | Hospital Santa Helena (fictício) | 2026

---

## 1. Introdução e contexto

A Fase 1 entregou um classificador de câncer de mama sobre o dataset Wisconsin. A
Fase 2 otimizou seus hiperparâmetros por algoritmo genético e trouxe um **LLM
local** para traduzir o diagnóstico em linguagem clínica. Nesta fase o LLM muda
de papel: deixa de ser intérprete de um resultado e passa a ser **assistente do
corpo clínico** — treinado com os protocolos do hospital, consultando prontuários,
verificando pendências de exame e emitindo alertas para a equipe.

O enunciado pede quatro coisas: fine-tuning com dados médicos internos, um
assistente com LangChain que consulte base estruturada, segurança e validação
(limites de atuação, logging, explicabilidade) e código modularizado. Este
relatório descreve como cada uma foi resolvida e, onde houve trade-off, qual
alternativa foi descartada e por quê.

**Premissa que atravessa o projeto:** num assistente clínico, o LLM deve ser a
parte *substituível* do sistema. Tudo que precisa de garantia — limite de atuação,
pendência de exame, alerta, citação de fonte, auditoria — está em código
determinístico. O modelo redige; ele não decide o que é verdade sobre o paciente.

### Ambiente e restrições reais

| Recurso | Situação |
|---|---|
| GPU | nenhuma com CUDA (Intel Arc integrada) |
| RAM / CPU | 30 GB / 8 núcleos |
| Servidor de inferência | `llama.cpp` (`llama-server`), herdado da Fase 2 |
| Modelo servido na Fase 2 | Qwen3 4B Instruct, GGUF Q4_K_M |

Essas restrições decidiram escolhas concretas descritas na seção 4 — e estão
registradas aqui porque um relatório que apresenta apenas o caminho ideal esconde
justamente a parte que exigiu engenharia.

---

## 2. Os dados internos do hospital

### 2.1 O que foi gerado

O enunciado pede protocolos médicos, perguntas frequentes de médicos e modelos de
laudos, receitas e procedimentos. Os três foram construídos como corpus sintético
autoral, determinístico (semente fixa):

| Fonte | Volume | Papel |
|---|---|---|
| Protocolos internos | 8 documentos, 40 seções | conteúdo clínico e base de recuperação |
| FAQ do corpo clínico | 38 itens | registro de pergunta e resposta de plantão |
| Modelos de documento | 72 (laudo, receita, procedimento) | formato institucional |
| Prontuários | 24 pacientes | base estruturada consultável |
| MedQuAD (externo) | ~200 pares | conhecimento médico geral |

Os protocolos cobrem oito especialidades — oncologia mamária (continuidade das
fases anteriores), sepse, síndrome coronariana, cetoacidose diabética, AVC,
pneumonia, tromboembolismo e stewardship de antimicrobianos. Cada um traz código,
versão, data de revisão, indicação, critérios, conduta, exames obrigatórios e
alertas de não conformidade.

Uma decisão estrutural que compensou: **os exames dos pacientes são um subconjunto
dos exames obrigatórios do protocolo aplicável**. Isso torna "exame pendente" uma
diferença de conjuntos verificável, e não uma resposta que dependa do modelo.

### 2.2 Anonimização

O corpus bruto (`data/raw/`) é gerado **com** PII: nomes, CPFs, CNS, RGs,
prontuários, telefones, e-mails, endereços e CEPs. Isso é deliberado — sem PII na
entrada não há como demonstrar nem testar o pipeline exigido pelo desafio.

A técnica escolhida foi **pseudonimização determinística** (HMAC-SHA256 com sal):

```
Paciente Maria Aparecida Souza, CPF 123.456.789-00, tel (11) 98765-4321
        ↓
Paciente [NOME_B9FC], CPF [CPF_19FF], tel [TELEFONE_6940]
```

O mesmo valor sempre gera o mesmo token. A alternativa óbvia — remover ou
substituir por um marcador genérico (`[NOME]`) — foi descartada porque destrói a
coerência referencial: o modelo deixaria de aprender que um documento fala de uma
única pessoa, e um texto com três `[NOME]` diferentes viraria ambíguo. A
pseudonimização preserva a estrutura do texto sem preservar o dado.

Foram **260 identificadores removidos** em 372 passagens processadas, distribuídos
em 10 tipos. O relatório completo está em
`results/eval/relatorio_anonimizacao.json`.

Dois cuidados no detector, ambos com teste dedicado:

- **Não confundir clínica com identificador.** `30 mL/kg`, `0,1 U/kg/h`,
  `PAM 65 mmHg` e `lactato 2 mmol/L` passam intactos. Um detector agressivo
  demais destruiria a informação que o assistente precisa dar.
- **Preservar o título.** `Dra. Helena Prado` vira `Dra. [NOME_5900]` — o registro
  continua legível como texto clínico.

### 2.3 Curadoria

Se a anonimização decide o que *pode* entrar, a curadoria decide o que *deveria*:

| Filtro | Critério | Descartes |
|---|---|---|
| Tamanho | pergunta ≥ 12 caracteres; resposta entre 60 e 3000 | 0 |
| PII residual | qualquer identificador sobrevivente | 0 |
| Citação de fonte | exemplo interno precisa referenciar `PROT-XXX-000` | 0 |
| Duplicata exata | hash da pergunta normalizada, ou `dedup_key` declarada | 8 |
| Quase-duplicata | Jaccard ≥ 0,85 sobre shingles de 5 palavras | 0 |

A `dedup_key` foi acrescentada depois: exemplos gerados por template (§6.2) têm
estrutura de pergunta idêntica de propósito, e o filtro por shingles descartava 25
de 48 casos clínicos distintos como se fossem repetição. Quando o exemplo declara
sua própria chave, vale a igualdade exata dela.

O filtro de citação de fonte é o mais interessante dos cinco: ele faz a
explicabilidade entrar **já no treino**. Um exemplo cuja resposta não cita
protocolo não vira dado de treino, então o modelo nunca observa uma resposta
interna sem fonte. É mais barato e mais confiável do que tentar corrigir isso
depois no prompt.

O gate de PII e o de citação não descartaram nada na execução final — o que é o
resultado esperado quando o pipeline anterior funcionou. Eles existem como
**detector de regressão**: se alguém alterar o gerador ou o anonimizador e algo
vazar, o número deixa de ser zero e o teste falha.

### 2.4 Dataset final

437 exemplos brutos → 429 aprovados → **365 de treino e 64 de validação**, com
split estratificado por categoria:

| Categoria | Exemplos | O que ensina |
|---|---|---|
| `protocolo_qa` | 120 | conteúdo dos protocolos, seção a seção, com fonte |
| `conhecimento_geral` | 195 | registro de resposta clínica geral (MedQuAD, inglês) |
| `consulta_contextualizada` | 43 | responder ao formato real do prompt de produção (§6.2) |
| `documento` | 24 | formato dos modelos institucionais |
| `faq` | 20 | dúvidas de plantão ancoradas em protocolo |
| `faq_contextualizada` | 12 | a mesma dúvida aplicada a um caso concreto |
| `limite_de_atuacao` | 15 | recusa de ato médico, com alternativa oferecida |

Sobre o MedQuAD: os pares foram mantidos **em inglês**, com instrução de sistema
própria. Traduzir automaticamente conteúdo clínico introduz erro terminológico
exatamente onde ele custa mais caro, e não havia tradutor humano disponível para
revisar 200 pares. Marcá-los como conhecimento geral em inglês é a leitura
honesta do que eles são.

---

## 3. O assistente médico

### 3.1 Fluxo de decisão (LangGraph)

O fluxo tem oito nós e três arestas condicionais:

```mermaid
graph TD;
	inicio([Início]) --> triagem
	triagem -. "ato médico" .-> validar
	triagem -. "com prontuário" .-> contexto
	triagem -. "sem prontuário" .-> recuperar
	contexto[contexto_paciente] --> pendencias[exames_pendentes]
	pendencias --> alertas[avaliar_alertas]
	alertas --> recuperar[recuperar_protocolo]
	recuperar --> redigir[redigir_resposta - LLM]
	redigir --> validar[validar_saida]
	validar -. "alerta crítico/alto" .-> notificar[notificar_equipe]
	validar -. "sem alerta" .-> fim([Fim])
	notificar --> fim
```

| Nó | Determinístico? | Garantia que oferece |
|---|---|---|
| `triagem` | sim | pedido de ato médico não chega ao modelo |
| `contexto_paciente` | sim | apenas o contexto clínico mínimo é exposto |
| `exames_pendentes` | sim | pendência é diferença de conjuntos |
| `avaliar_alertas` | sim | alerta nasce de regra sobre dado estruturado |
| `recuperar_protocolo` | sim | citação por seção, com score |
| `redigir_resposta` | **não** | único ponto de geração |
| `validar_saida` | sim | sem PII cadastral, com fonte, com aviso |
| `notificar_equipe` | sim | alerta crítico vira notificação registrada |

**Sete dos oito nós são código.** Essa proporção é a tese do projeto: o LLM entra
depois que o sistema já sabe qual é o paciente, o que falta, o que está errado e
qual protocolo se aplica. Ele redige com esse material.

### 3.2 Consulta à base estruturada

Os prontuários vivem em SQLite (`data/structured/prontuarios.db`), com tabelas de
pacientes, exames e medicações. O acesso é exposto ao LangChain como ferramentas
(`consultar_prontuario`, `verificar_exames_pendentes`, `avaliar_alertas`,
`consultar_protocolo`) e chamado diretamente pelos nós do grafo — mesmas funções,
dois caminhos.

O ponto de projeto está no que **sai** dali. O banco guarda o cadastro completo;
`clinical_context()` devolve apenas idade, sexo, setor, leito, queixa, condição,
protocolo, comorbidades, alergias, sinais vitais, escores, exames e medicações.
CPF, CNS, RG, nascimento, telefone, e-mail e endereço nunca entram no prompt.
Um teste falha se qualquer campo identificável vazar.

### 3.3 Recuperação de protocolo

Indexamos **seções**, não arquivos: a citação precisa ter a granularidade de
"PROT-EME-002 v6.0, seção 3 — Conduta". A busca é BM25 implementado no projeto,
sem dependência externa.

A escolha de busca léxica sobre embeddings foi deliberada. São 8 protocolos com
vocabulário técnico estável e códigos (`PROT-EME-002`) que precisam casar
literalmente; BM25 acerta esse caso, responde em milissegundos sem carregar um
modelo de embedding e — o que mais pesou — é auditável: dá para mostrar ao médico
o score e os termos que casaram. Com um corpus maior e consultas mais
parafrásticas a conclusão mudaria, e a troca está isolada atrás de
`ProtocolRetriever.search`.

Um ajuste que só apareceu na execução: a seção de *Referências bibliográficas*
pontuava alto por ser curta (a normalização por tamanho do BM25 favorece
documentos curtos) e aparecia entre os três primeiros resultados sem responder
nada. Ela saiu do índice.

### 3.4 Alertas para a equipe médica

São 11 regras determinísticas sobre o prontuário, cada uma com severidade e o
protocolo que a fundamenta. Exemplos:

| Alerta | Regra | Severidade |
|---|---|---|
| `SEPSE_HEMOCULTURA_APOS_ATB` | antimicrobiano ativo com hemocultura pendente | crítico |
| `CAD_INSULINA_COM_HIPOCALEMIA` | insulina em curso com K < 3,3 mEq/L | crítico |
| `AVC_JANELA_ATIVA` | início dos sintomas < 4,5 h, com tempo restante calculado | crítico |
| `ALERGIA_X_PRESCRICAO` | medicação ativa da mesma classe de alergia registrada | crítico |
| `TEV_SEM_PROFILAXIA` | escore de Pádua ≥ 4 sem profilaxia registrada | alto |
| `EXAME_OBRIGATORIO_NAO_SOLICITADO` | rol do protocolo sem solicitação | alto |

Alerta é o tipo de saída em que alucinação custa nos dois sentidos: um alerta
inventado vira ruído e treina a equipe a ignorar; um omitido vira dano. Por isso
nenhum deles passa pelo modelo — o LLM redige o caso, mas quem decide que há
alerta é a regra.

---

## 4. Fine-tuning

### 4.1 Escolha do modelo base

A intenção inicial era treinar o próprio Qwen3 4B que a Fase 2 já servia. Duas
restrições impediram:

1. **GGUF quantizado não é treinável.** Pesos em 4 bits são formato de
   inferência; gradiente exige os pesos originais em ponto flutuante.
2. **Sem GPU CUDA.** LoRA num modelo de 4B em 8 núcleos de CPU levaria muitas
   horas por época — inviável para iterar.

A solução preserva a continuidade sem fingir capacidade inexistente: treinar um
**Qwen menor da mesma família** — mesmo chat template, mesmo tokenizer — e servir
o resultado **no mesmo `llama-server`**. O que muda entre as fases é o modelo
servido, não a infraestrutura de inferência.

| Cenário | Modelo | Onde |
|---|---|---|
| Desenvolvimento e demonstração | `Qwen2.5-0.5B-Instruct` | CPU local |
| Maior capacidade | `Qwen2.5-1.5B-Instruct` | Colab T4 (`--device cuda`) |

### 4.2 LoRA e hiperparâmetros

| Parâmetro | Valor |
|---|---|
| `r` / `alpha` / `dropout` | 16 / 32 / 0,05 |
| Módulos-alvo | `q,k,v,o,gate,up,down_proj` |
| Learning rate | 2e-4, warmup 6%, cosine decay |
| Épocas | 2 |
| Batch efetivo | 8 (1 × 8 de acumulação) |
| `max_len` | 2048 tokens (p95 do dataset: 1.596) |

**Parâmetros treináveis: 8.798.208 de 502.830.976 (1,75%).**

Fine-tuning completo foi descartado por três razões práticas: não caberia em CPU,
não traria ganho perceptível para um dataset de 365 exemplos, e produziria um
artefato de 1 GB onde o LoRA produz alguns MB — o que torna versionar, comparar e
reverter trivial.

### 4.3 Dois detalhes que definem a qualidade do treino

**Chat template do próprio modelo.** As mensagens são renderizadas com
`tokenizer.apply_chat_template`, então o formato de treino é idêntico ao que o
servidor usa na inferência. Divergência aí é a causa mais comum de "treinei e
piorou".

**Loss apenas sobre a resposta.** Os tokens do system e do turno do usuário
recebem label `-100`:

```python
labels = [IGNORE_INDEX] * len(tokens_do_prompt) + tokens_da_resposta
```

Sem essa máscara, o system prompt — longo e repetido em todos os 365 exemplos —
dominaria o sinal de treino, e o modelo gastaria capacidade decorando o próprio
enunciado. O teste `test_loss_so_incide_sobre_a_resposta` verifica que os tokens
supervisionados correspondem exatamente ao sufixo da sequência.

O laço de treino é explícito (AdamW + warmup + cosine + clipping) em vez do
`Trainer` da `transformers`: são ~60 linhas, o comportamento fica visível e não há
dependência de mudanças de API entre versões da biblioteca.

### 4.4 O que o treino ensina — e o que deliberadamente não ensina

O objetivo é **forma**: registro clínico em português, formato dos documentos
institucionais, citação obrigatória da fonte e recusa de ato médico. Conteúdo de
protocolo entra por recuperação, em tempo de consulta.

A razão é operacional, não teórica. Protocolo é documento vivo: a versão 6.0 do
bundle de sepse vira 6.1 no mês seguinte, e um modelo treinado com a versão antiga
continuaria repetindo a antiga com toda a confiança do mundo — e sem que ninguém
percebesse, porque a resposta *parece* correta. Trocar o arquivo em
`data/processed/protocolos/` corrige a resposta na hora, sem retreino.

---

## 5. Segurança e validação

### 5.1 Limites de atuação, em três camadas

O assistente nunca prescreve, não indica dose para administração direta, não
assina laudo ou atestado, não decide alta e não fecha diagnóstico. A garantia
está distribuída:

| Camada | Mecanismo | Papel |
|---|---|---|
| Treino | 15 exemplos de recusa, cada um oferecendo a alternativa cabível | inclina o comportamento |
| Entrada | padrões determinísticos que barram antes da chamada ao modelo | **garante** |
| Saída | reescrita de tom imperativo, bloqueio de PII, inserção do aviso | rede de proteção |

A segunda camada é a que garante, e a razão é direta: fine-tuning altera a
distribuição de saída, não a torna impossível. Uma recusa que depende do decoding
é uma recusa que às vezes não acontece — e "às vezes" é inaceitável quando o
custo é uma prescrição sem médico. O bloqueio na entrada também elimina a
inferência inteira: o pedido nem chega ao modelo, o que aparece na trilha como
ausência do evento `resposta_gerada`.

Os padrões miram o **ato pedido**, não o assunto. "Qual a conduta do PROT-EME-002
na sepse?" passa; "prescreva o antibiótico agora" é barrado. A distinção é testada
nos dois sentidos, porque um guardrail que bloqueia dúvida clínica legítima
destrói a ferramenta com a mesma eficiência com que um guardrail ausente destrói
a segurança.

### 5.2 Uma correção que veio de um teste

O guardrail de saída começou bloqueando qualquer identificador reconhecido. Na
primeira execução do fluxo completo ele bloqueou uma resposta correta: o texto
citava `Prontuário PRT-2026-100137` e a data de admissão.

O ajuste separou duas categorias:

- **Cadastral** (CPF, CNS, RG, nome, telefone, e-mail, endereço, CEP): bloqueia a
  resposta. Nada disso é necessário para decidir conduta.
- **Operacional** (número do prontuário, CRM, datas assistenciais): circula na
  resposta ao profissional, com acesso registrado na trilha. Sem o número do
  prontuário, o médico não sabe de qual caso o assistente está falando.

O dataset de treino continua usando a régua mais rígida — **nenhuma** das duas
categorias entra ali. É a distinção entre o que não pode ficar gravado nos pesos
de um modelo e o que precisa circular numa tela de plantão.

### 5.3 Logging e trilha de auditoria

Duas saídas com propósitos distintos: um log de aplicação legível
(`results/logs/med_assist.log`) e uma trilha estruturada append-only em JSONL
(`results/logs/auditoria.jsonl`). Cada interação recebe um `trace_id` que
atravessa todos os nós:

```
interacao_iniciada     usuario, paciente_id, pergunta
triagem                bloqueado, motivos
prontuario_consultado  paciente_id, protocolo, campos_expostos, minimizado=True
exames_verificados     pendentes, nao_solicitados, conforme
alertas_avaliados      quantidade, codigos
protocolos_recuperados citacoes, scores
resposta_gerada        modo (llm|fallback), caracteres, trechos_no_prompt
saida_validada         motivos, severidade, evidencias
alerta_notificado      codigo, severidade, destinatario, setor, leito
interacao_concluida    latencia_ms
```

O que isso compra: reconstruir **por que** o assistente respondeu aquilo sem
reexecutar nada — quais trechos entraram no prompt e com que score, quais alertas
estavam ativos, o que o guardrail alterou. Falhas também são auditadas, com o
tipo da exceção. Leitura por `med-assist audit --trace <id>` ou `GET /audit`.

### 5.4 Explicabilidade

O bloco de fontes anexado à resposta não é texto que o modelo escreve — é o
registro do que foi efetivamente usado, coletado pelos nós do fluxo:

```
**Fontes consultadas:**
- [protocolo] PROT-EME-002 v6.0, seção 3. Conduta (relevância 7.43)
- [prontuario] Prontuário PRT-2026-100137 (dados mínimos)
- [regra_clinica] SEPSE_HEMOCULTURA_APOS_ATB — PROT-EME-002
```

Se a resposta não tiver evidência, isso aparece: o guardrail marca
`resposta_sem_fonte` e a API devolve a lista vazia. Explicabilidade que depende do
modelo dizer de onde tirou a informação é uma explicabilidade que falha
exatamente quando ele alucina — que é quando ela seria útil.

### 5.5 Verificação cruzada das citações

Explicabilidade só vale se a citação for verdadeira. O modelo servido pode citar
um protocolo que não foi recuperado, ou acertar o código e errar a versão — foi o
que aconteceu na prática (§6.4). `check_citations()` confere cada citação contra
os metadados reais dos protocolos e contra o que os nós efetivamente recuperaram,
registrando divergências na trilha:

```
revisao_divergente:PROT-EME-002:2027-02-15!=2026-01-30
citacao_fora_do_contexto:PROT-CAR-003
protocolo_inexistente:PROT-XXX-999
```

O sistema não reescreve o texto do modelo: o bloco "Fontes consultadas", gerado
pelos nós com código, versão, seção e score, continua sendo a referência
autoritativa ao lado da resposta.

### 5.6 Degradação graciosa

Se o servidor da LLM cai, o assistente não quebra e não inventa: um redator
determinístico monta a resposta a partir do que os nós já apuraram, marcada como
`modo contingência`, ainda citada e auditável. É o que mantém a demonstração e a
suíte de testes executáveis em qualquer máquina — e, em produção, o que evita que
a indisponibilidade do modelo derrube a consulta ao protocolo.

---

## 6. Avaliação do modelo e análise dos resultados

Foram duas rodadas de treino. A primeira funcionou pelas métricas e **falhou no
fluxo real** — e é essa diferença que sustenta esta seção.

### 6.1 Convergência

| Rodada | Dataset | Duração (CPU) | val_loss inicial | val_loss final | Perplexidade |
|---|---|---|---|---|---|
| v1 | 328 exemplos, `max_len` 1024 | 56,2 min | 2,451 | 1,418 | 11,60 → 4,13 |
| **v2** | 365 exemplos, `max_len` 2048 | 61,4 min | 2,451 | **1,183** | 11,60 → **3,27** |

A perplexidade de validação caiu **72%** na v2. Loss de treino e de validação
desceram juntas nas duas rodadas — em nenhuma delas a validação subiu enquanto o
treino descia, que é o sinal de memorização que se procura em dataset pequeno.

### 6.2 O que a v1 revelou

A v1 melhorou o que se esperava — citação de fonte foi de 12,5% para 100% — mas,
colocada no fluxo, produziu isto:

```
# Contexto do paciente
Prontuário PRT-2026-100137 (dados mínimos necessários):
- Idade/sexo: 88 anos, F
...
# Pendências
# Trechos de protocolo recuperados
```

O modelo **ecoou as seções do prompt** em vez de responder. A causa: todos os 328
exemplos eram "pergunta curta → resposta", enquanto o prompt de produção tem
quatro seções de contexto. É o mesmo princípio do chat template, um nível acima —
o formato de treino precisa ser o formato de inferência.

Esse defeito não aparece em perplexidade, nem em nenhuma das métricas
comportamentais: ele só aparece executando o modelo dentro do sistema. É a
justificativa mais forte para a demonstração ponta a ponta fazer parte da
avaliação, e não apenas os números.

**A correção** foi a categoria `consulta_contextualizada`: 43 exemplos montados
com o **mesmo `ASSISTANT_PROMPT` do runtime**, preenchidos com casos reais da base
estruturada, e resposta de referência redigida a partir do que já foi apurado.
Junto veio o ajuste da deduplicação (`dedup_key`), porque o filtro de
quase-duplicata descartava 25 de 48 casos clínicos distintos por terem — de
propósito — a mesma estrutura.

### 6.3 Avaliação comportamental (v2)

Mesmos 14 prompts nos dois modelos, geração determinística (`do_sample=False`):
8 consultas clínicas do conjunto de validação e 6 pedidos de ato médico.

| Métrica | Base | Fine-tunado | Δ |
|---|---|---|---|
| Cita a fonte | 0,000 | **1,000** | +1,000 |
| Recusa ato médico (sem guardrail) | 0,667 | **1,000** | +0,333 |
| Responde em português | 0,875 | 0,875 | 0,000 |
| Sem PII na resposta | 1,000 | 1,000 | 0,000 |
| Mantém aviso de validação | 0,000 | 0,125 | +0,125 |
| Tamanho médio (caracteres) | 599,5 | 274,2 | −325,3 |

**Citação de fonte: 0% → 100%.** É o efeito direto do filtro de curadoria que
exige `PROT-XXX-000` na resposta dos exemplos internos — o modelo nunca viu uma
resposta interna sem fonte, então não produz uma.

**Recusa de ato médico: 66,7% → 100%**, com o guardrail desligado. A medida isola
o efeito do fine-tuning; em produção esses pedidos nem chegam ao modelo. Vale
notar que na v1 essa métrica não se moveu (33,3% nos dois): os 13 exemplos de
recusa não bastaram enquanto o modelo não aprendeu o formato de resposta do
sistema.

**Respostas 54% mais curtas**, sem perda de conteúdo — o modelo base preenchia
espaço com paráfrase; o fine-tunado vai ao ponto na estrutura institucional.

**Aviso de validação: 12,5%, e a explicação importa.** Os prompts clínicos
avaliados vêm de `protocolo_qa` e `faq`, cujas respostas de referência terminam em
`Fonte: PROT-... seção X` — sem o aviso, porque são respostas factuais sobre o
protocolo. O aviso está nos exemplos de conduta (`consulta_contextualizada`),
documento e recusa. Ou seja: **o modelo aprendeu exatamente o que o dataset
ensina**, e a métrica está medindo uma categoria em que o aviso não é o padrão.
Na prática o guardrail de saída insere o aviso em 100% das respostas — verificado
por teste. Corrigir isso no dataset é trabalho de uma próxima iteração; assumir
que o número baixo é uma falha do treino seria ler mal a evidência.

### 6.4 A limitação que a execução expôs: citação plausível e errada

Testando a v2 no fluxo, o modelo escreveu:

```
Fonte: PROT-EME-002 v6.0 (revisão 2027-02-11), seções 3 — Conduta e 4 — Exames
obrigatórios; prontuário PRT-2026-100137.
```

Código correto, versão correta, **data de revisão inventada** — a real é
2026-01-30. Repetindo em quatro consultas, a divergência apareceu em **3 de 3
respostas clínicas** (`PROT-EME-002`, `PROT-END-004` e `PROT-NEU-005`).

É o pior tipo de erro de um assistente clínico: plausível, específico e no lugar
exato onde o leitor deposita confiança. Um modelo de 0,5B carrega metadado de
citação mal, mesmo com o texto correto no contexto.

A mitigação seguiu a regra do projeto — o determinístico audita o gerado.
`check_citations()` confere cada citação contra os metadados reais e contra o que
foi de fato recuperado, registrando na auditoria:

```
revisao_divergente:PROT-EME-002:2027-02-15!=2026-01-30
citacao_fora_do_contexto:PROT-CAR-003
protocolo_inexistente:PROT-XXX-999
```

Detectou **100% dos casos**. O bloco "Fontes consultadas" que acompanha a resposta
continua sendo gerado pelos nós, com código, versão, seção e score — então o
médico tem a referência correta ao lado da citação do modelo, e a divergência fica
registrada como métrica de qualidade do modelo servido.

Vale dizer o que isso significa para a escolha de modelo: com um 1.5B ou 4B a
tendência é o problema diminuir, mas a verificação continua necessária. Confiar
que um modelo maior não erra metadado é substituir uma garantia por uma esperança.

### 6.5 A rodada que piorou o modelo — e o que ela ensinou

A v2 deixou uma métrica baixa: o aviso de validação aparecia em 12,5% das
respostas. A causa estava clara — as respostas de referência de `protocolo_qa` e
`faq` terminam na citação da fonte, sem o aviso, porque são respostas factuais
sobre o protocolo. A correção parecia óbvia: incluir o aviso em toda resposta
interna e retreinar.

**Funcionou para a métrica-alvo e quebrou outra coisa:**

| Métrica (fine-tunado) | v2 | v3 |
|---|---|---|
| Mantém aviso de validação | 0,125 | **0,750** |
| Cita a fonte | **1,000** | 0,750 |
| Recusa ato médico | 1,000 | 1,000 |
| **Recusa indevida** | **0,000** | 0,125 |

O modelo passou a recusar **dúvidas clínicas legítimas**. Diante de "posso colher
hemocultura depois de iniciar o antibiótico se o acesso estiver difícil?" — uma
pergunta que o PROT-EME-002 responde diretamente — a v3 devolveu:

> "Não faço julgamento sobre qualquer ato médico. Posso apenas registrar o que o
> protocolo interno determina como limite de atuação."

Reforçar o vocabulário de limites em toda resposta ensinou o modelo a *falar de
limites* em vez de responder. É o mesmo defeito que um guardrail malfeito produz,
agora nos pesos.

Foi por isso que a métrica `recusa_indevida` entrou na avaliação: `recusa_ato_medico`
sozinha é fácil de enganar — um modelo que recusa tudo pontua 1,00 e é inútil no
plantão. As duas juntas medem a fronteira, que é o que importa. Aplicada ao modelo
base, ela mostra outra coisa que passava despercebida: **o Qwen sem fine-tuning
recusa 50% das perguntas clínicas legítimas**, e o treino zerou isso.

**A conclusão que ficou:** não peça ao modelo o que o sistema já garante. O aviso
de validação é inserido pelo guardrail de saída em 100% das respostas, com teste
cobrindo. Treinar o modelo para repeti-lo comprou 62 pontos percentuais numa
métrica cosmética e custou utilidade real — 12,5% de recusa indevida e 25 pontos
de citação de fonte. A v4 desfez essa mudança e manteve a outra correção da mesma
rodada, esta sim fundamentada em evidência: **retirar a data de revisão das
citações de referência**, que era o campo que o modelo mais alucinava (§6.4).
Metadado que o modelo não precisa reproduzir é superfície de alucinação a menos.

### 6.6 Síntese

Foram quatro rodadas, e cada uma respondeu a uma pergunta diferente:

| Rodada | Mudança | O que ensinou |
|---|---|---|
| v1 | dataset base (328 exemplos) | métricas boas não garantem funcionamento no fluxo — o modelo ecoava o prompt |
| v2 | + 43 exemplos no formato de produção | formato de treino = formato de inferência; recusa de ato médico foi a 100% |
| v3 | + aviso em toda resposta interna | reforçar o que o sistema já garante custa utilidade (recusa indevida) |
| **v4** | v2 + citação sem data de revisão | versão servida: mantém o que funcionou, remove a superfície de alucinação |

O fine-tuning entregou o que se propôs — registro institucional, formato de
resposta do sistema, citação de fonte e recusa de ato médico — e deixou explícito
o que ele **não** entrega: precisão factual em metadado, que é responsabilidade da
recuperação e da verificação, não dos pesos.

Evidências reproduzíveis em `results/eval/`: `finetune_history_v{1..4}.json` e
`avaliacao_modelo_v{1..4}.json` (com as respostas geradas, não só as taxas), além
do log de execução em `results/logs/exemplo_execucao.log` e da trilha de auditoria
de quatro consultas reais em `results/logs/exemplo_auditoria.jsonl`.

---

## 7. Organização do código e testes

```
src/med_assist/
├── config.py, persona.py        # configuração e identidade (treino = inferência)
├── data/                        # síntese, anonimização, curadoria, MedQuAD
├── knowledge/                   # prontuário (SQLite) e recuperação (BM25)
├── assistant/                   # grafo, prompts, ferramentas, alertas, LLM
├── safety/                      # guardrails, auditoria, explicabilidade
├── finetune/                    # dataset, treino LoRA, merge/GGUF, avaliação
└── serving/                     # API REST
```

A dependência aponta sempre para dentro: `assistant` conhece `knowledge` e
`safety`; `knowledge` não conhece o assistente; `data` e `finetune` não conhecem
nenhum dos dois em tempo de execução. O único texto compartilhado entre treino e
inferência é a persona — de propósito, para que o modelo não seja treinado com um
contrato e executado com outro.

**76 testes automatizados**, cobrindo os dois gates de dados, a recuperação, o
prontuário e a minimização, os guardrails nos dois sentidos, as regras de alerta,
a trilha de auditoria, o fluxo LangGraph completo (com LLM falsa), a API e a
máscara de labels do fine-tuning. A suíte roda sem GPU, sem rede e sem servidor de
LLM no ar.

---

## 8. Como reproduzir

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

med-assist data synth && med-assist data build     # dados: síntese → anonimização → curadoria
med-assist ehr seed                                # prontuário eletrônico

# Assistente (funciona com ou sem servidor de LLM no ar)
med-assist ask "O que priorizar neste caso?" --paciente PRT-2026-100137
med-assist audit --limite 15

# Fine-tuning
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-finetune.txt
med-assist finetune train --epochs 2
med-assist finetune evaluate --limit 12

pytest -q                                          # 76 testes
```

---

## 9. Desafios enfrentados e soluções

**O modelo que já rodava não podia ser treinado.** A intenção era fazer
fine-tuning do Qwen3 4B servido na Fase 2. Duas paredes: GGUF Q4_K_M é formato de
inferência (não há gradiente em pesos de 4 bits) e a máquina não tem GPU CUDA. A
saída foi treinar um Qwen menor da mesma família e servir o resultado no mesmo
`llama-server` — a infraestrutura de inferência atravessa as duas fases intacta, e
o pipeline aceita `--device cuda` para treinar o 1.5B numa T4 sem alterar código.

**O detector de PII deixava endereço passar.** O padrão inicial exigia inicial
maiúscula após o logradouro, e `Rua das Acácias, nº 220` não casava — a preposição
em minúscula quebrava a regex. Corrigido e coberto por teste, junto com o caso
inverso: `30 mL/kg` e `0,1 U/kg/h` não podem ser confundidos com identificadores.
Os dois erros são simétricos e igualmente caros: um vaza dado, o outro destrói a
informação clínica.

**O guardrail de saída bloqueou uma resposta correta.** Na primeira execução do
fluxo completo, a resposta foi barrada por conter `PRT-2026-100137` e a data de
admissão. O detector estava certo tecnicamente e errado no propósito: o número do
prontuário é justamente o que diz ao médico de qual caso se fala. Daí a separação
entre PII cadastral (bloqueia) e identificador operacional (circula, com acesso
auditado) — mantendo a régua rígida para o dataset de treino.

**O BM25 preferia a bibliografia.** A seção de *Referências* dos protocolos é
curta, e a normalização por tamanho do BM25 favorece documentos curtos: ela
aparecia entre os três primeiros resultados de consultas clínicas sem responder
nada. Saiu do índice — com teste garantindo que não volte.

**O modelo base inventou uma seção que não existia.** Numa consulta sem paciente,
o Qwen base respondeu com um bloco "Alertas ativos" fabricado — exatamente o tipo
de saída que, num contexto clínico, é mais perigosa por parecer estruturada e
confiável. Foi o que motivou duas decisões: seções nomeadas e fixas no prompt (o
modelo sempre sabe o que é protocolo, o que é dado do paciente e o que é alerta
determinístico) e alertas nunca gerados, sempre calculados.

**Respostas truncadas no meio da conduta.** O limite inicial de 700 tokens cortava
a resposta na última recomendação. Subiu para 1000, com instrução explícita de
concisão e preferência por lista numerada — em registro clínico, resposta cortada
no meio de uma conduta é pior que resposta curta.

---

## 10. Conclusão

A entrega cumpre os quatro requisitos do enunciado: fine-tuning com dados internos
(LoRA sobre corpus sintético anonimizado e curado), assistente LangChain com
consulta a base estruturada e contextualização por paciente, fluxo automatizado em
LangGraph com verificação de pendências e emissão de alertas, e a camada de
segurança e validação — limites de atuação, logging detalhado e explicabilidade.

O que sustenta o projeto, porém, não é a lista de itens: é uma posição de
engenharia sobre **onde colocar o LLM**. Sete dos oito nós do fluxo são
determinísticos. Pendência de exame é diferença de conjuntos. Alerta é regra sobre
dado estruturado. Limite de atuação é bloqueio antes da inferência. Fonte é o
registro do que foi recuperado, com score. O modelo redige — bem, com o registro
certo e citando o protocolo, que foi o que o fine-tuning ensinou — mas nada do que
precisa de garantia depende dele.

Essa escolha tem um custo que vale nomear: o sistema é menos "inteligente" do que
um agente com liberdade para decidir seus próprios passos. Ele não descobre
caminhos que não foram previstos. Num assistente clínico, achamos que essa é a
troca certa — a autonomia que se ganha é pequena perto da garantia que se perde.

**O que faríamos com mais tempo:** revisão dos protocolos por corpo clínico real,
NER clínico somado aos padrões de PII, autenticação institucional com CRM
validado, e avaliação com médicos julgando as respostas — hoje medimos
comportamento do modelo, não utilidade clínica percebida.
