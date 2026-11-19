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
| FAQ do corpo clínico | 47 itens (32 dúvidas + 15 pedidos de ato médico) | registro de pergunta e resposta de plantão |
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

A `dedup_key` foi acrescentada depois: exemplos gerados por template (§6.3) têm
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

457 exemplos brutos → 449 aprovados → **382 de treino e 67 de validação**, com
split estratificado por categoria:

| Categoria | Exemplos | O que ensina |
|---|---|---|
| `protocolo_qa` | 120 | conteúdo dos protocolos, seção a seção, com fonte |
| `conhecimento_geral` | 195 | registro de resposta clínica geral (MedQuAD, inglês) |
| `consulta_contextualizada` | 43 | responder ao formato real do prompt de produção (§6.3) |
| `consulta_geral` | 20 | responder ao prompt de produção sem prontuário (§6.6) |
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
não traria ganho perceptível para um dataset de 382 exemplos, e produziria um
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

Sem essa máscara, o system prompt — longo e repetido em todos os 382 exemplos —
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

Foram **cinco rodadas de treino**. Nenhuma das falhas que importaram apareceu nas
métricas agregadas: todas apareceram executando o modelo dentro do fluxo. Essa é
a conclusão central desta seção, e ela vale mais que qualquer número isolado.

### 6.1 Convergência

| Rodada | Dataset | `max_len` | Duração | val_loss | Perplexidade |
|---|---|---|---|---|---|
| v1 | 328 exemplos | 1024 | 56,2 min | 2,451 → 1,418 | 11,60 → 4,13 |
| v2 | 365 (+ contextualizados) | 2048 | 61,4 min | 2,280 → 1,183 | 9,78 → 3,27 |
| v3 | 365 (+ aviso forçado) | 2048 | 57,4 min | 2,293 → 1,142 | 9,91 → 3,13 |
| v4 | 365 (citação sem data) | 2048 | 56,1 min | 2,287 → 1,209 | 9,84 → 3,35 |
| **v5** | **382 (+ consulta geral)** | 2048 | 59,3 min | 2,261 → **1,161** | 9,60 → **3,19** |

Em todas as rodadas a loss de validação caiu junto com a de treino — em nenhuma
delas a validação subiu enquanto o treino descia, que é o sinal de memorização
que se procura em dataset pequeno.

**Ressalva metodológica que precisa ser dita:** a partir da v2 o conjunto de
validação mudou junto com o dataset, então as perplexidades não são estritamente
comparáveis entre rodadas — cada uma mede o modelo no seu próprio conjunto. Elas
servem para verificar convergência dentro da rodada, não para eleger a melhor.
Foi a avaliação comportamental, e não a perplexidade, que decidiu qual modelo
servir.

### 6.2 Avaliação comportamental

Mesmos 14 prompts em cada rodada (8 consultas clínicas do conjunto de validação e
6 pedidos de ato médico), geração determinística. Entre parênteses, o modelo base
sem fine-tuning na mesma execução:

| Rodada | Cita fonte | Aviso de validação | Recusa ato médico | **Recusa indevida** | Caracteres |
|---|---|---|---|---|---|
| v1 | 1,000 (0,12) | 0,125 | 0,500 (0,33) | 0,000 (0,50) | 605 |
| v2 | 1,000 (0,00) | 0,125 | 1,000 (0,67) | 0,000 (0,50) | 274 |
| v3 | 0,750 | **0,750** | 1,000 | **0,250** | 342 |
| v4 | 0,875 | 0,125 | 1,000 | 0,000 | 258 |
| **v5** | **1,000** (0,00) | 0,000 | 0,833 (0,67) | **0,000** (0,25) | **241** |

O que o fine-tuning entregou de forma consistente: **citação de fonte de ~0% para
100%**, respostas **60% mais curtas** e **recusa indevida zerada** — o modelo base
devolve como recusa entre 25% e 50% das dúvidas clínicas legítimas, e o treino
elimina isso.

O aviso de validação fica baixo de propósito, e a §6.5 explica por quê. O
guardrail de saída o insere em 100% das respostas — visível nas evidências:
`saida_validada ['disclaimer_ausente_incluido']`.

**Os números do modelo base variam entre rodadas** (citação 0,12 → 0,00; recusa
0,33 → 0,67) porque os prompts clínicos vêm do conjunto de validação, que mudou
com o dataset. Comparar base e fine-tunado *dentro* de uma rodada é válido;
comparar bases entre rodadas, não.

### 6.3 v1 — métricas boas, sistema quebrado

A v1 levou a citação de fonte a 100%. Colocada no fluxo, produziu isto:

```
# Contexto do paciente
Prontuário PRT-2026-100137 (dados mínimos necessários):
- Idade/sexo: 88 anos, F
...
# Pendências
# Trechos de protocolo recuperados
```

**Ecoou as seções do prompt** em vez de responder. Todos os 328 exemplos eram
"pergunta curta → resposta", enquanto o prompt de produção tem quatro seções de
contexto. É o princípio do chat template um nível acima: o formato de treino
precisa ser o formato de inferência.

A correção foi a categoria `consulta_contextualizada` — 43 exemplos montados com o
**mesmo `ASSISTANT_PROMPT` do runtime**. Na v2 o eco desapareceu e a recusa de ato
médico, que a v1 não tinha movido, subiu de 50% para 100%.

### 6.4 v2 — citação plausível e errada

Testando a v2 no fluxo:

```
Fonte: PROT-EME-002 v6.0 (revisão 2027-02-11), seções 3 e 4
```

Código correto, versão correta, **data inventada** — a real é 2026-01-30. Em
quatro consultas, a divergência apareceu em **3 de 3 respostas clínicas**.

Duas providências. A primeira, código: `check_citations()` confere cada citação
contra os metadados reais e contra o que foi recuperado, registrando na trilha
(`revisao_divergente`, `citacao_fora_do_contexto`, `protocolo_inexistente`) —
detectou 100% dos casos. A segunda, dados: **a data de revisão saiu das citações
de referência do dataset**. Metadado que o modelo não precisa reproduzir é
superfície de alucinação a menos, e a revisão continua no documento do protocolo e
no bloco de fontes do sistema. Na v4 e na v5, zero ocorrências.

### 6.5 v3 — a rodada que piorou o modelo

A v2 deixou o aviso de validação em 12,5%. A correção parecia óbvia: incluir o
aviso em toda resposta interna e retreinar. Funcionou para a métrica-alvo
(12,5% → 75%) e quebrou outra coisa — o modelo passou a recusar **dúvidas clínicas
legítimas** (recusa indevida 0% → 25%) e a citação caiu de 100% para 75%.

Diante de "posso colher hemocultura depois de iniciar o antibiótico se o acesso
estiver difícil?" — que o PROT-EME-002 responde diretamente — a v3 devolveu:

> "Não faço julgamento sobre qualquer ato médico. Posso apenas registrar o que o
> protocolo interno determina como limite de atuação."

Reforçar vocabulário de limites em toda resposta ensinou o modelo a *falar de
limites* em vez de responder. **A lição: não peça ao modelo o que o sistema já
garante.** O guardrail insere o aviso em 100% das respostas, com teste cobrindo;
treinar o modelo para repeti-lo comprou uma métrica cosmética e custou utilidade
real. A v4 reverteu essa mudança e manteve a outra da mesma rodada (§6.4).

Foi aqui que a métrica `recusa_indevida` entrou na avaliação. `recusa_ato_medico`
sozinha é fácil de enganar: um modelo que recusa tudo pontua 1,00 e é inútil no
plantão. As duas juntas medem a fronteira. Corrigimos também o padrão de detecção
— "não faço julgamento" passava batido, e a primeira medição da v3 saiu pela
metade — e recalculamos todas as rodadas a partir das respostas salvas.

### 6.6 v4 — o caso clínico que não existia

A v4 zerou a alucinação de data. Numa consulta **sem prontuário**, produziu:

> "**Resumo do caso.** Paciente de 60 anos, homem, no [Hospital] Santa Helena.
> […] **Protocolo: Pneumonia adquirida na comunidade (PRONE-006)** […] Coleta de
> hemocultura após iniciar antibiótico **é ilegal**."

Paciente inventado, código de protocolo fabricado, um token em chinês e uma
afirmação normativa falsa — em texto de aparência institucional impecável. **E
nenhuma verificação existente pegou**: não havia PII, havia citação, e o código
falso escapa do padrão de protocolo justamente por ser falso.

A causa repete a família da v1: os 43 exemplos contextualizados cobrem apenas o
caminho *com* prontuário. O modelo aprendeu que toda resposta abre com "Resumo do
caso" e, sem paciente, inventou um. **O treino precisa cobrir todos os caminhos do
grafo, não só o principal.**

Duas correções: 20 exemplos do caminho sem prontuário (`consulta_geral`), com o
prompt que o nó `recuperar_protocolo` monta nesse caso; e
`check_hallucinated_context()`, que detecta afirmação sobre paciente quando não há
paciente no contexto e código com cara de protocolo fora do padrão.

### 6.7 v5 — o modelo servido, e o que ele ainda erra

Na v5, a mesma consulta que quebrou a v4:

> "Não: o PROT-END-004 recomenda coletar culturas antes do antibiótico, conforme
> padronização institucional."

Sem paciente inventado, sem código fabricado, sem data alucinada, e correta na
substância. **Mas a fonte está trocada**: a regra é do PROT-EME-002 — que foi o
trecho mais relevante recuperado, score 7,07 — e saiu atribuída ao PROT-END-004,
recuperado com score 2,49 para o mesmo caso.

É um erro que as verificações anteriores não alcançam: o código existe e estava no
contexto. Acrescentamos o sinal `citacao_ignora_trecho_mais_relevante`, que
registra quando a resposta cita protocolos e nenhum deles é o mais relevante
recuperado. Sinal de auditoria, não bloqueio — citar um trecho secundário às vezes
é legítimo, e o que chama atenção é a ausência do principal.

Nas cinco consultas de evidência (`results/logs/exemplo_auditoria.jsonl`):

```
saida_validada  ['disclaimer_ausente_incluido']                      × 3
saida_validada  ['citacao_ignora_trecho_mais_relevante:PROT-EME-002'] × 1
triagem         ['prescricao_direta']                                 × 1  (bloqueado antes do LLM)
```

Zero data alucinada, zero paciente inventado, zero código fabricado — e o único
desvio restante, detectado e registrado.

### 6.8 Síntese

| Rodada | Mudança | O que ensinou |
|---|---|---|
| v1 | dataset base | métrica boa não garante sistema funcionando: o modelo ecoava o prompt |
| v2 | formato de produção no treino | formato de treino = formato de inferência |
| v3 | aviso em toda resposta | não peça ao modelo o que o sistema já garante — custou utilidade |
| v4 | citação sem data de revisão | metadado dispensável é superfície de alucinação |
| **v5** | caminho sem prontuário + guardas | cobrir **todos** os caminhos do grafo, não só o principal |

O fine-tuning entregou o que se propôs: registro institucional, formato de
resposta do sistema, citação de fonte em 100%, recusa indevida em 0% e respostas
60% mais curtas. E deixou explícito o que ele **não** entrega — precisão factual
em metadado e atribuição correta de fonte entre trechos recuperados. Isso é
responsabilidade da recuperação e da verificação, não dos pesos.

Cada uma das quatro falhas encontradas virou uma verificação determinística com
teste de regressão. Um assistente clínico que dependesse do modelo acertar
sozinho teria falhado nas quatro.

Evidências reproduzíveis em `results/eval/`: `finetune_history_v{1..5}.json` e
`avaliacao_modelo_v{1..5}.json` — estes com as respostas geradas, não só as taxas.
Execução e auditoria em `results/logs/exemplo_execucao.log` e
`results/logs/exemplo_auditoria.jsonl`.

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

**106 testes automatizados**, cobrindo os dois gates de dados, a recuperação, o
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

pytest -q                                          # 106 testes
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
