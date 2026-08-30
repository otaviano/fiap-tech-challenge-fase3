# Segurança, privacidade e validação

Contexto hospitalar: o custo de um erro aqui não é uma resposta ruim, é uma
conduta errada em cima de um paciente. As garantias do sistema são, por isso,
**determinísticas** — o modelo participa da redação, não da decisão sobre o que
é verdade.

## 1. Limites de atuação

O assistente nunca prescreve, não indica dose para administração direta, não
assina laudo ou atestado, não decide alta e não fecha diagnóstico.

Isso é garantido em três camadas independentes:

| Camada | Mecanismo | Onde |
|---|---|---|
| Treino | 15 exemplos de recusa no dataset, com a alternativa oferecida | `data/synth.py` |
| Entrada | padrões que barram o pedido **antes** da chamada ao modelo | `safety/guardrails.py` |
| Saída | reescrita de tom imperativo e inclusão do aviso de validação | `safety/guardrails.py` |

A camada de entrada é a que garante: fine-tuning *inclina* o comportamento, mas
uma recusa que depende do decoding é uma recusa que às vezes não acontece. Um
pedido de prescrição sequer chega ao LLM — o que também elimina o custo da
inferência e o risco de a formulação do pedido contornar o treino.

Os padrões miram o **ato pedido**, não o assunto: "qual a conduta do PROT-EME-002
na sepse?" passa; "prescreva o antibiótico agora" é barrado. Essa distinção é
testada nos dois sentidos.

## 2. Minimização de dados

O prontuário guarda o cadastro completo. O que chega ao prompt é outra coisa:

```python
CAMPOS_IDENTIFICAVEIS = ("nome", "cpf", "cns", "nascimento", "telefone",
                         "email", "endereco", "convenio", "medico_assistente")
```

`clinical_context()` devolve apenas idade, sexo, setor, leito, queixa, condição,
protocolo, comorbidades, alergias, sinais vitais, escores, exames e medicações —
o que sustenta a decisão clínica. Quem precisa do dado cadastral usa o prontuário
eletrônico, que tem controle de acesso próprio. Um teste falha se qualquer campo
identificável vazar para o contexto.

## 3. LLM local

O modelo roda em `llama-server` na infraestrutura do hospital, endpoint
OpenAI-compatível. Dados de paciente não saem da rede — alinhado à LGPD, que
trata dado de saúde como dado pessoal sensível (art. 5º, II) e exige base legal e
medidas técnicas compatíveis. Custo de API zero e o endpoint é plugável para
qualquer servidor compatível.

O fine-tuning também é local: o adapter LoRA é treinado em CPU ou GPU própria,
nunca em serviço de terceiros com upload do corpus interno.

## 4. Trilha de auditoria

Duas saídas, com propósitos distintos:

- `results/logs/med_assist.log` — log de aplicação, legível, para operação.
- `results/logs/auditoria.jsonl` — trilha estruturada, append-only, uma linha
  JSON por evento.

Cada interação recebe um `trace_id` que atravessa todos os nós do fluxo:

```
interacao_iniciada    usuario, paciente_id, pergunta
triagem               bloqueado, motivos
prontuario_consultado paciente_id, protocolo, campos_expostos, minimizado
exames_verificados    pendentes, nao_solicitados, conforme
alertas_avaliados     quantidade, codigos
protocolos_recuperados citacoes, scores
resposta_gerada       modo (llm|fallback), caracteres, trechos_no_prompt
saida_validada        motivos, severidade, evidencias
alerta_notificado     codigo, severidade, destinatario, setor, leito
interacao_concluida   latencia_ms
```

Com isso é possível reconstruir **por que** o assistente respondeu aquilo sem
reexecutar nada: quais trechos entraram no prompt, com que score, que alertas
estavam ativos e o que o guardrail alterou. Falhas também são auditadas
(`interacao_falhou`, com o tipo da exceção).

Leitura: `med-assist audit --trace <id>` ou `GET /audit?trace_id=<id>`.

## 5. Explicabilidade

Toda resposta carrega o bloco de fontes, construído a partir das evidências
coletadas pelos nós — não é um texto que o modelo escreve, é o registro do que
foi efetivamente usado:

```
**Fontes consultadas:**
- [protocolo] PROT-EME-002 v6.0, seção 3. Conduta (relevância 7.43)
- [prontuario] Prontuário PRT-2026-100137 (dados mínimos)
- [regra_clinica] SEPSE_HEMOCULTURA_APOS_ATB — PROT-EME-002
```

Se a resposta não tiver evidência, isso aparece: o guardrail de saída marca
`resposta_sem_fonte` e a lista volta vazia na API.

### Citação verdadeira, não só presente

Uma citação plausível e errada é pior que citação nenhuma — ela convida à
confiança. O modelo fine-tunado de 0,5B escreveu `PROT-EME-002 v6.0 (revisão
2027-02-11)` quando a revisão real é 2026-01-30: código certo, versão certa, data
inventada, em 3 de 3 respostas clínicas.

`check_citations()` confere cada citação contra os metadados reais e contra o que
foi de fato recuperado, e registra as divergências na auditoria
(`revisao_divergente`, `citacao_fora_do_contexto`, `protocolo_inexistente`).
Detectou 100% dos casos. O bloco de fontes gerado pelos nós continua ao lado da
resposta como referência autoritativa.

## 6. Alertas: por que não são gerados

Alerta é o tipo de saída em que alucinação custa caro nos dois sentidos — um
alerta inventado vira ruído e treina a equipe a ignorar; um alerta omitido vira
dano. Por isso cada uma das 11 regras é código sobre dado estruturado, com a
severidade e o protocolo que a fundamenta. Exemplos:

- antimicrobiano ativo com hemocultura pendente (inverte o bundle do PROT-EME-002);
- insulina em curso com potássio abaixo de 3,3 mEq/L (PROT-END-004: risco de arritmia fatal);
- paciente dentro da janela de trombólise, com o tempo restante calculado;
- medicação ativa da mesma classe de uma alergia registrada.

O LLM redige o caso; ele nunca decide se há alerta.

## 7. Degradação graciosa

Se o servidor da LLM cai, o assistente não quebra e não inventa: um redator
determinístico monta a resposta a partir do que os nós já apuraram — protocolo
recuperado, pendências, alertas. A resposta fica mais seca, sinalizada como
`modo contingência`, e continua citada e auditável. A demonstração e a suíte de
testes rodam em qualquer máquina, com ou sem modelo no ar.

## 8. O que este projeto **não** resolve

Sendo explícito sobre o escopo, porque prometer garantia demais em software
clínico é o próprio risco:

- **Não há autenticação real.** O campo `usuario` é registrado na auditoria, mas
  não é verificado. Em produção, entraria SSO institucional com o CRM validado.
- **A base de conhecimento é sintética.** Os protocolos são plausíveis e
  internamente consistentes, mas não foram revisados por corpo clínico.
- **O detector de PII é baseado em padrões.** Cobre os identificadores
  brasileiros estruturados e nomes com marcadores; texto livre com nome sem
  título e sem prenome conhecido pode escapar. Em produção, a régua seria um
  modelo de NER clínico somado a estes padrões — e a curadoria continuaria
  rejeitando o que escapasse.
- **Não há avaliação clínica dos desfechos.** Medimos comportamento do modelo
  (cita fonte, recusa ato médico, mantém aviso), não acurácia diagnóstica.
- **O modelo pequeno erra metadado de citação.** Está medido e mitigado por
  verificação cruzada, mas a mitigação detecta e registra — ela não impede que o
  texto da resposta traga a data errada. Quem lê deve usar o bloco de fontes.
