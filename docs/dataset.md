# Dataset: preparação, anonimização e curadoria

## De onde vêm os dados

| Fonte | O que é | Volume |
|---|---|---|
| Protocolos internos | 8 protocolos clínicos sintéticos do Hospital Santa Helena, com versão, revisão e responsável técnico | 8 documentos, 40 seções indexadas |
| Perguntas frequentes | dúvidas reais de plantão, ancoradas em protocolo, incluindo versões contextualizadas em pacientes | 38 itens |
| Modelos de documento | laudo, receituário e descrição de procedimento preenchidos | 72 documentos |
| Prontuários | 24 pacientes sintéticos com cadastro, exames, medicações, escores e sinais vitais | 24 pacientes |
| MedQuAD | perguntas e respostas de fontes oficiais dos NIH (conhecimento médico geral) | ~200 pares amostrados |

Tudo o que é interno é **autoral e fictício**, gerado por
[`data/synth.py`](../src/med_assist/data/synth.py) com semente fixa. Nenhum dado
real de paciente foi usado em nenhuma etapa.

O MedQuAD entra com papel delimitado: ensinar o registro de resposta clínica
geral (definição, sintomas, tratamento, exames). Os pares são mantidos **em
inglês**, com instrução de sistema própria — tradução automática em domínio
clínico introduz erro terminológico justamente onde ele é mais caro. Baixe com
`scripts/fetch_medquad.sh`; sem ele, o dataset é montado só com o corpus interno.

## Por que o corpus bruto tem PII

`data/raw/` é gerado **com** nomes, CPFs, CNS, RGs, prontuários, telefones,
e-mails e endereços. Sem PII na entrada não há como demonstrar — nem testar — o
pipeline de anonimização. Todos os identificadores são fabricados.

## Anonimização: pseudonimização determinística

Cada valor identificado vira um token derivado por HMAC-SHA256 com sal:

```
Paciente Maria Aparecida Souza, CPF 123.456.789-00, tel (11) 98765-4321
        ↓
Paciente [NOME_B9FC], CPF [CPF_19FF], tel [TELEFONE_6940]
```

O mesmo valor produz sempre o mesmo token. Isso preserva a **coerência
referencial** — duas menções ao mesmo médico continuam apontando para a mesma
pessoa, e o modelo aprende que um documento fala de um único paciente — sem
manter o dado original e sem reversibilidade prática sem o sal.

Tipos reconhecidos: CPF, CNS, RG, número de prontuário, CRM, e-mail, telefone,
CEP, endereço, data, nome próprio (com ou sem título clínico) e idade acima de 89
anos (generalizada, seguindo a lógica de *safe harbor*: idades muito altas são
reidentificáveis).

Dois cuidados no detector, ambos cobertos por teste:

- **Não engolir clínica.** `30 mL/kg`, `0,1 U/kg/h`, `PAM 65 mmHg` e
  `lactato 2 mmol/L` não são identificadores — o texto sai intacto.
- **Preservar o título.** `Dra. Helena Prado` vira `Dra. [NOME_5900]`, não
  `[NOME_5900]`: o registro clínico continua legível.

### Cadastral vs. operacional

Nem todo identificador tem o mesmo peso. O guardrail de saída distingue:

- **Cadastral** (CPF, CNS, RG, nome, telefone, e-mail, endereço, CEP): bloqueia a
  resposta. Nada disso é necessário para decidir conduta.
- **Operacional** (número do prontuário, CRM, datas assistenciais): circula na
  resposta ao profissional, com acesso registrado na trilha. Sem o número do
  prontuário, o médico não sabe de qual caso o assistente está falando.

O dataset de treino usa a régua mais rígida: **nenhum** dos dois grupos entra.

## Curadoria

A anonimização remove o que não pode entrar; a curadoria remove o que não
*deveria*. Filtros aplicados, na ordem em que são baratos de avaliar:

| Filtro | Critério |
|---|---|
| Tamanho | pergunta ≥ 12 caracteres; resposta entre 60 e 2000 |
| PII residual | qualquer identificador sobrevivente descarta o exemplo |
| Citação de fonte | exemplo interno precisa referenciar `PROT-XXX-000` |
| Duplicata exata | hash da pergunta normalizada (sem acento, sem pontuação) |
| Quase-duplicata | Jaccard ≥ 0,85 sobre shingles de 5 palavras |

O filtro de citação é o que ensina explicabilidade **no próprio treino**: um
exemplo cuja resposta não cita protocolo não vira dado de treino, então o modelo
nunca vê uma resposta interna sem fonte.

## Composição final

| Categoria | Exemplos | O que ensina |
|---|---|---|
| `protocolo_qa` | 120 | conteúdo dos protocolos, seção a seção, sempre com fonte |
| `conhecimento_geral` | 195 | registro de resposta clínica geral (MedQuAD, inglês) |
| `consulta_contextualizada` | 43 | responder ao **formato real do prompt de produção** (caminho com prontuário) |
| `consulta_geral` | 20 | responder ao prompt de produção **sem prontuário** |
| `documento` | 24 | formato dos modelos institucionais de laudo, receita e procedimento |
| `faq` | 20 | dúvidas de plantão, com a resposta ancorada em protocolo |
| `faq_contextualizada` | 12 | mesma dúvida aplicada a um caso concreto |
| `limite_de_atuacao` | 15 | recusa de ato médico, com a alternativa oferecida |

Split estratificado por categoria: **382 exemplos de treino, 67 de validação** —
a validação tem a mesma composição do treino, e nenhuma pergunta aparece nos dois
lados (verificado por teste).

### As duas categorias que só existem por causa de erros

A primeira versão do dataset era inteira de "pergunta curta → resposta". Treinado
assim, o modelo funcionou bem em perguntas soltas e **falhou no fluxo real**:
diante do prompt de produção — que tem quatro seções (contexto do paciente,
pendências, alertas, trechos de protocolo) — ele *ecoou as seções de volta* em vez
de responder.

O diagnóstico é o mesmo princípio que vale para o chat template, aplicado um nível
acima: **o formato de treino tem de ser o formato de inferência**. A correção foi
[`data/contextualized.py`](../src/med_assist/data/contextualized.py), que monta
exemplos com o **mesmo `ASSISTANT_PROMPT` do runtime**, preenchido com casos reais
da base estruturada, e uma resposta de referência redigida por template a partir
do que já foi apurado (alertas, pendências, conduta).

**A segunda metade da lição veio depois.** Com os exemplos contextualizados, o
modelo aprendeu a abrir toda resposta com "Resumo do caso" — e, numa pergunta
geral, **sem paciente nenhum**, inventou um: *"Paciente de 60 anos, homem"*, com
código de protocolo fabricado junto. Faltava cobrir o **caminho sem prontuário**
do grafo, que existe no runtime e não existia no treino. Daí `consulta_geral`, com
o prompt que o nó `recuperar_protocolo` monta quando não há paciente.

Duas honestidades sobre esses exemplos:

- A resposta de referência é **template, não redação humana** — supervisão fraca.
  O que ela ensina é estrutura e ancoragem: resumir o caso, começar pelos alertas,
  listar pendências, citar a fonte, fechar com o aviso de validação.
- O módulo depende de `knowledge` e `assistant`, invertendo a direção usual da
  arquitetura. É proposital: se o prompt do runtime mudar, o dataset acompanha.
  Um gerador que reimplementasse o template voltaria a divergir na primeira
  alteração — que é exatamente o defeito que ele existe para corrigir.

### Deduplicação com chave explícita

Esses exemplos quebraram o filtro de quase-duplicata: 25 dos 48 foram descartados,
porque a estrutura da pergunta é intencionalmente idêntica e a variação está no
conteúdo clínico. O filtro por shingles pressupõe pergunta em texto livre.

A solução foi permitir que o exemplo declare sua própria `dedup_key`
(`contextualizada:{paciente}:{foco}`): com ela, vale a igualdade exata da chave e
o teste de shingles é dispensado. Sobraram 43 — os 5 restantes eram repetição
verdadeira (mesmo paciente e mesmo foco sorteados).

Relatórios reproduzíveis em [`results/eval/`](../results/eval/):
`relatorio_anonimizacao.json`, `relatorio_curadoria.json`, `resumo_dataset.json`.

## Reproduzir

```bash
med-assist data synth    # regenera data/raw/ (determinístico)
med-assist data build    # anonimiza, cura, faz o split e escreve os relatórios
```
