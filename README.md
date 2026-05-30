# MCDM Dashboard — Manual Detalhado de Utilização
### Documento-base para Work Instruction

> Ferramenta de **apoio à decisão multicritério** (Multi-Criteria Decision Making). Compara alternativas segundo vários critérios usando 4 modelos clássicos (AHP, TOPSIS, PROMETHEE II, COPRAS) e consolida tudo num Dashboard interactivo. Este manual descreve **cada ecrã, cada botão, cada gráfico e como o ler**.

---

## ÍNDICE

1. [O que a ferramenta faz](#1-o-que-a-ferramenta-faz)
2. [Instalação e arranque](#2-instalação-e-arranque)
3. [Visão geral da interface](#3-visão-geral-da-interface)
4. [Aba 📋 Dados — introduzir dados](#4-aba--dados--introduzir-dados)
5. [Aba 🔍 AHP — pesos e consistência](#5-aba--ahp--pesos-e-consistência)
6. [Aba 🎯 TOPSIS](#6-aba--topsis)
7. [Aba 📈 PROMETHEE II](#7-aba--promethee-ii)
8. [Aba 📊 COPRAS](#8-aba--copras)
9. [Aba 🏆 Dashboard — leitura completa](#9-aba--dashboard--leitura-completa)
10. [Como interagir com os gráficos](#10-como-interagir-com-os-gráficos)
11. [Análise de Sensibilidade — como ler](#11-análise-de-sensibilidade--como-ler)
12. [Glossário e fórmulas](#12-glossário-e-fórmulas)
13. [Resolução de problemas](#13-resolução-de-problemas)
14. [Exemplo completo passo-a-passo (caso MCG)](#14-exemplo-completo-passo-a-passo-caso-mcg)

---

## 1. O que a ferramenta faz

A ferramenta responde à pergunta: **"De um conjunto de alternativas (ex.: A1…A9), quais são as melhores, segundo vários critérios com pesos diferentes?"**

- **Alternativas** = as opções a avaliar (oportunidades de negócio, fornecedores, projectos…).
- **Critérios** = os eixos de avaliação (valor, probabilidade, esforço…). Cada critério é **benefício (max** — quanto mais melhor**)** ou **custo (min** — quanto menos melhor**)**.
- **Pesos** = a importância relativa de cada critério, calculada pelo método **AHP**.
- **4 modelos** classificam as alternativas, cada um com a sua lógica matemática.
- **Dashboard** consolida os 4 rankings e diz qual o Top-3 e quão robusta é a decisão.

---

## 2. Instalação e arranque

**Pré-requisitos:** Python 3.9 ou superior.

```bash
# 1. Instalar as bibliotecas necessárias
pip install streamlit pandas numpy plotly

# 2. Arrancar a aplicação (a partir da pasta onde está o app.py)
streamlit run app.py
```

A aplicação abre automaticamente no browser em `http://localhost:8501`. Se não abrir, copie esse endereço para o browser.

Para parar: `Ctrl + C` no terminal.

---

## 3. Visão geral da interface

A aplicação tem **6 separadores (abas)** no topo. **Não há barra lateral** — toda a configuração está na aba Dados.

| Ordem | Aba | Função |
|---|---|---|
| 1ª | **🏆 Dashboard** | Página principal. Vista consolidada dos 4 modelos. |
| 2ª | **📋 Dados** | Onde se introduzem os dados (2 pastes) e se calcula o AHP. |
| 3ª | **🔍 AHP** | Detalhe dos pesos, consistência (CR) e sugestão de correcção. |
| 4ª | **🎯 TOPSIS** | Detalhe do modelo TOPSIS. |
| 5ª | **📈 PROMETHEE II** | Detalhe do modelo PROMETHEE II. |
| 6ª | **📊 COPRAS** | Detalhe do modelo COPRAS. |

> **Fluxo recomendado:** 📋 Dados → 🔍 AHP (verificar consistência) → 🏆 Dashboard → abas dos modelos (se quiser ver o detalhe).

> **Importante:** se abrir o Dashboard ou uma aba de modelo **antes** de carregar dados, aparece a mensagem azul *"📥 Sem dados carregados. Vá à aba 📋 Dados…"*. É normal — basta carregar os dados primeiro.

---

## 4. Aba 📋 Dados — introduzir dados

Esta aba tem 4 secções. Os dados entram por **copiar-colar a partir do Excel** (sem upload de ficheiros).

### Secção 1 — Matriz AHP dos Critérios (par-a-par + MAX/MIN)

Aqui cola a **matriz de comparação par-a-par** dos critérios. É esta matriz que define:
- **quais são os critérios** (os códigos no cabeçalho),
- **o tipo de cada critério** (coluna MAX/MIN),
- **os pesos** (calculados a partir das comparações).

**Formato exacto a colar** (copiado directamente do Excel, separado por tabulações):

```
         C1_VP    C2_PF    C3_EE    C4_FE    C5_UD    C6_RC    MAX/MIN ?
C1_VP    1.0000   4.0000   8.0000   6.0000   9.0000   5.0000   max
C2_PF    0.2500   1.0000   8.0000   3.0000   9.0000   2.0000   max
C3_EE    0.1250   0.1250   1.0000   0.1111   1.0000   0.1111   min
C4_FE    0.1667   0.3333   9.0000   1.0000   9.0000   0.1667   max
C5_UD    0.1111   0.1111   1.0000   0.1111   1.0000   0.1111   max
C6_RC    0.2000   0.5000   9.0000   6.0000   9.0000   1.0000   max
```

**Regras:**
- A **primeira linha** = cabeçalho com os códigos dos critérios. Pode ter ou não uma célula vazia no canto superior esquerdo — a ferramenta aceita os dois casos.
- A **última coluna** = `MAX/MIN ?` indica o tipo: escreva `max` (benefício) ou `min` (custo).
- Cada **linha** = um critério comparado com todos os outros.
- **Diagonal = 1** (cada critério comparado consigo próprio).
- **Escala de Saaty:** quanto é que o critério da linha é mais importante que o da coluna?
  - `1` = igualmente importante
  - `3` = moderadamente mais importante
  - `5` = fortemente mais importante
  - `7` = muito fortemente mais importante
  - `9` = extremamente mais importante
  - Valores **recíprocos** (quando a linha é *menos* importante): `0.5` (=1/2), `0.33333` (=1/3), `0.2` (=1/5), `0.14286` (=1/7), `0.11111` (=1/9).
- Aceita até **5 casas decimais** (ex.: `0.11111`).
- Não precisa preencher a metade de baixo da matriz com rigor — a ferramenta **recalcula automaticamente os recíprocos** a partir da metade de cima (`a_ji = 1/a_ij`).

### Secção 2 — Tabela das Alternativas × Critérios

Aqui cola a tabela com os **valores reais** de cada alternativa em cada critério.

**Formato exacto a colar:**

```
Alternativa  C1_VP        C2_PF   C3_EE   C4_FE   C5_UD   C6_RC
A1           250,000,000  0.25    24      4       180     4
A2           300,000      0.35    8       5       60      5
A3           900,000      0.50    8       3       60      5
A4           650,000      0.50    8       3       90      3
...          ...          ...     ...     ...     ...     ...
```

**Regras:**
- A **primeira coluna** chama-se `Alternativa` e contém os nomes (A1, A2…).
- As **colunas seguintes** têm de usar **exactamente os mesmos códigos** de critérios que estão na matriz AHP (Secção 1). Maiúsculas/minúsculas contam.
- Aceita valores com **vírgula decimal** (`0,462`), **ponto** (`0.462`), **símbolo €**, **percentagem %**, e **espaços nos milhares** (`250 000 000`).
- Pode ter linhas de rótulos por cima (ex.: "Critérios Quantitativos / Qualitativos") — são **ignoradas** automaticamente.
- Suporta até **50 alternativas** e **15 critérios**.

### Secção 3 — Processar dados

Tem um único botão: **🚀 Processar pastes (parse + calcular AHP + carregar)**.

**Ao clicar, a ferramenta:**
1. Faz o *parsing* (interpretação) dos dois quadros colados.
2. Calcula automaticamente os **pesos AHP** (média geométrica das linhas da matriz).
3. Calcula o **CR** (índice de consistência).
4. Carrega os 4 modelos com esses pesos.
5. Mostra **balões** 🎈 e um **banner verde permanente** com o resumo:

```
✅ Dados processados em 14:32:07
   • 9 alternativas × 6 critérios
   • Tipos: 5 benefício (max) · 1 custo (min)
   • AHP — CR = 0.15099 ✗ (inconsistente)
   • Pesos AHP injectados em TOPSIS, PROMETHEE II e COPRAS
```

> Este banner **fica visível** mesmo quando muda de aba e volta — confirma sempre o que foi carregado e quando.

**Se houver erro** (ex.: códigos não coincidem entre os dois pastes), aparece uma mensagem vermelha a explicar exactamente o problema. Corrija e clique outra vez.

### Secção 4 — Variação para Análise de Sensibilidade

Um **slider de 5% a 50%** (passo de 5%). Define a percentagem de variação ± aplicada aos pesos em **todas** as análises de sensibilidade da aplicação (Dashboard e abas dos modelos). O valor por defeito é **±20%** (o valor pedido na maioria dos enunciados académicos).

---

## 5. Aba 🔍 AHP — pesos e consistência

Esta aba **não tem editores** — a matriz vem da aba Dados. Serve para **ver e validar** os pesos e, se necessário, **corrigir a consistência**.

### Passo 1 — Matriz de Comparação Par-a-Par

Mostra a matriz AHP que foi carregada (apenas leitura). Cores azuis mais escuras = valores maiores. Para alterar valores, volte à aba Dados, edite o paste e re-processe.

### Passo 2 — Vector de Pesos (média geométrica)

Tabela com o **peso de cada critério** (w_j), em valor decimal e em %. A fórmula usada:

> wᵢ = (∏ⱼ aᵢⱼ)^(1/n) ÷ Σₖ (∏ⱼ aₖⱼ)^(1/n)

Ou seja: faz-se a média geométrica de cada linha e normaliza-se para somar 1 (100%).

### Passo 3 — Verificação de Consistência

Mostra 4 métricas:
- **n** = número de critérios.
- **λ_max** (lambda máximo) = maior valor próprio aproximado da matriz.
- **CI** (Consistency Index) = (λ_max − n) ÷ (n − 1).
- **CR** (Consistency Ratio) = CI ÷ RI, onde RI é o Índice Aleatório de Saaty (depende de n).

**Regra de ouro:** **CR < 0.10** → matriz **consistente** (os julgamentos são coerentes). **CR ≥ 0.10** → **inconsistente** (há contradições nos julgamentos).

#### Se CR ≥ 0.10 — caixa "🔧 Sugestão para tornar a matriz consistente"

A ferramenta identifica **o par de critérios mais problemático** (aquele cujo valor introduzido mais se afasta do que os pesos implicam) e propõe um valor da escala de Saaty. Mostra 4 métricas:
- **Par problemático** — ex.: "C4_FE vs C6_RC".
- **Valor actual** — o que está na matriz (ex.: 0.16670).
- **Valor ideal** — o valor matematicamente coerente (ex.: 0.49899).
- **Sugerido (Saaty)** — o valor da escala mais próximo do ideal (ex.: 0.50000), com a diferença (Δ).

Por baixo, um texto interpreta em linguagem natural: *"Está a dizer que C4_FE vale 0.16670× C6_RC, mas os pesos calculados sugerem ~0.49899×. Aplicar substitui o valor por 0.50000 (e o recíproco simétrico = 2.00000)."*

#### Botão "✏️ Aplicar sugestão"

Ao clicar:
1. Substitui o valor problemático na matriz pelo valor sugerido (e o recíproco na célula simétrica).
2. **Recalcula** os pesos, λ_max, CI e CR.
3. **Sincroniza** a matriz com a aba Dados (o paste é reconstruído).
4. **Regista a alteração no histórico** (ver abaixo).
5. Mostra uma de duas mensagens:
   - ✅ *"Sugestão aplicada. CR = 0.097 < 0.10 — matriz agora consistente!"*
   - ✓ *"Sugestão aplicada. CR baixou de 0.151 → 0.103. Ainda ≥ 0.10 — clique de novo se quiser continuar a iterar."*

> **Iteração:** carregue no botão **repetidamente** até CR < 0.10. Cada clique corrige o par mais problemático naquele momento. Tipicamente bastam 1 a 3 cliques.

#### 📜 Histórico de Iterações AHP

É um painel expansível (clicar para abrir) que **regista cada correcção aplicada**. É a "caixa-preta" das suas alterações — essencial para documentar a auditoria da decisão. Cada linha mostra:

| Coluna | Significado |
|---|---|
| **iteração** | Número sequencial (1, 2, 3…). |
| **par** | Que par de critérios foi corrigido (ex.: "C4_FE vs C6_RC"). |
| **valor antigo** | O valor que lá estava antes. |
| **valor novo** | O valor de Saaty aplicado. |
| **CR antes** | O CR antes desta correcção. |
| **CR depois** | O CR depois desta correcção. |

**Como ler o histórico:** percorra as linhas de cima para baixo para ver a "viagem" da matriz desde inconsistente (CR alto) até consistente (CR < 0.10). A coluna **CR depois** deve ir diminuindo a cada iteração. Exemplo típico:

| iteração | par | valor antigo | valor novo | CR antes | CR depois |
|---|---|---|---|---|---|
| 1 | C4_FE vs C6_RC | 0.16670 | 0.50000 | 0.15099 | 0.10281 |
| 2 | C1_VP vs C3_EE | 8.00000 | 9.00000 | 0.10281 | 0.09719 |

Lê-se: *"foram precisas 2 correcções; a primeira baixou o CR de 0.151 para 0.103, a segunda de 0.103 para 0.097 (< 0.10, consistente)."*

Há também um botão **🗑️ Limpar histórico** para recomeçar o registo do zero (não altera a matriz, só apaga o log).

### Passo 4 — Ranking das Alternativas (usando pesos AHP)

Aplica os pesos AHP às alternativas e produz um ranking. Score = Σⱼ wⱼ · uⱼ(xᵢⱼ), onde uⱼ é o valor normalizado (0 a 1) do critério j, já com o sentido max/min aplicado. Mostra tabela ordenada + a melhor alternativa em destaque.

### 📊 Gráficos AHP

Dois gráficos lado-a-lado:
- **Esquerda — Pesos dos Critérios:** barras verticais com o peso % de cada critério. As 2 barras mais importantes ficam azul-escuro.
- **Direita — Ranking das Alternativas:** barras horizontais com o score AHP. Top-3 a azul-escuro, posições 4-6 a azul-claro, restantes a vermelho.

Por baixo, uma legenda confirma o CR (✅ ou ❌).

---

## 6. Aba 🎯 TOPSIS

**TOPSIS** = *Technique for Order of Preference by Similarity to Ideal Solution*. Lógica: a melhor alternativa é a que está **mais perto da solução ideal** e **mais longe da anti-ideal**. É **compensatório** (um valor muito bom num critério compensa um valor mau noutro).

A aba mostra todos os passos do cálculo:

- **Passo 1 & 2 — Matriz Normalizada (vectorial):** divide cada valor pela norma euclidiana da coluna: rᵢⱼ = xᵢⱼ ÷ √(Σᵢ xᵢⱼ²).
- **Passo 3 — Matriz Ponderada:** multiplica cada coluna pelo peso AHP: vᵢⱼ = wⱼ · rᵢⱼ.
- **Passo 4 — Soluções Ideal A⁺ e Anti-Ideal A⁻:** para critérios **max**, A⁺ = máximo da coluna, A⁻ = mínimo. Para critérios **min**, é o **contrário** (A⁺ = mínimo).
- **Passo 5 & 6 — Distâncias e CC*:** calcula a distância de cada alternativa ao ideal (D⁺) e ao anti-ideal (D⁻), e o **coeficiente de proximidade**: CC* = D⁻ ÷ (D⁺ + D⁻). **CC* varia entre 0 e 1; quanto maior, melhor.**

### 📊 Gráfico — Score de Proximidade Relativa (CC*)

Barras horizontais, uma por alternativa, ordenadas da melhor (topo) para a pior. Comprimento da barra = CC*.
- 🟦 **Azul escuro** = Top-3
- 🟦 **Azul claro** = posições 4 a 6
- 🟥 **Vermelho** = últimas posições

O valor exacto (5 decimais) aparece à direita de cada barra.

Por baixo do gráfico está a **Análise de Sensibilidade** (ver [secção 11](#11-análise-de-sensibilidade--como-ler)).

---

## 7. Aba 📈 PROMETHEE II

**PROMETHEE II** = *Preference Ranking Organisation Method*. Lógica: compara **todas as alternativas par-a-par** e calcula um "fluxo de preferência" líquido. É **não-compensatório** (ver nota importante abaixo).

### Selector "Função de preferência"

Três opções (botões de rádio):
- **Tipo I (Usual)** — o mais comum nos enunciados. Se a alternativa A é estritamente melhor que B num critério, conta `1`; senão, `0`. **Ignora a magnitude da diferença.**
- **Tipo V (Linear)** — a preferência cresce linearmente com a diferença até um limiar.
- **Tipo VI (Gaussiana)** — a preferência segue uma curva suave (gaussiana).

> **⚠️ Nota importante (compensatório vs não-compensatório):** com **Tipo I**, o PROMETHEE II só conta *quem ganha cada critério*, não *por quanto*. Por isso uma alternativa que domina massivamente um critério (ex.: valor 16× maior) pode **não** ficar em 1.º se perder em vários outros critérios. Isto é **matematicamente correcto** e explica porque o PROMETHEE II pode dar um Top-1 diferente do TOPSIS/AHP/COPRAS (que são compensatórios e premeiam a magnitude). **Não é um erro** — é a natureza do método.

### Secções de cálculo

- **Matriz π(a, b):** a preferência agregada de cada alternativa sobre cada outra. π(a,b) = Σⱼ wⱼ · Pⱼ(a,b).
- **Fluxos φ⁺, φ⁻ e φ líquido:**
  - φ⁺(a) = quanto `a` é preferida a todas as outras (fluxo positivo / "força").
  - φ⁻(a) = quanto as outras são preferidas a `a` (fluxo negativo / "fraqueza").
  - **φ(a) = φ⁺ − φ⁻** = fluxo líquido. **Varia entre −1 e +1; quanto maior, melhor.** A soma de todos os φ é sempre 0.

### 📊 Gráfico — Fluxo Líquido φ

Barras horizontais com o φ de cada alternativa, ordenadas da melhor para a pior. Há uma **linha vertical no zero**:
- 🟦 **Azul** (à direita do zero) = fluxo **positivo** → alternativa globalmente **preferida**.
- 🟥 **Vermelho** (à esquerda do zero) = fluxo **negativo** → alternativa globalmente **preterida**.

O valor aparece com sinal (ex.: `+0.50790`, `−0.39620`).

---

## 8. Aba 📊 COPRAS

**COPRAS** = *Complex Proportional Assessment*. Lógica: separa a contribuição dos **benefícios** (S⁺) e dos **custos** (S⁻) e calcula um grau de utilidade proporcional. É **compensatório**.

### Secções de cálculo

- **S⁺ (Benefícios) e S⁻ (Custos):** soma ponderada normalizada dos critérios de benefício (S⁺) e de custo (S⁻), por alternativa.
- **Q_i e U_i (%):**
  - Q_i = S⁺ᵢ + [ (S⁻ₘᵢₙ · ΣS⁻) ÷ (S⁻ᵢ · Σ(S⁻ₘᵢₙ/S⁻)) ] — combina benefícios e custos.
  - **U_i (%) = Q_i ÷ Q_max × 100** — grau de utilidade. A melhor alternativa tem **U = 100%**; as outras são uma percentagem dela.

### 📊 Gráfico — Grau de Utilidade U_i (%)

Barras horizontais com o U% de cada alternativa, ordenadas. Mesmo esquema de cores (azul-escuro Top-3, azul-claro 4-6, vermelho últimas). A barra de topo está sempre a 100%.

---

## 9. Aba 🏆 Dashboard — leitura completa

A página principal. Recalcula **tudo dinamicamente** a partir dos dados e pesos actuais — se mudar os dados ou os pesos AHP, o Dashboard actualiza-se sozinho. Está organizado em **4 linhas**.

### Barra de título (topo)

Faixa azul com o título e, à direita, o estado dos pesos: *"Pesos: AHP"* (ou *"Pesos iguais (fallback)"* se o AHP ainda não foi calculado) e a variação de sensibilidade activa.

> Se o AHP ainda não estiver calculado, aparece um aviso amarelo e o Dashboard usa **pesos iguais (1/n)** como recurso temporário.

### LINHA 1 — quatro colunas

#### Coluna 1 — 🔧 Filtros & Parâmetros
Três selectores que controlam o resto do Dashboard:
- **Modelo destacado** (botões de rádio: AHP / TOPSIS / PROMETHEE II / COPRAS) — escolhe qual o modelo cujos scores aparecem na tabela (coluna 2) e no gráfico de sensibilidade (coluna 4).
- **Critério (sensibilidade)** — escolhe o critério a analisar na coluna 4.
- **Alternativa (radar)** — escolhe a alternativa a destacar no radar (coluna 3).
- Mostra ainda o **Σ pesos** (deve ser 1.00000) e a variação ± activa.

#### Coluna 2 — 🏆 Ranking Consolidado das Alternativas
A tabela central (requisito D1). Cada linha é uma alternativa. Colunas:
- **Medalha** — 🥇🥈🥉 para o Top-3.
- **Alternativa** — o nome.
- **AHP / TOPSIS / PROMETHEE II / COPRAS** — a posição (rank) da alternativa em cada modelo. Cores: verde = bom rank (1.º), vermelho = mau rank.
- **Score {modelo destacado}** — o score do modelo escolhido na coluna 1 (verde = alto).
- **Posição Média** — a média das 4 posições (o "score final composto").
- **Ranking Final** — a ordenação consensual final (1 = melhor posição média).

> **Como ordenar:** clique no cabeçalho de qualquer coluna para ordenar por ela (requisito D1 — ordenação por qualquer coluna).

> **Como ler:** a alternativa com **menor Posição Média** é a vencedora consensual. Se uma alternativa for 1.º em 3 modelos mas 5.º num, a média "penaliza" ligeiramente, mas continua provavelmente no topo.

#### Coluna 3 — 🎯 Perfil Multicritério (radar)
Gráfico de radar/teia (requisito D2). Cada eixo é um critério; os valores estão normalizados de 0 (centro) a 1 (extremo). Mostra:
- O **Top-3** consensual em **ouro, prata e bronze**.
- A **alternativa escolhida** na coluna 1 (se não estiver no Top-3) com uma linha **tracejada roxa**.

> **Como ler:** quanto mais "preenchido" e "esticado" para fora estiver o polígono de uma alternativa, melhor o seu desempenho global. Picos num eixo = forte nesse critério; reentrâncias = fraco. Compare as formas para perceber **onde** cada alternativa ganha ou perde.

#### Coluna 4 — 🌪️ Sensibilidade do critério escolhido
Barras horizontais que mostram o score do **modelo destacado** quando o **peso do critério escolhido** sobe a percentagem do slider (ex.: +20%). As barras do Top-3 ficam azul-escuro. Serve para ver **como o ranking reage** se valorizar mais esse critério.

### LINHA 2 — 📈 Scores por Modelo (quatro gráficos)
Um gráfico de barras horizontais **por cada modelo** (requisito D3/D4 — visualização individual), lado a lado: AHP, TOPSIS, PROMETHEE II, COPRAS. Cada um mostra o score de cada alternativa segundo esse modelo, com o vencedor identificado no título (ex.: *"TOPSIS · 🥇 A1"*).
- Cores normais: azul-escuro Top-3, azul-claro 4-6, vermelho últimas.
- **Excepção PROMETHEE II:** azul = φ positivo, vermelho = φ negativo, com linha no zero.

> **Como ler:** compare os 4 gráficos. Se o mesmo nome estiver no topo dos 4 → decisão muito consensual. Se variar (ex.: A1 no topo de 3, A9 no topo do PROMETHEE) → ver a nota sobre compensatório/não-compensatório na [secção 7](#7-aba--promethee-ii).

### LINHA 3 — 🎯 Análise de Sensibilidade por Critério (caixas)
Uma **caixa por critério** (requisito D3/D5), que testa a **robustez do Top-1 consensual**. Para cada critério, varia-se o seu peso e recalculam-se os 4 modelos + a consolidação. Cada caixa mostra:
- **Nome do critério** (topo).
- **Estado a ±X%** (a percentagem do slider): 🟢 **Robusto** (Top-1 não muda) ou 🔴 **Sensível** (Top-1 muda). A borda da caixa fica verde ou vermelha.
- **Margem de segurança** (duas linhas):
  - Linha `+`: até quanto se pode **subir** o peso sem mudar o Top-1. Ex.: `+>50% estável` (mesmo subindo 50% nada muda) ou `+30% → A9` (a partir de +30% o Top-1 passa a A9).
  - Linha `−`: até quanto se pode **baixar** o peso. Ex.: `-55% → A9`.

> **Como ler:** caixas todas verdes com `>50% estável` = decisão **muito robusta** (o vencedor não muda mesmo com grandes variações de peso). Caixas vermelhas ou com margens pequenas (ex.: `+10% → A9`) = decisão **frágil** nesse critério — pequenas mudanças de peso mudam o vencedor.

### LINHA 4 — duas colunas

#### Coluna esquerda — 🎯 Recomendação & Notas
Painel automático (requisito D8) com:
- **Cartão azul** com o **Top-3 consensual** (🥇🥈🥉).
- **Barra de convergência** colorida: 🟢 ALTA (≥70%), 🟡 MODERADA (40-69%), 🔴 BAIXA (<40%). A convergência mede quantas vezes o Top-3 consensual aparece no Top-3 de cada modelo individual.
- **Robustez SA:** quantos critérios mantêm o Top-1 (ex.: "6/6 critérios mantêm A1 como Top-1").
- **Margem de segurança mínima:** o "elo mais fraco" — a menor variação de peso que mudaria o vencedor.
- **Veredicto final:** ✅ MUITO ROBUSTA / 🟡 MODERADAMENTE ROBUSTA / ⚠️ SENSÍVEL.

#### Coluna direita — 📊 Critérios e Pesos Activos
Tabela com cada critério, o seu tipo (max/min) e o peso AHP em valor e %, ordenada por importância. Confirma de onde vêm os pesos (AHP ou fallback).

---

## 10. Como interagir com os gráficos

Todos os gráficos são **interactivos** (tecnologia Plotly). Passe o rato sobre um gráfico e aparece uma pequena barra de ferramentas no canto superior direito.

| Acção | Como fazer |
|---|---|
| **Ver valor exacto** | Passar o rato (*hover*) sobre uma barra/ponto → aparece uma etiqueta com o valor. |
| **Ampliar (zoom)** | Arrastar o rato para desenhar um rectângulo sobre a zona de interesse. |
| **Repor zoom** | Duplo-clique no gráfico, ou botão "🏠 Reset axes" na barra. |
| **Mover (pan)** | Botão da "mãozinha" na barra, depois arrastar. |
| **Exportar imagem** | Botão da "📷 câmara" na barra → guarda um PNG do gráfico. |
| **Ligar/desligar séries** | (No radar) clicar numa entrada da legenda esconde/mostra essa alternativa — útil para comparar só 2 de cada vez. |

**Nas tabelas:** clique nos cabeçalhos das colunas para **ordenar**; as cores em gradiente (verde→vermelho) ajudam a ver rapidamente melhor/pior.

---

## 11. Análise de Sensibilidade — como ler

A análise de sensibilidade responde a: **"Se mudarmos um pouco os pesos, o resultado muda?"** Uma decisão é **robusta** se o vencedor se mantém mesmo variando os pesos.

A ferramenta tem **dois tipos** de análise de sensibilidade:

### A) Dentro de cada aba de modelo — Tabela de cenários (tornado)

No fim das abas AHP, TOPSIS, PROMETHEE II e COPRAS há uma tabela onde:
- **Cada linha** = uma alternativa.
- **Coluna "Base"** (cinzenta) = o ranking original.
- **As outras colunas** = o ranking quando cada critério varia +X% e −X% (isoladamente, com os outros renormalizados para somar 1).

**Cores das células:**
- 🟢 **Verde** = a alternativa **subiu** no ranking nesse cenário.
- 🔴 **Vermelho** = a alternativa **desceu** no ranking.
- ⚪ **Branco** = manteve a posição.
- ⬜ **Cinzento** (coluna Base) = a referência.

Por baixo há um **Resumo de Robustez por Alternativa**:
- **Inversões** = quantos cenários mudaram a posição daquela alternativa.
- **Robustez:** 🟢 ESTÁVEL (0 inversões), 🟡 MODERADA (1-2), 🔴 INSTÁVEL (3+).

> **Como ler:** uma alternativa com 0 inversões mantém sempre a posição → muito fiável. Muitas inversões → a posição depende muito dos pesos exactos.

### B) No Dashboard — Caixas de margem de segurança por critério

Já descrito na [Linha 3 do Dashboard](#linha-3--análise-de-sensibilidade-por-critério-caixas). A diferença é que aqui testa-se a robustez do **Top-1 consensual** (dos 4 modelos juntos), e mostra-se **até que percentagem** de variação o vencedor aguenta. É a leitura mais "executiva" da robustez.

> **Regra prática:** se a **margem de segurança mínima** (no painel de recomendação) for `>±50%`, a decisão é extremamente sólida. Se for pequena (ex.: `±10%`), comunique aos decisores que o resultado é sensível a esse critério.

---

## 12. Glossário e fórmulas

| Termo | Significado |
|---|---|
| **Alternativa** | Opção a avaliar (A1, A2…). |
| **Critério** | Eixo de avaliação (C1_VP, C2_PF…). |
| **Benefício (max)** | Critério em que mais é melhor (ex.: valor do contrato). |
| **Custo (min)** | Critério em que menos é melhor (ex.: esforço, risco). |
| **Peso (wⱼ)** | Importância relativa de um critério (somam 1). |
| **Escala de Saaty** | Escala 1-9 para comparações par-a-par no AHP. |
| **CR (Consistency Ratio)** | Mede a coerência dos julgamentos AHP. Deve ser < 0.10. |
| **λ_max** | Maior valor próprio aproximado da matriz AHP. |
| **Normalização** | Pôr todos os critérios na mesma escala (0-1) para serem comparáveis. |
| **CC\*** | Coeficiente de proximidade do TOPSIS (0-1, maior=melhor). |
| **φ (phi)** | Fluxo líquido do PROMETHEE II (−1 a +1, maior=melhor). |
| **U_i (%)** | Grau de utilidade do COPRAS (0-100%, maior=melhor). |
| **Posição Média** | Média das posições nos 4 modelos (o score final composto). |
| **Robustez** | Quão pouco o resultado muda quando se variam os pesos. |

**Fórmulas-chave dos 4 modelos:**

| Modelo | Fórmula do score |
|---|---|
| **AHP** | Sᵢ = Σⱼ wⱼ · uⱼ(xᵢⱼ) |
| **TOPSIS** | CC*ᵢ = D⁻ᵢ ÷ (D⁺ᵢ + D⁻ᵢ) |
| **PROMETHEE II** | φᵢ = φ⁺ᵢ − φ⁻ᵢ |
| **COPRAS** | Qᵢ = S⁺ᵢ + (S⁻ₘᵢₙ·ΣS⁻)÷(S⁻ᵢ·Σ(S⁻ₘᵢₙ/S⁻)) |

Todos respeitam o tipo (max/min) de cada critério.

---

## 13. Resolução de problemas

| Mensagem / Sintoma | Causa | Solução |
|---|---|---|
| *"📥 Sem dados carregados…"* | Ainda não processou dados | Vá à aba Dados, cole os 2 quadros, clique Processar |
| *"Não encontrei linha de cabeçalho…"* | O paste da matriz AHP não tem os códigos no cabeçalho | Inclua a linha de cabeçalho com os códigos dos critérios |
| *"Critério 'XX' não aparece na tabela de alternativas"* | Os códigos não coincidem entre os 2 pastes | Use **exactamente** os mesmos códigos (maiúsculas contam) |
| *"Códigos das linhas ≠ códigos das colunas"* | A matriz AHP não é quadrada ou usa códigos diferentes em linhas vs colunas | Confirme que linhas e colunas usam os mesmos códigos |
| **CR ≥ 0.10 (inconsistente)** | Julgamentos contraditórios | Aba AHP → botão "Aplicar sugestão" repetidamente até CR < 0.10 |
| **Top-1 nunca muda na sensibilidade** | Decisão genuinamente robusta | Normal. Ver a margem de segurança — se for `>50%`, é mesmo sólida |
| **PROMETHEE II dá vencedor diferente** | Tipo I é não-compensatório | Normal e correcto (ver nota na secção 7) |
| **Pesos aparecem como "iguais (1/n)"** | AHP ainda não calculado | Processe os dados na aba Dados (calcula o AHP automaticamente) |

---

## 14. Exemplo completo passo-a-passo (caso MCG)

Cenário: priorizar 9 oportunidades de negócio (A1-A9) segundo 6 critérios.

**Passo 1 — Abrir a aba 📋 Dados.**

**Passo 2 — Na Secção 1, colar a matriz AHP** (já vem um exemplo pré-preenchido que pode usar):

```
        C1_VP   C2_PF   C3_EE    C4_FE    C5_UD    C6_RC   MAX/MIN ?
C1_VP   1.0000  4.0000  8.0000   6.0000   9.0000   5.0000  max
C2_PF   0.2500  1.0000  8.0000   3.0000   9.0000   2.0000  max
C3_EE   0.1250  0.1250  1.0000   0.1111   1.0000   0.1111  min
C4_FE   0.1667  0.3333  9.0000   1.0000   9.0000   0.1667  max
C5_UD   0.1111  0.1111  1.0000   0.1111   1.0000   0.1111  max
C6_RC   0.2000  0.5000  9.0000   6.0000   9.0000   1.0000  max
```

**Passo 3 — Na Secção 2, colar as alternativas:**

```
Alternativa  C1_VP        C2_PF   C3_EE   C4_FE   C5_UD   C6_RC
A1           250000000    0.25    24      4       180     4
A2           300000       0.35    8       5       60      5
A3           900000       0.50    8       3       60      5
A4           650000       0.50    8       3       90      3
A5           5000000      0.40    24      4       30      3
A6           1350000      0.50    8       3       60      5
A7           10500000     0.40    16      3       180     4
A8           3450000      0.40    8       3       60      4
A9           15000000     0.60    24      4       300     3
```

**Passo 4 — Clicar 🚀 Processar pastes.** Aparecem balões e o banner verde: *"9 alternativas × 6 critérios · CR = 0.15099 ✗ (inconsistente)"*.

**Passo 5 — Ir à aba 🔍 AHP.** O CR = 0.151 está a vermelho. Clicar **"✏️ Aplicar sugestão"**:
- 1.º clique: corrige C4_FE vs C6_RC (0.1667 → 0.5), CR baixa para 0.103.
- 2.º clique: corrige C1_VP vs C3_EE (8 → 9), CR baixa para 0.097 ✅ consistente.
- Abrir o **📜 Histórico de Iterações** para confirmar as 2 correcções registadas.

**Passo 6 — Ir à aba 🏆 Dashboard.** Ler:
- **Ranking consolidado:** 🥇 A1, 🥈 A9, 🥉 A6.
- **Convergência:** mostra a percentagem de acordo entre modelos.
- **Caixas de sensibilidade:** todas verdes (A1 mantém-se Top-1 mesmo a ±50% em qualquer critério) → decisão muito robusta.
- **Nota:** no gráfico do PROMETHEE II o vencedor é A9 (não A1) — é o comportamento não-compensatório, correcto.

**Passo 7 — Explorar os modelos** (abas TOPSIS, PROMETHEE II, COPRAS) para ver o detalhe de cada cálculo e as tabelas de sensibilidade individuais.

**Resultado final:** A oportunidade **A1** é a recomendada (vence em 3 dos 4 modelos e é Top-1 consensual robusto), seguida de A9 e A6.

---

*MCDM Dashboard · Manual Detalhado · ISEL · MEGI · Modelos de Decisão 2025/2026*
