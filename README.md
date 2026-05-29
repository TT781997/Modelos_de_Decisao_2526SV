# MCDM Dashboard — Guia Rápido

Ferramenta de **apoio à decisão multicritério** (MCDM). A *homepage* é o **🏆 Dashboard** — todo o resto (Dados, AHP, TOPSIS, PROMETHEE II, COPRAS) é detalhe consultável.

---

## 1. Como instalar e correr

```bash
pip install streamlit pandas numpy plotly
streamlit run app.py
```

Abre automaticamente no browser em `http://localhost:8501`.

---

## 2. Como alimentar a ferramenta — 3 passos

Toda a entrada de dados está na aba **📋 Dados**.

### Passo 1 — Colar a matriz AHP dos critérios

Copie do Excel a sua **matriz de comparação par-a-par** dos critérios, **com cabeçalho** e uma coluna **MAX/MIN** no fim. Exemplo:

```
         C1_VP    C2_PF    C3_EE    C4_FE    C5_UD    C6_RC    MAX/MIN ?
C1_VP    1.0000   4.0000   8.0000   6.0000   9.0000   5.0000   max
C2_PF    0.2500   1.0000   8.0000   3.0000   9.0000   2.0000   max
C3_EE    0.1250   0.1250   1.0000   0.1111   1.0000   0.1111   min
C4_FE    0.1667   0.3333   9.0000   1.0000   9.0000   0.1667   max
C5_UD    0.1111   0.1111   1.0000   0.1111   1.0000   0.1111   max
C6_RC    0.2000   0.5000   9.0000   6.0000   9.0000   1.0000   max
```

- Escala Saaty: **1**=igual · **3**=moderada · **5**=forte · **7**=muito forte · **9**=extrema. Recíprocos: **0.11111** (=1/9), **0.1667** (=1/6) etc.
- Coluna **MAX/MIN**: `max` = benefício (mais é melhor), `min` = custo (menos é melhor).
- Aceita até **5 casas decimais**.
- Cole na 1ª caixa de texto da aba 📋 Dados.

### Passo 2 — Colar a tabela das alternativas

Copie do Excel a tabela `Alternativa × Critério`. Os códigos de critério têm de ser **exactamente os mesmos** da matriz AHP. Exemplo:

```
Alternativa  C1_VP        C2_PF   C3_EE   C4_FE   C5_UD   C6_RC
A1           250,000,000  0.25    24      4       180     4
A2           300,000      0.35    8       5       60      5
A3           900,000      0.50    8       3       60      5
...          ...          ...     ...     ...     ...     ...
```

- Aceita formatos com `€`, `%`, vírgulas decimais (`0,462`) e espaços nos milhares (`250 000 000`).
- Aceita rótulos como "Critérios Quantitativos / Qualitativos" acima do cabeçalho — são ignorados.
- Cole na 2ª caixa de texto da aba 📋 Dados.

### Passo 3 — Premir **🚀 Processar pastes**

A ferramenta:
1. Faz parsing dos dois quadros
2. Calcula automaticamente os **pesos AHP** (média geométrica)
3. Calcula o **CR** (índice de consistência); se ≥ 0.10 mostra a sugestão de correcção na aba 🔍 AHP
4. Carrega TOPSIS, PROMETHEE II e COPRAS com os pesos AHP

Pronto. Vá ao **🏆 Dashboard**.

---

## 3. Como ler o Dashboard

Layout em 4 zonas, todas **recalculadas dinamicamente** sempre que os dados ou os pesos mudam.

### Zona 1 — Topo (4 colunas)

| Coluna | O que faz |
|---|---|
| **🔧 Filtros** | Escolha o **modelo destacado** (AHP/TOPSIS/PROMETHEE/COPRAS) → propaga para a tabela e gráfico de sensibilidade. Escolha o **critério** para a análise de sensibilidade. Escolha uma **alternativa** para destacar no radar. |
| **🏆 Ranking Consolidado** | Tabela com rank de cada alternativa em cada modelo, posição média, **Ranking Final** consensual, e Score do modelo destacado. Cores: verde = melhor, vermelho = pior. |
| **🎯 Perfil Multicritério** | Gráfico radar com o Top-3 consensual em ouro/prata/bronze + alternativa que escolheu (linha tracejada roxa). Eixos = critérios. Eixo de 0 a 1 (normalizado). |
| **🌪️ Sensibilidade do critério** | O score do modelo destacado quando o peso do critério escolhido sobe X% (X = slider na 📋 Dados). |

### Zona 2 — Gráficos por Modelo (4 colunas)

Um gráfico de barras horizontais por modelo. Para cada um:
- **TOPSIS** mostra CC* (proximidade)
- **PROMETHEE II** mostra φ líquido (azul positivo, vermelho negativo, linha vertical no zero)
- **AHP** mostra Score AHP
- **COPRAS** mostra Q_i

Cores: azul-escuro = Top-3, azul-claro = posições 4-6, vermelho = piores.

### Zona 3 — Sensibilidade por Critério (6 caixinhas)

Para cada critério, mostra:
- **Linha 1** (azul): nome do critério
- **Linha 2**: estado a ±X% do slider (verde "Robusto" ou vermelho "Sensível")
- **Linha 3-4 — Margem de Segurança**: até **quanto** se pode variar o peso isoladamente sem mudar o Top-1 consensual. Ex.: `+>50% estável` = peso pode subir 50% e o Top-1 não muda; `-55% → A9` = se baixar 55%, o Top-1 passa a A9.

### Zona 4 — Recomendação e Pesos

- **Cartão azul**: Top-3 final (🥇🥈🥉)
- **Barra colorida**: convergência inter-modelo (🟢 ALTA / 🟡 MODERADA / 🔴 BAIXA)
- **Margem de Segurança Mínima**: o "elo mais fraco" da decisão
- **Tabela à direita**: critérios + tipo + pesos AHP em %

---

## 4. Como interagir com os gráficos (Plotly)

Todos os gráficos são Plotly e suportam:

- **🖱️ Hover** sobre uma barra/ponto → tooltip com valor exacto
- **🔍 Zoom** → arrastar uma caixa sobre a área de interesse; duplo-click para reset
- **📷 Camera** (canto superior direito do gráfico) → exporta PNG
- **🪟 Pan** (ferramenta de mão) → mover o gráfico
- **👁️ Legenda** → clicar num item na legenda esconde/mostra essa série (útil no radar)

No radar:
- Clique numa entrada da legenda para esconder/mostrar (compare apenas 2 alternativas, por exemplo)

Nas tabelas:
- Clique nos cabeçalhos para **ordenar**
- Cores degradadas mostram melhor/pior por coluna

---

## 5. Análise de Sensibilidade — slider global

Na aba **📋 Dados**, secção 4, há um slider **Variação ± nos pesos (%)** entre 5% e 50%. Define o nível de perturbação que vai ser aplicado:
- Na coluna 4 do Dashboard (efeito de variar o critério escolhido)
- Nas 6 caixinhas (estado a ±X% — linha verde/vermelha)
- Nas abas dos modelos individuais (Tornado interno)

A **margem de segurança** nas caixinhas é independente do slider — varre sempre até ±50%.

---

## 6. Fluxo típico de uso

1. **Aba 📋 Dados** → cole a matriz AHP + tabela de alternativas → prima Processar
2. **Aba 🔍 AHP** → confirme **CR < 0.10** (se não, aplique a sugestão na sua matriz original e cole novamente)
3. **Aba 🏆 Dashboard** → leia o Top-3, a convergência e a margem de segurança
4. **Abas 🎯 TOPSIS / 📈 PROMETHEE II / 📊 COPRAS** → consulte detalhe de cada modelo (passos, normalizações, ranking, tornado de sensibilidade individual)

---

## 7. Resolução de problemas

| Sintoma | Causa | Solução |
|---|---|---|
| "Não encontrei linha de cabeçalho..." | O paste da matriz AHP não tem códigos de critério no cabeçalho | Cole com a linha do cabeçalho incluída |
| "Critério 'XX' não aparece na tabela de alternativas" | Códigos não coincidem entre os 2 pastes | Use **exactamente** os mesmos códigos (case-sensitive) |
| "Códigos das linhas ≠ códigos das colunas" | A matriz AHP não é quadrada ou tem códigos diferentes em linhas vs colunas | Verifique que as linhas e as colunas da matriz AHP usam os mesmos códigos |
| Top-1 nunca muda em sensibilidade | Decisão genuinamente robusta — o Top-1 domina muito o critério mais pesado | Veja a "Margem de Segurança" — se for `>50%` em todos os critérios, é mesmo robusta |
| CR ≥ 0.10 (inconsistente) | Os seus julgamentos são contraditórios | Vá à aba 🔍 AHP — mostra **qual par** está inconsistente e que **valor Saaty** colocar |

---

## 8. Limites técnicos

- Até **50 alternativas × 15 critérios**
- Precisão de **5 casas decimais** em todos os inputs
- AHP CR usa tabela RI até n=15 (Saaty)
- Os pesos são sempre **normalizados** para Σ=1 antes de qualquer cálculo

---

## 9. Os 4 modelos — em uma linha cada

| Modelo | Conceito | Fórmula final |
|---|---|---|
| **AHP** (Saaty 1980) | Comparação par-a-par → vector de pesos validado por CR | Score = Σ wⱼ · uⱼ(xᵢⱼ) |
| **TOPSIS** (Hwang & Yoon 1981) | Distância à solução ideal (A⁺) e anti-ideal (A⁻) | CC* = D⁻ / (D⁺ + D⁻) |
| **PROMETHEE II** (Brans 1985) | Fluxos de preferência par-a-par | φ = φ⁺ − φ⁻ |
| **COPRAS** (Zavadskas 1996) | Proporção benefícios / custos | Q = S⁺ + (S⁻min · Σ S⁻) / (S⁻ · Σ 1/S⁻) |

Todos respeitam o **tipo (max/min)** definido na coluna MAX/MIN.

---

*MCDM Dashboard · Ferramenta de Apoio à Decisão · ISEL · MEGI · Modelos de Decisão 2025/2026*