# MCDM Dashboard — Sistema de Apoio à Decisão Multicritério

Aplicação web interactiva em **Streamlit** que implementa **13 modelos de decisão multicritério** e produz um dashboard consolidado de ranking. Desenvolvida para a unidade curricular **Modelos de Decisão** do Mestrado em Engenharia e Gestão Industrial (MEGI) do ISEL, com o caso de estudo MCG (priorização de oportunidades comerciais no sector Automotive).

A app é totalmente dinâmica: aceita qualquer Excel com N alternativas × M critérios e recalcula tudo sem editar código.

---

## Índice

1. [Pré-requisitos](#pré-requisitos)
2. [Instalação e execução](#instalação-e-execução)
3. [Formato do ficheiro Excel](#formato-do-ficheiro-excel)
4. [Como usar a webapp](#como-usar-a-webapp)
5. [Modelos implementados](#modelos-implementados)
6. [Estrutura das 14 abas](#estrutura-das-14-abas)
7. [Deploy online (live)](#deploy-online-live)
8. [Troubleshooting](#troubleshooting)
9. [Notas metodológicas](#notas-metodológicas)

---

## Pré-requisitos

- **Python 3.9 ou superior** (testado em 3.10 / 3.11 / 3.12)
- **pip** actualizado (`python -m pip install --upgrade pip`)
- Browser moderno (Chrome, Firefox, Edge, Safari)

Verifica a versão do Python com:

```bash
python --version
```

---

## Instalação e execução

### 1. Clonar/descarregar o projecto

Coloca `app.py`, `requirements.txt` e `README.md` na mesma pasta. Estrutura mínima:

```
mcdm-dashboard/
├── app.py
├── requirements.txt
└── README.md
```

### 2. (Recomendado) Criar ambiente virtual

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

### 4. Correr a aplicação

```bash
streamlit run app.py
```

A app abre automaticamente em **http://localhost:8501**. Se não abrir, abre manualmente esse URL no browser.

Para parar a app: `Ctrl + C` no terminal.

---

## Formato do ficheiro Excel

A app exige um ficheiro `.xlsx` com pelo menos uma folha chamada `Dados`. Opcionalmente pode incluir uma folha `Pesos`.

### Folha `Dados` (obrigatória)

| Alternativa | C1_VP       | C2_PF | C3_EE | C4_FE | C5_UD | C6_RC |
|:------------|------------:|------:|------:|------:|------:|------:|
| A1          | 250 000 000 | 0,25  | 24    | 4     | 180   | 4     |
| A2          | 300 000     | 0,35  | 8     | 5     | 60    | 5     |
| A3          | 900 000     | 0,50  | 8     | 3     | 60    | 5     |
| ...         | ...         | ...   | ...   | ...   | ...   | ...   |

**Regras:**

- A **primeira coluna** identifica a alternativa (pode ser texto: `A1`, `Oportunidade Be`, etc.).
- As **restantes colunas** são os critérios e têm de ser **numéricas**.
- Não há limite de linhas (alternativas) nem de colunas (critérios). A app adapta-se dinamicamente.
- Linhas vazias na primeira coluna são ignoradas.
- O nome das colunas dos critérios é livre (`C1_VP`, `Valor`, `revenue_2025`, etc.) — só a posição importa.

### Folha `Pesos` (opcional)

Vector de pesos na mesma ordem das colunas de critérios da folha `Dados`. Aceita linha ou coluna:

| 0,4615 |
|-------:|
| 0,1987 |
| 0,0230 |
| 0,0972 |
| 0,0217 |
| 0,1979 |

ou em formato horizontal:

| 0,4615 | 0,1987 | 0,0230 | 0,0972 | 0,0217 | 0,1979 |

Os pesos são automaticamente normalizados (somatório = 1). Se a folha `Pesos` não existir, são aplicados pesos uniformes (1/M) e podes editá-los na sidebar.

### Modo de demonstração

Se não tiveres ficheiro à mão, activa na sidebar **"Usar dados de demonstração MCG"** — carrega os 9 alts × 6 critérios do caso de estudo com pesos AHP pré-calculados.

---

## Como usar a webapp

### Sidebar (painel esquerdo)

1. **Upload** — carrega o `.xlsx`.
2. **Modo demo** — checkbox para activar dados MCG.
3. **Parâmetros dos modelos:**
   - ELECTRE: limiares de concordância `c` (default 0,65) e discordância `d` (default 0,35).
   - PROMETHEE: função de preferência (`usual`, `linear` ou `gaussian`).
   - VIKOR: peso da estratégia `v` (default 0,5).
   - Sensibilidade TOPSIS: variação ±% nos pesos (default 20%).
4. **Sentido dos critérios** — `max` (benefício) ou `min` (custo) para cada critério. A app aplica heurística para detectar critérios de custo (ex.: contém "EE", "custo", "esforço") mas podes corrigir.
5. **Pesos editáveis** — ajusta manualmente; normalização automática.

### Área principal (14 abas)

Cada aba mostra:
- Tabelas intermédias (matrizes normalizadas, ponderadas, distâncias, etc.).
- Score final e ranking por alternativa.
- Visualização (gráfico de barras, heatmap, radar, etc.).

A última aba — **Dashboard Consolidado** — cruza os rankings de todos os modelos e gera uma recomendação agregada.

---

## Modelos implementados

### Modelos clássicos (crisp)

#### 1. AHP — Analytic Hierarchy Process
Determina pesos dos critérios através de uma matriz de comparação par-a-par usando a escala de Saaty (1-9). Calcula o autovector principal e o **Consistency Ratio (CR)**: se CR < 0,10 os julgamentos são consistentes. Útil para extrair preferências subjectivas de decisores.

#### 2. ANP — Analytic Network Process
Extensão do AHP que captura dependências entre critérios (rede em vez de hierarquia). Esta implementação usa uma **aproximação data-driven**: estima a influência inter-critério via correlações nas alternativas observadas, constrói a supermatriz e eleva-a a potências até convergir. Os pesos AHP são modulados pela matriz limite.

#### 3. TOPSIS — Technique for Order Preference by Similarity to Ideal Solution
Cada alternativa é comparada com a solução ideal (A+) e anti-ideal (A−) num espaço euclidiano ponderado. O **coeficiente Ci\*** mede proximidade ao ideal (0 = pior, 1 = melhor). Inclui análise de sensibilidade ±% nos pesos com error bars.

#### 4. ELECTRE I — ELimination Et Choix Traduisant la REalité
Constrói relações de **sobreclassificação**: a alternativa A "sobreclassifica" B se a concordância (peso dos critérios em que A ≥ B) excede um limiar `c` e a discordância (maior diferença em critérios desfavoráveis a A) está abaixo de `d`. Devolve um **kernel** (subconjunto de não-dominadas). Não produz ranking total — ideal para identificar candidatas robustas.

#### 5. PROMETHEE II — Preference Ranking Organization METHod for Enrichment Evaluations
Compara pares de alternativas por critério através de **funções de preferência** (usual, linear, gaussiana). Agrega num **fluxo líquido φ = φ⁺ − φ⁻** que ordena todas as alternativas. Mais informativo que ELECTRE I quando se quer ranking completo.

#### 6. VIKOR — VIseKriterijumska Optimizacija I Kompromisno Resenje
Procura a solução de compromisso minimizando o arrependimento. Calcula:
- **S**: distância à solução ideal (utilidade de grupo)
- **R**: pior desvio individual (arrependimento)
- **Q = v·S + (1−v)·R**: índice de compromisso

O parâmetro `v` (default 0,5) regula a estratégia: `v=1` privilegia utilidade, `v=0` privilegia equidade.

#### 7. MAUT — Multi-Attribute Utility Theory
Modelo de **utilidade linear aditiva**: normaliza valores via min-max (invertendo critérios de custo) e calcula U = Σ wᵢ · uᵢ(x). Simples, transparente e estável.

#### 8. COPRAS — COmplex PRoportional ASsessment
Separa critérios em benefícios (S⁺) e custos (S⁻), calcula utilidade relativa Q usando a razão entre os dois, e normaliza para uma escala N% (100 = melhor). Boa interpretação económica quando há trade-off claro entre receita e custo.

#### 9. DEMATEL — Decision Making Trial and Evaluation Laboratory
Modela influências causais entre critérios. Constrói a matriz de relação total **T = X(I−X)⁻¹** e produz:
- **D + R** (prominência): importância global do critério
- **D − R** (relação): se positivo → critério causal, se negativo → critério efeito

Aqui usa-se proxy data-driven (correlações) na ausência de matriz de influência directa. A prominência modula os pesos para o ranking final.

### Modelos Fuzzy (incerteza)

#### 10. Fuzzy AHP
Substitui os julgamentos crisp por **números triangulares fuzzy** (l, m, u) — gerados a partir dos pesos crisp com spread ±20% como aproximação. A defuzzificação usa o **método do centro de área**.

#### 11. Fuzzy TOPSIS
TOPSIS com valores tratados como números triangulares (val·(1−s), val, val·(1+s)). Calcula soluções ideais fuzzy (FPIS e FNIS) e distâncias pelo **método do vértice**. O slider de spread permite explorar diferentes níveis de incerteza.

#### 12. Fuzzy ANP
Combina pesos fuzzy do Fuzzy AHP com o ajuste por supermatriz de influência do ANP. Útil quando há simultaneamente dependência entre critérios e incerteza nas avaliações.

### Modelo agregado

#### 13. Ranking Consolidado (Borda invertido)
No dashboard final, todas as posições por modelo são agregadas por **média de posições** (método de Borda invertido). O modelo declara "convergência" Top-3 quando ≥ 60% dos modelos colocam a alternativa nas 3 primeiras posições.

---

## Estrutura das 14 abas

| # | Aba | Conteúdo principal |
|---|-----|--------------------|
| 1 | **Visão Geral** | Matriz de decisão, pesos, sentidos, descritivas, heatmap normalizado |
| 2 | **AHP** | Matriz Saaty editável, λ_max, CI, CR, pesos AHP, ranking, botão para aplicar pesos globalmente |
| 3 | **ANP** | Pesos ajustados, matriz limite, ranking |
| 4 | **TOPSIS** | Matrizes normalizada/ponderada, A+/A−, D+/D−, Ci*, ranking, sensibilidade com error bars |
| 5 | **ELECTRE** | Matrizes C e D, sobreclassificação, kernel, mapa de estabilidade c×d |
| 6 | **PROMETHEE** | Matriz π, φ+/φ−/φ líquido, comparação entre 3 funções de preferência |
| 7 | **VIKOR** | S, R, Q com slider v |
| 8 | **MAUT** | Utilidades parciais + utilidade global |
| 9 | **COPRAS** | S⁺, S⁻, Q, N% |
| 10 | **DEMATEL** | Matriz T, prominência, diagrama causa-efeito |
| 11 | **Fuzzy AHP** | Pesos fuzzy triangulares + defuzzificação |
| 12 | **Fuzzy TOPSIS** | d+/d−, CC, slider de spread |
| 13 | **Fuzzy ANP** | Pesos fuzzy + ajuste supermatriz |
| 14 | **Dashboard** | Tabela cruzada, heatmap de posições, radar Top-3, painel de recomendação, export Excel |

O **export Excel** na aba 14 gera um ficheiro `mcdm_resultados.xlsx` com folhas para Dados, Pesos_e_Tipos, Rankings e Scores, pronto para anexar ao relatório.

---

## Deploy online (live)

> **Importante:** o GitHub Pages (`*.github.io`) **não consegue correr esta app**. GitHub Pages só serve ficheiros estáticos (HTML/CSS/JS) e o Streamlit precisa de um servidor Python a correr o backend. Mas há uma alternativa praticamente igual, gratuita e oficial: o **Streamlit Community Cloud**.

### Opção recomendada — Streamlit Community Cloud (gratuito)

Liga directamente ao teu repositório GitHub e gera um URL público (`https://<utilizador>-<repo>-app-<hash>.streamlit.app`).

**Passos:**

1. **Cria um repositório no GitHub** com os ficheiros:
   ```
   mcdm-dashboard/
   ├── app.py
   ├── requirements.txt
   └── README.md
   ```

2. **Push para o GitHub** (terminal):
   ```bash
   cd mcdm-dashboard
   git init
   git add .
   git commit -m "Initial MCDM dashboard"
   git branch -M main
   git remote add origin https://github.com/<teu-utilizador>/mcdm-dashboard.git
   git push -u origin main
   ```

3. **Vai a https://share.streamlit.io** e faz login com a conta GitHub.

4. **"New app"** → seleciona o repositório `mcdm-dashboard` → branch `main` → ficheiro principal `app.py` → **Deploy!**

5. Em 1-2 minutos a app fica online num URL público. Partilhas com o professor / colegas.

**Restrições do tier gratuito:** 1 GB de RAM, app dorme se ficar inactiva mas reactiva automaticamente ao primeiro acesso. Mais que suficiente para este caso.

### Alternativas (também gratuitas)

| Plataforma | Vantagens | Desvantagens |
|------------|-----------|--------------|
| **Hugging Face Spaces** | Suporta Streamlit nativamente, GPU opcional | URL menos amigável |
| **Render** | Free tier; bom uptime | Configuração mais complexa (precisa Procfile) |
| **Railway** | Deploy via GitHub, generoso | 5 USD crédito/mês depois pago |

Para o âmbito do trabalho académico, **Streamlit Community Cloud** é claramente o caminho mais limpo.

### E o github.io?

Se mesmo assim quiseres usar **github.io**, terias de re-implementar a app como pura HTML/JS (sem Python), provavelmente com bibliotecas tipo `mathjs` ou portar os modelos para JavaScript. Não recomendo — perdes a vantagem do ecossistema científico do Python (numpy, pandas) e o esforço é desproporcionado para o objectivo.

### Alternativa híbrida: badge no github.io

O que **podes** fazer no github.io: criar uma landing page estática (com texto, screenshots, vídeo demo) que tem um botão "🚀 Abrir App" a apontar para o URL Streamlit Cloud. Aproveitas a vanity URL `<user>.github.io/mcdm` e mantens a app a correr no Streamlit Cloud.

---

## Troubleshooting

### A app não abre / `streamlit: command not found`
- Confirma que o ambiente virtual está activado.
- Reinstala: `pip install --upgrade streamlit`.

### Erro `ModuleNotFoundError: openpyxl`
- O Excel `.xlsx` precisa de `openpyxl`. Instala: `pip install openpyxl`.

### Erro `Folha 'Dados' não foi encontrada`
- Verifica o nome da folha no Excel — tem de ser exactamente `Dados` (sensível a maiúsculas/minúsculas).

### Critério de custo está a ser tratado como benefício
- Na sidebar, em "Sentido dos critérios", muda manualmente para `min` o critério em causa.

### AHP devolve CR > 0,10 (inconsistente)
- A matriz par-a-par tem julgamentos contraditórios. Identifica as comparações mais extremas (valores ≥ 7) e ajusta-as gradualmente. A app continua a calcular pesos mesmo com CR alto, mas reporta-o.

### A app fica lenta com muitas alternativas (>100)
- O ELECTRE e PROMETHEE têm complexidade O(N²·M). Acima de 200 alternativas considera filtrar previamente ou usar apenas TOPSIS/MAUT (lineares).

### Sensibilidade TOPSIS mostra resultados estranhos
- Se um critério tiver peso muito baixo (< 0,03) a variação ±20% é desprezável e os error bars ficam quase invisíveis. Aumenta a percentagem de sensibilidade na sidebar.

---

## Notas metodológicas

### Limitações conhecidas

- **ANP, DEMATEL e Fuzzy ANP** usam um proxy data-driven baseado em correlações entre critérios. Numa aplicação rigorosa, estas matrizes seriam elicitadas directamente do decisor. Esta abordagem é defensável academicamente quando se cita explicitamente como "automated dependency estimation" e se reconhece a limitação.

- **Fuzzy AHP/TOPSIS** usam spread fixo (±20% / ±10%) como aproximação dos números triangulares. Implementações rigorosas requerem que o decisor defina o spread por julgamento.

- **AHP com matriz fornecida no caso MCG** (Q5.2 do questionário) apresenta CR ≈ 0,15. Devem ser iterados 1-2 julgamentos com o BU Manager para obter CR < 0,10 antes da publicação final.

### Convenções

- Todas as escalas ordinais (1-5) são tratadas como contínuas para fins de normalização.
- A1 (caso MCG) tem valor potencial ~17x superior ao próximo. Modelos baseados em normalização vectorial (TOPSIS) e min-max (MAUT, ANP, DEMATEL) suavizam o efeito; PROMETHEE com função `usual` e ELECTRE com limiares apertados mantêm a dominância. Esta divergência **é informativa**, não um erro.

### Referências

- Saaty, T. L. (1980). *The Analytic Hierarchy Process*. McGraw-Hill.
- Hwang, C.-L., Yoon, K. (1981). *Multiple Attribute Decision Making: Methods and Applications*. Springer.
- Roy, B. (1996). *Multicriteria Methodology for Decision Aiding*. Kluwer.
- Brans, J. P., Vincke, P. (1985). A Preference Ranking Organisation Method. *Management Science*, 31(6).
- Opricovic, S., Tzeng, G.-H. (2004). Compromise solution by MCDM methods: A comparative analysis of VIKOR and TOPSIS. *EJOR*, 156(2).
- Zavadskas, E. K., Kaklauskas, A. (1996). *Multiple Criteria Evaluation of Buildings*. Vilnius Tech.
- Gabus, A., Fontela, E. (1972). *World Problems, an Invitation to Further Thought*. Battelle Institute.
- Chang, D.-Y. (1996). Applications of the extent analysis method on fuzzy AHP. *EJOR*, 95(3).

---

## Licença

Material académico desenvolvido para a UC Modelos de Decisão (MEGI ISEL 2025/2026). Caso de estudo MCG. Livre para uso não-comercial com atribuição.
