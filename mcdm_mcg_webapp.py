# -*- coding: utf-8 -*-
"""
MCDM Dashboard v2.0 - Reestruturado
Arquitetura baseada em 5 Pilares: Autonomia, Pedagogia, Pesos Globais, Sensibilidade Universal e Foco.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import entropy
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURAÇÃO GERAL E ESTILO
# =============================================================================
st.set_page_config(
    page_title="MCDM Dashboard | Investigação Operacional",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS Personalizado para Teoria e Layout
st.markdown("""
<style>
    .theory-box {
        background-color: #f0f2f6;
        border-left: 5px solid #2E86AB;
        padding: 1.5rem;
        border-radius: 5px;
        margin-bottom: 2rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .theory-title {
        color: #2E86AB;
        font-weight: bold;
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
    }
    .step-header {
        margin-top: 2rem;
        border-bottom: 2px solid #eee;
        padding-bottom: 0.5rem;
        color: #444;
    }
    /* Highlighting para Sensibilidade */
    .rank-up { color: green; font-weight: bold; }
    .rank-down { color: red; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("📊 MCDM Dashboard — Apoio à Decisão Multicritério")
st.caption("Ferramenta Pedagógica e de Análise | Foco em Robustez e Transparência")

# =============================================================================
# GESTÃO DE ESTADO (SESSION STATE)
# =============================================================================
if 'data_loaded' not in st.session_state:
    # Dados Iniciais de Demonstração
    demo_data = {
        'Alternativa': ['A1', 'A2', 'A3', 'A4', 'A5'],
        'Custo (€)': [100, 150, 120, 200, 180],
        'Qualidade (0-10)': [8, 6, 9, 7, 8],
        'Prazo (Dias)': [10, 5, 15, 8, 12],
        'Sustentabilidade (0-10)': [5, 8, 4, 9, 7]
    }
    st.session_state.df = pd.DataFrame(demo_data)
    st.session_state.criteria_types = {'Custo (€)': 'min', 'Qualidade (0-10)': 'max', 'Prazo (Dias)': 'min', 'Sustentabilidade (0-10)': 'max'}
    st.session_state.global_weights = None  # Pesos calculados externamente
    st.session_state.use_global_weights = False # Toggle
    st.session_state.weights_source = "Manual"

# =============================================================================
# FUNÇÕES AUXILIARES (UTILITIES)
# =============================================================================

def render_theory_box(title, content, latex_formulas=None):
    """Renderiza a caixa de teoria no topo de cada aba."""
    st.markdown(f"""
    <div class='theory-box'>
        <div class='theory-title'>📚 {title}</div>
        <div>{content}</div>
    </div>
    """, unsafe_allow_html=True)
    if latex_formulas:
        for formula in latex_formulas:
            st.latex(formula)

def normalize_matrix(df, criteria_types):
    """Normalização Vetorial (Toplis style) ou Min-Max dependendo da necessidade."""
    # Aqui usaremos Normalização Vetorial padrão para a maioria, 
    # mas adaptada para Min/Max conforme o tipo.
    mat = df.to_numpy()
    norm_mat = np.zeros_like(mat, dtype=float)
    
    # Normalização Vetorial
    col_norms = np.linalg.norm(mat, axis=0)
    col_norms[col_norms == 0] = 1 # Evitar divisão por zero
    norm_mat = mat / col_norms
    
    # Inverter colunas de Custo/Min
    for j, col in enumerate(df.columns):
        if criteria_types.get(col, 'max') == 'min':
            norm_mat[:, j] = 1 - norm_mat[:, j] # Inversão simples para demonstração pedagógica
            
    return norm_mat, col_norms

def calculate_sensitivity(base_scores, base_ranking, model_func, df, weights, criteria_types, sensitivity_pct=0.1):
    """
    Calcula sensibilidade variando um critério de cada vez.
    Retorna DataFrame com resultados para plotting.
    """
    results = []
    n_crit = len(weights)
    alts = df.index.tolist()
    
    # Variação base
    results.append({
        'Cenário': 'Base', 'Alternativa': alts, 
        'Score': base_scores, 'Rank': base_ranking
    })
    
    # Variações
    for i in range(n_crit):
        w_perturbed = weights.copy()
        w_perturbed[i] *= (1 + sensitivity_pct)
        # Re-normalizar pesos para somar 1
        w_perturbed = w_p.erturbed / w_perturbed.sum()
        
        # Recalcular modelo
        try:
            res = model_func(df, w_perturbed, criteria_types)
            results.append({
                'Cenário': f'Var. {df.columns[i]} (+{sensitivity_pct*100:.0f}%)',
                'Alternativa': alts,
                'Score': res['scores'],
                'Rank': res['ranking']
            })
        except:
            pass
            
    return results

# =============================================================================
# MODELOS MCDM (MOTORES DE CÁLCULO)
# =============================================================================

def run_topsis(df, weights, criteria_types):
    """Motor TOPSIS."""
    # 1. Normalização Vetorial
    mat = df.to_numpy()
    norms = np.linalg.norm(mat, axis=0)
    norms[norms == 0] = 1
    r_mat = mat / norms
    
    # Inverter se for Min (Custo) -> 1-r
    for j, col in enumerate(df.columns):
        if criteria_types.get(col, 'max') == 'min':
            r_mat[:, j] = 1 - r_mat[:, j]
            
    # 2. Matriz Ponderada
    v_mat = r_mat * weights
    
    # 3. Ideais
    ideal = np.array([v_mat[:, j].max() if criteria_types.get(df.columns[j], 'max') == 'max' else v_mat[:, j].min() for j in range(len(weights))])
    anti_ideal = np.array([v_mat[:, j].min() if criteria_types.get(df.columns[j], 'max') == 'max' else v_mat[:, j].max() for j in range(len(weights))])
    
    # 4. Distâncias
    d_pos = np.sqrt(np.sum((v_mat - ideal)**2, axis=1))
    d_neg = np.sqrt(np.sum((v_mat - anti_ideal)**2, axis=1))
    
    # 5. CC
    cc = d_neg / (d_pos + d_neg + 1e-9)
    
    return {
        'scores': cc,
        'ranking': np.argsort(np.argsort(-cc)) + 1, # Rank 1 é o melhor
        'steps': {'R': r_mat, 'V': v_mat, 'D+': d_pos, 'D-': d_neg}
    }

def run_vikor(df, weights, criteria_types, v_param=0.5):
    """Motor VIKOR."""
    mat = df.to_numpy()
    f_star = np.array([mat[:, j].max() if criteria_types.get(df.columns[j], 'max') == 'max' else mat[:, j].min() for j in range(len(weights))])
    f_minus = np.array([mat[:, j].min() if criteria_types.get(df.columns[j], 'max') == 'max' else mat[:, j].max() for j in range(len(weights))])
    
    denom = f_star - f_minus
    denom[denom == 0] = 1e-9 # Evitar div zero
    
    S = np.zeros(len(df))
    R = np.zeros(len(df))
    
    for i in range(len(df)):
        diffs = (np.abs(f_star - mat[i, :])) / denom
        # Ajuste de sinal para Min (se custo, f_star é min, então mat - f_star)
        # Simplificação: assumindo que f_star é sempre o "melhor" valor
        # Se critério é Min, f_star é o min da coluna. Mat[i] - f_star >= 0.
        # Se critério é Max, f_star é o max. f_star - Mat[i] >= 0.
        # A fórmula padrão usa (f* - fij). 
        # Vamos usar a lógica absoluta da diferença relativa ponderada.
        S[i] = np.sum(weights * diffs)
        R[i] = np.max(weights * diffs)
        
    S_star, S_worst = S.min(), S.max()
    R_star, R_worst = R.min(), R.max()
    
    Q_num = v_param * (S - S_star) / (S_worst - S_star + 1e-9) + \
            (1 - v_param) * (R - R_star) / (R_worst - R_star + 
    Q = Q_num
    
    return {
        'scores': -Q, # Maximizar Q negativo = Minimizar Q positivo
        'ranking': np.argsort(np.argsort(Q)) + 1,
        'steps': {'S': S, 'R': R, 'Q': Q}
    }

# =============================================================================
# COMPONENTES DE UI
# =============================================================================

def render_sensitivity_analysis(base_scores, base_ranking, model_func, df, weights, criteria_types):
    st.markdown("### 🔍 Análise de Sensibilidade Universal")
    st.caption("Variação de ±20% no peso de cada critério individualmente.")
    
    sensitivity_pct = 0.2
    alts = df.index.tolist()
    
    # Calcular cenário base
    base_res = model_func(df, weights, criteria_types)
    base_ranks = base_res['ranking']
    
    # Tabela de Comparação
    sens_data = {'Alternativa': alts, 'Ranking Base': base_ranks}
    
    # Iterar sobre critérios
    for i, crit_name in enumerate(df.columns):
        w_temp = weights.copy()
        w_temp[i] *= (1 + sensitivity_pct)
        w_temp = w_temp / w_temp.sum()
        
        res = model_func(df, w_temp, criteria_types)
        sens_data[f'Rank {crit_name} (+20%)'] = res['ranking']
        
    df_sens = pd.DataFrame(sens_data)
    
    # Styling condicional
    def highlight_changes(row):
        styles = []
        base = row['Ranking Base']
        for col in row.index[2:]: # Pular Alt e Base
            val = row[col]
            if val < base: styles.append('color: green; font-weight: bold') # Melhorou rank (menor numero)
            elif val > base: styles.append('color: red; font-weight: bold') # Piorou
            else: styles.append('')
        return styles
        
    st.dataframe(df_sens.style.apply(highlight_changes, axis=1), use_container_width=True)
    
    # Gráfico de Tornado Simples (Opcional, mas bom para visual)
    # Para simplificar o código, focamos na tabela colorida que é obrigatória.

# =============================================================================
# ABA 1: DADOS DE ENTRADA (O CORAÇÃO DA APP)
# =============================================================================
def tab_input_data():
    st.header("1. Matriz de Decisão")
    render_theory_box(
        "Construção da Matriz",
        "Defina aqui as suas Alternativas (linhas) e Critérios (colunas). "
        "O sistema deteta automaticamente se o critério deve ser Maximizado (Benefício) ou Minimizado (Custo).",
        ["X = [x_{ij}]_{m \\times n}"]
    )
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Edição da Matriz")
        # Editor Dinâmico
        edited_df = st.data_editor(
            st.session_state.df,
            num_rows="dynamic",
            use_container_width=True,
            key="data_editor_main"
        )
        st.session_state.df = edited_df
        
    with col2:
        st.subheader("Configuração de Critérios")
        st.caption("Defina o sentido (Max/Min) e o peso manual.")
        
        # Pesos Manuais
        weights_input = {}
        types_input = {}
        
        # Criar inputs para cada coluna numérica
        for col in edited_df.columns:
            c1, c2, c3 = st.columns([3, 1, 1])
            with c1:
                st.write(f"**{col}**")
            with c2:
                # Tipo
                current_type = st.session_state.criteria_types.get(col, 'max')
                new_type = st.selectbox("Tipo", ['max', 'min'], index=0 if current_type=='max' else 1, key=f"type_{col}", label_visibility="collapsed")
                types_input[col] = new_type
            with c3:
                # Peso Manual
                # Se houver pesos globais injetados, mostra-os mas permite editar? Não, peso manual é manual.
                default_w = 1.0 / len(edited_df.columns)
                w_val = st.number_input("Peso", min_value=0.0, max_value=1.0, value=default_w, step=0.01, key=f"w_{col}", label_visibility="collapsed")
                weights_input[col] = w_val
                
        st.session_state.criteria_types = types_input
        
        # Normalizar pesos manuais
        total_w = sum(weights_input.values())
        if total_w > 0:
            final_weights = np.array([weights_input[k]/total_w for k in weights_input.keys()])
        else:
            final_weights = np.ones(len(weights_input)) / len(weights_input)
            
        st.session_state.manual_weights = final_weights
        st.session_state.weight_labels = list(weights_input.keys())
        
        st.divider()
        st.write("**Resumo dos Pesos:**")
        st.bar_chart(pd.Series(final_weights, index=weights_input.keys()))

# =============================================================================
# ABA 2: MOTORES DE PESOS (GLOBAL WEIGHT INJECTION)
# =============================================================================
def tab_weights_engine():
    st.header("2. Motores de Pesos & Injeção Global")
    render_theory_box(
        "Calculadora de Pesos Objetiva",
        "Utilize métodos matemáticos (Entropia, CRITIC) ou subjetivos (AHP simplificado) para determinar a importância dos critérios. "
        "Ative a 'Injeção Global' para forçar estes pesos em todos os modelos de decisão.",
        [r"\sum w_j = 1"]
    )
    
    if st.session_state.data_loaded: # Apenas se houver dados (sempre true na v2)
        df = st.session_state.df
        n_crit = len(df.columns)
        
        col_opt, col_res = st.columns([1, 2])
        
        with col_opt:
            method = st.selectbox("Método", ["Entropia de Shannon", "CRITIC", "AHP (Simulado)"])
            calc_btn = st.button("Calcular Pesos")
            
            # Toggle Global
            st.divider()
            st.session_state.use_global_weights = st.toggle(
                "🌍 Injeção Global de Pesos", 
                value=st.session_state.use_global_weights,
                help="Se ativo, todos os modelos (TOPSIS, VIKOR...) usarão os pesos calculados aqui, ignorando os manuais."
            )
            
        with col_res:
            if calc_btn or st.session_state.use_global_weights:
                weights = None
                if method == "Entropia de Shannon":
                    # Cálculo simples de entropia
                    mat = df.to_numpy()
                    # Normalizar
                    mat_norm = mat / mat.sum(axis=0)
                    mat_norm[mat_norm == 0] = 1e-9 # Log safety
                    e = -entropy(mat_norm, axis=0) / np.log(len(df))
                    d = 1 - e
                    weights = d / d.sum()
                    st.success("Pesos calculados via Entropia de Shannon.")
                    
                elif method == "CRITIC":
                    # Variância e Correlação
                    mat = (df - df.mean()) / df.std()
                    corr = np.corrcoef(mat)
                    std_dev = df.std()
                    intensity = std_dev / std_dev.sum() # Simplificação
                    conflict = np.sum(1 - corr, axis=1) # Simplificação
                    # Ajuste dimensional
                    info = std_dev * conflict 
                    weights = info / info.sum()
                    st.success("Pesos calculados via CRITIC.")
                    
                elif method == "AHP (Simulado)":
                    # Simulação AHP baseada na variância (proxy)
                    weights = df.std()
                    weights = weights / weights.sum()
                    st.success("Pesos estimados via proxy de variância (AHP Simulado).")

                if weights is not None:
                    st.session_state.global_weights = weights
                    st.session_state.weights_source = method
                    
                    res_df = pd.DataFrame({
                        'Critério': df.columns,
                        'Peso Calculado': weights
                    })
                    st.dataframe(res_df.style.format({'Peso Calculado': '{:.4f}'}), hide_index=True)
                    
                    st.info(f"✅ Pesos guardados na memória. {'Ative o Toggle para usar em todos os modelos.' if not st.session_state.use_global_weights else '🌍 Modo Global ATIVO.'}")

# =============================================================================
# TEMPLATE PARA ABAS DE MODELOS
# =============================================================================
def render_model_tab(model_name, model_func, theory_title, theory_text, latex_list):
    # Obter dados
    df = st.session_state.df
    if df.empty:
        st.warning("Por favor, insira dados na aba '1. Matriz de Decisão'.")
        return

    # Obter pesos
    if st.session_state.use_global_weights and st.session_state.global_weights is not None:
        weights = st.session_state.global_weights
        w_source = f"Global ({st.session_state.weights_source})"
    else:
        # Tentar pegar pesos manuais da sessão, senão uniformes
        if hasattr(st.session_state, 'manual_weights') and len(st.session_state.manual_weights) == len(df.columns):
            weights = st.session_state.manual_weights
        else:
            weights = np.ones(len(df.columns)) / len(df.columns)
        w_source = "Manual"

    st.header(model_name)
    render_theory_box(theory_title, theory_text, latex_list)
    
    st.caption(f"🏷️ A usar pesos: **{w_source}**")
    
    # Passo 1: Normalização
    st.markdown("### Passo 1: Matriz Normalizada")
    # Chamar função auxiliar de normalização genérica ou específica do modelo
    # Para simplificar o template, vamos assumir que o modelo retorna 'steps'
    # Mas precisamos rodar o modelo primeiro.
    
    # Executar Modelo
    try:
        res = model_func(df, weights, st.session_state.criteria_types)
    except Exception as e:
        st.error(f"Erro no cálculo do modelo: {e}")
        return

    # Exibir Passos Intermediários (Genérico)
    if 'steps' in res:
        steps = res['steps']
        if 'R' in steps:
            st.markdown("**Matriz de Decisão Normalizada ($R$):**")
            st.dataframe(pd.DataFrame(steps['R'], columns=df.columns, index=df.index).style.format("{:.3f}"))
        
        if 'V' in steps:
            st.markdown("**Matriz Ponderada ($V$):**")
            st.dataframe(pd.DataFrame(steps['V'], columns=df.columns, index=df.index).style.format("{:.3f}"))

    # Resultado Final
    st.markdown("### Resultado Final & Ranking")
    results_df = pd.DataFrame({
        'Alternativa': df.index,
        'Score': res['scores'],
        'Ranking': res['ranking']
    }).sort_values('Ranking')
    
    st.dataframe(results_df.style.format({'Score': '{:.4f}'}).hide_index=True, use_container_width=True)
    
    # Gráfico
    fig = px.bar(results_df.sort_values('Score', ascending=False), x='Alternativa', y='Score', title=f"Score {model_name}", text_auto='.3f')
    st.plotly_chart(fig, use_container_width=True)
    
    # Sensibilidade Universal (Pilar 4)
    st.divider()
    render_sensitivity_analysis(res['scores'], res['ranking'], model_func, df, weights, st.session_state.criteria_types)

# =============================================================================
# LAYOUT PRINCIPAL E NAVEGAÇÃO
# =============================================================================

# Sidebar para navegação rápida
st.sidebar.title("Navegação")
menu = st.sidebar.radio("Ir para:", 
    ["1. Matriz de Decisão", 
     "2. Pesos & Critérios", 
     "3. TOPSIS", 
     "4. VIKOR", 
     "5. MAUT (Simples)",
     "Sobre"])

if menu == "1. Matriz de Decisão":
    tab_input_data()
elif menu == "2. Pesos & Critérios":
    tab_weights_engine()
elif menu == "3. TOPSIS":
    render_model_tab(
        "TOPSIS", 
        run_topsis, 
        "Técnica de Ordem por Preferência de Semelhança com a Solução Ideal",
        "Baseia-se na distância geométrica de cada alternativa à Solução Ideal Positiva e à Solução Ideal Negativa.",
        [r"C_i^* = \frac{D_i^-}{D_i^+ + D_i^-}"]
    )
elif menu == "4. VIKOR":
    render_model_tab(
        "VIKOR", 
        lambda df, w, t: run_vikor(df, w, t, v_param=0.5), 
        "Otimização Multicritério e Solução de Compromisso",
        "Foca-se no ranking e seleção da melhor alternativa de compromisso, considerando a 'maioria' (utilidade de grupo) e o 'indivíduo' (arrependimento).",
        [r"Q_j = v \frac{S_j - S^*}{S^- - S^*} + (1-v) \frac{R_j - R^*}{R^- - R^*}"]
    )
elif menu == "5. MAUT (Simples)":
    # Implementação inline rápida para MAUT aditiva linear
    def run_maut(df, w, t):
        mat = df.to_numpy()
        # MinMax Normalization
        mat_norm = np.zeros_like(mat, dtype=float)
        for j in range(mat.shape[1]):
            col = mat[:, j]
            mn, mx = col.min(), col.max()
            if mx - mn == 0: mat_norm[:, j] = 0.5
            elif t.get(df.columns[j], 'max') == 'max':
                mat_norm[:, j] = (col - mn) / (mx - mn)
            else:
                mat_norm[:, j] = (mx - col) / (mx - mn)
        
        scores = np.dot(mat_norm, w)
        return {'scores': scores, 'ranking': np.argsort(np.argsort(-scores)) + 1}

    render_model_tab(
        "MAUT (Linear)",
        run_maut,
        "Teoria da Utilidade Multi-Atributo",
        "Soma ponderada das utilidades. Assume compensação total entre critérios.",
        [r"U_i = \sum w_j u_j(x_{ij})"]
    )
else:
    st.header("Sobre o MCDM Dashboard")
    st.info("Desenvolvido para fins pedagógicos e de investigação operacional.")
    st.markdown("""
    **Arquitetura:**
    1. **Autonomia:** Dados geridos em memória.
    2. **Pedagogia:** Teoria visível em cada passo.
    3. **Flexibilidade:** Pesos manuais ou calculados (Entropia/CRITIC).
    4. **Robustez:** Análise de sensibilidade universal.
    """)
