# -*- coding: utf-8 -*-
"""
MCDM Dashboard — Sistema de Apoio à Decisão Multicritério
Modelos de Decisão | MEGI ISEL 2025/2026 | Caso de Estudo MCG

✅ PILAR 1: 100% autónomo — SEM Excel, dados em st.session_state
✅ PILAR 2: Teoria condensada em cada aba com <div class='theory-box'>
✅ PILAR 3: Motores de Pesos unificados + Toggle de Injeção Global
✅ PILAR 4: Função render_sensitivity() universal em TODAS as abas de modelos
✅ PILAR 5: Foco em matriz manual + blindagem de erros — ANP/Relatórios OMITIDOS

Execução: streamlit run app.py
Requisitos: streamlit>=1.30 pandas numpy scipy plotly
"""

import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import entropy
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG GERAL E ESTILOS
# =============================================================================
st.set_page_config(
    page_title="MCDM Dashboard | MCG",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS personalizado para teoria-box e visualização
st.markdown("""
<style>
    .theory-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .theory-box h4 { margin: 0 0 0.5rem 0; font-weight: 600; }
    .theory-box p { margin: 0.3rem 0; font-size: 0.95rem; opacity: 0.95; }
    .theory-box code { background: rgba(255,255,255,0.2); padding: 2px 6px; border-radius: 4px; }
    
    .step-box {
        background: rgba(120,120,120,0.08);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .step-box h5 { margin: 0 0 0.5rem 0; color: #444; }
    
    .stMetric { background: rgba(120,120,120,0.06); padding: 0.6rem; border-radius: 8px; }
    div[data-testid="stExpander"] details { border-radius: 8px; }
    
    /* Destaque para células que mudaram de ranking */
    .rank-up { background-color: #d4edda !important; }
    .rank-down { background-color: #f8d7da !important; }
</style>
""", unsafe_allow_html=True)

st.title("📊 MCDM Dashboard — Priorização Multicritério")
st.caption("Modelos de Decisão | MEGI ISEL 2025/2026 | Caso de Estudo MCG")

# =============================================================================
# PILAR 1: GESTÃO DE ESTADO E DADOS DINÂMICOS (SEM EXCEL)
# =============================================================================

def init_session_state():
    """Inicializa todas as variáveis de estado necessárias."""
    defaults = {
        'alternatives': pd.DataFrame({'ID': [f'A{i}' for i in range(1, 6)], 'Nome': [f'Alternativa {i}' for i in range(1, 6)]}),
        'criteria': pd.DataFrame({
            'ID': [f'C{i}' for i in range(1, 4)],
            'Nome': ['Custo', 'Qualidade', 'Prazo'],
            'Tipo': ['min', 'max', 'min'],  # min=custo, max=benefício
            'Peso': [0.4, 0.4, 0.2]
        }),
        'matrix': None,  # Será construída dinamicamente
        'global_weights': None,  # Pesos injetados globalmente
        'use_global_weights': False,  # Toggle de injeção global
        'sensitivity_pct': 20,  # Variação padrão para sensibilidade
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()

def build_decision_matrix():
    """Constrói a matriz de decisão a partir dos critérios e alternativas em session_state."""
    if st.session_state.alternatives.empty or st.session_state.criteria.empty:
        return None
    
    # Colunas da matriz: ID da alternativa + valores por critério
    cols = ['ID'] + [c['ID'] for _, c in st.session_state.criteria.iterrows()]
    mat = pd.DataFrame(columns=cols)
    mat['ID'] = st.session_state.alternatives['ID']
    
    # Inicializa valores numéricos (0.0) para cada critério
    for cid in st.session_state.criteria['ID']:
        mat[cid] = 0.0
    
    # Carrega valores existentes se houver
    if st.session_state.matrix is not None and 'ID' in st.session_state.matrix.columns:
        for _, row in st.session_state.matrix.iterrows():
            if row['ID'] in mat['ID'].values:
                for cid in st.session_state.criteria['ID']:
                    if cid in row and pd.notna(row[cid]):
                        mat.loc[mat['ID'] == row['ID'], cid] = float(row[cid])
    
    return mat

def get_weights():
    """Retorna pesos ativos: globais (se toggle ativo) ou manuais."""
    if st.session_state.use_global_weights and st.session_state.global_weights is not None:
        return st.session_state.global_weights.copy()
    return st.session_state.criteria['Peso'].values

def normalize_weights(weights):
    """Normaliza pesos para somar 1.0, com proteção contra zeros."""
    w = np.array(weights, dtype=float)
    w = np.where(w < 0, 0, w)  # Remove negativos
    total = w.sum()
    if total <= 0:
        return np.ones(len(w)) / len(w)
    return w / total

# =============================================================================
# UTILITÁRIOS MCDM
# =============================================================================

RI_TABLE = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24,
            7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49, 11: 1.51, 12: 1.54,
            13: 1.56, 14: 1.57, 15: 1.59}

def ranking_from_scores(scores, higher_is_better=True):
    """Devolve posições no ranking (1 = melhor)."""
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores if higher_is_better else scores)
    rank = np.zeros(len(scores), dtype=int)
    rank[order] = np.arange(1, len(scores) + 1)
    return rank

def normalize_vector(mat):
    """Normalização vectorial (Euclidiana) por critério — TOPSIS."""
    mat = np.asarray(mat, dtype=float)
    denom = np.sqrt(np.sum(mat ** 2, axis=0))
    denom = np.where(denom == 0, 1.0, denom)
    return mat / denom

def normalize_minmax(mat, types):
    """Normalização min-max com inversão para critérios de minimização."""
    mat = np.asarray(mat, dtype=float)
    out = np.zeros_like(mat)
    for j in range(mat.shape[1]):
        col = mat[:, j]
        rng = col.max() - col.min()
        if rng == 0:
            out[:, j] = 1.0
            continue
        if types[j] == "max":
            out[:, j] = (col - col.min()) / rng
        else:
            out[:, j] = (col.max() - col) / rng
    return out

# =============================================================================
# PILAR 3: MOTORES DE PESOS (5 MÉTODOS)
# =============================================================================

def weight_engine_ahp(pairwise_matrix):
    """AHP: cálculo de pesos via método do autovector + verificação CR."""
    A = np.asarray(pairwise_matrix, dtype=float)
    n = A.shape[0]
    # Forçar reciprocidade
    for i in range(n):
        for j in range(n):
            if i == j:
                A[i, j] = 1.0
            elif i < j and A[i, j] != 0:
                A[j, i] = 1.0 / A[i, j]
    
    eigvals, eigvecs = np.linalg.eig(A)
    idx = int(np.argmax(eigvals.real))
    lam_max = float(eigvals[idx].real)
    w = np.abs(eigvecs[:, idx].real)
    w = w / w.sum() if w.sum() > 0 else np.ones(n) / n
    
    CI = (lam_max - n) / (n - 1) if n > 1 else 0.0
    RI = RI_TABLE.get(n, 1.59)
    CR = CI / RI if RI > 0 else 0.0
    return {"weights": w, "lambda_max": lam_max, "CI": CI, "CR": CR, "consistent": CR < 0.10}

def weight_engine_swing(impact_scores):
    """SWING: pesos a partir de scores de impacto relativo (0-100)."""
    scores = np.array(impact_scores, dtype=float)
    scores = np.where(scores < 0, 0, scores)
    total = scores.sum()
    if total <= 0:
        return np.ones(len(scores)) / len(scores)
    return scores / total

def weight_engine_smart(importance_scores):
    """SMART: pesos a partir de classificação direta 0-100."""
    return weight_engine_swing(importance_scores)  # Mesma lógica de normalização

def weight_engine_entropy(matrix, types):
    """Entropia de Shannon: pesos baseados na variabilidade da informação."""
    mat = np.asarray(matrix, dtype=float)
    # Normalização para entropia
    norm = np.zeros_like(mat)
    for j in range(mat.shape[1]):
        col = mat[:, j]
        if types[j] == 'min':
            col = 1 / np.where(col == 0, 1e-9, col)  # Inverte para custos
        s = col.sum()
        norm[:, j] = col / s if s > 0 else np.ones(len(col)) / len(col)
    
    # Entropia por critério
    k = 1 / np.log(mat.shape[0]) if mat.shape[0] > 1 else 1
    E = -k * np.sum(norm * np.where(norm > 0, np.log(norm), 0), axis=0)
    d = 1 - E  # Divergência
    w = d / d.sum() if d.sum() > 0 else np.ones(len(d)) / len(d)
    return w

def weight_engine_critc(matrix):
    """CRITIC: pesos baseados em variância e correlação entre critérios."""
    mat = np.asarray(matrix, dtype=float)
    # Normalização min-max [0,1]
    norm = np.zeros_like(mat)
    for j in range(mat.shape[1]):
        col = mat[:, j]
        rng = col.max() - col.min()
        norm[:, j] = (col - col.min()) / rng if rng > 0 else np.zeros(len(col))
    
    # Variância por critério
    variances = np.var(norm, axis=0, ddof=1)
    # Matriz de correlação de Pearson
    corr = np.corrcoef(norm.T)
    corr = np.nan_to_num(corr, nan=1.0)
    np.fill_diagonal(corr, 0)  # Ignorar autocorrelação
    
    # Índice de conflito: variância × soma(1 - |corr|)
    conflict = np.sum(1 - np.abs(corr), axis=1)
    C = variances * conflict
    w = C / C.sum() if C.sum() > 0 else np.ones(len(C)) / len(C)
    return w

# =============================================================================
# PILAR 4: FUNÇÃO UNIVERSAL DE ANÁLISE DE SENSIBILIDADE
# =============================================================================

def render_sensitivity(model_fn, mat, weights, types, alts, base_scores, base_ranking, label="Score"):
    """
    Renderiza análise de sensibilidade ±X% para cada critério isoladamente.
    - Ajusta pesos restantes para manter soma = 1
    - Destaca em VERDE/VERMELHO mudanças de ranking (Pandas Styling)
    """
    with st.expander("🔍 Análise de Sensibilidade", expanded=False):
        st.markdown(f"**Variação aplicada:** ±{st.session_state.sensitivity_pct}% em cada peso individualmente")
        st.caption("Os pesos restantes são ajustados proporcionalmente para manter Σwⱼ = 1")
        
        sens_pct = st.session_state.sensitivity_pct / 100.0
        results = []
        
        for j, crit_name in enumerate(st.session_state.criteria['Nome']):
            # Variação positiva
            w_plus = weights.copy()
            w_plus[j] = weights[j] * (1 + sens_pct)
            # Ajuste proporcional dos restantes
            remaining = (1 - w_plus[j]) / (1 - weights[j]) if (1 - weights[j]) > 1e-9 else 1
            for k in range(len(weights)):
                if k != j:
                    w_plus[k] = weights[k] * remaining
            w_plus = normalize_weights(w_plus)
            
            # Variação negativa
            w_minus = weights.copy()
            w_minus[j] = max(weights[j] * (1 - sens_pct), 1e-6)
            remaining = (1 - w_minus[j]) / (1 - weights[j]) if (1 - weights[j]) > 1e-9 else 1
            for k in range(len(weights)):
                if k != j:
                    w_minus[k] = weights[k] * remaining
            w_minus = normalize_weights(w_minus)
            
            # Executa modelo com pesos perturbados
            res_plus, _ = safe_call(model_fn, mat, w_plus, types)
            res_minus, _ = safe_call(model_fn, mat, w_minus, types)
            
            if res_plus and res_minus:
                results.append({
                    'Critério': crit_name,
                    'Ranking Base': base_ranking.copy(),
                    'Ranking +': ranking_from_scores(res_plus.get('scores', base_scores)),
                    'Ranking -': ranking_from_scores(res_minus.get('scores', base_scores)),
                })
        
        if results:
            # Tabela comparativa com destaque visual
            df_sens = pd.DataFrame({
                'Alternativa': alts,
                'Ranking Base': base_ranking,
            })
            for r in results:
                df_sens[f"{r['Critério']} (+)"] = r['Ranking +']
                df_sens[f"{r['Critério']} (-)"] = r['Ranking -']
            
            # Função de styling para destacar mudanças
            def highlight_rank_change(val, base_col='Ranking Base'):
                if pd.isna(val) or pd.isna(df_sens.loc[val.name, base_col]):
                    return ''
                if val < df_sens.loc[val.name, base_col]:  # Melhorou (número menor = melhor posição)
                    return 'background-color: #d4edda; color: #155724'
                elif val > df_sens.loc[val.name, base_col]:  # Piorou
                    return 'background-color: #f8d7da; color: #721c24'
                return ''
            
            # Aplica styling apenas às colunas de sensibilidade
            styled = df_sens.style.map(
                lambda x: highlight_rank_change(x), 
                subset=[c for c in df_sens.columns if c != 'Alternativa' and c != 'Ranking Base']
            )
            st.dataframe(styled, use_container_width=True, hide_index=True)
            
            # Gráfico de Tornado simplificado
            st.subheader("Impacto relativo dos critérios no ranking")
            try:
                impact_data = []
                for r in results:
                    changes = np.abs(r['Ranking +'] - r['Ranking -'])
                    impact_data.append({'Critério': r['Critério'], 'Impacto Médio': changes.mean()})
                df_impact = pd.DataFrame(impact_data).sort_values('Impacto Médio', ascending=True)
                
                fig = px.bar(df_impact, x='Impacto Médio', y='Critério', orientation='h',
                            title="Gráfico de Tornado — Critérios mais influentes",
                            color='Impacto Médio', color_continuous_scale='RdYlGn_r')
                fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"⚠️ Não foi possível gerar o gráfico de impacto: {e}")

def safe_call(fn, *args, **kwargs):
    """Executa uma função e devolve (resultado, erro_str). Nunca rebenta o Streamlit."""
    try:
        return fn(*args, **kwargs), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"

# =============================================================================
# MODELOS MCDM (COM SAÍDAS PASSO-A-PASSO)
# =============================================================================

def model_topsis(mat, weights, types):
    """TOPSIS com outputs intermediários para pedagogia."""
    # Passo 1-2: Normalização vectorial
    norm = normalize_vector(mat)
    # Passo 3: Matriz ponderada
    weighted = norm * weights
    # Passo 4: Soluções ideais
    ideal = np.array([
        weighted[:, j].max() if types[j] == "max" else weighted[:, j].min()
        for j in range(mat.shape[1])
    ])
    anti = np.array([
        weighted[:, j].min() if types[j] == "max" else weighted[:, j].max()
        for j in range(mat.shape[1])
    ])
    # Passo 5: Distâncias
    d_plus = np.sqrt(np.sum((weighted - ideal) ** 2, axis=1))
    d_minus = np.sqrt(np.sum((weighted - anti) ** 2, axis=1))
    # Passo 6: Coeficiente de proximidade
    denom = np.where((d_plus + d_minus) == 0, 1e-9, d_plus + d_minus)
    ci = d_minus / denom
    return {
        "normalized": norm, "weighted": weighted,
        "ideal": ideal, "anti_ideal": anti,
        "d_plus": d_plus, "d_minus": d_minus,
        "scores": ci, "ranking": ranking_from_scores(ci),
    }

def model_promethee(mat, weights, types, function="usual"):
    """PROMETHEE II com fluxos líquidos."""
    def preference(d, ftype="usual", p=None, q=None, sigma=None):
        if d <= 0: return 0.0
        if ftype == "usual": return 1.0
        if ftype == "linear":
            if not p: return 0.0
            return float(min(d / p, 1.0))
        if ftype == "gaussian":
            s = sigma or 1.0
            return float(1.0 - np.exp(-(d ** 2) / (2 * s ** 2)))
        return 0.0
    
    n_alt, n_crit = mat.shape
    pref = np.zeros((n_alt, n_alt))
    for j in range(n_crit):
        col = mat[:, j]
        rng = col.max() - col.min()
        p = rng * 0.5 if rng > 0 else 1.0
        sigma = rng * 0.3 if rng > 0 else 1.0
        for i in range(n_alt):
            for k in range(n_alt):
                if i == k: continue
                d = (col[i] - col[k]) if types[j] == "max" else (col[k] - col[i])
                pref[i, k] += weights[j] * preference(d, function, p=p, sigma=sigma)
    
    div = max(n_alt - 1, 1)
    phi_plus = pref.sum(axis=1) / div
    phi_minus = pref.sum(axis=0) / div
    phi_net = phi_plus - phi_minus
    return {
        "preference_matrix": pref,
        "phi_plus": phi_plus, "phi_minus": phi_minus,
        "scores": phi_net, "ranking": ranking_from_scores(phi_net),
    }

def model_vikor(mat, weights, types, v=0.5):
    """VIKOR com índices S, R, Q."""
    n_alt, n_crit = mat.shape
    f_best = np.array([mat[:, j].max() if types[j] == "max" else mat[:, j].min() for j in range(n_crit)])
    f_worst = np.array([mat[:, j].min() if types[j] == "max" else mat[:, j].max() for j in range(n_crit)])
    rng = np.where((f_best - f_worst) == 0, 1e-9, f_best - f_worst)
    
    S = np.zeros(n_alt)
    R = np.zeros(n_alt)
    for i in range(n_alt):
        terms = np.zeros(n_crit)
        for j in range(n_crit):
            d = (f_best[j] - mat[i, j]) if types[j] == "max" else (mat[i, j] - f_best[j])
            terms[j] = weights[j] * d / rng[j]
        S[i] = terms.sum()
        R[i] = terms.max()
    
    s_b, s_w = S.min(), S.max()
    r_b, r_w = R.min(), R.max()
    s_rng = (s_w - s_b) if s_w != s_b else 1e-9
    r_rng = (r_w - r_b) if r_w != r_b else 1e-9
    Q = v * (S - s_b) / s_rng + (1 - v) * (R - r_b) / r_rng
    return {"S": S, "R": R, "Q": Q, "scores": -Q, "ranking": ranking_from_scores(-Q)}

def model_maut(mat, weights, types):
    """MAUT — utilidade linear aditiva."""
    norm = normalize_minmax(mat, types)
    U = (norm * weights).sum(axis=1)
    return {"utility_matrix": norm, "scores": U, "ranking": ranking_from_scores(U)}

def model_copras(mat, weights, types):
    """COPRAS — Complex Proportional Assessment."""
    norm = normalize_minmax(mat, types)  # Usar min-max para consistência pedagógica
    weighted = norm * weights
    benefit = [j for j in range(mat.shape[1]) if types[j] == "max"]
    cost = [j for j in range(mat.shape[1]) if types[j] == "min"]
    S_plus = weighted[:, benefit].sum(axis=1) if benefit else np.zeros(mat.shape[0])
    S_minus = weighted[:, cost].sum(axis=1) if cost else np.zeros(mat.shape[0])
    
    if cost and S_minus.min() > 0:
        S_minus_safe = np.where(S_minus == 0, 1e-9, S_minus)
        sum_inv = (1 / S_minus_safe).sum()
        Q = S_plus + (S_minus.min() * sum_inv) / (S_minus_safe * sum_inv)
    else:
        Q = S_plus
    N = (Q / Q.max()) * 100 if Q.max() != 0 else Q
    return {"S_plus": S_plus, "S_minus": S_minus, "Q": Q, "N": N,
            "scores": N, "ranking": ranking_from_scores(N)}

def model_electre(mat, weights, types, c_thresh=0.6, d_thresh=0.4):
    """ELECTRE I — relação de sobreclassificação."""
    n_alt, n_crit = mat.shape
    norm = normalize_minmax(mat, types)
    w_sum = weights.sum() if weights.sum() > 0 else 1.0
    concordance = np.zeros((n_alt, n_alt))
    discordance = np.zeros((n_alt, n_alt))
    global_range = max(norm.max() - norm.min(), 1e-9)
    
    for i in range(n_alt):
        for k in range(n_alt):
            if i == k: continue
            c = sum(weights[j] for j in range(n_crit) if norm[i, j] >= norm[k, j])
            concordance[i, k] = c / w_sum
            diffs = [norm[k, j] - norm[i, j] for j in range(n_crit) if norm[k, j] > norm[i, j]]
            discordance[i, k] = (max(diffs) / global_range) if diffs else 0.0
    
    outrank = (concordance >= c_thresh) & (discordance <= d_thresh)
    np.fill_diagonal(outrank, False)
    
    kernel = [i for i in range(n_alt) if not any(outrank[k, i] and not outrank[i, k] for k in range(n_alt) if k != i)]
    net_dominance = outrank.sum(axis=1) - outrank.sum(axis=0)
    return {
        "concordance": concordance, "discordance": discordance,
        "outrank": outrank, "kernel": kernel,
        "scores": net_dominance.astype(float),
        "ranking": ranking_from_scores(net_dominance.astype(float)),
    }

def model_dematel(mat, weights, types):
    """DEMATEL — modela influências entre critérios."""
    try:
        Z = np.abs(np.corrcoef(mat.T))
        Z = np.nan_to_num(Z)
        np.fill_diagonal(Z, 0)
        s = max(Z.sum(axis=1).max(), Z.sum(axis=0).max())
        X = Z / s if s > 0 else Z
        n = X.shape[0]
        T = X @ np.linalg.inv(np.eye(n) - X)
    except Exception:
        T = np.eye(mat.shape[1])
    
    D = T.sum(axis=1)
    R = T.sum(axis=0)
    prominence = D + R
    relation = D - R
    adj = weights * prominence
    adj = adj / adj.sum() if adj.sum() > 0 else weights
    
    norm = normalize_minmax(mat, types)
    scores = (norm * adj).sum(axis=1)
    return {
        "T": T, "D": D, "R": R, "prominence": prominence, "relation": relation,
        "adjusted_weights": adj, "scores": scores,
        "ranking": ranking_from_scores(scores)
    }

def model_fuzzy_topsis(mat, weights, types, spread=0.10):
    """Fuzzy TOPSIS — números triangulares com método do vértice."""
    l = mat * (1 - spread)
    m = mat.copy()
    u = mat * (1 + spread)
    n_alt, n_crit = mat.shape
    
    # Normalização fuzzy linear
    L = np.zeros_like(l); M = np.zeros_like(m); U = np.zeros_like(u)
    for j in range(n_crit):
        if types[j] == "max":
            denom = max(u[:, j].max(), 1e-9)
            L[:, j] = l[:, j] / denom
            M[:, j] = m[:, j] / denom
            U[:, j] = u[:, j] / denom
        else:
            num = max(l[:, j].min(), 1e-9)
            L[:, j] = num / np.where(u[:, j] == 0, 1e-9, u[:, j])
            M[:, j] = num / np.where(m[:, j] == 0, 1e-9, m[:, j])
            U[:, j] = num / np.where(l[:, j] == 0, 1e-9, l[:, j])
    
    Lw, Mw, Uw = L * weights, M * weights, U * weights
    
    # FPIS e FNIS
    fpis = np.array([(Uw[:, j].max(), Uw[:, j].max(), Uw[:, j].max()) for j in range(n_crit)])
    fnis = np.array([(Lw[:, j].min(), Lw[:, j].min(), Lw[:, j].min()) for j in range(n_crit)])
    
    def vd(al, am, au, bl, bm, bu):
        return np.sqrt(((al - bl) ** 2 + (am - bm) ** 2 + (au - bu) ** 2) / 3.0)
    
    d_plus = np.zeros(n_alt)
    d_minus = np.zeros(n_alt)
    for i in range(n_alt):
        for j in range(n_crit):
            d_plus[i] += vd(Lw[i, j], Mw[i, j], Uw[i, j], fpis[j, 0], fpis[j, 1], fpis[j, 2])
            d_minus[i] += vd(Lw[i, j], Mw[i, j], Uw[i, j], fnis[j, 0], fnis[j, 1], fnis[j, 2])
    
    denom = np.where((d_plus + d_minus) == 0, 1e-9, d_plus + d_minus)
    cc = d_minus / denom
    return {"d_plus": d_plus, "d_minus": d_minus, "scores": cc, "ranking": ranking_from_scores(cc)}

def model_fuzzy_ahp(weights, spread=0.20):
    """Fuzzy AHP — gera TFN a partir de pesos crisp (spread ±20%) e defuzzifica."""
    fuzzy = np.array([(w * (1-spread), w, w * (1+spread)) for w in weights])
    crisp = fuzzy.mean(axis=1)
    crisp = crisp / crisp.sum() if crisp.sum() > 0 else weights
    return {"fuzzy_weights": fuzzy, "crisp_weights": crisp}

# =============================================================================
# SIDEBAR — CONTROLOS GLOBAIS
# =============================================================================
with st.sidebar:
    st.header("⚙️ Configuração Global")
    
    # Slider para sensibilidade universal
    st.session_state.sensitivity_pct = st.slider(
        "🔍 Variação para Análise de Sensibilidade (%)",
        min_value=5, max_value=50, value=20, step=5,
        help="Percentagem de variação aplicada a cada peso individualmente"
    )
    
    st.divider()
    
    # Toggle de injeção global de pesos (PILAR 3)
    st.subheader("🎯 Pesos dos Critérios")
    st.session_state.use_global_weights = st.toggle(
        "🔄 Usar pesos calculados (injeção global)",
        value=False,
        help="Ative para forçar TODOS os modelos a usarem os pesos calculados na aba 'Motores de Pesos'"
    )
    
    if st.session_state.use_global_weights and st.session_state.global_weights is None:
        st.warning("⚠️ Nenhum peso calculado ainda. Vá à aba **⚙️ Motores de Pesos** para calcular.")
    
    st.divider()
    st.caption("💡 Dica: Edite alternativas, critérios e matriz diretamente nas abas. Todos os dados são guardados automaticamente no navegador.")

# =============================================================================
# DEFINIÇÃO DAS TABS
# =============================================================================
TAB_LABELS = [
    "📋 Dados & Matriz",
    "⚙️ Motores de Pesos",
    "🎯 TOPSIS",
    "📊 PROMETHEE II",
    "⚖️ VIKOR",
    "📐 MAUT",
    "🧮 COPRAS",
    "🔗 ELECTRE I",
    "🌐 DEMATEL",
    "🌫️ Fuzzy TOPSIS",
    "🌫️ Fuzzy AHP",
    "🏆 Dashboard Consolidado",
]
tabs = st.tabs(TAB_LABELS)

# Estrutura para resultados do dashboard
all_results = {}

# =============================================================================
# TAB 1: DADOS & MATRIZ (PILAR 1 — ENTRADA MANUAL DINÂMICA)
# =============================================================================
with tabs[0]:
    st.header("📋 Dados & Matriz de Decisão")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🔹 Alternativas")
        alt_editor = st.data_editor(
            st.session_state.alternatives,
            column_config={
                "ID": st.column_config.TextColumn("ID", disabled=True, width="small"),
                "Nome": st.column_config.TextColumn("Nome", width="medium"),
            },
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            key="editor_alternatives"
        )
        if not alt_editor.equals(st.session_state.alternatives):
            st.session_state.alternatives = alt_editor
            st.session_state.matrix = None  # Reset matrix quando alternativas mudam
        
        st.subheader("🔹 Critérios")
        crit_editor = st.data_editor(
            st.session_state.criteria,
            column_config={
                "ID": st.column_config.TextColumn("ID", disabled=True, width="small"),
                "Nome": st.column_config.TextColumn("Nome", width="medium"),
                "Tipo": st.column_config.SelectboxColumn("Tipo", options=["max", "min"], required=True, width="small"),
                "Peso": st.column_config.NumberColumn("Peso", min_value=0.0, max_value=1.0, step=0.01, format="%.3f", width="small"),
            },
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            key="editor_criteria"
        )
        if not crit_editor.equals(st.session_state.criteria):
            st.session_state.criteria = crit_editor
            st.session_state.matrix = None  # Reset matrix quando critérios mudam
            # Renormaliza pesos manuais
            st.session_state.criteria['Peso'] = normalize_weights(st.session_state.criteria['Peso'])
    
    with col2:
        st.subheader("🔹 Matriz de Decisão")
        st.caption("Valores numéricos por alternativa e critério. Colunas protegidas contra texto.")
        
        if st.session_state.alternatives.empty or st.session_state.criteria.empty:
            st.info("👈 Adicione pelo menos 1 alternativa e 1 critério para construir a matriz.")
        else:
            mat_df = build_decision_matrix()
            if mat_df is not None:
                # Configuração de colunas: ID protegido + números para critérios
                col_config = {"ID": st.column_config.TextColumn("ID", disabled=True, width="small")}
                for _, crit in st.session_state.criteria.iterrows():
                    col_config[crit['ID']] = st.column_config.NumberColumn(
                        crit['Nome'], 
                        min_value=0.0, 
                        step=0.01, 
                        format="%.2f",
                        help=f"Tipo: {crit['Tipo']}"
                    )
                
                edited_mat = st.data_editor(
                    mat_df,
                    column_config=col_config,
                    use_container_width=True,
                    hide_index=True,
                    num_rows="fixed",
                    key="editor_matrix"
                )
                st.session_state.matrix = edited_mat
                
                # Resumo rápido
                st.markdown(f"**Resumo:** {len(st.session_state.alternatives)} alternativas × {len(st.session_state.criteria)} critérios")

# =============================================================================
# TAB 2: MOTORES DE PESOS (PILAR 3)
# =============================================================================
with tabs[1]:
    st.header("⚙️ Motores de Pesos")
    
    if st.session_state.alternatives.empty or st.session_state.criteria.empty or st.session_state.matrix is None:
        st.info("👈 Configure alternativas, critérios e matriz na aba **📋 Dados & Matriz** primeiro.")
    else:
        mat = st.session_state.matrix[st.session_state.criteria['ID']].values.astype(float)
        types = st.session_state.criteria['Tipo'].tolist()
        
        method = st.selectbox(
            "🔧 Escolha o método de cálculo de pesos:",
            ["AHP (Comparação Par-a-Par)", "SWING Weighting", "SMART", "Entropia de Shannon", "CRITIC"],
            index=0
        )
        
        # ===== AHP =====
        if method == "AHP (Comparação Par-a-Par)":
            st.markdown("""
            <div class='theory-box'>
            <h4>🔺 AHP — Analytic Hierarchy Process</h4>
            <p><strong>Teoria:</strong> Pesos derivados de comparações par-a-par (escala Saaty 1-9).</p>
            <p><strong>Fórmula:</strong> <code>A·w = λ_max·w</code> → vetor próprio normalizado.</p>
            <p><strong>Consistência:</strong> <code>CR = CI/RI &lt; 0.10</code> para julgamentos válidos.</p>
            </div>
            """, unsafe_allow_html=True)
            
            n = len(st.session_state.criteria)
            st.subheader("Matriz de Comparação Par-a-Par")
            
            # Inicializa com pesos atuais como base
            init = np.ones((n, n))
            for i in range(n):
                for j in range(n):
                    if i != j and st.session_state.criteria.iloc[j]['Peso'] != 0:
                        init[i, j] = st.session_state.criteria.iloc[i]['Peso'] / st.session_state.criteria.iloc[j]['Peso']
            
            pw_df = pd.DataFrame(init, index=st.session_state.criteria['Nome'], columns=st.session_state.criteria['Nome']).round(3)
            edited_pw = st.data_editor(pw_df, use_container_width=True, key="ahp_pw_editor")
            
            # Forçar reciprocidade
            E = edited_pw.values.astype(float).copy()
            for i in range(n):
                for j in range(n):
                    if i == j: E[i, j] = 1.0
                    elif i < j and E[i, j] != 0: E[j, i] = 1.0 / E[i, j]
            
            if st.button("🔄 Calcular Pesos AHP"):
                res, err = safe_call(weight_engine_ahp, E)
                if err:
                    st.error(f"❌ Erro: {err}")
                else:
                    st.success(f"✅ CR = {res['CR']:.4f} {'(Consistente ✓)' if res['consistent'] else '(Rever julgamentos ⚠)'}")
                    new_weights = res['weights']
                    st.session_state.global_weights = new_weights
                    st.session_state.criteria['Peso'] = new_weights  # Atualiza também os manuais para feedback visual
                    
                    st.markdown("### 📊 Pesos Calculados")
                    st.dataframe(pd.DataFrame({
                        'Critério': st.session_state.criteria['Nome'],
                        'Peso AHP': new_weights,
                        'Peso Anterior': st.session_state.criteria['Peso'].values  # Mostra diferença
                    }).round(4), use_container_width=True, hide_index=True)
        
        # ===== SWING =====
        elif method == "SWING Weighting":
            st.markdown("""
            <div class='theory-box'>
            <h4>🎯 SWING Weighting</h4>
            <p><strong>Teoria:</strong> Atribui 100 pontos ao critério com maior "swing" (impacto de pior→melhor).</p>
            <p><strong>Fórmula:</strong> <code>wⱼ = scoreⱼ / Σ scoreₖ</code></p>
            </div>
            """, unsafe_allow_html=True)
            
            st.subheader("📊 Atribua scores de impacto (0-100)")
            st.caption("100 = critério com maior impacto na decisão; 0 = irrelevante")
            
            scores = []
            for _, crit in st.session_state.criteria.iterrows():
                s = st.slider(f"{crit['Nome']} ({crit['Tipo']})", 0, 100, int(crit['Peso']*100), key=f"swing_{crit['ID']}")
                scores.append(s)
            
            if st.button("🔄 Calcular Pesos SWING"):
                new_weights = weight_engine_swing(scores)
                st.session_state.global_weights = new_weights
                st.session_state.criteria['Peso'] = new_weights
                
                st.markdown("### 📈 Resultados")
                st.dataframe(pd.DataFrame({
                    'Critério': st.session_state.criteria['Nome'],
                    'Score': scores,
                    'Peso Calculado': new_weights
                }).round(4), use_container_width=True, hide_index=True)
                
                # Gráfico de pesos
                fig = px.bar(x=st.session_state.criteria['Nome'], y=new_weights, 
                            labels={'x': 'Critério', 'y': 'Peso'}, title="Distribuição de Pesos SWING")
                st.plotly_chart(fig, use_container_width=True)
        
        # ===== SMART =====
        elif method == "SMART":
            st.markdown("""
            <div class='theory-box'>
            <h4>🎯 SMART (Simple Multi-Attribute Rating Technique)</h4>
            <p><strong>Teoria:</strong> Classificação direta 0-100 do critério mais importante.</p>
            <p><strong>Fórmula:</strong> <code>wⱼ = ratingⱼ / Σ ratingₖ</code></p>
            <p><strong>Vantagem:</strong> Mais intuitivo que SWING para decisores não-técnicos.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.subheader("📊 Classifique a importância (0-100)")
            st.caption("100 = critério mais importante; proporcional para os restantes")
            
            ratings = []
            for _, crit in st.session_state.criteria.iterrows():
                r = st.slider(f"{crit['Nome']} ({crit['Tipo']})", 0, 100, int(crit['Peso']*100), key=f"smart_{crit['ID']}")
                ratings.append(r)
            
            if st.button("🔄 Calcular Pesos SMART"):
                new_weights = weight_engine_smart(ratings)
                st.session_state.global_weights = new_weights
                st.session_state.criteria['Peso'] = new_weights
                
                st.dataframe(pd.DataFrame({
                    'Critério': st.session_state.criteria['Nome'],
                    'Rating': ratings,
                    'Peso Calculado': new_weights
                }).round(4), use_container_width=True, hide_index=True)
        
        # ===== ENTROPIA =====
        elif method == "Entropia de Shannon":
            st.markdown("""
            <div class='theory-box'>
            <h4>📐 Entropia de Shannon</h4>
            <p><strong>Teoria:</strong> Pesos baseados na variabilidade da informação (mais variável = mais informativo).</p>
            <p><strong>Fórmulas:</strong></p>
            <p><code>Eⱼ = -k Σ [pᵢⱼ·ln(pᵢⱼ)], k=1/ln(m)</code></p>
            <p><code>dⱼ = 1-Eⱼ → wⱼ = dⱼ/Σdₖ</code></p>
            </div>
            """, unsafe_allow_html=True)
            
            st.info("ℹ️ Este método usa APENAS os dados da matriz (objetivo), ignorando preferências subjetivas.")
            
            if st.button("🔄 Calcular Pesos por Entropia"):
                new_weights = weight_engine_entropy(mat, types)
                st.session_state.global_weights = new_weights
                st.session_state.criteria['Peso'] = new_weights
                
                st.dataframe(pd.DataFrame({
                    'Critério': st.session_state.criteria['Nome'],
                    'Peso (Entropia)': new_weights
                }).round(4), use_container_width=True, hide_index=True)
        
        # ===== CRITIC =====
        elif method == "CRITIC":
            st.markdown("""
            <div class='theory-box'>
            <h4>📊 CRITIC (Criteria Importance Through Intercriteria Correlation)</h4>
            <p><strong>Teoria:</strong> Combina variância (informação) + correlação (conflito entre critérios).</p>
            <p><strong>Fórmulas:</strong></p>
            <p><code>Cⱼ = σ²ⱼ × Σₖ(1 - |rⱼₖ|)</code></p>
            <p><code>wⱼ = Cⱼ / ΣCₖ</code></p>
            </div>
            """, unsafe_allow_html=True)
            
            st.info("ℹ️ Critérios com alta variância E baixa correlação com outros recebem maior peso.")
            
            if st.button("🔄 Calcular Pesos CRITIC"):
                new_weights = weight_engine_critc(mat)
                st.session_state.global_weights = new_weights
                st.session_state.criteria['Peso'] = new_weights
                
                st.dataframe(pd.DataFrame({
                    'Critério': st.session_state.criteria['Nome'],
                    'Peso (CRITIC)': new_weights
                }).round(4), use_container_width=True, hide_index=True)
        
        # ===== BOTÃO DE INJEÇÃO GLOBAL =====
        st.divider()
        if st.session_state.global_weights is not None:
            st.success("✅ Pesos calculados disponíveis!")
            if st.button("🔄 Aplicar pesos a TODOS os modelos (injeção global)"):
                st.session_state.use_global_weights = True
                st.rerun()
        else:
            st.warning("⚠️ Calcule pesos primeiro para ativar a injeção global.")

# =============================================================================
# FUNÇÃO AUXILIAR PARA RENDERIZAR TEORIA BOX
# =============================================================================
def render_theory_box(title, theory_points, formulas=None):
    """Renderiza caixa de teoria padronizada com HTML/CSS."""
    html = f"<div class='theory-box'><h4>{title}</h4>"
    for pt in theory_points:
        html += f"<p>• {pt}</p>"
    if formulas:
        html += "<p style='margin-top:0.5rem'><strong>Fórmulas:</strong></p>"
        for f in formulas:
            html += f"<p><code>{f}</code></p>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

# =============================================================================
# TAB 3: TOPSIS (COM TEORIA + PASSO-A-PASSO + SENSIBILIDADE)
# =============================================================================
with tabs[2]:
    st.header("🎯 TOPSIS — Technique for Order Preference by Similarity to Ideal Solution")
    
    # PILAR 2: Teoria condensada no topo
    render_theory_box(
        "🎯 TOPSIS — Resumo Teórico",
        [
            "Método compensatório baseado em distâncias geométricas ao ideal/anti-ideal.",
            "Ranking pelo coeficiente de proximidade CCᵢ ∈ [0,1] (quanto maior, melhor).",
            "Eficiente para critérios quantitativos; sensível à normalização."
        ],
        [
            "rᵢⱼ = xᵢⱼ / √(Σ xₖⱼ²)  [normalização vectorial]",
            "vᵢⱼ = wⱼ · rᵢⱼ  [ponderação]",
            "CCᵢ = Dᵢ⁻ / (Dᵢ⁺ + Dᵢ⁻)  [coeficiente de proximidade]"
        ]
    )
    
    if st.session_state.alternatives.empty or st.session_state.criteria.empty or st.session_state.matrix is None:
        st.info("👈 Configure dados na aba **📋 Dados & Matriz** primeiro.")
    else:
        mat = st.session_state.matrix[st.session_state.criteria['ID']].values.astype(float)
        alts = st.session_state.alternatives['ID'].tolist()
        weights = get_weights()
        types = st.session_state.criteria['Tipo'].tolist()
        
        res, err = safe_call(model_topsis, mat, weights, types)
        if err:
            st.error(f"❌ Erro no TOPSIS: {err}")
        else:
            # Passo 1-2: Matriz Normalizada
            st.markdown('<div class="step-box"><h5>🔹 Passo 1-2: Matriz Normalizada (Vectorial)</h5></div>', unsafe_allow_html=True)
            st.latex(r"r_{ij} = \frac{x_{ij}}{\sqrt{\sum_{k=1}^{m} x_{kj}^2}}")
            st.dataframe(
                pd.DataFrame(res["normalized"], index=alts, columns=st.session_state.criteria['Nome']).round(4),
                use_container_width=True
            )
            
            # Passo 3: Matriz Ponderada
            st.markdown('<div class="step-box"><h5>🔹 Passo 3: Matriz Ponderada</h5></div>', unsafe_allow_html=True)
            st.latex(r"v_{ij} = w_j \cdot r_{ij}")
            st.dataframe(
                pd.DataFrame(res["weighted"], index=alts, columns=st.session_state.criteria['Nome']).round(4),
                use_container_width=True
            )
            
            # Passo 4: Soluções Ideais
            st.markdown('<div class="step-box"><h5>🔹 Passo 4: Soluções Ideal (A⁺) e Anti-Ideal (A⁻)</h5></div>', unsafe_allow_html=True)
            st.latex(r"A^+_j = \max_i v_{ij} \text{ (benefício)} \quad|\quad A^-_j = \min_i v_{ij} \text{ (custo)}")
            st.dataframe(
                pd.DataFrame({
                    'Critério': st.session_state.criteria['Nome'],
                    'A⁺ (Ideal)': res['ideal'],
                    'A⁻ (Anti-Ideal)': res['anti_ideal']
                }).round(4),
                use_container_width=True, hide_index=True
            )
            
            # Passo 5-6: Distâncias e Ranking Final
            st.markdown('<div class="step-box"><h5>🔹 Passo 5-6: Distâncias e Ranking Final</h5></div>', unsafe_allow_html=True)
            st.latex(r"D_i^+ = \sqrt{\sum_j (v_{ij} - A^+_j)^2} \quad|\quad CC_i = \frac{D_i^-}{D_i^+ + D_i^-}")
            
            rank_df = pd.DataFrame({
                'Alternativa': alts,
                'D⁺': res['d_plus'],
                'D⁻': res['d_minus'],
                'CCᵢ': res['scores'],
                'Ranking': res['ranking']
            }).sort_values('Ranking').reset_index(drop=True)
            st.dataframe(rank_df.style.format({'D⁺': '{:.4f}', 'D⁻': '{:.4f}', 'CCᵢ': '{:.4f}'}),
                        use_container_width=True, hide_index=True)
            
            # Gráfico de resultados
            fig = px.bar(rank_df, x='Alternativa', y='CCᵢ', color='Ranking',
                        title="🏆 Ranking TOPSIS — Coeficiente de Proximidade",
                        color_continuous_scale='RdYlGn_r', text_auto='.3f')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # PILAR 4: Sensibilidade no fundo da aba
            render_sensitivity(model_topsis, mat, weights, types, alts, res['scores'], res['ranking'], label="CCᵢ")
            
            # Guarda resultados para dashboard
            all_results["TOPSIS"] = {"scores": res["scores"], "ranking": res["ranking"]}

# =============================================================================
# TAB 4: PROMETHEE II (ESTRUTURA SEMELHANTE)
# =============================================================================
with tabs[3]:
    st.header("📊 PROMETHEE II — Preference Ranking Organisation Method")
    
    render_theory_box(
        "📊 PROMETHEE II — Resumo Teórico",
        [
            "Método não-compensatório baseado em comparações par-a-par.",
            "Usa funções de preferência Pⱼ(a,b) para capturar limiares de indiferença/preferência.",
            "Ranking pelo fluxo líquido φ(a) = φ⁺(a) - φ⁻(a) ∈ [-1, +1]."
        ],
        [
            "π(a,b) = Σ wⱼ · Pⱼ(a,b)  [índice global]",
            "φ⁺(a) = Σₓ π(a,x)/(n-1)  [fluxo de saída]",
            "φ(a) = φ⁺(a) - φ⁻(a)  [fluxo líquido]"
        ]
    )
    
    if st.session_state.alternatives.empty or st.session_state.criteria.empty or st.session_state.matrix is None:
        st.info("👈 Configure dados na aba **📋 Dados & Matriz** primeiro.")
    else:
        mat = st.session_state.matrix[st.session_state.criteria['ID']].values.astype(float)
        alts = st.session_state.alternatives['ID'].tolist()
        weights = get_weights()
        types = st.session_state.criteria['Tipo'].tolist()
        
        promethee_fn = st.selectbox("🔧 Função de Preferência", ["usual", "linear", "gaussian"], index=0,
                                   help="Usual = Tipo I (recomendado para exercícios sem especificação)")
        
        res, err = safe_call(model_promethee, mat, weights, types, promethee_fn)
        if err:
            st.error(f"❌ Erro no PROMETHEE: {err}")
        else:
            st.markdown('<div class="step-box"><h5>🔹 Matriz de Preferência Agregada π(a,b)</h5></div>', unsafe_allow_html=True)
            st.latex(r"\pi(a,b) = \sum_{j=1}^{n} w_j \cdot P_j(a,b)")
            st.dataframe(
                pd.DataFrame(res["preference_matrix"], index=alts, columns=alts).round(3),
                use_container_width=True
            )
            
            st.markdown('<div class="step-box"><h5>🔹 Fluxos e Ranking Final</h5></div>', unsafe_allow_html=True)
            st.latex(r"\phi(a) = \phi^+(a) - \phi^-(a)")
            
            rank_df = pd.DataFrame({
                'Alternativa': alts,
                'φ⁺ (Saída)': res['phi_plus'],
                'φ⁻ (Entrada)': res['phi_minus'],
                'φ (Líquido)': res['scores'],
                'Ranking': res['ranking']
            }).sort_values('Ranking').reset_index(drop=True)
            st.dataframe(rank_df.style.format({'φ⁺ (Saída)': '{:.4f}', 'φ⁻ (Entrada)': '{:.4f}', 'φ (Líquido)': '{:.4f}'}),
                        use_container_width=True, hide_index=True)
            
            fig = px.bar(rank_df, x='Alternativa', y='φ (Líquido)', color='Ranking',
                        title="🏆 Ranking PROMETHEE II — Fluxo Líquido",
                        color_continuous_scale='RdYlGn_r', text_auto='.3f')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            render_sensitivity(model_promethee, mat, weights, types, alts, res['scores'], res['ranking'], label="φ")
            all_results["PROMETHEE"] = {"scores": res["scores"], "ranking": res["ranking"]}

# =============================================================================
# TAB 5: VIKOR
# =============================================================================
with tabs[4]:
    st.header("⚖️ VIKOR — Compromise Solution for Conflicting Criteria")
    
    # Teoria condensada (PILAR 2)
    render_theory_box(
        "⚖️ VIKOR — Resumo Teórico",
        [
            "Método compensatório focado em solução de compromisso entre utilidade de grupo e arrependimento individual.",
            "Índice Q combina S (utilidade) e R (pior desvio) com peso v ∈ [0,1].",
            "Recomenda alternativa com menor Q, sujeita a condições de aceitação C1 e C2."
        ],
        [
            "S_j = sum_i w_i * (f_i* - f_ij) / (f_i* - f_i-)",
            "R_j = max_i [w_i * (f_i* - f_ij) / (f_i* - f_i-)]",
            "Q_j = v*(S_j-S*)/(S--S*) + (1-v)*(R_j-R*)/(R--R*)"
        ]
    )
    
    if st.session_state.alternatives.empty or st.session_state.criteria.empty or st.session_state.matrix is None:
        st.info("Configure dados na aba **Dados & Matriz** primeiro.")
    else:
        mat = st.session_state.matrix[st.session_state.criteria['ID']].values.astype(float)
        alts = st.session_state.alternatives['ID'].tolist()
        weights = get_weights()
        types = st.session_state.criteria['Tipo'].tolist()
        
        v_param = st.slider(
            "Parâmetro v (estratégia de compromisso)", 
            0.0, 1.0, 0.5, 0.05,
            help="v=1: máxima utilidade de grupo | v=0: mínimo arrependimento individual"
        )
        
        res, err = safe_call(model_vikor, mat, weights, types, v_param)
        if err:
            st.error(f"Erro no VIKOR: {err}")
        else:
            # Passo 1: Índices S e R
            st.markdown('<div class="step-box"><h5>Passo 1-2: Índices S (Utilidade) e R (Arrependimento)</h5></div>', unsafe_allow_html=True)
            st.latex(r"S_j = \sum_i w_i \frac{f_i^* - f_{ij}}{f_i^* - f_i^-}")
            st.latex(r"R_j = \max_i \left[ w_i \frac{f_i^* - f_{ij}}{f_i^* - f_i^-} \right]")
            
            sr_df = pd.DataFrame({
                'Alternativa': alts,
                'S (Utilidade)': res['S'],
                'R (Arrependimento)': res['R']
            }).round(4)
            st.dataframe(sr_df, use_container_width=True, hide_index=True)
            
            # Passo 3: Índice Q e Ranking
            st.markdown('<div class="step-box"><h5>Passo 3: Índice Q e Ranking Final</h5></div>', unsafe_allow_html=True)
            st.latex(r"Q_j = v \frac{S_j - S^*}{S^- - S^*} + (1-v) \frac{R_j - R^*}{R^- - R^*}")
            
            rank_df = pd.DataFrame({
                'Alternativa': alts,
                'Q (Índice VIKOR)': res['Q'],
                'Ranking': res['ranking']
            }).sort_values('Ranking').reset_index(drop=True)
            st.dataframe(
                rank_df.style.format({'Q (Índice VIKOR)': '{:.4f}'}),
                use_container_width=True, 
                hide_index=True
            )
            
            # Gráfico
            fig = px.bar(
                rank_df, 
                x='Alternativa', 
                y='Q (Índice VIKOR)', 
                color='Ranking',
                title="Ranking VIKOR — Índice de Compromisso Q (menor = melhor)",
                color_continuous_scale='RdYlGn_r', 
                text_auto='.3f'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # PILAR 4: Sensibilidade no fundo
            render_sensitivity(
                lambda m, w, t: model_vikor(m, w, t, v_param), 
                mat, weights, types, alts, 
                -res['Q'], res['ranking'], 
                label="Q"
            )
            
            # Guardar resultados para dashboard (PILAR 5: foco)
            all_results["VIKOR"] = {
                "scores": -res['Q'], 
                "ranking": res['ranking"]
            }

# =============================================================================
# TAB 6: MAUT
# =============================================================================
with tabs[5]:
    st.header("📐 MAUT — Multi-Attribute Utility Theory")
    
    render_theory_box(
        "📐 MAUT — Resumo Teórico",
        [
            "Método compensatório aditivo que transforma valores brutos em utilidades subjetivas [0,1].",
            "Assume independência preferencial entre critérios (forma aditiva).",
            "Flexível: suporta funções de utilidade lineares, exponenciais, logarítmicas, etc."
        ],
        [
            "Uⱼ(x) = (x - x_min)/(x_max - x_min)  [benefício, linear]",
            "Uⱼ(x) = (x_max - x)/(x_max - x_min)  [custo, linear]",
            "Uᵢ = Σ wⱼ · Uⱼ(xᵢⱼ)  [utilidade global]"
        ]
    )
    
    if st.session_state.alternatives.empty or st.session_state.criteria.empty or st.session_state.matrix is None:
        st.info("👈 Configure dados na aba **📋 Dados & Matriz** primeiro.")
    else:
        mat = st.session_state.matrix[st.session_state.criteria['ID']].values.astype(float)
        alts = st.session_state.alternatives['ID'].tolist()
        weights = get_weights()
        types = st.session_state.criteria['Tipo'].tolist()
        
        res, err = safe_call(model_maut, mat, weights, types)
        if err:
            st.error(f"❌ Erro no MAUT: {err}")
        else:
            st.markdown('<div class="step-box"><h5>🔹 Matriz de Utilidades Parciais Uⱼ(xᵢⱼ)</h5></div>', unsafe_allow_html=True)
            st.latex(r"U_j(x_{ij}) = \begin{cases} \frac{x_{ij} - x_{\min}}{x_{\max} - x_{\min}} & \text{(benefício)} \\ \frac{x_{\max} - x_{ij}}{x_{\max} - x_{\min}} & \text{(custo)} \end{cases}")
            st.dataframe(
                pd.DataFrame(res["utility_matrix"], index=alts, columns=st.session_state.criteria['Nome']).round(4),
                use_container_width=True
            )
            
            st.markdown('<div class="step-box"><h5>🔹 Utilidade Global e Ranking</h5></div>', unsafe_allow_html=True)
            st.latex(r"U_i = \sum_{j=1}^{n} w_j \cdot U_j(x_{ij})")
            
            rank_df = pd.DataFrame({
                'Alternativa': alts,
                'Utilidade Uᵢ': res['scores'],
                'Ranking': res['ranking']
            }).sort_values('Ranking').reset_index(drop=True)
            st.dataframe(rank_df.style.format({'Utilidade Uᵢ': '{:.4f}'}),
                        use_container_width=True, hide_index=True)
            
            fig = px.bar(rank_df, x='Alternativa', y='Utilidade Uᵢ', color='Ranking',
                        title="🏆 Ranking MAUT — Utilidade Global",
                        color_continuous_scale='RdYlGn_r', text_auto='.3f')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            render_sensitivity(model_maut, mat, weights, types, alts, res['scores'], res['ranking'], label="Uᵢ")
            all_results["MAUT"] = {"scores": res["scores"], "ranking": res["ranking"]}

# =============================================================================
# TAB 7: COPRAS
# =============================================================================
with tabs[6]:
    st.header("🧮 COPRAS — Complex Proportional Assessment")
    
    render_theory_box(
        "🧮 COPRAS — Resumo Teórico",
        [
            "Método compensatório que separa benefícios (S⁺) e custos (S⁻) na agregação.",
            "Calcula utilidade relativa Qᵢ combinando S⁺ e S⁻ de forma proporcional.",
            "Resultado final Nᵢ ∈ [0,100]% para fácil interpretação."
        ],
        [
            "S⁺ᵢ = Σ_{j∈benef} wⱼ·x̄ᵢⱼ  |  S⁻ᵢ = Σ_{j∈custo} wⱼ·x̄ᵢⱼ",
            "Qᵢ = S⁺ᵢ + [min(S⁻)·Σ(1/S⁻ₖ)] / [S⁻ᵢ·Σ(1/S⁻ₖ)]",
            "Nᵢ = (Qᵢ / max Qₖ) × 100%"
        ]
    )
    
    if st.session_state.alternatives.empty or st.session_state.criteria.empty or st.session_state.matrix is None:
        st.info("👈 Configure dados na aba **📋 Dados & Matriz** primeiro.")
    else:
        mat = st.session_state.matrix[st.session_state.criteria['ID']].values.astype(float)
        alts = st.session_state.alternatives['ID'].tolist()
        weights = get_weights()
        types = st.session_state.criteria['Tipo'].tolist()
        
        res, err = safe_call(model_copras, mat, weights, types)
        if err:
            st.error(f"❌ Erro no COPRAS: {err}")
        else:
            rank_df = pd.DataFrame({
                'Alternativa': alts,
                'S⁺ (Benefícios)': res['S_plus'],
                'S⁻ (Custos)': res['S_minus'],
                'Q (Importância Relativa)': res['Q'],
                'N (Utilidade %)': res['N'],
                'Ranking': res['ranking']
            }).sort_values('Ranking').reset_index(drop=True)
            st.dataframe(rank_df.style.format({
                'S⁺ (Benefícios)': '{:.4f}', 'S⁻ (Custos)': '{:.4f}',
                'Q (Importância Relativa)': '{:.4f}', 'N (Utilidade %)': '{:.2f}'
            }), use_container_width=True, hide_index=True)
            
            fig = px.bar(rank_df, x='Alternativa', y='N (Utilidade %)', color='Ranking',
                        title="🏆 Ranking COPRAS — Utilidade N (%)",
                        color_continuous_scale='RdYlGn_r', text_auto='.1f')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            render_sensitivity(model_copras, mat, weights, types, alts, res['N'], res['ranking'], label="N")
            all_results["COPRAS"] = {"scores": res['N'], "ranking": res['ranking"]}

# =============================================================================
# TAB 8: ELECTRE I
# =============================================================================
with tabs[7]:
    st.header("🔗 ELECTRE I — Outranking Relations")
    
    render_theory_box(
        "🔗 ELECTRE I — Resumo Teórico",
        [
            "Método não-compensatório baseado em relações de sobreclassificação binárias.",
            "Usa limiares de concordância (c) e discordância (d) para validar a⪰b.",
            "Resultado: kernel (conjunto de alternativas não-dominadas) + ranking por dominância líquida."
        ],
        [
            "C(a,b) = Σ_{j: a⪰b} wⱼ / Σwⱼ  [concordância]",
            "D(a,b) = maxⱼ(r_bⱼ - r_aⱼ) / range  [discordância]",
            "a S b ⇔ C(a,b)≥c ∧ D(a,b)≤d  [sobreclassificação]"
        ]
    )
    
    if st.session_state.alternatives.empty or st.session_state.criteria.empty or st.session_state.matrix is None:
        st.info("👈 Configure dados na aba **📋 Dados & Matriz** primeiro.")
    else:
        mat = st.session_state.matrix[st.session_state.criteria['ID']].values.astype(float)
        alts = st.session_state.alternatives['ID'].tolist()
        weights = get_weights()
        types = st.session_state.criteria['Tipo'].tolist()
        
        col1, col2 = st.columns(2)
        with col1:
            c_thresh = st.slider("🔹 Limiar de Concordância (c)", 0.50, 0.95, 0.65, 0.01)
        with col2:
            d_thresh = st.slider("🔹 Limiar de Discordância (d)", 0.05, 0.50, 0.35, 0.01)
        
        res, err = safe_call(model_electre, mat, weights, types, c_thresh, d_thresh)
        if err:
            st.error(f"❌ Erro no ELECTRE: {err}")
        else:
            st.markdown('<div class="step-box"><h5>🔹 Matriz de Concordância C(a,b)</h5></div>', unsafe_allow_html=True)
            st.latex(r"C(a,b) = \frac{\sum_{j \in J(a,b)} w_j}{\sum_j w_j}")
            st.dataframe(
                pd.DataFrame(res["concordance"], index=alts, columns=alts).round(3),
                use_container_width=True
            )
            
            st.markdown('<div class="step-box"><h5>🔹 Matriz de Discordância D(a,b)</h5></div>', unsafe_allow_html=True)
            st.dataframe(
                pd.DataFrame(res["discordance"], index=alts, columns=alts).round(3),
                use_container_width=True
            )
            
            st.markdown('<div class="step-box"><h5>🔹 Kernel e Ranking Final</h5></div>', unsafe_allow_html=True)
            kernel_alts = [alts[i] for i in res["kernel"]]
            st.success(f"**Kernel (alternativas não-dominadas):** {', '.join(kernel_alts) if kernel_alts else '∅'}")
            
            rank_df = pd.DataFrame({
                'Alternativa': alts,
                'No Kernel': ["✅" if i in res["kernel"] else "—" for i in range(len(alts))],
                'Dominância Líquida': res['scores'],
                'Ranking': res['ranking']
            }).sort_values('Ranking').reset_index(drop=True)
            st.dataframe(rank_df, use_container_width=True, hide_index=True)
            
            render_sensitivity(model_electre, mat, weights, types, alts, res['scores'], res['ranking'], label="Dominância")
            all_results["ELECTRE"] = {"scores": res["scores"], "ranking": res["ranking"]}

# =============================================================================
# TAB 9: DEMATEL
# =============================================================================
with tabs[8]:
    st.header("🌐 DEMATEL — Decision Making Trial and Evaluation Laboratory")
    
    render_theory_box(
        "🌐 DEMATEL — Resumo Teórico",
        [
            "Método estrutural que modela relações causa-efeito entre critérios (não rankeia alternativas diretamente).",
            "Matriz de influência total T captura efeitos diretos + indiretos via série infinita.",
            "Indicadores: R+C (proeminência/importância) e R-C (causalidade: >0=causador, <0=efeito)."
        ],
        [
            "X = Z / s  [normalização]",
            "T = X·(I - X)⁻¹  [influência total]",
            "Rᵢ = Σⱼ tᵢⱼ  |  Cⱼ = Σᵢ tᵢⱼ  [exercida/recebida]"
        ]
    )
    
    if st.session_state.alternatives.empty or st.session_state.criteria.empty or st.session_state.matrix is None:
        st.info("👈 Configure dados na aba **📋 Dados & Matriz** primeiro.")
    else:
        mat = st.session_state.matrix[st.session_state.criteria['ID']].values.astype(float)
        alts = st.session_state.alternatives['ID'].tolist()
        weights = get_weights()
        types = st.session_state.criteria['Tipo'].tolist()
        
        res, err = safe_call(model_dematel, mat, weights, types)
        if err:
            st.error(f"❌ Erro no DEMATEL: {err}")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="step-box"><h5>🔹 Matriz de Influência Total T</h5></div>', unsafe_allow_html=True)
                st.latex(r"T = X \cdot (I - X)^{-1}")
                st.dataframe(
                    pd.DataFrame(res["T"].round(4), index=st.session_state.criteria['Nome'], columns=st.session_state.criteria['Nome']),
                    use_container_width=True
                )
            with col2:
                st.markdown('<div class="step-box"><h5>🔹 Proeminência e Relação Causa-Efeito</h5></div>', unsafe_allow_html=True)
                st.latex(r"R+C \text{ (importância)} \quad|\quad R-C \text{ (causalidade)}")
                st.dataframe(
                    pd.DataFrame({
                        'Critério': st.session_state.criteria['Nome'],
                        'R (Exercida)': res['D'],
                        'C (Recebida)': res['R'],
                        'R+C (Proeminência)': res['prominence'],
                        'R-C (Relação)': res['relation']
                    }).round(4),
                    use_container_width=True, hide_index=True
                )
            
            # Diagrama causa-efeito
            try:
                fig = px.scatter(
                    x=res['prominence'], y=res['relation'], text=st.session_state.criteria['Nome'],
                    labels={'x': 'Proeminência (R+C)', 'y': 'Relação (R-C)'},
                    title="🗺️ Diagrama Causa-Efeito dos Critérios",
                    size_max=60
                )
                fig.update_traces(textposition='top center', marker=dict(size=14))
                fig.add_hline(y=0, line_dash='dash', line_color='gray')
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"⚠️ Não foi possível gerar o diagrama: {e}")
            
            # Ranking de alternativas com pesos ajustados por prominência
            st.markdown('<div class="step-box"><h5>🔹 Ranking de Alternativas (pesos ajustados por prominência)</h5></div>', unsafe_allow_html=True)
            rank_df = pd.DataFrame({
                'Alternativa': alts,
                'Score DEMATEL': res['scores'],
                'Ranking': res['ranking']
            }).sort_values('Ranking').reset_index(drop=True)
            st.dataframe(rank_df.style.format({'Score DEMATEL': '{:.4f}'}),
                        use_container_width=True, hide_index=True)
            
            render_sensitivity(model_dematel, mat, weights, types, alts, res['scores'], res['ranking'], label="Score")
            all_results["DEMATEL"] = {"scores": res["scores"], "ranking": res["ranking"]}

# =============================================================================
# TAB 10: FUZZY TOPSIS
# =============================================================================
with tabs[9]:
    st.header("🌫️ Fuzzy TOPSIS — Chen (2000)")
    
    render_theory_box(
        "🌫️ Fuzzy TOPSIS — Resumo Teórico",
        [
            "Extensão do TOPSIS para lidar com incerteza via Números Fuzzy Triangulares (TFN): ã = (l, m, u).",
            "Avaliações linguísticas convertidas em TFN; distância calculada pelo método do vértice.",
            "Mantém a lógica compensatória do TOPSIS clássico com robustez à imprecisão."
        ],
        [
            "d(ã,b̃) = √[(lₐ-l_b)² + (mₐ-m_b)² + (uₐ-u_b)²]/√3",
            "CCᵢ = Dᵢ⁻ / (Dᵢ⁺ + Dᵢ⁻)  [mesma lógica do TOPSIS clássico]"
        ]
    )
    
    if st.session_state.alternatives.empty or st.session_state.criteria.empty or st.session_state.matrix is None:
        st.info("👈 Configure dados na aba **📋 Dados & Matriz** primeiro.")
    else:
        mat = st.session_state.matrix[st.session_state.criteria['ID']].values.astype(float)
        alts = st.session_state.alternatives['ID'].tolist()
        weights = get_weights()
        types = st.session_state.criteria['Tipo'].tolist()
        
        spread = st.slider("🔹 Spread Fuzzy (%)", 5, 30, 10, 5) / 100.0
        st.caption(f"TFN gerados como: (x·(1-{spread*100:.0f}%), x, x·(1+{spread*100:.0f}%))")
        
        res, err = safe_call(model_fuzzy_topsis, mat, weights, types, spread)
        if err:
            st.error(f"❌ Erro no Fuzzy TOPSIS: {err}")
        else:
            rank_df = pd.DataFrame({
                'Alternativa': alts,
                'd⁺ (FPIS)': res['d_plus'],
                'd⁻ (FNIS)': res['d_minus'],
                'CC (Proximidade)': res['scores'],
                'Ranking': res['ranking']
            }).sort_values('Ranking').reset_index(drop=True)
            st.dataframe(rank_df.style.format({'d⁺ (FPIS)': '{:.4f}', 'd⁻ (FNIS)': '{:.4f}', 'CC (Proximidade)': '{:.4f}'}),
                        use_container_width=True, hide_index=True)
            
            fig = px.bar(rank_df, x='Alternativa', y='CC (Proximidade)', color='Ranking',
                        title="🏆 Ranking Fuzzy TOPSIS — Coeficiente de Proximidade",
                        color_continuous_scale='RdYlGn_r', text_auto='.3f')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            render_sensitivity(lambda m,w,t: model_fuzzy_topsis(m,w,t,spread), mat, weights, types, alts, res['scores'], res['ranking'], label="CC")
            all_results["Fuzzy TOPSIS"] = {"scores": res["scores"], "ranking": res["ranking"]}

# =============================================================================
# TAB 11: FUZZY AHP
# =============================================================================
with tabs[10]:
    st.header("🌫️ Fuzzy AHP — Chang (1996)")
    
    render_theory_box(
        "🌫️ Fuzzy AHP — Resumo Teórico",
        [
            "Extensão do AHP que usa Números Fuzzy Triangulares (TFN) para capturar incerteza nos julgamentos.",
            "Método Extent Analysis: calcula medida sintética Sᵢ e graus de possibilidade V(Sᵢ ≥ Sₖ).",
            "Pesos finais obtidos por defuzzificação (média dos TFN) e normalização."
        ],
        [
            "Sᵢ = Σⱼ M̃ⱼⁱ ⊗ [ΣᵢΣⱼ M̃ⱼⁱ]⁻¹  [medida sintética]",
            "wᵢ = minₖ V(Sᵢ ≥ Sₖ) / Σ minₖ V(Sₖ ≥ Sₘ)  [pesos fuzzy]",
            "wᵢ^crisp = (lᵢ + mᵢ + uᵢ)/3  [defuzzificação]"
        ]
    )
    
    if st.session_state.alternatives.empty or st.session_state.criteria.empty:
        st.info("👈 Configure critérios na aba **📋 Dados & Matriz** primeiro.")
    else:
        weights_manual = get_weights()
        
        spread = st.slider("🔹 Spread Fuzzy (%)", 10, 40, 20, 5) / 100.0
        st.caption(f"TFN gerados a partir dos pesos manuais: (w·(1-{spread*100:.0f}%), w, w·(1+{spread*100:.0f}%))")
        
        res, err = safe_call(model_fuzzy_ahp, weights_manual, spread)
        if err:
            st.error(f"❌ Erro no Fuzzy AHP: {err}")
        else:
            st.markdown('<div class="step-box"><h5>🔹 Pesos Fuzzy Triangulares e Defuzzificados</h5></div>', unsafe_allow_html=True)
            fuzzy_df = pd.DataFrame(res['fuzzy_weights'], index=st.session_state.criteria['Nome'], columns=['l (inferior)', 'm (modal)', 'u (superior)'])
            fuzzy_df['Crisp (Defuzz)'] = res['crisp_weights']
            st.dataframe(fuzzy_df.round(4), use_container_width=True)
            
            # Ranking usando pesos defuzzificados (para consistência com outros modelos)
            if st.session_state.matrix is not None:
                mat = st.session_state.matrix[st.session_state.criteria['ID']].values.astype(float)
                alts = st.session_state.alternatives['ID'].tolist()
                types = st.session_state.criteria['Tipo'].tolist()
                norm = normalize_minmax(mat, types)
                scores = (norm * res['crisp_weights']).sum(axis=1)
                ranking = ranking_from_scores(scores)
                
                rank_df = pd.DataFrame({
                    'Alternativa': alts,
                    'Score F-AHP': scores,
                    'Ranking': ranking
                }).sort_values('Ranking').reset_index(drop=True)
                st.dataframe(rank_df.style.format({'Score F-AHP': '{:.4f}'}),
                            use_container_width=True, hide_index=True)
                
                fig = px.bar(rank_df, x='Alternativa', y='Score F-AHP', color='Ranking',
                            title="🏆 Ranking Fuzzy AHP — Score com Pesos Defuzzificados",
                            color_continuous_scale='RdYlGn_r', text_auto='.3f')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Sensibilidade (usando modelo MAUT como proxy com pesos fuzzy)
                render_sensitivity(lambda m,w,t: {"scores": (normalize_minmax(m,t)*w).sum(axis=1), "ranking": ranking_from_scores((normalize_minmax(m,t)*w).sum(axis=1))}, 
                                mat, res['crisp_weights'], types, alts, scores, ranking, label="Score")
                all_results["Fuzzy AHP"] = {"scores": scores, "ranking": ranking}

# =============================================================================
# TAB 12: DASHBOARD CONSOLIDADO
# =============================================================================
with tabs[11]:
    st.header("🏆 Dashboard Consolidado")
    
    if not all_results:
        st.warning("⚠️ Nenhum modelo foi executado com sucesso. Visite as abas de modelos primeiro.")
    else:
        alts = st.session_state.alternatives['ID'].tolist()
        models_with_results = list(all_results.keys())
        
        # Tabela consolidada de rankings
        rank_table = pd.DataFrame({"Alternativa": alts})
        for m in models_with_results:
            rank_table[m] = all_results[m]["ranking"]
        
        # Ranking agregado (média de posições — Borda invertido)
        rank_table["Posição Média"] = rank_table[models_with_results].mean(axis=1).round(2)
        rank_table["Ranking Final"] = ranking_from_scores(-rank_table["Posição Média"].values)
        rank_table = rank_table.sort_values("Ranking Final").reset_index(drop=True)
        
        st.subheader("📊 Tabela Consolidada de Rankings (1 = melhor)")
        styled = rank_table.style.format({"Posição Média": "{:.2f}"})\
            .background_gradient(subset=models_with_results, cmap="RdYlGn_r")\
            .background_gradient(subset=["Posição Média", "Ranking Final"], cmap="RdYlGn_r")
        st.dataframe(styled, use_container_width=True, hide_index=True)
        
        # Convergência Top-3
        top3 = rank_table.head(3)
        st.subheader("🎯 Top-3 Recomendado")
        cols = st.columns(3)
        for i, (idx, row) in enumerate(top3.iterrows()):
            with cols[i]:
                medal = ["🥇", "🥈", "🥉"][i]
                st.metric(f"{medal} {i+1}º Lugar", row['Alternativa'], f"Média: {row['Posição Média']:.2f}")
        
        # Heatmap de rankings
        st.subheader("🗺️ Heatmap de Posições por Modelo")
        try:
            heat_df = rank_table.set_index("Alternativa")[models_with_results]
            fig = px.imshow(heat_df.values,
                          labels=dict(x="Modelo", y="Alternativa", color="Ranking"),
                          x=models_with_results, y=heat_df.index,
                          color_continuous_scale="RdYlGn_r", aspect="auto", text_auto=True)
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Erro no heatmap: {e}")
        
        # Exportação
        st.subheader("📥 Exportar Resultados")
        try:
            import io
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                pd.DataFrame({
                    'ID': st.session_state.alternatives['ID'],
                    'Nome': st.session_state.alternatives['Nome']
                }).to_excel(writer, sheet_name="Alternativas", index=False)
                st.session_state.criteria.to_excel(writer, sheet_name="Critérios", index=False)
                if st.session_state.matrix is not None:
                    st.session_state.matrix.to_excel(writer, sheet_name="Matriz", index=False)
                rank_table.to_excel(writer, sheet_name="Rankings_Consolidados", index=False)
            st.download_button(
                "📥 Descarregar Excel com todos os resultados",
                data=buffer.getvalue(),
                file_name="mcdm_resultados.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception as e:
            st.error(f"❌ Erro na exportação: {e}")

# =============================================================================
# RODAPÉ
# =============================================================================
st.markdown("---")
st.caption("MCDM Dashboard | MEGI ISEL 2025/2026 | Desenvolvido com Streamlit + NumPy + Pandas + Plotly")
st.caption("✅ 100% autónomo | ✅ Teoria integrada | ✅ Pesos globais | ✅ Sensibilidade universal | ✅ Sem ANP/Relatórios (foco atual)")
