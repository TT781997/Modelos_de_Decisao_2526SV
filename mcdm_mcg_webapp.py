"""
MCDM Dashboard v2 — Ferramenta de Apoio à Decisão Multicritério
================================================================
Reestruturada conforme pedido:
  • Sidebar: 3 fontes de dados (manual + paste, Excel, demo) + N_alt/N_crit + sensibilidade
  • AHP movido para a sua própria aba com iterações até CR < 0.10
  • Motores de Pesos com inputs específicos da teoria
  • Aba Gráficos com visualizações Plotly bonitas
  • Sensibilidade explícita e clara em cada aba
  • Relatório visual final
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO, BytesIO

# =============================================================================
# CONFIG
# =============================================================================
st.set_page_config(page_title="MCDM Dashboard v2", page_icon="📊", layout="wide",
                   initial_sidebar_state="expanded")

CSS = """
<style>
.theory-box {
    background: linear-gradient(135deg, #f0f7ff 0%, #e0ecff 100%);
    border-left: 4px solid #1F4E78; padding: 16px 20px; border-radius: 6px;
    margin: 12px 0 20px 0; font-size: 14px; line-height: 1.5; color: #1F4E78;
}
.theory-box h4 { color: #1F4E78; margin-top: 0; font-size: 16px; font-weight: 700; }
.theory-box ul, .theory-box ol { margin: 8px 0 0 20px; }
.theory-box code { background: #fff; padding: 2px 6px; border-radius: 3px; font-size: 13px; color: #c7254e; }
.step-header {
    background: #2E75B6; color: white; padding: 8px 14px; border-radius: 4px;
    font-weight: 600; margin: 16px 0 8px 0; font-size: 15px;
}
.sensitivity-box {
    background: linear-gradient(135deg, #fff8e1 0%, #ffecb3 100%);
    border: 2px solid #f57c00; padding: 18px; border-radius: 8px; margin: 24px 0 12px 0;
}
.sensitivity-box h3 { color: #e65100; margin-top: 0; font-size: 18px; }
.injection-active {
    background: #fce4ec; border: 2px solid #c2185b; padding: 8px 14px;
    border-radius: 6px; color: #c2185b; font-weight: 700; text-align: center; margin: 12px 0;
}
.result-box {
    background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
    border: 2px solid #2e7d32; padding: 18px; border-radius: 8px; margin: 16px 0;
    color: #1b5e20; font-size: 16px; font-weight: 600;
}
.warning-box {
    background: #fff3e0; border-left: 3px solid #ef6c00; padding: 10px 16px;
    border-radius: 4px; margin: 8px 0; color: #e65100;
}
.stTabs [data-baseweb="tab"] { padding: 10px 18px; font-weight: 600; }
.stTabs [aria-selected="true"] { background-color: #1F4E78 !important; color: white !important; }
.sidebar-section {
    background: #f5f5f5; padding: 12px; border-radius: 6px; margin: 10px 0;
    border-left: 3px solid #1F4E78;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# =============================================================================
# DEMO PRESETS
# =============================================================================
DEMO_PRESETS = {
    "Caso MCG (9 alts × 6 crits)": {
        "criteria": pd.DataFrame({
            "Critério": ["C1_VP", "C2_PF", "C3_EE", "C4_FE", "C5_UD", "C6_RC"],
            "Tipo": ["max", "max", "min", "max", "max", "max"],
            "Peso Manual": [0.462, 0.218, 0.024, 0.114, 0.023, 0.159],
        }),
        "matrix": pd.DataFrame({
            "Alternativa": [f"A{i}" for i in range(1, 10)],
            "C1_VP": [250_000_000, 300_000, 900_000, 650_000, 5_000_000, 1_350_000, 10_500_000, 3_450_000, 15_000_000],
            "C2_PF": [0.25, 0.35, 0.50, 0.50, 0.40, 0.50, 0.40, 0.40, 0.60],
            "C3_EE": [24, 8, 8, 8, 24, 8, 16, 8, 24],
            "C4_FE": [4, 5, 3, 3, 4, 3, 3, 3, 4],
            "C5_UD": [180, 60, 60, 90, 30, 60, 180, 60, 300],
            "C6_RC": [4, 5, 5, 3, 3, 5, 4, 4, 3],
        }),
    },
    "Selecção de Fornecedor (5 × 4)": {
        "criteria": pd.DataFrame({
            "Critério": ["Custo", "Qualidade", "Prazo", "Sustentabilidade"],
            "Tipo": ["min", "max", "min", "max"],
            "Peso Manual": [0.30, 0.30, 0.20, 0.20],
        }),
        "matrix": pd.DataFrame({
            "Alternativa": ["Forn. A", "Forn. B", "Forn. C", "Forn. D", "Forn. E"],
            "Custo": [1200.0, 1500.0, 1100.0, 1300.0, 1400.0],
            "Qualidade": [8.0, 6.0, 9.0, 7.0, 5.0],
            "Prazo": [15.0, 20.0, 18.0, 12.0, 25.0],
            "Sustentabilidade": [7.0, 5.0, 8.0, 6.0, 4.0],
        }),
    },
    "Investimento (4 × 5)": {
        "criteria": pd.DataFrame({
            "Critério": ["Retorno (%)", "Risco", "Liquidez", "Maturidade (anos)", "Custo fees"],
            "Tipo": ["max", "min", "max", "min", "min"],
            "Peso Manual": [0.30, 0.25, 0.20, 0.15, 0.10],
        }),
        "matrix": pd.DataFrame({
            "Alternativa": ["Acções", "Obrigações", "Imobiliário", "ETFs"],
            "Retorno (%)": [12.0, 4.5, 7.0, 8.5],
            "Risco": [8.0, 2.0, 5.0, 4.0],
            "Liquidez": [9.0, 7.0, 3.0, 8.0],
            "Maturidade (anos)": [1.0, 10.0, 15.0, 3.0],
            "Custo fees": [1.5, 0.5, 3.0, 0.3],
        }),
    },
}

# =============================================================================
# ESTADO INICIAL
# =============================================================================
def init_state():
    if "criteria_df" not in st.session_state:
        preset = DEMO_PRESETS["Selecção de Fornecedor (5 × 4)"]
        st.session_state.criteria_df = preset["criteria"].copy()
        st.session_state.matrix_df = preset["matrix"].copy()
    if "global_injection_on" not in st.session_state:
        st.session_state.global_injection_on = False
    if "global_injection_engine" not in st.session_state:
        st.session_state.global_injection_engine = "AHP"
    if "engine_weights" not in st.session_state:
        st.session_state.engine_weights = {}
    if "sensitivity_pct" not in st.session_state:
        st.session_state.sensitivity_pct = 20
    if "ahp_history" not in st.session_state:
        st.session_state.ahp_history = []
    if "data_source" not in st.session_state:
        st.session_state.data_source = "Demo"

init_state()

# =============================================================================
# HELPERS
# =============================================================================
def get_decision_matrix():
    crit_df = st.session_state.criteria_df.copy()
    crit_df = crit_df.dropna(subset=["Critério"])
    crit_df = crit_df[crit_df["Critério"].astype(str).str.strip() != ""]
    crits = crit_df["Critério"].astype(str).tolist()
    types = crit_df["Tipo"].fillna("max").astype(str).tolist()

    m_df = st.session_state.matrix_df.copy()
    m_df = m_df.dropna(subset=["Alternativa"])
    m_df = m_df[m_df["Alternativa"].astype(str).str.strip() != ""]
    alts = m_df["Alternativa"].astype(str).tolist()

    matrix = []
    for crit in crits:
        if crit not in m_df.columns:
            matrix.append([0.0] * len(alts))
        else:
            col = pd.to_numeric(m_df[crit], errors="coerce").fillna(0.0).values
            matrix.append(col)
    matrix = np.array(matrix).T if matrix else np.zeros((0, 0))
    return matrix, alts, crits, types

def get_active_weights():
    _, _, crits, _ = get_decision_matrix()
    n = len(crits)
    if (st.session_state.global_injection_on
        and st.session_state.global_injection_engine in st.session_state.engine_weights):
        w = st.session_state.engine_weights[st.session_state.global_injection_engine]
        if len(w) == n:
            w = np.array(w, dtype=float)
            return w / w.sum() if w.sum() > 0 else np.ones(n) / n
    crit_df = st.session_state.criteria_df.copy()
    crit_df = crit_df.dropna(subset=["Critério"])
    crit_df = crit_df[crit_df["Critério"].astype(str).str.strip() != ""]
    w = pd.to_numeric(crit_df["Peso Manual"], errors="coerce").fillna(0).values
    return w / w.sum() if w.sum() > 0 else np.ones(n) / n

def show_active_weights_banner():
    w = get_active_weights()
    _, _, crits, _ = get_decision_matrix()
    if st.session_state.global_injection_on:
        engine = st.session_state.global_injection_engine
        st.markdown(
            f'<div class="injection-active">🔌 Injecção Global ACTIVA — pesos do motor <b>{engine}</b></div>',
            unsafe_allow_html=True)
    cols = st.columns([3, 1])
    with cols[0]:
        df_w = pd.DataFrame({"Critério": crits, "Peso (%)": [f"{x*100:.2f}%" for x in w]})
        st.dataframe(df_w, hide_index=True, use_container_width=False)
    with cols[1]:
        st.metric("Σ pesos", f"{w.sum():.4f}")

def theory_box(title, html):
    st.markdown(f'<div class="theory-box"><h4>📚 {title}</h4>{html}</div>', unsafe_allow_html=True)

def step_header(text):
    st.markdown(f'<div class="step-header">{text}</div>', unsafe_allow_html=True)

def check_valid_input():
    matrix, alts, crits, types = get_decision_matrix()
    if len(alts) < 2 or len(crits) < 2:
        st.warning("⚠️ Defina pelo menos 2 alternativas e 2 critérios.")
        return False
    if matrix.size == 0 or np.all(matrix == 0):
        st.warning("⚠️ Matriz vazia ou só com zeros.")
        return False
    return True

# Normalizações
def normalize_vector(m):
    d = np.sqrt((m ** 2).sum(axis=0))
    return m / np.where(d == 0, 1, d)

def normalize_minmax(m, t):
    out = np.zeros_like(m, dtype=float)
    for j in range(m.shape[1]):
        col = m[:, j]; mn, mx = col.min(), col.max()
        if mx == mn: out[:, j] = 0.5
        elif t[j] == "max": out[:, j] = (col - mn) / (mx - mn)
        else: out[:, j] = (mx - col) / (mx - mn)
    return out

def normalize_sum(m):
    s = m.sum(axis=0)
    return m / np.where(s == 0, 1, s)

# =============================================================================
# SENSIBILIDADE UNIVERSAL — destacada
# =============================================================================
def render_sensitivity(score_function, alts, crits, base_weights, higher_is_better=True, key_suffix=""):
    """Sensibilidade ±X% nos pesos. Apresentação destacada e clara."""
    st.markdown(
        '<div class="sensitivity-box"><h3>🎯 Análise de Sensibilidade ± X% nos Pesos</h3>'
        '<p style="margin-bottom:0; color:#bf360c;">Variamos o peso de <b>cada critério isoladamente</b> '
        'em ±X% e renormalizamos os restantes. Para cada cenário recalculamos o ranking e comparamos com o Base.</p>'
        '</div>',
        unsafe_allow_html=True
    )

    # Slider lê do session_state (controlo na sidebar)
    variation_pct = st.session_state.sensitivity_pct

    bw = np.array(base_weights, dtype=float)
    bw = bw / bw.sum()
    try:
        base_scores = np.array(score_function(bw))
    except Exception as e:
        st.error(f"Erro a calcular scores base: {e}")
        return
    base_ranks = pd.Series(base_scores).rank(ascending=not higher_is_better, method='min').astype(int).values

    factor_pos = 1 + variation_pct / 100
    factor_neg = 1 - variation_pct / 100
    scenarios = {"Base": base_ranks}

    for j, crit in enumerate(crits):
        for sign, factor in [("+", factor_pos), ("-", factor_neg)]:
            new_w = bw.copy()
            new_w[j] = bw[j] * factor
            other_sum_old = bw.sum() - bw[j]
            other_sum_new = 1 - new_w[j]
            if other_sum_old > 0 and other_sum_new > 0:
                for k in range(len(new_w)):
                    if k != j:
                        new_w[k] = bw[k] * (other_sum_new / other_sum_old)
            ws = new_w.sum()
            new_w = new_w / ws if ws > 0 else np.ones_like(new_w) / len(new_w)
            try:
                sc = np.array(score_function(new_w))
                rk = pd.Series(sc).rank(ascending=not higher_is_better, method='min').astype(int).values
            except Exception:
                rk = [None] * len(alts)
            scenarios[f"{crit} {sign}{variation_pct}%"] = rk

    df_sens = pd.DataFrame(scenarios, index=alts)
    df_sens.index.name = "Alternativa"

    def style_row(row):
        base = row["Base"]
        styles = []
        for col in row.index:
            if col == "Base":
                styles.append("background-color: #f0f0f0; font-weight: 700;")
            else:
                val = row[col]
                try:
                    if val is None or pd.isna(val):
                        styles.append("background-color: #fafafa; color: #999;")
                    elif val < base:
                        styles.append("background-color: #C6EFCE; color: #006100; font-weight: 600;")
                    elif val > base:
                        styles.append("background-color: #FFC7CE; color: #9C0006; font-weight: 600;")
                    else:
                        styles.append("")
                except Exception:
                    styles.append("")
        return styles

    st.markdown(f"**Variação ± aplicada:** {variation_pct}% (ajustável na barra lateral)")
    st.markdown("**Legenda:** 🟢 sobe no ranking · 🔴 desce no ranking · ⚪ sem alteração")
    st.dataframe(df_sens.style.apply(style_row, axis=1), use_container_width=True)

    # Robustez
    base_vals = df_sens["Base"].values
    others = df_sens.drop(columns=["Base"])
    n_changes = []
    for i in range(len(alts)):
        c = 0
        for col in others.columns:
            v = others.iloc[i][col]
            try:
                if v is not None and not pd.isna(v) and v != base_vals[i]:
                    c += 1
            except Exception:
                pass
        n_changes.append(c)
    df_robust = pd.DataFrame({
        "Alternativa": alts,
        "Rank Base": base_ranks,
        "Inversões": n_changes,
        "Robustez": ["🟢 ESTÁVEL" if c == 0 else ("🟡 MODERADA" if c <= 2 else "🔴 INSTÁVEL") for c in n_changes]
    })
    st.markdown("**Resumo de Robustez por Alternativa:**")
    st.dataframe(df_robust, hide_index=True, use_container_width=True)

# Guardar score-functions e rankings por método para o Dashboard/Relatório
if "all_results" not in st.session_state:
    st.session_state.all_results = {}

def store_result(method_name, scores, ranking, higher_is_better=True):
    st.session_state.all_results[method_name] = {
        "scores": np.array(scores),
        "ranking": np.array(ranking),
        "higher_is_better": higher_is_better,
    }

# =============================================================================
# TÍTULO
# =============================================================================
st.title("📊 MCDM Dashboard")
st.markdown(
    "**Decisão Multicritério** · 3 modos de entrada de dados · 9 modelos MCDM · "
    "Motores de pesos · Sensibilidade universal · Relatório final"
)


# =============================================================================
# SIDEBAR — 3 fontes de dados + dimensões + sensibilidade
# =============================================================================
with st.sidebar:
    st.header("⚙️ Configuração")

    # ============== SECÇÃO 1: FONTE DE DADOS ==============
    st.markdown("### 📥 Fonte de Dados")
    data_source = st.radio(
        "Como quer fornecer os dados?",
        ["📋 Demo (pré-definidos)", "✏️ Manual (editor + paste)", "📁 Carregar Excel"],
        key="data_source_radio",
        help="3 modos: usar caso demo, criar manualmente (com paste do Excel), ou carregar ficheiro"
    )

    if data_source == "📋 Demo (pré-definidos)":
        preset_name = st.selectbox(
            "Escolher caso:",
            list(DEMO_PRESETS.keys()),
            key="preset_selector"
        )
        if st.button("📥 Carregar este caso", use_container_width=True, type="primary"):
            preset = DEMO_PRESETS[preset_name]
            st.session_state.criteria_df = preset["criteria"].copy()
            st.session_state.matrix_df = preset["matrix"].copy()
            st.session_state.engine_weights = {}
            st.session_state.ahp_history = []
            st.success(f"✓ Carregado: {preset_name}")
            st.rerun()

    elif data_source == "✏️ Manual (editor + paste)":
        st.markdown("**Opção A — Dimensões:**")
        col_a, col_b = st.columns(2)
        n_alt_input = col_a.number_input("N.º Alternativas", min_value=2, max_value=30, value=5, step=1, key="n_alt_input")
        n_crit_input = col_b.number_input("N.º Critérios", min_value=2, max_value=15, value=4, step=1, key="n_crit_input")
        if st.button("🆕 Criar matriz vazia", use_container_width=True):
            new_crits = pd.DataFrame({
                "Critério": [f"C{i+1}" for i in range(n_crit_input)],
                "Tipo": ["max"] * n_crit_input,
                "Peso Manual": [1.0 / n_crit_input] * n_crit_input,
            })
            new_matrix = pd.DataFrame({"Alternativa": [f"Alt {i+1}" for i in range(n_alt_input)]})
            for c in new_crits["Critério"]:
                new_matrix[c] = 0.0
            st.session_state.criteria_df = new_crits
            st.session_state.matrix_df = new_matrix
            st.session_state.engine_weights = {}
            st.success(f"✓ Matriz {n_alt_input}×{n_crit_input} criada (preencher abaixo)")
            st.rerun()

        st.markdown("**Opção B — Colar do Excel:**")
        st.caption("Cole valores separados por TAB (copy directo do Excel). Primeira linha = nomes dos critérios. Primeira coluna = nomes das alternativas.")
        paste_text = st.text_area(
            "Colar aqui (Ctrl+V):",
            height=140,
            placeholder="Alternativa\tC1\tC2\tC3\nAlt 1\t8\t1200\t15\nAlt 2\t6\t1500\t20\n...",
            key="paste_area"
        )
        if st.button("📋 Processar dados colados", use_container_width=True):
            try:
                df = pd.read_csv(StringIO(paste_text), sep="\t")
                first_col = df.columns[0]
                crits = list(df.columns[1:])
                df = df.rename(columns={first_col: "Alternativa"})
                new_crits = pd.DataFrame({
                    "Critério": crits,
                    "Tipo": ["max"] * len(crits),
                    "Peso Manual": [1.0 / len(crits)] * len(crits),
                })
                for c in crits:
                    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
                st.session_state.criteria_df = new_crits
                st.session_state.matrix_df = df
                st.session_state.engine_weights = {}
                st.success(f"✓ Carregado: {len(df)} alts × {len(crits)} crits")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Erro ao processar: {e}\n\nVerifique que o texto está separado por TAB.")

    else:  # Carregar Excel
        uploaded = st.file_uploader(
            "Carregar Excel (.xlsx)",
            type=["xlsx", "xls"],
            help="Folha 1 deve ter: 1ª coluna = nomes alternativas, restantes = critérios numéricos"
        )
        if uploaded is not None:
            if st.button("📁 Processar Excel", use_container_width=True):
                try:
                    df = pd.read_excel(uploaded)
                    first_col = df.columns[0]
                    crits = list(df.columns[1:])
                    df = df.rename(columns={first_col: "Alternativa"})
                    new_crits = pd.DataFrame({
                        "Critério": crits,
                        "Tipo": ["max"] * len(crits),
                        "Peso Manual": [1.0 / len(crits)] * len(crits),
                    })
                    for c in crits:
                        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
                    st.session_state.criteria_df = new_crits
                    st.session_state.matrix_df = df
                    st.session_state.engine_weights = {}
                    st.success(f"✓ Carregado: {len(df)} alts × {len(crits)} crits")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Erro: {e}")

    st.markdown("---")

    # ============== SECÇÃO 2: SENSIBILIDADE ==============
    st.markdown("### 🎯 Análise de Sensibilidade")
    st.session_state.sensitivity_pct = st.slider(
        "Variação ± nos pesos (%):",
        5, 50, st.session_state.sensitivity_pct, 5,
        key="sens_slider_sidebar",
        help="Aplicada em TODAS as abas. Cada peso é variado isoladamente; restantes ajustados para Σ=1."
    )

    st.markdown("---")

    # ============== SECÇÃO 3: INJECÇÃO GLOBAL ==============
    st.markdown("### 🔌 Injecção Global de Pesos")
    st.session_state.global_injection_on = st.toggle(
        "Activar pesos do motor",
        value=st.session_state.global_injection_on,
        help="ON: todos os modelos usam pesos do motor seleccionado. OFF: usam 'Peso Manual'."
    )
    if st.session_state.global_injection_on:
        available = list(st.session_state.engine_weights.keys())
        if not available:
            st.caption("⚠️ Nenhum motor calculado ainda. Vá às abas dos motores.")
        else:
            st.session_state.global_injection_engine = st.selectbox(
                "Motor activo:",
                available,
                index=available.index(st.session_state.global_injection_engine)
                      if st.session_state.global_injection_engine in available else 0,
                key="engine_selector_sidebar"
            )

    st.markdown("---")

    # ============== SECÇÃO 4: EDITORES (sempre acessíveis) ==============
    with st.expander("📋 Editor de Critérios", expanded=False):
        edited_crit = st.data_editor(
            st.session_state.criteria_df,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            key="criteria_editor_sb",
            column_config={
                "Critério": st.column_config.TextColumn("Critério", required=True),
                "Tipo": st.column_config.SelectboxColumn("Tipo", options=["max", "min"], required=True),
                "Peso Manual": st.column_config.NumberColumn(
                    "Peso Manual", min_value=0.0, max_value=1.0, step=0.01, format="%.4f"),
            }
        )
        if edited_crit is not None and not edited_crit.equals(st.session_state.criteria_df):
            valid = edited_crit.dropna(subset=["Critério"])
            valid = valid[valid["Critério"].astype(str).str.strip() != ""]
            new_crits = valid["Critério"].astype(str).tolist()
            old_matrix = st.session_state.matrix_df.copy()
            new_matrix = pd.DataFrame({"Alternativa": old_matrix["Alternativa"]})
            for crit in new_crits:
                if crit in old_matrix.columns:
                    new_matrix[crit] = old_matrix[crit]
                else:
                    new_matrix[crit] = 0.0
            st.session_state.criteria_df = valid.reset_index(drop=True)
            st.session_state.matrix_df = new_matrix
            st.rerun()

        w_manual = pd.to_numeric(st.session_state.criteria_df["Peso Manual"], errors="coerce").fillna(0)
        soma = w_manual.sum()
        if abs(soma - 1.0) > 0.01:
            st.caption(f"⚠️ Σ = **{soma:.4f}** (será renormalizado)")
        else:
            st.caption(f"✅ Σ = **{soma:.4f}**")

    with st.expander("🔢 Editor de Matriz de Decisão", expanded=False):
        crit_names = st.session_state.criteria_df["Critério"].astype(str).tolist()
        matrix_col_config = {
            "Alternativa": st.column_config.TextColumn("Alternativa", required=True, width="small"),
        }
        for crit in crit_names:
            matrix_col_config[crit] = st.column_config.NumberColumn(crit, format="%.4f", required=False)
        edited_matrix = st.data_editor(
            st.session_state.matrix_df, num_rows="dynamic", use_container_width=True,
            hide_index=True, key="matrix_editor_sb", column_config=matrix_col_config,
        )
        if edited_matrix is not None and not edited_matrix.equals(st.session_state.matrix_df):
            st.session_state.matrix_df = edited_matrix.reset_index(drop=True)
            st.rerun()


# =============================================================================
# TABS
# =============================================================================
TAB_LABELS = [
    "📋 Dados",
    "⚖️ Motores de Pesos",
    "🔍 AHP",
    "🎯 TOPSIS",
    "📈 PROMETHEE II",
    "⚖️ VIKOR",
    "📊 COPRAS",
    "🚫 ELECTRE III",
    "💡 MAUT",
    "🌐 DEMATEL",
    "🌫️ Fuzzy TOPSIS",
    "🧮 Fuzzy AHP",
    "📊 Gráficos",
    "🏆 Dashboard",
    "📄 Relatório",
]
tabs = st.tabs(TAB_LABELS)


# =============================================================================
# TAB 1: DADOS
# =============================================================================
with tabs[0]:
    st.header("📋 Dados de Entrada")
    theory_box(
        "Como começar",
        """
        <ol>
            <li>Na <b>barra lateral</b>, escolha a <b>fonte de dados</b>:
                <ul>
                    <li><b>Demo</b>: 3 casos pré-definidos (MCG, Fornecedor, Investimento)</li>
                    <li><b>Manual</b>: defina N alts × N crits OU cole valores do Excel (separado por TAB)</li>
                    <li><b>Excel</b>: carregue um ficheiro .xlsx</li>
                </ul>
            </li>
            <li>Ajuste tipos (max/min) e pesos no editor de critérios</li>
            <li>Edite valores no editor de matriz</li>
            <li>Configure a <b>sensibilidade</b> (slider na sidebar)</li>
            <li>Veja resultados em cada aba de modelo</li>
        </ol>
        """
    )

    matrix, alts, crits, types = get_decision_matrix()
    if not check_valid_input():
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Alternativas", len(alts))
    c2.metric("Critérios", len(crits))
    c3.metric("Max", types.count("max"))
    c4.metric("Min", types.count("min"))

    st.subheader("Matriz de Decisão")
    display_df = pd.DataFrame(matrix, index=alts, columns=crits)
    st.dataframe(display_df.style.format("{:.4f}").background_gradient(cmap="Blues", axis=0),
                use_container_width=True)

    st.subheader("Critérios e Pesos Activos")
    show_active_weights_banner()

    st.subheader("Heatmap normalizado (min-max, sentido aplicado)")
    norm = normalize_minmax(matrix, types)
    norm_df = pd.DataFrame(norm, index=alts, columns=crits)
    st.dataframe(norm_df.style.format("{:.3f}").background_gradient(cmap="RdYlGn", axis=None),
                use_container_width=True)
    st.caption("1.0 = melhor; 0.0 = pior (com inversão automática para critérios de min).")


# =============================================================================
# TAB 2: MOTORES DE PESOS (SEM AHP — está na aba dedicada)
# =============================================================================
with tabs[1]:
    st.header("⚖️ Motores de Pesos")
    theory_box(
        "4 métodos para calcular pesos automaticamente",
        """
        <p>Em vez de definir pesos manualmente, pode calculá-los matematicamente:</p>
        <ul>
            <li><b>SWING</b>: swing pior→melhor → 100 pontos no mais impactante → pontuações relativas → normalização</li>
            <li><b>SMART</b>: pontuação directa 0-100 (simplificação do SWING)</li>
            <li><b>Entropia de Shannon</b>: pesos pela variabilidade dos dados (objectivo, baseado na matriz)</li>
            <li><b>CRITIC</b>: combina variância + correlações (objectivo, considera dependências)</li>
        </ul>
        <p><b>Nota:</b> o método <b>AHP</b> tem aba dedicada (mais à direita) por ser mais complexo
        (matriz par-a-par + validação de consistência + iterações).</p>
        <p><b>Injecção Global:</b> active o toggle na sidebar para forçar os modelos a usar
        os pesos calculados aqui (ou na aba AHP).</p>
        """
    )

    if not check_valid_input():
        st.stop()

    matrix, alts, crits, types = get_decision_matrix()
    n = len(crits)

    engine = st.radio(
        "Motor a usar:",
        ["SWING", "SMART", "Entropia", "CRITIC"],
        horizontal=True, key="engine_radio"
    )

    # ----------- SWING -----------
    if engine == "SWING":
        st.subheader("🎢 SWING Weighting")
        theory_box(
            "Como funciona (von Winterfeldt & Edwards, 1986)",
            """
            <p>Imagine que tem a alternativa onde <b>todos os critérios estão no pior nível</b>.
            Para cada critério, defina:</p>
            <ul>
                <li><b>Nível pior</b> e <b>nível melhor</b> do critério (limites realistas do problema)</li>
                <li><b>Pontuação SWING</b>: 100 ao mais impactante; relativos aos restantes</li>
            </ul>
            """
        )
        st.latex(r"w_j = \frac{p_j}{\sum_k p_k},\quad p_j \in [0, 100]")

        # Auto-fill pior/melhor a partir da matriz
        swing_key = f"swing_data_{'_'.join(crits)}"
        if swing_key not in st.session_state or len(st.session_state[swing_key]) != n:
            pior = [matrix[:, j].min() if types[j] == "max" else matrix[:, j].max() for j in range(n)]
            melhor = [matrix[:, j].max() if types[j] == "max" else matrix[:, j].min() for j in range(n)]
            st.session_state[swing_key] = pd.DataFrame({
                "Critério": crits,
                "Nível Pior": pior,
                "Nível Melhor": melhor,
                "Swing Score (0-100)": [100] + [50] * (n - 1)
            })

        edited = st.data_editor(
            st.session_state[swing_key], use_container_width=True, hide_index=True,
            key="swing_edit",
            column_config={
                "Critério": st.column_config.TextColumn("Critério", disabled=True),
                "Nível Pior": st.column_config.NumberColumn("Nível Pior", format="%.4f"),
                "Nível Melhor": st.column_config.NumberColumn("Nível Melhor", format="%.4f"),
                "Swing Score (0-100)": st.column_config.NumberColumn(
                    "Swing Score", min_value=0.0, max_value=100.0, step=5.0,
                    help="100 = critério onde o swing pior→melhor é mais impactante"),
            },
            disabled=["Critério"]
        )
        st.session_state[swing_key] = edited

        pts = pd.to_numeric(edited["Swing Score (0-100)"], errors="coerce").fillna(0).values
        w_swing = pts / pts.sum() if pts.sum() > 0 else np.ones(n) / n

        st.markdown("**Pesos SWING calculados:**")
        df_w = pd.DataFrame({"Critério": crits, "Score": pts, "Peso": w_swing,
                             "%": [f"{x*100:.2f}%" for x in w_swing]})
        st.dataframe(df_w.style.format({"Score": "{:.1f}", "Peso": "{:.4f}"}),
                    hide_index=True, use_container_width=True)

        # Gráfico de barras
        fig = px.bar(df_w, x="Critério", y="Peso", text=df_w["%"],
                     color="Peso", color_continuous_scale="Viridis",
                     title="Pesos SWING")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

        st.session_state.engine_weights["SWING"] = w_swing
        st.success(f"💾 Pesos SWING guardados (Σ={w_swing.sum():.4f}). Active a injecção global na sidebar para usar.")

    # ----------- SMART -----------
    elif engine == "SMART":
        st.subheader("📐 SMART — Simple Multi-Attribute Rating Technique")
        theory_box(
            "Como funciona (von Winterfeldt & Edwards, 1986)",
            """
            <p>Simplificação directa do SWING: atribua <b>pontuação 0-100</b> a cada critério
            conforme a importância. Não há comparação de "swings" — é classificação directa.</p>
            """
        )
        st.latex(r"w_j = \frac{p_j}{\sum_k p_k}")

        smart_key = f"smart_data_{'_'.join(crits)}"
        if smart_key not in st.session_state or len(st.session_state[smart_key]) != n:
            st.session_state[smart_key] = pd.DataFrame({
                "Critério": crits,
                "Pontuação (0-100)": [80] * n
            })
        edited = st.data_editor(
            st.session_state[smart_key], use_container_width=True, hide_index=True, key="smart_edit",
            column_config={
                "Critério": st.column_config.TextColumn("Critério", disabled=True),
                "Pontuação (0-100)": st.column_config.NumberColumn(
                    "Pontuação", min_value=0.0, max_value=100.0, step=5.0),
            }, disabled=["Critério"]
        )
        st.session_state[smart_key] = edited
        pts = pd.to_numeric(edited["Pontuação (0-100)"], errors="coerce").fillna(0).values
        w_smart = pts / pts.sum() if pts.sum() > 0 else np.ones(n) / n
        df_w = pd.DataFrame({"Critério": crits, "Score": pts, "Peso": w_smart,
                             "%": [f"{x*100:.2f}%" for x in w_smart]})
        st.dataframe(df_w.style.format({"Score": "{:.1f}", "Peso": "{:.4f}"}),
                    hide_index=True, use_container_width=True)
        fig = px.bar(df_w, x="Critério", y="Peso", text=df_w["%"],
                     color="Peso", color_continuous_scale="Plasma", title="Pesos SMART")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
        st.session_state.engine_weights["SMART"] = w_smart
        st.success(f"💾 Pesos SMART guardados (Σ={w_smart.sum():.4f}).")

    # ----------- ENTROPIA -----------
    elif engine == "Entropia":
        st.subheader("📊 Entropia de Shannon")
        theory_box(
            "Como funciona (Shannon, 1948)",
            """
            <p>Mede a <b>quantidade de informação</b> de cada critério via variabilidade.
            Critérios com mais variabilidade nos dados → maior peso (mais informação para discriminar).</p>
            <p><b>É 100% objectivo</b>: depende só da matriz, sem julgamentos.</p>
            """
        )
        st.markdown("**Passo 1 — Normalização (max → proporção; min → inverso)**")
        st.latex(r"x'_{ij} = \frac{x_{ij}}{\sum_i x_{ij}}\;\text{(max)};\quad x'_{ij} = \frac{1/x_{ij}}{\sum_i 1/x_{ij}}\;\text{(min)}")

        try:
            m = matrix.shape[0]
            X_norm = np.zeros_like(matrix, dtype=float)
            for j in range(n):
                if types[j] == "max":
                    s = matrix[:, j].sum()
                    X_norm[:, j] = matrix[:, j] / s if s > 0 else 1/m
                else:
                    inv = 1.0 / np.where(matrix[:, j] == 0, 1e-9, matrix[:, j])
                    X_norm[:, j] = inv / inv.sum() if inv.sum() > 0 else 1/m

            st.markdown("**Passo 2 — Entropia E_j**")
            st.latex(r"E_j = -k \sum_i x'_{ij} \ln(x'_{ij}),\quad k = 1/\ln(m)")
            k = 1.0 / np.log(m) if m > 1 else 1.0
            E = np.array([-k * np.sum(np.where(X_norm[:, j] > 0, X_norm[:, j] * np.log(X_norm[:, j]), 0))
                          for j in range(n)])

            st.markdown("**Passo 3 — Divergência e pesos**")
            st.latex(r"d_j = 1 - E_j,\quad w_j = d_j / \sum_k d_k")
            d = 1 - E
            w_ent = d / d.sum() if d.sum() > 0 else np.ones(n) / n
            df_w = pd.DataFrame({
                "Critério": crits, "Entropia E_j": E, "Divergência d_j": d,
                "Peso": w_ent, "%": [f"{x*100:.2f}%" for x in w_ent]
            })
            st.dataframe(df_w.style.format({"Entropia E_j": "{:.4f}", "Divergência d_j": "{:.4f}", "Peso": "{:.4f}"}),
                        hide_index=True, use_container_width=True)
            fig = px.bar(df_w, x="Critério", y="Peso", text=df_w["%"],
                         color="Peso", color_continuous_scale="Greens", title="Pesos Entropia")
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)
            st.session_state.engine_weights["Entropia"] = w_ent
            st.success(f"💾 Pesos Entropia guardados (Σ={w_ent.sum():.4f}).")
        except Exception as e:
            st.error(f"Erro Entropia: {e}")

    # ----------- CRITIC -----------
    else:  # CRITIC
        st.subheader("🔬 CRITIC")
        theory_box(
            "Como funciona (Diakoulaki, 1995)",
            """
            <p>Combina dois efeitos:</p>
            <ul>
                <li><b>Contraste</b>: desvio-padrão σ (variabilidade)</li>
                <li><b>Conflito</b>: correlação Pearson com outros critérios</li>
            </ul>
            <p>Pesos maiores para critérios com alta variabilidade <b>e</b> baixa correlação com outros
            (informação única, não redundante).</p>
            """
        )
        st.markdown("**Passo 1 — Normalização min-max**")
        st.latex(r"r_{ij} \in [0,1] \text{ via min-max}")
        try:
            R = normalize_minmax(matrix, types)
            st.markdown("**Passo 2 — σ e correlação Pearson**")
            sigma = R.std(axis=0, ddof=0)
            corr = np.corrcoef(R.T)
            corr = np.nan_to_num(corr)
            st.markdown("**Passo 3 — Conflito C_j e pesos**")
            st.latex(r"C_j = \sigma_j \sum_k (1 - r_{jk}),\quad w_j = C_j / \sum_l C_l")
            conflict = (1 - corr).sum(axis=1)
            C = sigma * conflict
            w_c = C / C.sum() if C.sum() > 0 else np.ones(n) / n
            df_w = pd.DataFrame({
                "Critério": crits, "σ_j": sigma, "Σ(1-r_jk)": conflict, "C_j": C,
                "Peso": w_c, "%": [f"{x*100:.2f}%" for x in w_c]
            })
            st.dataframe(df_w.style.format({"σ_j": "{:.4f}", "Σ(1-r_jk)": "{:.4f}", "C_j": "{:.4f}", "Peso": "{:.4f}"}),
                        hide_index=True, use_container_width=True)
            with st.expander("Ver matriz de correlação"):
                st.dataframe(pd.DataFrame(corr, index=crits, columns=crits)
                              .style.format("{:.3f}").background_gradient(cmap="RdBu_r", vmin=-1, vmax=1),
                            use_container_width=True)
            fig = px.bar(df_w, x="Critério", y="Peso", text=df_w["%"],
                         color="Peso", color_continuous_scale="Oranges", title="Pesos CRITIC")
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)
            st.session_state.engine_weights["CRITIC"] = w_c
            st.success(f"💾 Pesos CRITIC guardados (Σ={w_c.sum():.4f}).")
        except Exception as e:
            st.error(f"Erro CRITIC: {e}")

    # Comparação
    st.markdown("---")
    st.subheader("📊 Comparação dos motores calculados")
    if st.session_state.engine_weights:
        comp = {"Critério": crits}
        for eng, w in st.session_state.engine_weights.items():
            if len(w) == n: comp[eng] = w
        df_comp = pd.DataFrame(comp)
        st.dataframe(df_comp.style.format({c: "{:.4f}" for c in df_comp.columns if c != "Critério"})
                              .background_gradient(cmap="Blues", axis=None,
                                                   subset=[c for c in df_comp.columns if c != "Critério"]),
                    hide_index=True, use_container_width=True)
        # Radar dos motores
        try:
            fig = go.Figure()
            for eng in df_comp.columns:
                if eng != "Critério":
                    fig.add_trace(go.Scatterpolar(
                        r=list(df_comp[eng]) + [df_comp[eng].iloc[0]],
                        theta=crits + [crits[0]],
                        fill="toself", name=eng
                    ))
            fig.update_layout(title="Comparação dos motores (radar)",
                              polar=dict(radialaxis=dict(visible=True, range=[0, max(df_comp[df_comp.columns[1:]].values.max(), 0.5)])),
                              height=420)
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass
    else:
        st.caption("Nenhum motor calculado ainda. Use as opções acima.")


# =============================================================================
# TAB 3: AHP (FULL — matriz Saaty + consistência + iterações)
# =============================================================================
with tabs[2]:
    st.header("🔍 AHP — Analytic Hierarchy Process (Saaty, 1980)")

    theory_box(
        "Teoria condensada",
        """
        <p>Determina pesos via <b>comparação par-a-par</b> usando a <b>escala Saaty 1-9</b>:</p>
        <ul>
            <li>1 = igual · 3 = moderadamente · 5 = fortemente · 7 = muito fortemente · 9 = extremamente</li>
            <li>a<sub>ji</sub> = 1/a<sub>ij</sub> (recíproco automático); diagonal = 1</li>
            <li>Vector de pesos = média geométrica das linhas (normalizada)</li>
            <li><b>Validação obrigatória</b>: CR = CI/RI < 0.10 (senão, REVER julgamentos)</li>
        </ul>
        <p><b>Iteração:</b> se CR ≥ 0.10, a app identifica o <b>par mais inconsistente</b> e
        sugere o valor que reduz CR. Pode aplicar a sugestão ou ajustar manualmente até convergir.</p>
        """
    )

    if not check_valid_input():
        st.stop()

    matrix, alts, crits, types = get_decision_matrix()
    n = len(crits)
    RI_TABLE = {1: 0, 2: 0, 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41,
                9: 1.45, 10: 1.49, 11: 1.51, 12: 1.54, 13: 1.56, 14: 1.57, 15: 1.59}

    step_header("Passo 1: Matriz de Comparação Par-a-Par")
    st.latex(r"A = [a_{ij}],\quad a_{ji} = 1/a_{ij},\quad a_{ii} = 1")
    st.markdown("**Edite APENAS o triângulo superior (i < j). O inferior é recíproco automático.**")

    ahp_key = f"ahp_full_{'_'.join(crits)}"
    if ahp_key not in st.session_state:
        st.session_state[ahp_key] = pd.DataFrame(np.ones((n, n)), index=crits, columns=crits)
    if st.session_state[ahp_key].shape != (n, n):
        st.session_state[ahp_key] = pd.DataFrame(np.ones((n, n)), index=crits, columns=crits)

    edited_pw = st.data_editor(
        st.session_state[ahp_key],
        use_container_width=True, key="ahp_pw_full_editor",
        column_config={c: st.column_config.NumberColumn(c, min_value=1/9, max_value=9.0, step=0.5, format="%.4f") for c in crits}
    )

    A = edited_pw.values.astype(float).copy()
    for i in range(n):
        A[i, i] = 1.0
        for j in range(n):
            if i < j and A[i, j] > 0:
                A[j, i] = 1.0 / A[i, j]
    st.session_state[ahp_key] = pd.DataFrame(A, index=crits, columns=crits)

    step_header("Passo 2: Cálculo do Vector de Pesos (média geométrica)")
    st.latex(r"w_i = \frac{(\prod_j a_{ij})^{1/n}}{\sum_k (\prod_j a_{kj})^{1/n}}")
    geomean = np.prod(A, axis=1) ** (1.0 / n)
    w_ahp = geomean / geomean.sum()
    st.dataframe(pd.DataFrame({"Critério": crits, "Peso w_j": w_ahp,
                               "%": [f"{x*100:.2f}%" for x in w_ahp]})
                  .style.format({"Peso w_j": "{:.4f}"}),
                hide_index=True, use_container_width=True)

    step_header("Passo 3: Verificação de Consistência")
    st.latex(r"\lambda_{max},\;CI = \frac{\lambda_{max}-n}{n-1},\;CR = CI/RI(n)")
    Aw = A @ w_ahp
    lam_max = (Aw / np.where(w_ahp == 0, 1e-9, w_ahp)).mean()
    CI = (lam_max - n) / (n - 1) if n > 1 else 0
    RI = RI_TABLE.get(n, 1.59)
    CR = CI / RI if RI > 0 else 0

    cc1, cc2, cc3, cc4 = st.columns(4)
    cc1.metric("n", n)
    cc2.metric("λ_max", f"{lam_max:.4f}")
    cc3.metric("CI", f"{CI:.4f}")
    cc4.metric("CR", f"{CR:.4f}",
              delta="✓ Consistente" if CR < 0.10 else "✗ Inconsistente",
              delta_color="normal" if CR < 0.10 else "inverse")

    # =========== ITERAÇÃO PARA REDUZIR CR ===========
    if CR >= 0.10:
        st.markdown(
            f'<div class="warning-box"><b>⚠️ CR = {CR:.4f} ≥ 0.10 — INCONSISTENTE.</b><br>'
            'A teoria de Saaty exige <b>CR < 0.10</b>. Reveja os julgamentos par-a-par. '
            'A app calcula abaixo o par <b>mais problemático</b> e sugere o valor que reduz CR.</div>',
            unsafe_allow_html=True
        )

        # Encontrar o par mais inconsistente: maior |log(a_ij) - log(w_i/w_j)|
        worst_i, worst_j, worst_dev = -1, -1, 0
        suggested_value = 1.0
        for i in range(n):
            for j in range(i+1, n):
                if w_ahp[j] != 0:
                    expected = w_ahp[i] / w_ahp[j]
                    observed = A[i, j]
                    if expected > 0 and observed > 0:
                        dev = abs(np.log(observed / expected))
                        if dev > worst_dev:
                            worst_dev = dev
                            worst_i, worst_j = i, j
                            # Sugerir valor que aproxima o ideal — arredondar para escala Saaty
                            ideal = expected
                            saaty_scale = [1/9, 1/7, 1/5, 1/3, 1/2, 1, 2, 3, 5, 7, 9]
                            suggested_value = min(saaty_scale, key=lambda x: abs(np.log(x) - np.log(ideal)))

        if worst_i >= 0:
            st.markdown(f"### 🔧 Sugestão de Iteração")
            colA, colB, colC = st.columns(3)
            colA.metric("Par problemático", f"{crits[worst_i]} vs {crits[worst_j]}")
            colB.metric("Valor actual", f"{A[worst_i, worst_j]:.2f}")
            colC.metric("Valor sugerido (Saaty)", f"{suggested_value:.4f}",
                       delta=f"Δ = {suggested_value - A[worst_i, worst_j]:+.2f}")

            if st.button(f"✏️ Aplicar sugestão ({crits[worst_i]} vs {crits[worst_j]} → {suggested_value:.2f})",
                        type="primary"):
                new_df = st.session_state[ahp_key].copy()
                new_df.iloc[worst_i, worst_j] = suggested_value
                new_df.iloc[worst_j, worst_i] = 1.0 / suggested_value
                st.session_state[ahp_key] = new_df
                st.session_state.ahp_history.append({
                    "iteration": len(st.session_state.ahp_history) + 1,
                    "CR_before": CR, "pair": f"{crits[worst_i]} vs {crits[worst_j]}",
                    "old_value": A[worst_i, worst_j], "new_value": suggested_value
                })
                st.success("✓ Sugestão aplicada. Recarregue a aba para ver novo CR.")
                st.rerun()
    else:
        st.markdown(
            f'<div class="result-box">✅ <b>Matriz CONSISTENTE</b> — CR = {CR:.4f} < 0.10. Pesos AHP válidos.</div>',
            unsafe_allow_html=True
        )

    if st.session_state.ahp_history:
        with st.expander("📜 Histórico de iterações AHP"):
            st.dataframe(pd.DataFrame(st.session_state.ahp_history), hide_index=True, use_container_width=True)

    # Guardar pesos AHP
    st.session_state.engine_weights["AHP"] = w_ahp

    # ============== RANKING das alternativas usando pesos AHP ==============
    st.markdown("---")
    step_header("Passo 4: Ranking das Alternativas")
    st.latex(r"S_i = \sum_{j=1}^n w_j^{AHP} \cdot u_j(x_{ij})")
    st.markdown("(utilidade min-max com inversão para Custos)")

    # Aqui usamos os pesos AHP (não os activos — porque esta é a aba AHP)
    U = normalize_minmax(matrix, types)
    S = (U * w_ahp).sum(axis=1)
    rank = pd.Series(S).rank(ascending=False, method='min').astype(int).values
    df_res = pd.DataFrame({"Alternativa": alts, "Score AHP": S, "% do máx": S / S.max() * 100 if S.max() > 0 else S,
                           "Ranking": rank}).sort_values("Ranking")
    st.dataframe(df_res.style.format({"Score AHP": "{:.4f}", "% do máx": "{:.1f}%"})
                  .background_gradient(cmap="RdYlGn", subset=["Score AHP"]),
                hide_index=True, use_container_width=True)

    best = df_res.iloc[0]["Alternativa"]
    st.markdown(
        f'<div class="result-box">🏆 Melhor segundo AHP: <b>{best}</b> '
        f'(Score = {df_res.iloc[0]["Score AHP"]:.4f}) '
        f'| CR = {CR:.4f} {"✓" if CR < 0.10 else "✗"}</div>',
        unsafe_allow_html=True
    )

    store_result("AHP", S, rank, higher_is_better=True)

    # Sensibilidade
    def ahp_score_fn(w):
        U = normalize_minmax(matrix, types)
        return (U * w).sum(axis=1)
    render_sensitivity(ahp_score_fn, alts, crits, w_ahp, higher_is_better=True, key_suffix="ahp")


# =============================================================================
# TAB 4: TOPSIS
# =============================================================================
with tabs[3]:
    st.header("🎯 TOPSIS")
    theory_box("Teoria (Hwang & Yoon, 1981)",
        """<p>Método compensatório baseado em <b>distâncias</b> à solução ideal A⁺ e anti-ideal A⁻.
        A melhor alternativa é simultaneamente <b>mais próxima de A⁺ e mais afastada de A⁻</b>.</p>""")

    if not check_valid_input(): st.stop()
    matrix, alts, crits, types = get_decision_matrix()
    weights = get_active_weights()
    show_active_weights_banner()

    def topsis_calc(W):
        R = normalize_vector(matrix); V = R * W
        Ap = np.array([V[:, j].max() if types[j] == "max" else V[:, j].min() for j in range(len(crits))])
        An = np.array([V[:, j].min() if types[j] == "max" else V[:, j].max() for j in range(len(crits))])
        Dp = np.sqrt(((V - Ap) ** 2).sum(axis=1)); Dn = np.sqrt(((V - An) ** 2).sum(axis=1))
        denom = np.where(Dp + Dn == 0, 1e-9, Dp + Dn)
        return Dn / denom, R, V, Ap, An, Dp, Dn

    CC, R, V, Ap, An, Dp, Dn = topsis_calc(weights)

    step_header("Passo 1 & 2: Matriz Normalizada (vectorial Euclidiana)")
    st.latex(r"r_{ij} = x_{ij} / \sqrt{\sum_k x_{kj}^2}")
    st.dataframe(pd.DataFrame(R, index=alts, columns=crits).style.format("{:.4f}"), use_container_width=True)

    step_header("Passo 3: Matriz Ponderada")
    st.latex(r"v_{ij} = w_j \cdot r_{ij}")
    st.dataframe(pd.DataFrame(V, index=alts, columns=crits).style.format("{:.4f}"), use_container_width=True)

    step_header("Passo 4: Soluções Ideal A⁺ e Anti-Ideal A⁻")
    st.dataframe(pd.DataFrame({"Critério": crits, "Tipo": types, "A⁺": Ap, "A⁻": An})
                  .style.format({"A⁺": "{:.4f}", "A⁻": "{:.4f}"}),
                hide_index=True, use_container_width=True)

    step_header("Passo 5 & 6: Distâncias e CC*")
    st.latex(r"D_i^{\pm} = \sqrt{\sum_j (v_{ij} - A_j^{\pm})^2};\quad CC_i = D_i^- / (D_i^+ + D_i^-)")
    rank = pd.Series(CC).rank(ascending=False, method='min').astype(int).values
    df_res = pd.DataFrame({"Alternativa": alts, "D⁺": Dp, "D⁻": Dn, "CC*": CC, "Ranking": rank}).sort_values("Ranking")
    st.dataframe(df_res.style.format({"D⁺": "{:.4f}", "D⁻": "{:.4f}", "CC*": "{:.4f}"})
                  .background_gradient(cmap="RdYlGn", subset=["CC*"]),
                hide_index=True, use_container_width=True)
    best = df_res.iloc[0]["Alternativa"]
    st.markdown(f'<div class="result-box">🏆 Melhor segundo TOPSIS: <b>{best}</b> (CC* = {df_res.iloc[0]["CC*"]:.4f})</div>', unsafe_allow_html=True)
    store_result("TOPSIS", CC, rank, True)

    render_sensitivity(lambda w: topsis_calc(w)[0], alts, crits, weights, True, "topsis")


# =============================================================================
# TAB 5: PROMETHEE II
# =============================================================================
with tabs[4]:
    st.header("📈 PROMETHEE II")
    theory_box("Teoria (Brans, 1985)",
        """<p>Método <b>não-compensatório</b> baseado em fluxos de preferência par-a-par.
        Para cada par (a, b), agrega preferências em φ(a) = φ⁺(a) − φ⁻(a) ∈ [-1, 1].</p>
        <p><b>Funções de preferência</b>: Tipo I (Usual), Tipo V (Linear), Tipo VI (Gaussiana).</p>""")

    if not check_valid_input(): st.stop()
    matrix, alts, crits, types = get_decision_matrix()
    weights = get_active_weights()
    show_active_weights_banner()
    m = len(alts); n = len(crits)

    pref_type = st.radio("Função de preferência:", ["Tipo I (Usual)", "Tipo V (Linear)", "Tipo VI (Gaussiana)"],
                         horizontal=True, key="prom_pref")

    def pref(d, ftype, p=1.0, sigma=1.0):
        if d <= 0: return 0.0
        if ftype.startswith("Tipo I"): return 1.0
        elif ftype.startswith("Tipo V"): return min(d / p, 1.0)
        else: return 1.0 - np.exp(-d**2 / (2 * sigma**2))

    def prom_calc(W):
        params_p = [(matrix[:, j].max() - matrix[:, j].min()) * 0.5 if matrix[:, j].max() > matrix[:, j].min() else 1.0 for j in range(n)]
        params_s = [(matrix[:, j].max() - matrix[:, j].min()) * 0.3 if matrix[:, j].max() > matrix[:, j].min() else 1.0 for j in range(n)]
        pi = np.zeros((m, m))
        for a in range(m):
            for b in range(m):
                if a == b: continue
                for j in range(n):
                    d = matrix[a, j] - matrix[b, j] if types[j] == "max" else matrix[b, j] - matrix[a, j]
                    pi[a, b] += W[j] * pref(d, pref_type, params_p[j], params_s[j])
        phi_p = pi.sum(axis=1) / (m - 1) if m > 1 else pi.sum(axis=1)
        phi_n = pi.sum(axis=0) / (m - 1) if m > 1 else pi.sum(axis=0)
        return phi_p - phi_n, pi, phi_p, phi_n

    phi, pi, phi_p, phi_n = prom_calc(weights)

    step_header("Passo 1: Função de Preferência (P_j)")
    if pref_type.startswith("Tipo I"):
        st.latex(r"P_j(a,b) = 1 \text{ se } a > b, \text{ senão } 0")
    elif pref_type.startswith("Tipo V"):
        st.latex(r"P_j = d/p \text{ se } 0 < d < p, \text{ senão } 0 \text{ ou } 1")
    else:
        st.latex(r"P_j = 1 - e^{-d^2/(2\sigma^2)},\;\sigma = 30\% \text{ do intervalo}")

    step_header("Passo 2: Matriz π(a, b) — preferência agregada")
    st.latex(r"\pi(a,b) = \sum_j w_j P_j(a,b)")
    st.dataframe(pd.DataFrame(pi, index=alts, columns=alts).style.format("{:.4f}")
                  .background_gradient(cmap="Greens"), use_container_width=True)

    step_header("Passo 3: Fluxos φ⁺, φ⁻ e φ líquido")
    st.latex(r"\phi^{\pm}(a) = \frac{1}{m-1}\sum_{b\ne a} \pi(a,b) / \pi(b,a);\quad \phi = \phi^+ - \phi^-")
    rank = pd.Series(phi).rank(ascending=False, method='min').astype(int).values
    df_res = pd.DataFrame({"Alternativa": alts, "φ⁺": phi_p, "φ⁻": phi_n, "φ líquido": phi, "Ranking": rank}).sort_values("Ranking")
    st.dataframe(df_res.style.format({"φ⁺": "{:.4f}", "φ⁻": "{:.4f}", "φ líquido": "{:.4f}"})
                  .background_gradient(cmap="RdYlGn", subset=["φ líquido"]),
                hide_index=True, use_container_width=True)
    best = df_res.iloc[0]["Alternativa"]
    st.markdown(f'<div class="result-box">🏆 Melhor segundo PROMETHEE II: <b>{best}</b> (φ = {df_res.iloc[0]["φ líquido"]:.4f})</div>', unsafe_allow_html=True)
    store_result("PROMETHEE II", phi, rank, True)

    render_sensitivity(lambda w: prom_calc(w)[0], alts, crits, weights, True, "prom")


# =============================================================================
# TAB 6: VIKOR
# =============================================================================
with tabs[5]:
    st.header("⚖️ VIKOR")
    theory_box("Teoria (Opricovic & Tzeng, 2004)",
        """<p>Procura <b>solução de compromisso</b> entre utilidade global (S) e arrependimento individual (R).
        Q_i combina ambos via v ∈ [0,1]; menor Q = melhor.</p>""")

    if not check_valid_input(): st.stop()
    matrix, alts, crits, types = get_decision_matrix()
    weights = get_active_weights()
    show_active_weights_banner()

    v_p = st.slider("Parâmetro v:", 0.0, 1.0, 0.5, 0.05, key="vikor_v")

    def vikor_calc(W, v=0.5):
        n_c = len(crits)
        fb = np.array([matrix[:, j].max() if types[j] == "max" else matrix[:, j].min() for j in range(n_c)])
        fw = np.array([matrix[:, j].min() if types[j] == "max" else matrix[:, j].max() for j in range(n_c)])
        den = np.where(fb - fw == 0, 1e-9, fb - fw)
        terms = np.zeros_like(matrix, dtype=float)
        for j in range(n_c): terms[:, j] = W[j] * np.abs(fb[j] - matrix[:, j]) / abs(den[j])
        S = terms.sum(axis=1); R = terms.max(axis=1)
        Sr = (S.max() - S.min()) if S.max() != S.min() else 1e-9
        Rr = (R.max() - R.min()) if R.max() != R.min() else 1e-9
        Q = v * (S - S.min()) / Sr + (1 - v) * (R - R.min()) / Rr
        return Q, S, R, fb, fw

    Q, S, R, fb, fw = vikor_calc(weights, v_p)

    step_header("Passo 1: Melhores f* e Piores f⁻")
    st.dataframe(pd.DataFrame({"Critério": crits, "f*": fb, "f⁻": fw})
                  .style.format({"f*": "{:.4f}", "f⁻": "{:.4f}"}),
                hide_index=True, use_container_width=True)

    step_header("Passo 2: Índices S e R")
    st.latex(r"S_i = \sum_j w_j (f_j^* - f_{ij})/(f_j^* - f_j^-);\;\;R_i = \max_j[\cdot]")

    step_header("Passo 3: Índice de Compromisso Q")
    st.latex(r"Q_i = v\frac{S_i-S^*}{S^- - S^*} + (1-v)\frac{R_i-R^*}{R^- - R^*}")
    rank = pd.Series(Q).rank(ascending=True, method='min').astype(int).values
    df_res = pd.DataFrame({"Alternativa": alts, "S": S, "R": R, "Q": Q, "Ranking (menor Q)": rank}).sort_values("Ranking (menor Q)")
    st.dataframe(df_res.style.format({"S": "{:.4f}", "R": "{:.4f}", "Q": "{:.4f}"})
                  .background_gradient(cmap="RdYlGn_r", subset=["Q"]),
                hide_index=True, use_container_width=True)

    # Condições C1 e C2
    sorted_q = np.sort(Q)
    if len(sorted_q) >= 2:
        dq = sorted_q[1] - sorted_q[0]; thresh = 1.0 / max(len(alts) - 1, 1)
        c1 = dq >= thresh
        bi = int(np.argmin(Q)); si = int(np.argmin(S)); ri = int(np.argmin(R))
        c2 = (bi == si) or (bi == ri)
        st.markdown(f"**C1 (vantagem)**: ΔQ = {dq:.4f} vs 1/(J-1) = {thresh:.4f} → {'✅' if c1 else '❌'}")
        st.markdown(f"**C2 (estabilidade)**: melhor em S ou R → {'✅' if c2 else '❌'}")

    best = df_res.iloc[0]["Alternativa"]
    st.markdown(f'<div class="result-box">🏆 Melhor segundo VIKOR: <b>{best}</b> (Q = {df_res.iloc[0]["Q"]:.4f})</div>', unsafe_allow_html=True)
    store_result("VIKOR", -Q, rank, True)  # -Q porque menor Q = melhor mas higher_is_better=True

    render_sensitivity(lambda w: -vikor_calc(w, v_p)[0], alts, crits, weights, True, "vikor")


# =============================================================================
# TAB 7: COPRAS
# =============================================================================
with tabs[6]:
    st.header("📊 COPRAS")
    theory_box("Teoria (Zavadskas & Kaklauskas, 1996)",
        """<p>Avalia alternativas como função proporcional entre <b>benefícios (S⁺)</b> e <b>custos (S⁻)</b>.
        Resultado: índice Q_i e grau de utilidade U_i (%) onde 100% = óptimo absoluto.</p>""")

    if not check_valid_input(): st.stop()
    matrix, alts, crits, types = get_decision_matrix()
    weights = get_active_weights()
    show_active_weights_banner()

    def copras_calc(W):
        Xn = normalize_sum(matrix); V = Xn * W
        bi = [j for j, t in enumerate(types) if t == "max"]
        ci = [j for j, t in enumerate(types) if t == "min"]
        Sp = V[:, bi].sum(axis=1) if bi else np.zeros(len(alts))
        Sm = V[:, ci].sum(axis=1) if ci else np.zeros(len(alts))
        if Sm.sum() > 0 and (Sm > 0).all():
            sm = Sm.min(); ssm = Sm.sum(); si = (sm / Sm).sum()
            Q = Sp + (sm * ssm) / (Sm * si) if si > 0 else Sp
        else:
            Q = Sp
        U = (Q / Q.max() * 100) if Q.max() > 0 else Q * 0
        return Q, U, Xn, V, Sp, Sm

    Q, U, Xn, V, Sp, Sm = copras_calc(weights)

    step_header("Passo 1: Normalização por Soma")
    st.latex(r"x'_{ij} = x_{ij} / \sum_i x_{ij}")
    st.dataframe(pd.DataFrame(Xn, index=alts, columns=crits).style.format("{:.4f}"), use_container_width=True)

    step_header("Passo 2: Matriz Ponderada")
    st.latex(r"\hat{x}_{ij} = w_j \cdot x'_{ij}")
    st.dataframe(pd.DataFrame(V, index=alts, columns=crits).style.format("{:.4f}"), use_container_width=True)

    step_header("Passo 3: S⁺ (Benefícios) e S⁻ (Custos)")
    st.dataframe(pd.DataFrame({"Alternativa": alts, "S⁺": Sp, "S⁻": Sm})
                  .style.format({"S⁺": "{:.4f}", "S⁻": "{:.4f}"}),
                hide_index=True, use_container_width=True)

    step_header("Passo 4 & 5: Q_i (fórmula oficial) e U_i (%)")
    st.latex(r"Q_i = S_i^+ + \frac{S_{\min}^- \cdot \sum_i S_i^-}{S_i^- \cdot \sum_i (S_{\min}^-/S_i^-)};\quad U_i = Q_i/Q_{\max} \times 100")
    rank = pd.Series(Q).rank(ascending=False, method='min').astype(int).values
    df_res = pd.DataFrame({"Alternativa": alts, "S⁺": Sp, "S⁻": Sm, "Q_i": Q, "U_i (%)": U, "Ranking": rank}).sort_values("Ranking")
    st.dataframe(df_res.style.format({"S⁺": "{:.4f}", "S⁻": "{:.4f}", "Q_i": "{:.4f}", "U_i (%)": "{:.2f}"})
                  .background_gradient(cmap="RdYlGn", subset=["U_i (%)"]),
                hide_index=True, use_container_width=True)
    best = df_res.iloc[0]["Alternativa"]
    st.markdown(f'<div class="result-box">🏆 Melhor segundo COPRAS: <b>{best}</b> (U = {df_res.iloc[0]["U_i (%)"]:.2f}%)</div>', unsafe_allow_html=True)
    store_result("COPRAS", Q, rank, True)

    render_sensitivity(lambda w: copras_calc(w)[0], alts, crits, weights, True, "copras")


# =============================================================================
# TAB 8: ELECTRE III
# =============================================================================
with tabs[7]:
    st.header("🚫 ELECTRE III")
    theory_box("Teoria (Roy, 1968+)",
        """<p>Método de <b>sobreclassificação</b>. Para cada par (a, b), avalia se há
        evidência suficiente que a "supera" b, usando 3 limiares: <b>q</b> (indiferença), <b>p</b> (preferência),
        <b>v</b> (veto). Não força ranking total — permite incomparabilidades.</p>""")

    if not check_valid_input(): st.stop()
    matrix, alts, crits, types = get_decision_matrix()
    weights = get_active_weights()
    show_active_weights_banner()
    n = len(crits); m = len(alts)

    st.markdown("**Limiares % do intervalo de cada critério:**")
    c1, c2, c3 = st.columns(3)
    qp = c1.slider("q (indiferença) %", 0, 30, 5, 1, key="elec_q")
    pp = c2.slider("p (preferência) %", 5, 50, 20, 1, key="elec_p")
    vp = c3.slider("v (veto) %", 30, 80, 50, 1, key="elec_v")

    def elec_calc(W):
        q = np.zeros(n); p = np.zeros(n); v = np.zeros(n)
        for j in range(n):
            rng = matrix[:, j].max() - matrix[:, j].min()
            q[j] = rng * qp / 100; p[j] = rng * pp / 100; v[j] = rng * vp / 100
            if p[j] <= q[j]: p[j] = q[j] + 0.001
            if v[j] <= p[j]: v[j] = p[j] + 0.001
        cP = np.zeros((m, m, n)); dP = np.zeros((m, m, n))
        for a in range(m):
            for b in range(m):
                if a == b: continue
                for j in range(n):
                    ga, gb = (matrix[a, j], matrix[b, j]) if types[j] == "max" else (-matrix[a, j], -matrix[b, j])
                    diff = ga + q[j] - gb; diff2 = ga + p[j] - gb
                    if diff >= 0: cP[a, b, j] = 1.0
                    elif diff2 <= 0: cP[a, b, j] = 0.0
                    else: cP[a, b, j] = (p[j] + ga - gb) / (p[j] - q[j])
                    dif = gb - ga - p[j]; dif2 = gb - ga - v[j]
                    if dif <= 0: dP[a, b, j] = 0.0
                    elif dif2 >= 0: dP[a, b, j] = 1.0
                    else: dP[a, b, j] = (gb - ga - p[j]) / (v[j] - p[j])
        C = (cP * W).sum(axis=2)
        Sc = C.copy()
        for a in range(m):
            for b in range(m):
                if a == b: continue
                for j in range(n):
                    if dP[a, b, j] > C[a, b]:
                        Sc[a, b] *= (1 - dP[a, b, j]) / (1 - C[a, b]) if C[a, b] < 1 else 0
        return Sc, C, q, p, v, dP.max(axis=2)

    Sc, C, q_abs, p_abs, v_abs, dmax = elec_calc(weights)

    step_header("Passo 1: Limiares Absolutos por Critério")
    st.dataframe(pd.DataFrame({"Critério": crits, "q": q_abs, "p": p_abs, "v": v_abs})
                  .style.format({"q": "{:.4f}", "p": "{:.4f}", "v": "{:.4f}"}),
                hide_index=True, use_container_width=True)

    step_header("Passo 2: Concordância Global C(a, b)")
    st.latex(r"C(a,b) = \sum_j w_j c_j(a,b)")
    st.dataframe(pd.DataFrame(C, index=alts, columns=alts).style.format("{:.3f}").background_gradient(cmap="Greens"),
                use_container_width=True)

    step_header("Passo 3: Matriz de Credibilidade S(a, b)")
    st.dataframe(pd.DataFrame(Sc, index=alts, columns=alts).style.format("{:.3f}").background_gradient(cmap="RdYlGn"),
                use_container_width=True)

    step_header("Passo 4: Ranking por Dominância Líquida")
    cutoff = st.slider("Limiar de corte λ:", 0.5, 0.95, 0.7, 0.05, key="elec_lam")
    outrank = (Sc >= cutoff) & (np.eye(m) == 0)
    net = outrank.sum(axis=1) - outrank.sum(axis=0)
    rank = pd.Series(net).rank(ascending=False, method='min').astype(int).values
    df_res = pd.DataFrame({"Alternativa": alts, "Sobreclassifica": outrank.sum(axis=1),
                           "Sobreclassificada por": outrank.sum(axis=0),
                           "Dominância líquida": net, "Ranking": rank}).sort_values("Ranking")
    st.dataframe(df_res.style.background_gradient(cmap="RdYlGn", subset=["Dominância líquida"]),
                hide_index=True, use_container_width=True)
    best = df_res.iloc[0]["Alternativa"]
    st.markdown(f'<div class="result-box">🏆 Melhor segundo ELECTRE III: <b>{best}</b></div>', unsafe_allow_html=True)
    store_result("ELECTRE III", net, rank, True)

    def elec_score_fn(w):
        Sc2, *_ = elec_calc(w)
        out = (Sc2 >= cutoff) & (np.eye(m) == 0)
        return out.sum(axis=1) - out.sum(axis=0)
    render_sensitivity(elec_score_fn, alts, crits, weights, True, "elec")


# =============================================================================
# TAB 9: MAUT
# =============================================================================
with tabs[8]:
    st.header("💡 MAUT")
    theory_box("Teoria (Keeney & Raiffa, 1976)",
        """<p>Cada valor é convertido em <b>utilidade</b> u_j(x) ∈ [0,1] via função (linear, exponencial, potência),
        e agregado: U_i = Σ w_j · u_j(x_ij). Maior U = melhor.</p>""")

    if not check_valid_input(): st.stop()
    matrix, alts, crits, types = get_decision_matrix()
    weights = get_active_weights()
    show_active_weights_banner()

    func = st.radio("Função de utilidade:", ["Linear", "Exponencial (k=2)", "Potência (p=0.5, côncava)", "Potência (p=2, convexa)"],
                    horizontal=True, key="maut_func")

    def maut_calc(W):
        U0 = normalize_minmax(matrix, types)
        if func == "Linear": U = U0
        elif func.startswith("Exp"): U = (1 - np.exp(-2 * U0)) / (1 - np.exp(-2))
        elif func.startswith("Potência (p=0.5"): U = U0 ** 0.5
        else: U = U0 ** 2
        return (U * W).sum(axis=1), U

    S, U = maut_calc(weights)

    step_header("Passo 1: Utilidades Parciais u_j(x_ij)")
    if func == "Linear": st.latex(r"u(x) = (x - x_{min})/(x_{max} - x_{min})")
    elif func.startswith("Exp"): st.latex(r"u(x) = (1 - e^{-2\tilde{x}})/(1 - e^{-2})")
    elif "p=0.5" in func: st.latex(r"u(x) = \tilde{x}^{0.5} \text{ (côncava — favorece ganhos pequenos)}")
    else: st.latex(r"u(x) = \tilde{x}^{2} \text{ (convexa — penaliza valores baixos)}")
    st.dataframe(pd.DataFrame(U, index=alts, columns=crits).style.format("{:.4f}"), use_container_width=True)

    step_header("Passo 2: Utilidade Global U_i")
    st.latex(r"U_i = \sum_j w_j \cdot u_j(x_{ij})")
    rank = pd.Series(S).rank(ascending=False, method='min').astype(int).values
    df_res = pd.DataFrame({"Alternativa": alts, "U_i": S, "% do máx": S / S.max() * 100 if S.max() > 0 else S, "Ranking": rank}).sort_values("Ranking")
    st.dataframe(df_res.style.format({"U_i": "{:.4f}", "% do máx": "{:.1f}%"})
                  .background_gradient(cmap="RdYlGn", subset=["U_i"]),
                hide_index=True, use_container_width=True)
    best = df_res.iloc[0]["Alternativa"]
    st.markdown(f'<div class="result-box">🏆 Melhor segundo MAUT: <b>{best}</b> (U = {df_res.iloc[0]["U_i"]:.4f})</div>', unsafe_allow_html=True)
    store_result("MAUT", S, rank, True)

    render_sensitivity(lambda w: maut_calc(w)[0], alts, crits, weights, True, "maut")


# =============================================================================
# TAB 10: DEMATEL
# =============================================================================
with tabs[9]:
    st.header("🌐 DEMATEL")
    theory_box("Teoria (Gabus & Fontela, 1972)",
        """<p>Modela <b>relações causa-efeito</b> entre critérios. Aqui, na ausência de elicitação directa,
        usa-se correlação absoluta como proxy. Output: R+C (proeminência) e R−C (causa/efeito).</p>""")

    if not check_valid_input(): st.stop()
    matrix, alts, crits, types = get_decision_matrix()
    weights = get_active_weights()
    show_active_weights_banner()
    n = len(crits)

    def dem_calc(W):
        try:
            Z = np.abs(np.corrcoef(matrix.T)); Z = np.nan_to_num(Z); np.fill_diagonal(Z, 0)
        except Exception: Z = np.zeros((n, n))
        s = max(Z.sum(axis=1).max(), Z.sum(axis=0).max(), 1e-9); X = Z / s
        try: T = X @ np.linalg.inv(np.eye(n) - X)
        except Exception: T = np.eye(n)
        R = T.sum(axis=1); C = T.sum(axis=0)
        prom = R + C; rel = R - C
        if prom.sum() > 0:
            Wa = W * prom; Wa = Wa / Wa.sum()
        else: Wa = W
        U = normalize_minmax(matrix, types); S = (U * Wa).sum(axis=1)
        return S, T, R, C, prom, rel, Wa

    S, T, R_v, C_v, prom, rel, Wa = dem_calc(weights)

    step_header("Passo 1: Matriz Total T = X(I - X)⁻¹")
    st.dataframe(pd.DataFrame(T, index=crits, columns=crits).style.format("{:.4f}").background_gradient(cmap="Blues"),
                use_container_width=True)

    step_header("Passo 2: R+C (Proeminência) e R−C (Causa/Efeito)")
    df_rc = pd.DataFrame({"Critério": crits, "R": R_v, "C": C_v, "R+C": prom, "R-C": rel,
                          "Tipo": ["🎯 Causa" if r > 0 else "📥 Efeito" for r in rel]})
    st.dataframe(df_rc.style.format({"R": "{:.4f}", "C": "{:.4f}", "R+C": "{:.4f}", "R-C": "{:.4f}"}),
                hide_index=True, use_container_width=True)

    # Diagrama causa-efeito
    fig = go.Figure(go.Scatter(x=prom, y=rel, mode="markers+text", text=crits, textposition="top center",
                                marker=dict(size=14, color=rel, colorscale="RdBu", showscale=True, colorbar=dict(title="R-C"))))
    fig.add_hline(y=0, line_dash="dash", line_color="grey")
    fig.update_layout(title="Diagrama Causa-Efeito DEMATEL",
                      xaxis_title="Proeminência (R+C)", yaxis_title="Relação (R-C)",
                      height=400)
    st.plotly_chart(fig, use_container_width=True)

    step_header("Passo 3: Ranking com Pesos Ajustados")
    rank = pd.Series(S).rank(ascending=False, method='min').astype(int).values
    df_res = pd.DataFrame({"Alternativa": alts, "Score": S, "Ranking": rank}).sort_values("Ranking")
    st.dataframe(df_res.style.format({"Score": "{:.4f}"}).background_gradient(cmap="RdYlGn", subset=["Score"]),
                hide_index=True, use_container_width=True)
    best = df_res.iloc[0]["Alternativa"]
    st.markdown(f'<div class="result-box">🏆 Melhor segundo DEMATEL: <b>{best}</b></div>', unsafe_allow_html=True)
    store_result("DEMATEL", S, rank, True)

    render_sensitivity(lambda w: dem_calc(w)[0], alts, crits, weights, True, "dem")


# =============================================================================
# TAB 11: FUZZY TOPSIS
# =============================================================================
with tabs[10]:
    st.header("🌫️ Fuzzy TOPSIS")
    theory_box("Teoria (Chen, 2000)",
        """<p>TOPSIS com <b>números fuzzy triangulares</b> (l, m, u). Captura imprecisão.
        Distância pelo método do vértice; CC_i mantém-se como ranking.</p>""")

    if not check_valid_input(): st.stop()
    matrix, alts, crits, types = get_decision_matrix()
    weights = get_active_weights()
    show_active_weights_banner()

    spread = st.slider("Spread fuzzy (% do valor):", 5, 40, 15, 5, key="ft_spread")

    def ftopsis_calc(W, s_pct=15):
        s = s_pct / 100.0
        L = matrix * (1 - s); M = matrix.copy(); U_ = matrix * (1 + s)
        nc = matrix.shape[1]
        Ln = np.zeros_like(L); Mn = np.zeros_like(M); Un = np.zeros_like(U_)
        for j in range(nc):
            if types[j] == "max":
                d = max(U_[:, j].max(), 1e-9)
                Ln[:, j] = L[:, j] / d; Mn[:, j] = M[:, j] / d; Un[:, j] = U_[:, j] / d
            else:
                num = max(L[:, j].min(), 1e-9)
                Ln[:, j] = num / np.where(U_[:, j] == 0, 1e-9, U_[:, j])
                Mn[:, j] = num / np.where(M[:, j] == 0, 1e-9, M[:, j])
                Un[:, j] = num / np.where(L[:, j] == 0, 1e-9, L[:, j])
        Lw, Mw, Uw = Ln * W, Mn * W, Un * W
        fpis = np.array([Uw[:, j].max() for j in range(nc)])
        fnis = np.array([Lw[:, j].min() for j in range(nc)])
        def vd(a, b, c, t): return np.sqrt(((a-t)**2 + (b-t)**2 + (c-t)**2) / 3.0)
        Dp = np.zeros(len(alts)); Dn = np.zeros(len(alts))
        for i in range(len(alts)):
            for j in range(nc):
                Dp[i] += vd(Lw[i, j], Mw[i, j], Uw[i, j], fpis[j])
                Dn[i] += vd(Lw[i, j], Mw[i, j], Uw[i, j], fnis[j])
        den = np.where(Dp + Dn == 0, 1e-9, Dp + Dn)
        return Dn / den, Dp, Dn

    CC, Dp, Dn = ftopsis_calc(weights, spread)

    step_header("Passo 1-3: Matriz Fuzzy + FPIS/FNIS + Distâncias")
    st.latex(r"\tilde{x}_{ij} = (x(1-s),\, x,\, x(1+s));\quad d_v(\tilde{a},\tilde{b}) = \sqrt{\frac{1}{3}[(l_a-l_b)^2 + (m_a-m_b)^2 + (u_a-u_b)^2]}")

    step_header("Passo 4: CC* e Ranking")
    st.latex(r"CC_i = D_i^- / (D_i^+ + D_i^-)")
    rank = pd.Series(CC).rank(ascending=False, method='min').astype(int).values
    df_res = pd.DataFrame({"Alternativa": alts, "D⁺": Dp, "D⁻": Dn, "CC*": CC, "Ranking": rank}).sort_values("Ranking")
    st.dataframe(df_res.style.format({"D⁺": "{:.4f}", "D⁻": "{:.4f}", "CC*": "{:.4f}"})
                  .background_gradient(cmap="RdYlGn", subset=["CC*"]),
                hide_index=True, use_container_width=True)
    best = df_res.iloc[0]["Alternativa"]
    st.markdown(f'<div class="result-box">🏆 Melhor segundo Fuzzy TOPSIS: <b>{best}</b> (CC* = {df_res.iloc[0]["CC*"]:.4f})</div>', unsafe_allow_html=True)
    store_result("Fuzzy TOPSIS", CC, rank, True)

    render_sensitivity(lambda w: ftopsis_calc(w, spread)[0], alts, crits, weights, True, "ft")


# =============================================================================
# TAB 12: FUZZY AHP
# =============================================================================
with tabs[11]:
    st.header("🧮 Fuzzy AHP")
    theory_box("Teoria (Chang, 1996)",
        """<p>AHP com TFN nos pesos. Defuzzificação por centro de área: w<sub>crisp</sub> = (l+m+u)/3.</p>""")

    if not check_valid_input(): st.stop()
    matrix, alts, crits, types = get_decision_matrix()
    weights = get_active_weights()
    show_active_weights_banner()

    fs = st.slider("Spread fuzzy dos pesos (%):", 5, 40, 20, 5, key="fa_spread")

    def fahp_calc(W, s_pct=20):
        s = s_pct / 100.0; L = W * (1 - s); M = W.copy(); U_ = W * (1 + s)
        Wc = (L + M + U_) / 3; Wc = Wc / Wc.sum()
        Um = normalize_minmax(matrix, types)
        return (Um * Wc).sum(axis=1), Wc, L, M, U_

    S, Wc, L, M, U_ = fahp_calc(weights, fs)

    step_header("Passo 1: TFN dos pesos")
    st.latex(r"\tilde{w}_j = (w(1-s),\, w,\, w(1+s))")
    st.dataframe(pd.DataFrame({"Critério": crits, "l": L, "m": M, "u": U_})
                  .style.format({"l": "{:.4f}", "m": "{:.4f}", "u": "{:.4f}"}),
                hide_index=True, use_container_width=True)

    step_header("Passo 2: Defuzzificação")
    st.latex(r"w^{crisp} = (l + m + u) / 3; \text{ normalizar para } \sum = 1")
    st.dataframe(pd.DataFrame({"Critério": crits, "w crisp": Wc, "%": [f"{x*100:.2f}%" for x in Wc]})
                  .style.format({"w crisp": "{:.4f}"}),
                hide_index=True, use_container_width=True)

    step_header("Passo 3: Ranking")
    rank = pd.Series(S).rank(ascending=False, method='min').astype(int).values
    df_res = pd.DataFrame({"Alternativa": alts, "Score": S, "Ranking": rank}).sort_values("Ranking")
    st.dataframe(df_res.style.format({"Score": "{:.4f}"}).background_gradient(cmap="RdYlGn", subset=["Score"]),
                hide_index=True, use_container_width=True)
    best = df_res.iloc[0]["Alternativa"]
    st.markdown(f'<div class="result-box">🏆 Melhor segundo Fuzzy AHP: <b>{best}</b></div>', unsafe_allow_html=True)
    store_result("Fuzzy AHP", S, rank, True)

    render_sensitivity(lambda w: fahp_calc(w, fs)[0], alts, crits, weights, True, "fa")


# =============================================================================
# TAB 13: GRÁFICOS (Plotly bonitos e decisivos)
# =============================================================================
with tabs[12]:
    st.header("📊 Gráficos para Decisão")
    theory_box(
        "Visualizações para uma decisão rápida e clara",
        """
        <p>Esta aba reúne os <b>gráficos mais impactantes</b> para decidir:</p>
        <ul>
            <li><b>Radar</b>: perfil multicritério do Top-3 (forma → identifica trade-offs)</li>
            <li><b>Heatmap de Rankings</b>: como cada modelo posiciona cada alternativa</li>
            <li><b>Tornado de Sensibilidade</b>: que critérios mais influenciam o Top-1</li>
            <li><b>Bar Race</b>: comparação directa de scores entre modelos</li>
        </ul>
        """
    )

    if not check_valid_input(): st.stop()
    matrix, alts, crits, types = get_decision_matrix()
    weights = get_active_weights()

    if not st.session_state.all_results:
        st.warning("⚠️ Nenhum modelo executado. Vá às abas de modelo (TOPSIS, AHP, etc.) primeiro.")
        st.stop()

    methods = list(st.session_state.all_results.keys())

    # ============== GRÁFICO 1: HEATMAP DE RANKINGS ==============
    st.subheader("🗺️ Heatmap de Rankings por Método")
    st.caption("Mostra como cada modelo classifica cada alternativa. Verde = topo, Vermelho = fundo.")
    rank_data = {"Alternativa": alts}
    for m in methods:
        rank_data[m] = st.session_state.all_results[m]["ranking"]
    df_ranks = pd.DataFrame(rank_data).set_index("Alternativa")

    fig_heat = px.imshow(
        df_ranks.values, x=methods, y=alts,
        color_continuous_scale="RdYlGn_r", aspect="auto", text_auto=True,
        labels=dict(x="Modelo", y="Alternativa", color="Ranking"),
        title=f"Rankings das {len(alts)} alternativas em {len(methods)} modelos"
    )
    fig_heat.update_layout(height=max(360, 28 * len(alts)),
                            margin=dict(l=10, r=10, t=50, b=10))
    fig_heat.update_traces(textfont=dict(size=14, color="black"))
    st.plotly_chart(fig_heat, use_container_width=True)

    # ============== GRÁFICO 2: RADAR DO TOP-3 ==============
    st.subheader("🎯 Radar — Perfil Multicritério do Top-3")
    st.caption("Cada alternativa é desenhada como polígono. Áreas maiores = melhor desempenho global.")
    # Calcular top-3 pelo ranking médio
    avg_rank = df_ranks.mean(axis=1)
    top3_alts = avg_rank.sort_values().head(3).index.tolist()
    norm = normalize_minmax(matrix, types)
    norm_df = pd.DataFrame(norm, index=alts, columns=crits)

    fig_radar = go.Figure()
    colors = ["#FFD700", "#C0C0C0", "#CD7F32"]
    medals = ["🥇 1º", "🥈 2º", "🥉 3º"]
    for i, alt in enumerate(top3_alts):
        vals = list(norm_df.loc[alt]) + [norm_df.loc[alt].iloc[0]]
        cats = crits + [crits[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=cats, fill="toself",
            name=f"{medals[i]} {alt}",
            line=dict(color=colors[i], width=2),
            opacity=0.6
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=480, showlegend=True,
        title="Top-3 — perfil normalizado (1=melhor, 0=pior por critério)"
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ============== GRÁFICO 3: TORNADO DE SENSIBILIDADE DO TOP-1 ==============
    st.subheader("🌪️ Tornado de Sensibilidade — Top-1")
    st.caption(f"Para a alternativa Top-1 ({top3_alts[0]}), mostra quanto cada peso afecta o seu rank. "
               "Barras mais longas = critério mais influente.")

    # Usar AHP/TOPSIS como referência (o que estiver disponível primeiro)
    ref_method = "TOPSIS" if "TOPSIS" in st.session_state.all_results else methods[0]
    top1_idx = alts.index(top3_alts[0])
    base_scores = st.session_state.all_results[ref_method]["scores"]
    base_score_top1 = base_scores[top1_idx]
    sens_pct = st.session_state.sensitivity_pct

    tornado_data = []
    for j, crit in enumerate(crits):
        # Variar +sens% e -sens%
        for sign, factor in [("+", 1 + sens_pct/100), ("-", 1 - sens_pct/100)]:
            new_w = weights.copy()
            new_w[j] = weights[j] * factor
            new_w = new_w / new_w.sum()
            # Recalcular com TOPSIS rápido
            R = normalize_vector(matrix); V = R * new_w
            Ap = np.array([V[:, k].max() if types[k] == "max" else V[:, k].min() for k in range(len(crits))])
            An = np.array([V[:, k].min() if types[k] == "max" else V[:, k].max() for k in range(len(crits))])
            Dp = np.sqrt(((V - Ap) ** 2).sum(axis=1)); Dn = np.sqrt(((V - An) ** 2).sum(axis=1))
            den = np.where(Dp + Dn == 0, 1e-9, Dp + Dn)
            new_cc = Dn / den
            delta = new_cc[top1_idx] - base_score_top1
            tornado_data.append({"Critério": crit, "Direcção": f"{sign}{sens_pct}%", "Δ Score": delta})
    df_tornado = pd.DataFrame(tornado_data)

    fig_tornado = px.bar(df_tornado, x="Δ Score", y="Critério", color="Direcção",
                         orientation="h", barmode="group",
                         color_discrete_map={f"+{sens_pct}%": "#1F77B4", f"-{sens_pct}%": "#FF7F0E"},
                         title=f"Tornado — sensibilidade do score de {top3_alts[0]} ({ref_method})")
    fig_tornado.update_layout(height=max(280, 50 * len(crits)),
                               margin=dict(l=10, r=10, t=50, b=10))
    fig_tornado.add_vline(x=0, line_dash="dash", line_color="grey")
    st.plotly_chart(fig_tornado, use_container_width=True)

    # ============== GRÁFICO 4: SCORES NORMALIZADOS POR MODELO ==============
    st.subheader("📊 Scores Normalizados por Modelo (comparação directa)")
    st.caption("Cada modelo é normalizado para [0,1]. Permite comparar a posição relativa entre métodos.")
    score_data = {"Alternativa": alts}
    for m in methods:
        sc = st.session_state.all_results[m]["scores"]
        if sc.max() > sc.min():
            score_data[m] = (sc - sc.min()) / (sc.max() - sc.min())
        else:
            score_data[m] = np.zeros_like(sc)
    df_sc = pd.DataFrame(score_data)
    # Long format para px
    df_long = df_sc.melt(id_vars="Alternativa", var_name="Modelo", value_name="Score Normalizado")
    fig_sc = px.bar(df_long, x="Alternativa", y="Score Normalizado", color="Modelo",
                    barmode="group", title="Scores normalizados [0,1] por modelo")
    fig_sc.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_sc, use_container_width=True)

    # ============== GRÁFICO 5: CONVERGÊNCIA TOP-3 ==============
    st.subheader("🎯 Convergência Top-3")
    st.caption("Quantos modelos colocam cada alternativa no Top-3? Indicador de robustez consensual.")
    top3_count = (df_ranks <= 3).sum(axis=1).sort_values(ascending=False)
    fig_conv = px.bar(x=top3_count.index, y=top3_count.values,
                      labels={"x": "Alternativa", "y": f"N.º de modelos com Top-3 (de {len(methods)})"},
                      color=top3_count.values, color_continuous_scale="Viridis",
                      title=f"Convergência Top-3 — análise de robustez")
    fig_conv.update_layout(height=380, showlegend=False)
    st.plotly_chart(fig_conv, use_container_width=True)


# =============================================================================
# TAB 14: DASHBOARD CONSOLIDADO
# =============================================================================
with tabs[13]:
    st.header("🏆 Dashboard Consolidado")
    theory_box(
        "Consolidação dos modelos",
        """<p>Aplica-se <b>Borda invertido</b> (média de posições) para agregar todos os modelos.
        A alternativa com <b>menor posição média</b> é a recomendação consensual.</p>"""
    )

    if not check_valid_input(): st.stop()
    matrix, alts, crits, types = get_decision_matrix()
    weights = get_active_weights()
    show_active_weights_banner()

    if not st.session_state.all_results:
        st.warning("⚠️ Execute primeiro pelo menos um modelo (TOPSIS, AHP, etc.).")
        st.stop()

    methods = list(st.session_state.all_results.keys())
    df_dash = pd.DataFrame({"Alternativa": alts})
    for m in methods:
        df_dash[m] = st.session_state.all_results[m]["ranking"]
    df_dash["Posição Média"] = df_dash[methods].mean(axis=1).round(2)
    df_dash["Top-3 em N modelos"] = (df_dash[methods] <= 3).sum(axis=1)
    df_dash["Ranking Final"] = pd.Series(df_dash["Posição Média"]).rank(ascending=True, method='min').astype(int).values
    df_dash = df_dash.sort_values("Ranking Final")

    st.dataframe(df_dash.style.background_gradient(cmap="RdYlGn_r", subset=methods + ["Posição Média", "Ranking Final"]),
                hide_index=True, use_container_width=True)

    st.subheader("🥇 Recomendação Final — Top-3")
    top3 = df_dash.head(3)["Alternativa"].tolist()
    c1, c2, c3 = st.columns(3)
    medals = ["🥇 1º lugar", "🥈 2º lugar", "🥉 3º lugar"]
    for k, (col, medal) in enumerate(zip([c1, c2, c3], medals)):
        if k < len(top3):
            col.metric(medal, top3[k], delta=f"Pos média: {df_dash.iloc[k]['Posição Média']}")

    total_top3 = sum(df_dash.head(3)["Top-3 em N modelos"].values)
    max_conv = 3 * len(methods)
    conv_pct = (total_top3 / max_conv * 100) if max_conv else 0
    st.info(f"**Convergência inter-modelo**: {total_top3}/{max_conv} ({conv_pct:.0f}%)\n\n"
            f"Modelos avaliados: {', '.join(methods)}")


# =============================================================================
# TAB 15: RELATÓRIO
# =============================================================================
with tabs[14]:
    st.header("📄 Relatório Executivo")
    theory_box(
        "Relatório consolidado",
        """<p>Resumo visual e descarregável da análise: dados de entrada, pesos, rankings,
        convergência, robustez e recomendação final.</p>"""
    )

    if not check_valid_input(): st.stop()
    matrix, alts, crits, types = get_decision_matrix()
    weights = get_active_weights()

    if not st.session_state.all_results:
        st.warning("⚠️ Execute pelo menos um modelo antes de gerar o relatório.")
        st.stop()

    methods = list(st.session_state.all_results.keys())
    df_dash = pd.DataFrame({"Alternativa": alts})
    for m in methods:
        df_dash[m] = st.session_state.all_results[m]["ranking"]
    df_dash["Posição Média"] = df_dash[methods].mean(axis=1).round(2)
    df_dash["Ranking Final"] = pd.Series(df_dash["Posição Média"]).rank(ascending=True, method='min').astype(int).values
    df_dash["Top-3 em N modelos"] = (df_dash[methods] <= 3).sum(axis=1)
    df_dash = df_dash.sort_values("Ranking Final")

    top1 = df_dash.iloc[0]["Alternativa"]
    top1_pos_avg = df_dash.iloc[0]["Posição Média"]
    top1_top3_count = df_dash.iloc[0]["Top-3 em N modelos"]
    conv_pct_top1 = (top1_top3_count / len(methods) * 100) if methods else 0

    # ============== CARD DE DECISÃO ==============
    st.markdown(
        f"""<div style="background: linear-gradient(135deg, #1F4E78 0%, #2E75B6 100%);
        color: white; padding: 30px; border-radius: 12px; margin: 20px 0; text-align: center;">
        <div style="font-size: 14px; opacity: 0.9; text-transform: uppercase; letter-spacing: 2px;">Recomendação Final</div>
        <div style="font-size: 56px; font-weight: 700; margin: 10px 0;">🏆 {top1}</div>
        <div style="font-size: 18px; opacity: 0.95;">
            Posição média: <b>{top1_pos_avg}</b> · Top-3 em <b>{top1_top3_count}/{len(methods)}</b> modelos ({conv_pct_top1:.0f}%)
        </div></div>""",
        unsafe_allow_html=True
    )

    # ============== SECÇÃO 1: SUMÁRIO ==============
    st.subheader("1. Sumário Executivo")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Alternativas avaliadas", len(alts))
    c2.metric("Critérios", len(crits))
    c3.metric("Modelos aplicados", len(methods))
    c4.metric("Convergência Top-1", f"{conv_pct_top1:.0f}%")

    if conv_pct_top1 >= 70:
        verdict = "🟢 **ALTA convergência** — recomendação ROBUSTA, decisão com elevado grau de confiança."
        verdict_color = "#388e3c"
    elif conv_pct_top1 >= 40:
        verdict = "🟡 **Convergência MODERADA** — recomendação aceitável, mas analise sensibilidade antes de decidir."
        verdict_color = "#f57c00"
    else:
        verdict = "🔴 **BAIXA convergência** — Top-1 instável. Reveja pesos ou alargue conjunto de alternativas."
        verdict_color = "#c62828"

    st.markdown(f"<div style='padding:12px; background:#f5f5f5; border-left:4px solid {verdict_color}; "
                f"border-radius:4px; font-size:16px;'>{verdict}</div>", unsafe_allow_html=True)

    # ============== SECÇÃO 2: TOP-3 + PERFIL RADAR ==============
    st.subheader("2. Top-3 e Perfil Multicritério")
    top3 = df_dash.head(3)["Alternativa"].tolist()
    norm = normalize_minmax(matrix, types)
    norm_df = pd.DataFrame(norm, index=alts, columns=crits)

    cc1, cc2 = st.columns([2, 3])
    with cc1:
        for k, alt in enumerate(top3):
            medal = ["🥇", "🥈", "🥉"][k]
            pos = df_dash.iloc[k]["Posição Média"]
            tc = df_dash.iloc[k]["Top-3 em N modelos"]
            st.markdown(f"### {medal} **{alt}**  \nPos média: {pos}  ·  Top-3: {tc}/{len(methods)}")
    with cc2:
        fig = go.Figure()
        colors = ["#FFD700", "#C0C0C0", "#CD7F32"]
        for i, alt in enumerate(top3):
            vals = list(norm_df.loc[alt]) + [norm_df.loc[alt].iloc[0]]
            cats = crits + [crits[0]]
            fig.add_trace(go.Scatterpolar(r=vals, theta=cats, fill="toself", name=alt,
                                            line=dict(color=colors[i], width=2), opacity=0.6))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                          height=380, showlegend=True, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # ============== SECÇÃO 3: TABELA CONSOLIDADA ==============
    st.subheader("3. Rankings Detalhados por Modelo")
    st.dataframe(df_dash.style.background_gradient(cmap="RdYlGn_r",
                  subset=methods + ["Posição Média", "Ranking Final"]),
                hide_index=True, use_container_width=True)

    # ============== SECÇÃO 4: PESOS USADOS ==============
    st.subheader("4. Pesos dos Critérios Aplicados")
    eng_src = "Manual" if not st.session_state.global_injection_on else f"Motor: {st.session_state.global_injection_engine}"
    st.caption(f"Fonte dos pesos: **{eng_src}**")
    df_w = pd.DataFrame({"Critério": crits, "Tipo": types, "Peso": weights,
                          "%": [f"{x*100:.2f}%" for x in weights]})
    cc1, cc2 = st.columns([2, 3])
    with cc1:
        st.dataframe(df_w.style.format({"Peso": "{:.4f}"}), hide_index=True, use_container_width=True)
    with cc2:
        fig_pie = px.pie(df_w, values="Peso", names="Critério", title="Distribuição dos pesos",
                          color_discrete_sequence=px.colors.qualitative.Set3)
        fig_pie.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    # ============== SECÇÃO 5: ROBUSTEZ ==============
    st.subheader("5. Análise de Robustez")
    st.caption(f"Variação ± {st.session_state.sensitivity_pct}% nos pesos (de cada critério isoladamente).")

    # Calcular robustez com TOPSIS (rápido)
    def quick_topsis(W):
        R = normalize_vector(matrix); V = R * W
        Ap = np.array([V[:, j].max() if types[j] == "max" else V[:, j].min() for j in range(len(crits))])
        An = np.array([V[:, j].min() if types[j] == "max" else V[:, j].max() for j in range(len(crits))])
        Dp = np.sqrt(((V - Ap) ** 2).sum(axis=1)); Dn = np.sqrt(((V - An) ** 2).sum(axis=1))
        return Dn / np.where(Dp + Dn == 0, 1e-9, Dp + Dn)

    base = quick_topsis(weights)
    base_rk = pd.Series(base).rank(ascending=False, method='min').astype(int).values
    sp = st.session_state.sensitivity_pct
    n_inv = []
    for i_alt in range(len(alts)):
        count = 0
        for j in range(len(crits)):
            for f in [1 + sp/100, 1 - sp/100]:
                nw = weights.copy(); nw[j] *= f
                nw = nw / nw.sum()
                sc = quick_topsis(nw)
                rk = pd.Series(sc).rank(ascending=False, method='min').astype(int).values
                if rk[i_alt] != base_rk[i_alt]: count += 1
        n_inv.append(count)
    df_rob = pd.DataFrame({
        "Alternativa": alts, "Rank Base": base_rk, "Inversões (12 cenários)": n_inv,
        "Robustez": ["🟢 ESTÁVEL" if c == 0 else ("🟡 MODERADA" if c <= 3 else "🔴 INSTÁVEL") for c in n_inv]
    }).sort_values("Rank Base")
    st.dataframe(df_rob, hide_index=True, use_container_width=True)

    # ============== SECÇÃO 6: METODOLOGIA APLICADA ==============
    st.subheader("6. Metodologia")
    st.markdown(f"""
    Foram aplicados **{len(methods)} modelos MCDM**:

    {chr(10).join(f"- **{m}**" for m in methods)}

    Os rankings foram agregados por **Borda invertido** (média de posições): a alternativa
    com menor posição média é a recomendada.

    A análise de sensibilidade ± {st.session_state.sensitivity_pct}% foi aplicada a cada modelo
    para verificar robustez dos resultados.
    """)

    # ============== EXPORT ==============
    st.subheader("📥 Exportar Relatório")

    # CSV consolidado
    csv_buffer = StringIO()
    df_dash.to_csv(csv_buffer, index=False)

    # Excel consolidado
    excel_buf = BytesIO()
    with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
        df_dash.to_excel(writer, sheet_name="Rankings", index=False)
        pd.DataFrame(matrix, index=alts, columns=crits).to_excel(writer, sheet_name="Matriz")
        df_w.to_excel(writer, sheet_name="Pesos", index=False)
        df_rob.to_excel(writer, sheet_name="Robustez", index=False)
        # Scores por método
        sc_data = {"Alternativa": alts}
        for m in methods:
            sc_data[m] = st.session_state.all_results[m]["scores"]
        pd.DataFrame(sc_data).to_excel(writer, sheet_name="Scores", index=False)
    excel_buf.seek(0)

    # Markdown report
    md_lines = [
        f"# Relatório MCDM\n",
        f"**Data:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n",
        f"## 🏆 Recomendação Final: {top1}\n",
        f"- Posição média: {top1_pos_avg}",
        f"- Top-3 em {top1_top3_count}/{len(methods)} modelos ({conv_pct_top1:.0f}%)",
        f"- {verdict}\n",
        f"## Sumário",
        f"- Alternativas: {len(alts)}",
        f"- Critérios: {len(crits)}",
        f"- Modelos aplicados: {', '.join(methods)}",
        f"- Sensibilidade: ±{st.session_state.sensitivity_pct}%\n",
        f"## Top-3 Recomendado",
    ]
    for k, alt in enumerate(top3):
        medal = ["🥇", "🥈", "🥉"][k]
        md_lines.append(f"{k+1}. {medal} **{alt}** — Pos média: {df_dash.iloc[k]['Posição Média']}, Top-3 em {df_dash.iloc[k]['Top-3 em N modelos']}/{len(methods)}")
    md_lines.append("\n## Rankings por Modelo")
    md_lines.append(df_dash.to_markdown(index=False))
    md_lines.append("\n## Pesos Activos")
    md_lines.append(f"Fonte: {eng_src}\n")
    md_lines.append(df_w.to_markdown(index=False))
    md_lines.append("\n## Robustez")
    md_lines.append(df_rob.to_markdown(index=False))
    md_report = "\n".join(md_lines)

    ec1, ec2, ec3 = st.columns(3)
    ec1.download_button("📥 CSV (rankings)", csv_buffer.getvalue(), "mcdm_rankings.csv", "text/csv",
                        use_container_width=True)
    ec2.download_button("📥 Excel (completo)", excel_buf.getvalue(), "mcdm_relatorio.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True)
    ec3.download_button("📥 Markdown (relatório)", md_report.encode("utf-8"),
                        "mcdm_relatorio.md", "text/markdown", use_container_width=True)
