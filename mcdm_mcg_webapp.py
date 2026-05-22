"""
MCDM Dashboard — Ferramenta de Apoio à Decisão Multicritério
=============================================================
100% autónoma (sem Excel), pedagógica, com motores de pesos e injecção global,
e análise de sensibilidade universal em todos os métodos.

Arquitectura (5 pilares):
  1. Autonomia: matrizes em st.session_state, st.data_editor dinâmico
  2. Pedagogia: teoria no topo de cada aba + fórmulas LaTeX antes das tabelas
  3. Motores de Pesos: AHP, SWING, SMART, Entropia, CRITIC + injecção global
  4. Sensibilidade Universal: render_sensitivity() em todos os modelos
  5. Foco: sem ANP, sem Fuzzy ANP, sem relatórios em texto longo
"""
import streamlit as st
import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURAÇÃO DA PÁGINA
# =============================================================================
st.set_page_config(
    page_title="MCDM Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CSS
# =============================================================================
CSS = """
<style>
.theory-box {
    background: linear-gradient(135deg, #f0f7ff 0%, #e0ecff 100%);
    border-left: 4px solid #1F4E78;
    padding: 16px 20px;
    border-radius: 6px;
    margin: 12px 0 20px 0;
    font-size: 14px;
    line-height: 1.5;
    color: #1F4E78;
}
.theory-box h4 { color: #1F4E78; margin-top: 0; font-size: 16px; font-weight: 700; }
.theory-box ul, .theory-box ol { margin: 8px 0 0 20px; }
.theory-box code { background: #fff; padding: 2px 6px; border-radius: 3px; font-size: 13px; color: #c7254e; }
.step-header {
    background: #2E75B6; color: white; padding: 8px 14px; border-radius: 4px;
    font-weight: 600; margin: 16px 0 8px 0; font-size: 15px;
}
.injection-active {
    background: #fce4ec; border: 2px solid #c2185b; padding: 8px 14px;
    border-radius: 6px; color: #c2185b; font-weight: 700; text-align: center; margin: 12px 0;
}
.stTabs [data-baseweb="tab"] { padding: 10px 18px; font-weight: 600; }
.stTabs [aria-selected="true"] { background-color: #1F4E78 !important; color: white !important; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# =============================================================================
# ESTADO INICIAL
# =============================================================================
def init_state():
    if "criteria_df" not in st.session_state:
        st.session_state.criteria_df = pd.DataFrame({
            "Critério": ["Custo", "Qualidade", "Prazo", "Sustentabilidade"],
            "Tipo": ["min", "max", "min", "max"],
            "Peso Manual": [0.30, 0.30, 0.20, 0.20],
        })
    if "matrix_df" not in st.session_state:
        st.session_state.matrix_df = pd.DataFrame({
            "Alternativa": ["Alt 1", "Alt 2", "Alt 3", "Alt 4", "Alt 5"],
            "Custo": [1200.0, 1500.0, 1100.0, 1300.0, 1400.0],
            "Qualidade": [8.0, 6.0, 9.0, 7.0, 5.0],
            "Prazo": [15.0, 20.0, 18.0, 12.0, 25.0],
            "Sustentabilidade": [7.0, 5.0, 8.0, 6.0, 4.0],
        })
    if "global_injection_on" not in st.session_state:
        st.session_state.global_injection_on = False
    if "global_injection_engine" not in st.session_state:
        st.session_state.global_injection_engine = "AHP"
    if "engine_weights" not in st.session_state:
        st.session_state.engine_weights = {}
    if "sensitivity_pct" not in st.session_state:
        st.session_state.sensitivity_pct = 20

init_state()


# =============================================================================
# HELPERS
# =============================================================================
def get_decision_matrix():
    """Devolve (matrix, alts, crits, types) de forma robusta."""
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
    """Pesos activos: do motor (se injecção ON) ou manuais."""
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
    if w.sum() == 0:
        return np.ones(n) / n
    return w / w.sum()


def show_active_weights_banner():
    """Banner com os pesos activos."""
    w = get_active_weights()
    _, _, crits, _ = get_decision_matrix()
    if st.session_state.global_injection_on:
        engine = st.session_state.global_injection_engine
        st.markdown(
            f'<div class="injection-active">🔌 Injecção Global ACTIVA — usando pesos do motor <b>{engine}</b></div>',
            unsafe_allow_html=True
        )
    df_w = pd.DataFrame({"Critério": crits, "Peso": w, "%": (w * 100).round(2)})
    df_w["%"] = df_w["%"].astype(str) + "%"
    cols = st.columns([3, 1])
    with cols[0]:
        st.dataframe(df_w[["Critério", "%"]], hide_index=True, use_container_width=False)
    with cols[1]:
        st.metric("Σ pesos", f"{w.sum():.4f}")


def theory_box(title, html):
    st.markdown(f'<div class="theory-box"><h4>📚 {title}</h4>{html}</div>', unsafe_allow_html=True)


def step_header(text):
    st.markdown(f'<div class="step-header">{text}</div>', unsafe_allow_html=True)


def check_valid_input():
    """Verifica se há dados suficientes para correr os métodos."""
    matrix, alts, crits, types = get_decision_matrix()
    if len(alts) < 2 or len(crits) < 2:
        st.warning("⚠️ Defina pelo menos 2 alternativas e 2 critérios na aba 'Dados'.")
        return False
    if matrix.size == 0 or np.all(matrix == 0):
        st.warning("⚠️ A matriz de decisão está vazia ou só tem zeros.")
        return False
    return True


# =============================================================================
# NORMALIZAÇÕES
# =============================================================================
def normalize_vector(matrix):
    denom = np.sqrt((matrix ** 2).sum(axis=0))
    denom = np.where(denom == 0, 1, denom)
    return matrix / denom


def normalize_minmax(matrix, types):
    n_alts, n_crits = matrix.shape
    norm = np.zeros_like(matrix, dtype=float)
    for j in range(n_crits):
        col = matrix[:, j]
        mn, mx = col.min(), col.max()
        if mx == mn:
            norm[:, j] = 0.5
        elif types[j] == "max":
            norm[:, j] = (col - mn) / (mx - mn)
        else:
            norm[:, j] = (mx - col) / (mx - mn)
    return norm


def normalize_sum(matrix):
    sums = matrix.sum(axis=0)
    sums = np.where(sums == 0, 1, sums)
    return matrix / sums


# =============================================================================
# FUNÇÃO UNIVERSAL DE SENSIBILIDADE (PILAR 4)
# =============================================================================
def render_sensitivity(score_function, alts, crits, base_weights, higher_is_better=True, key_suffix=""):
    """
    Sensibilidade universal: ±X% no peso de cada critério (renormalizando restantes).
    score_function: f(weights) → np.array de scores
    """
    st.markdown("---")
    st.markdown("### 🎯 Análise de Sensibilidade — ±X% nos Pesos")

    theory_box(
        "Como funciona",
        """
        <p>Variamos o peso de <b>cada critério isoladamente</b> em ±X% e renormalizamos os restantes
        para manter Σw = 1. Para cada cenário recalculamos o ranking:</p>
        <ul>
            <li>🟢 <b>VERDE</b>: a alternativa <b>sobe</b> no ranking (melhora)</li>
            <li>🔴 <b>VERMELHO</b>: a alternativa <b>desce</b> no ranking (piora)</li>
            <li>⚪ Sem cor: ranking inalterado</li>
        </ul>
        """
    )

    variation_pct = st.slider(
        "Variação ± nos pesos (%):", 5, 50, st.session_state.sensitivity_pct, 5,
        key=f"sens_pct_{key_suffix}"
    )

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
                styles.append("background-color: #f0f0f0; font-weight: 600;")
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

    st.dataframe(df_sens.style.apply(style_row, axis=1), use_container_width=True)

    # Robustez por alternativa
    n_changes = []
    base_vals = df_sens["Base"].values
    others = df_sens.drop(columns=["Base"])
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
        "Robustez": ["🟢 Estável" if c == 0 else ("🟡 Moderada" if c <= 2 else "🔴 Instável") for c in n_changes]
    })
    st.markdown("**Resumo de Robustez:**")
    st.dataframe(df_robust, hide_index=True, use_container_width=True)


# =============================================================================
# TÍTULO
# =============================================================================
st.title("📊 MCDM Dashboard")
st.markdown(
    "**Ferramenta de Apoio à Decisão Multicritério** · 100% autónoma · "
    "pedagógica · com motores de pesos, injecção global e sensibilidade universal"
)


# =============================================================================
# SIDEBAR — Editor de Critérios + Matriz (PILAR 1: estado total)
# =============================================================================
with st.sidebar:
    st.header("⚙️ Configuração")

    # Toggle de injecção global
    st.markdown("### 🔌 Injecção Global de Pesos")
    st.session_state.global_injection_on = st.toggle(
        "Activar injecção dos motores de pesos",
        value=st.session_state.global_injection_on,
        help="Quando ON: todos os modelos usam pesos calculados pelo motor escolhido. "
             "Quando OFF: usam os 'Peso Manual' definidos abaixo."
    )
    if st.session_state.global_injection_on:
        available_engines = list(st.session_state.engine_weights.keys())
        if not available_engines:
            st.warning("⚠️ Nenhum motor calculado ainda. Vá à aba 'Motores de Pesos'.")
        else:
            st.session_state.global_injection_engine = st.selectbox(
                "Motor activo:",
                available_engines,
                index=available_engines.index(st.session_state.global_injection_engine)
                      if st.session_state.global_injection_engine in available_engines else 0
            )

    st.markdown("---")
    st.markdown("### 📋 Critérios")
    st.caption("Edite nomes, tipo (max/min) e pesos manuais. Adicione/remova linhas com ➕.")

    edited_crit = st.data_editor(
        st.session_state.criteria_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key="criteria_editor",
        column_config={
            "Critério": st.column_config.TextColumn("Critério", required=True),
            "Tipo": st.column_config.SelectboxColumn(
                "Tipo", options=["max", "min"], required=True,
                help="max = quanto maior melhor; min = quanto menor melhor"
            ),
            "Peso Manual": st.column_config.NumberColumn(
                "Peso Manual", min_value=0.0, max_value=1.0, step=0.01, format="%.4f",
                help="Σ deve ser ~1.0 (será renormalizado)"
            ),
        }
    )

    # Sincronização: detectar mudanças nos critérios e ajustar matriz
    if edited_crit is not None and not edited_crit.equals(st.session_state.criteria_df):
        # Filtrar linhas válidas
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

    # Mostrar soma dos pesos
    w_manual = pd.to_numeric(st.session_state.criteria_df["Peso Manual"], errors="coerce").fillna(0)
    if len(w_manual) > 0:
        soma = w_manual.sum()
        if abs(soma - 1.0) > 0.01:
            st.caption(f"⚠️ Σ pesos manuais = **{soma:.4f}** (será renormalizado para 1.0)")
        else:
            st.caption(f"✅ Σ pesos manuais = **{soma:.4f}**")

    st.markdown("---")
    st.markdown("### 🔢 Matriz de Decisão")
    st.caption("Adicione alternativas com ➕. Colunas correspondem aos critérios definidos acima.")

    # Construir column_config dinâmico para NumberColumn em cada critério (PROTECÇÃO)
    crit_names = st.session_state.criteria_df["Critério"].astype(str).tolist()
    matrix_col_config = {
        "Alternativa": st.column_config.TextColumn("Alternativa", required=True, width="small"),
    }
    for crit in crit_names:
        matrix_col_config[crit] = st.column_config.NumberColumn(
            crit, format="%.4f", help=f"Valor numérico para {crit}", required=False
        )

    edited_matrix = st.data_editor(
        st.session_state.matrix_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key="matrix_editor",
        column_config=matrix_col_config,
    )
    if edited_matrix is not None and not edited_matrix.equals(st.session_state.matrix_df):
        st.session_state.matrix_df = edited_matrix.reset_index(drop=True)
        st.rerun()

    st.markdown("---")
    if st.button("🔄 Repor defaults", use_container_width=True):
        for key in ["criteria_df", "matrix_df", "engine_weights", "global_injection_on"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()


# =============================================================================
# TABS PRINCIPAIS
# =============================================================================
TAB_LABELS = [
    "📋 Dados",
    "⚖️ Motores de Pesos",
    "🎯 TOPSIS",
    "📈 PROMETHEE II",
    "⚖️ VIKOR",
    "📊 COPRAS",
    "🔍 AHP",
    "🚫 ELECTRE III",
    "💡 MAUT",
    "🌐 DEMATEL",
    "🌫️ Fuzzy TOPSIS",
    "🧮 Fuzzy AHP",
    "🏆 Dashboard",
]
tabs = st.tabs(TAB_LABELS)


# =============================================================================
# TAB 1: DADOS — Visão geral da matriz e critérios
# =============================================================================
with tabs[0]:
    st.header("📋 Dados de Entrada")

    theory_box(
        "Como definir o problema",
        """
        <p>Defina o seu problema de decisão multicritério na <b>barra lateral</b>:</p>
        <ol>
            <li><b>Critérios</b>: nome, tipo (max/min) e peso manual (Σ=1)</li>
            <li><b>Matriz de Decisão</b>: alternativas vs. critérios (valores numéricos)</li>
            <li>Use o botão <code>➕</code> para adicionar linhas dinamicamente</li>
        </ol>
        <p>O sistema bloqueia inserção de texto nas colunas numéricas — só aceita números —
        para evitar que cálculos rebentem.</p>
        """
    )

    matrix, alts, crits, types = get_decision_matrix()

    if len(alts) < 2 or len(crits) < 2:
        st.warning("⚠️ Defina pelo menos 2 alternativas e 2 critérios na barra lateral.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Alternativas", len(alts))
        c2.metric("Critérios", len(crits))
        c3.metric("Tipos (max/min)", f"{types.count('max')} / {types.count('min')}")

        st.subheader("Matriz de Decisão Actual")
        display_df = pd.DataFrame(matrix, index=alts, columns=crits)
        st.dataframe(display_df.style.format("{:.4f}"), use_container_width=True)

        st.subheader("Critérios e Pesos")
        w_active = get_active_weights()
        crit_summary = pd.DataFrame({
            "Critério": crits,
            "Tipo": types,
            "Peso Manual": pd.to_numeric(
                st.session_state.criteria_df["Peso Manual"], errors="coerce"
            ).fillna(0).values[:len(crits)],
            "Peso Activo": w_active,
        })
        st.dataframe(crit_summary, hide_index=True, use_container_width=True)

        st.subheader("Estatísticas Descritivas")
        stats = display_df.describe().T
        st.dataframe(stats, use_container_width=True)

        st.subheader("Heatmap normalizado (min-max, sentido aplicado)")
        norm = normalize_minmax(matrix, types)
        norm_df = pd.DataFrame(norm, index=alts, columns=crits)
        st.dataframe(
            norm_df.style.format("{:.3f}").background_gradient(cmap="RdYlGn", axis=None),
            use_container_width=True
        )
        st.caption("Valores normalizados em [0, 1]: 1 = melhor, 0 = pior (com inversão para min).")


# =============================================================================
# TAB 2: MOTORES DE PESOS (PILAR 3)
# =============================================================================
with tabs[1]:
    st.header("⚖️ Motores de Pesos")

    theory_box(
        "5 métodos para determinar os pesos dos critérios",
        """
        <p>Em vez de definir pesos manualmente, pode calculá-los matematicamente:</p>
        <ul>
            <li><b>AHP</b>: comparação par-a-par (escala Saaty 1-9) → autovector + CR</li>
            <li><b>SWING</b>: swing do pior→melhor → 100 pontos no mais impactante → pontuações relativas</li>
            <li><b>SMART</b>: pontuação directa 0-100 (mais simples que SWING)</li>
            <li><b>Entropia de Shannon</b>: pesos pela variabilidade dos dados (objectivo)</li>
            <li><b>CRITIC</b>: combina variância + correlações (objectivo)</li>
        </ul>
        <p><b>Injecção Global</b>: active o toggle na barra lateral para forçar os outros modelos
        (TOPSIS, VIKOR, etc.) a usarem os pesos calculados aqui.</p>
        """
    )

    matrix, alts, crits, types = get_decision_matrix()
    if not check_valid_input():
        st.stop()

    n = len(crits)
    engine_tab = st.radio(
        "Seleccione o motor:",
        ["AHP", "SWING", "SMART", "Entropia", "CRITIC"],
        horizontal=True,
        key="engine_selector"
    )

    # ============================ AHP ============================
    if engine_tab == "AHP":
        st.subheader("🔺 AHP — Analytic Hierarchy Process")
        theory_box(
            "Como funciona",
            """
            <p>Construa uma matriz <b>par-a-par</b> A onde a<sub>ij</sub> = quão mais importante é o critério i face a j
            (escala Saaty 1-9). O vector de pesos sai do <b>autovector principal</b>:</p>
            <ul>
                <li>1 = igual; 3 = moderadamente; 5 = fortemente; 7 = muito; 9 = extremamente</li>
                <li>a<sub>ji</sub> = 1/a<sub>ij</sub> (recíproco automático)</li>
                <li>Validar com <b>CR < 0.10</b> (Saaty); senão rever julgamentos</li>
            </ul>
            """
        )

        st.markdown("**Passo 1: Matriz de Comparação Par-a-Par**")
        st.latex(r"A = [a_{ij}],\quad a_{ji} = 1/a_{ij},\quad a_{ii} = 1")

        # Inicializar matriz par-a-par no session_state (por crits)
        ahp_key = f"ahp_matrix_{'_'.join(crits)}"
        if ahp_key not in st.session_state:
            init_pw = np.ones((n, n))
            st.session_state[ahp_key] = pd.DataFrame(init_pw, index=crits, columns=crits)

        # Garantir tamanho correcto
        if st.session_state[ahp_key].shape != (n, n):
            st.session_state[ahp_key] = pd.DataFrame(np.ones((n, n)), index=crits, columns=crits)

        st.caption("Edite APENAS o triângulo superior (i < j). O inferior actualiza-se automaticamente.")
        edited_pw = st.data_editor(
            st.session_state[ahp_key],
            use_container_width=True,
            key="ahp_pw_editor",
            column_config={c: st.column_config.NumberColumn(c, min_value=1/9, max_value=9.0, step=0.5, format="%.4f") for c in crits}
        )

        # Forçar reciprocidade e diagonal
        A = edited_pw.values.astype(float).copy()
        for i in range(n):
            A[i, i] = 1.0
            for j in range(n):
                if i < j and A[i, j] > 0:
                    A[j, i] = 1.0 / A[i, j]
        st.session_state[ahp_key] = pd.DataFrame(A, index=crits, columns=crits)

        st.markdown("**Passo 2: Cálculo do Vector de Pesos (média geométrica das linhas)**")
        st.latex(r"w_i = \frac{\sqrt[n]{\prod_{j=1}^n a_{ij}}}{\sum_{k=1}^n \sqrt[n]{\prod_{j=1}^n a_{kj}}}")
        try:
            geomean = np.prod(A, axis=1) ** (1.0 / n)
            w_ahp = geomean / geomean.sum()
            st.dataframe(pd.DataFrame({"Critério": crits, "Peso AHP": w_ahp, "%": [f"{x*100:.2f}%" for x in w_ahp]}),
                        hide_index=True, use_container_width=True)

            st.markdown("**Passo 3: Verificação de Consistência (Saaty)**")
            st.latex(r"\lambda_{max} = \text{média}(A\cdot w / w),\quad CI = \frac{\lambda_{max}-n}{n-1},\quad CR = CI / RI")
            Aw = A @ w_ahp
            lam_max = (Aw / np.where(w_ahp == 0, 1e-9, w_ahp)).mean()
            CI = (lam_max - n) / (n - 1) if n > 1 else 0
            RI_TABLE = {1: 0, 2: 0, 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41,
                        9: 1.45, 10: 1.49, 11: 1.51, 12: 1.54, 13: 1.56, 14: 1.57, 15: 1.59}
            RI = RI_TABLE.get(n, 1.59)
            CR = CI / RI if RI > 0 else 0
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("n", n)
            c2.metric("λ_max", f"{lam_max:.4f}")
            c3.metric("CI", f"{CI:.4f}")
            c4.metric("CR", f"{CR:.4f}", delta="✓ Consistente" if CR < 0.10 else "✗ Inconsistente",
                     delta_color="normal" if CR < 0.10 else "inverse")
            if CR >= 0.10:
                st.warning(f"⚠️ CR = {CR:.4f} ≥ 0.10 — reveja os julgamentos par-a-par para reduzir inconsistência.")
            else:
                st.success(f"✅ Matriz consistente (CR = {CR:.4f} < 0.10).")

            # Guardar pesos
            st.session_state.engine_weights["AHP"] = w_ahp
            st.success(f"💾 Pesos AHP guardados (disponíveis para injecção global).")
        except Exception as e:
            st.error(f"Erro AHP: {e}")

    # ============================ SWING ============================
    elif engine_tab == "SWING":
        st.subheader("🎢 SWING Weighting")
        theory_box(
            "Como funciona",
            """
            <p>Imagine que tem a alternativa onde <b>todos os critérios estão no pior nível</b>.
            Pergunte: <i>"que critério gostaria de 'fazer swing' do pior para o melhor?"</i>
            Esse critério recebe <b>100 pontos</b>. Os restantes recebem pontuações relativas.</p>
            """
        )
        st.latex(r"w_j = \frac{p_j}{\sum_k p_k},\quad p_j \in [0, 100]")

        swing_key = f"swing_scores_{'_'.join(crits)}"
        if swing_key not in st.session_state or len(st.session_state[swing_key]) != n:
            st.session_state[swing_key] = pd.DataFrame({
                "Critério": crits,
                "Pontuação SWING (0-100)": [100] + [50] * (n - 1)
            })

        edited_swing = st.data_editor(
            st.session_state[swing_key], use_container_width=True, hide_index=True,
            key="swing_editor",
            column_config={
                "Critério": st.column_config.TextColumn("Critério", disabled=True),
                "Pontuação SWING (0-100)": st.column_config.NumberColumn(
                    "Pontuação", min_value=0.0, max_value=100.0, step=5.0, format="%.1f",
                    help="100 = critério mais impactante"
                ),
            },
            disabled=["Critério"]
        )
        st.session_state[swing_key] = edited_swing

        try:
            pts = pd.to_numeric(edited_swing["Pontuação SWING (0-100)"], errors="coerce").fillna(0).values
            if pts.sum() == 0:
                w_swing = np.ones(n) / n
            else:
                w_swing = pts / pts.sum()
            df_w = pd.DataFrame({"Critério": crits, "Pontuação": pts, "Peso SWING": w_swing,
                                "%": [f"{x*100:.2f}%" for x in w_swing]})
            st.dataframe(df_w, hide_index=True, use_container_width=True)
            st.session_state.engine_weights["SWING"] = w_swing
            st.success("💾 Pesos SWING guardados.")
        except Exception as e:
            st.error(f"Erro SWING: {e}")

    # ============================ SMART ============================
    elif engine_tab == "SMART":
        st.subheader("📐 SMART — Simple Multi-Attribute Rating Technique")
        theory_box(
            "Como funciona",
            """
            <p>Simplificação directa do SWING: atribua <b>pontuação 0-100</b> a cada critério
            conforme a sua importância. Não há comparação relativa de "swings" — é uma classificação directa.</p>
            <p>É mais rápido e intuitivo que SWING, mas menos rigoroso.</p>
            """
        )
        st.latex(r"w_j = \frac{p_j}{\sum_k p_k}")

        smart_key = f"smart_scores_{'_'.join(crits)}"
        if smart_key not in st.session_state or len(st.session_state[smart_key]) != n:
            st.session_state[smart_key] = pd.DataFrame({
                "Critério": crits,
                "Pontuação (0-100)": [80] * n
            })

        edited_smart = st.data_editor(
            st.session_state[smart_key], use_container_width=True, hide_index=True,
            key="smart_editor",
            column_config={
                "Critério": st.column_config.TextColumn("Critério", disabled=True),
                "Pontuação (0-100)": st.column_config.NumberColumn(
                    "Pontuação", min_value=0.0, max_value=100.0, step=5.0, format="%.1f"
                ),
            },
            disabled=["Critério"]
        )
        st.session_state[smart_key] = edited_smart

        try:
            pts = pd.to_numeric(edited_smart["Pontuação (0-100)"], errors="coerce").fillna(0).values
            w_smart = pts / pts.sum() if pts.sum() > 0 else np.ones(n) / n
            df_w = pd.DataFrame({"Critério": crits, "Pontuação": pts, "Peso SMART": w_smart,
                                "%": [f"{x*100:.2f}%" for x in w_smart]})
            st.dataframe(df_w, hide_index=True, use_container_width=True)
            st.session_state.engine_weights["SMART"] = w_smart
            st.success("💾 Pesos SMART guardados.")
        except Exception as e:
            st.error(f"Erro SMART: {e}")

    # ============================ ENTROPIA ============================
    elif engine_tab == "Entropia":
        st.subheader("📊 Entropia de Shannon")
        theory_box(
            "Como funciona",
            """
            <p>Mede o <b>conteúdo informacional</b> de cada critério via variabilidade.
            Critérios com mais variabilidade nos dados recebem maior peso (mais informação para discriminar alternativas).</p>
            <p><b>É objectivo</b>: depende apenas da matriz de decisão, não de julgamentos.</p>
            """
        )

        st.markdown("**Passo 1: Normalização por soma (max) ou inverso (min)**")
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

            st.markdown("**Passo 2: Entropia E_j**")
            st.latex(r"E_j = -k \sum_i x'_{ij} \ln(x'_{ij}),\quad k = 1/\ln(m)")
            k = 1.0 / np.log(m)
            E = np.zeros(n)
            for j in range(n):
                vals = X_norm[:, j]
                terms = np.where(vals > 0, vals * np.log(vals), 0)
                E[j] = -k * terms.sum()

            st.markdown("**Passo 3: Divergência e pesos**")
            st.latex(r"d_j = 1 - E_j,\quad w_j = d_j / \sum_k d_k")
            d = 1 - E
            w_ent = d / d.sum() if d.sum() > 0 else np.ones(n) / n

            df_ent = pd.DataFrame({
                "Critério": crits,
                "Entropia E_j": E,
                "Divergência d_j": d,
                "Peso Entropia": w_ent,
                "%": [f"{x*100:.2f}%" for x in w_ent]
            })
            st.dataframe(df_ent.style.format({"Entropia E_j": "{:.4f}", "Divergência d_j": "{:.4f}", "Peso Entropia": "{:.4f}"}),
                        hide_index=True, use_container_width=True)
            st.session_state.engine_weights["Entropia"] = w_ent
            st.success("💾 Pesos Entropia guardados.")
        except Exception as e:
            st.error(f"Erro Entropia: {e}")

    # ============================ CRITIC ============================
    elif engine_tab == "CRITIC":
        st.subheader("🔬 CRITIC — Criteria Importance Through Inter-criteria Correlation")
        theory_box(
            "Como funciona",
            """
            <p>Combina dois efeitos:</p>
            <ul>
                <li><b>Contraste interno</b>: variância dos valores no critério (σ²)</li>
                <li><b>Conflito</b>: correlação com outros critérios — quanto menos correlacionado, mais informação única traz</li>
            </ul>
            <p>É objectivo, como a Entropia, mas considera dependências entre critérios.</p>
            """
        )

        st.markdown("**Passo 1: Normalização min-max**")
        st.latex(r"r_{ij} = \frac{x_{ij} - \min_i x_{ij}}{\max_i x_{ij} - \min_i x_{ij}}\;\text{(max)};\;\;1 - r_{ij}\;\text{(min)}")

        try:
            R = normalize_minmax(matrix, types)
            st.markdown("**Passo 2: Variância e correlação Pearson**")
            st.latex(r"\sigma_j = \sqrt{\text{Var}(r_{:,j})},\quad r(j,k) = \text{Pearson}(r_{:,j}, r_{:,k})")
            sigma = R.std(axis=0, ddof=0)
            corr = np.corrcoef(R.T)
            if np.isnan(corr).any():
                corr = np.nan_to_num(corr)

            st.markdown("**Passo 3: Conflito C_j e pesos**")
            st.latex(r"C_j = \sigma_j \sum_{k=1}^n (1 - r_{jk}),\quad w_j = C_j / \sum_l C_l")
            conflict_sum = (1 - corr).sum(axis=1)
            C = sigma * conflict_sum
            w_critic = C / C.sum() if C.sum() > 0 else np.ones(n) / n

            df_c = pd.DataFrame({
                "Critério": crits,
                "σ_j": sigma,
                "Σ(1-r_jk)": conflict_sum,
                "C_j": C,
                "Peso CRITIC": w_critic,
                "%": [f"{x*100:.2f}%" for x in w_critic]
            })
            st.dataframe(df_c.style.format({"σ_j": "{:.4f}", "Σ(1-r_jk)": "{:.4f}", "C_j": "{:.4f}", "Peso CRITIC": "{:.4f}"}),
                        hide_index=True, use_container_width=True)

            with st.expander("Ver matriz de correlação"):
                st.dataframe(pd.DataFrame(corr, index=crits, columns=crits).style.format("{:.3f}").background_gradient(cmap="RdBu_r", vmin=-1, vmax=1),
                           use_container_width=True)

            st.session_state.engine_weights["CRITIC"] = w_critic
            st.success("💾 Pesos CRITIC guardados.")
        except Exception as e:
            st.error(f"Erro CRITIC: {e}")

    # =========== Comparação de motores e botão de injecção rápida ============
    st.markdown("---")
    st.subheader("📊 Comparação dos motores calculados")
    if st.session_state.engine_weights:
        comp = {"Critério": crits}
        for engine, w in st.session_state.engine_weights.items():
            if len(w) == n:
                comp[engine] = w
        df_comp = pd.DataFrame(comp)
        st.dataframe(
            df_comp.style.format({c: "{:.4f}" for c in df_comp.columns if c != "Critério"})
                          .background_gradient(cmap="Blues", axis=None, subset=[c for c in df_comp.columns if c != "Critério"]),
            hide_index=True, use_container_width=True
        )

        st.info(
            "💡 Para usar um destes vectores em todos os modelos (TOPSIS, VIKOR, etc.), "
            "active o **🔌 Toggle de Injecção Global** na barra lateral e escolha o motor."
        )
    else:
        st.caption("Nenhum motor calculado ainda. Use as abas acima.")


# =============================================================================
# TAB 3: TOPSIS
# =============================================================================
with tabs[2]:
    st.header("🎯 TOPSIS — Technique for Order Preference by Similarity to Ideal Solution")
    theory_box(
        "Teoria condensada (Hwang & Yoon, 1981)",
        """
        <p>Método compensatório baseado em <b>distâncias geométricas</b>. A melhor alternativa é a
        que está simultaneamente <b>mais perto da solução ideal</b> e <b>mais longe da anti-ideal</b>.</p>
        <p><b>6 passos:</b></p>
        <ol>
            <li>Matriz de decisão + pesos</li>
            <li><b>Normalização vectorial</b> (Euclidiana)</li>
            <li><b>Matriz ponderada</b>: v<sub>ij</sub> = w<sub>j</sub> · r<sub>ij</sub></li>
            <li><b>Soluções ideal A⁺ e anti-ideal A⁻</b> (max/min conforme critério)</li>
            <li><b>Distâncias</b> Euclidianas D⁺ e D⁻ a A⁺ e A⁻</li>
            <li><b>Coeficiente de proximidade</b> CC = D⁻/(D⁺+D⁻); ranking decrescente</li>
        </ol>
        """
    )

    if not check_valid_input():
        st.stop()

    matrix, alts, crits, types = get_decision_matrix()
    weights = get_active_weights()
    show_active_weights_banner()

    def topsis_calculate(W):
        R = normalize_vector(matrix)
        V = R * W
        A_plus = np.array([V[:, j].max() if types[j] == "max" else V[:, j].min() for j in range(len(crits))])
        A_minus = np.array([V[:, j].min() if types[j] == "max" else V[:, j].max() for j in range(len(crits))])
        D_plus = np.sqrt(((V - A_plus) ** 2).sum(axis=1))
        D_minus = np.sqrt(((V - A_minus) ** 2).sum(axis=1))
        denom = D_plus + D_minus
        denom = np.where(denom == 0, 1e-9, denom)
        CC = D_minus / denom
        return CC, R, V, A_plus, A_minus, D_plus, D_minus

    CC, R, V, A_plus, A_minus, D_plus, D_minus = topsis_calculate(weights)

    step_header("Passo 1 & 2: Matriz Normalizada (vectorial Euclidiana)")
    st.latex(r"r_{ij} = \frac{x_{ij}}{\sqrt{\sum_{k=1}^m x_{kj}^2}}")
    st.dataframe(pd.DataFrame(R, index=alts, columns=crits).style.format("{:.4f}"),
                use_container_width=True)

    step_header("Passo 3: Matriz Ponderada")
    st.latex(r"v_{ij} = w_j \cdot r_{ij}")
    st.dataframe(pd.DataFrame(V, index=alts, columns=crits).style.format("{:.4f}"),
                use_container_width=True)

    step_header("Passo 4: Soluções Ideal A⁺ e Anti-Ideal A⁻")
    st.latex(r"A_j^+ = \max_i v_{ij} \text{ (max) ou } \min_i v_{ij} \text{ (min)};\quad A_j^- = \text{oposto}")
    st.dataframe(pd.DataFrame({"Critério": crits, "Tipo": types, "A⁺": A_plus, "A⁻": A_minus})
                  .style.format({"A⁺": "{:.4f}", "A⁻": "{:.4f}"}),
                hide_index=True, use_container_width=True)

    step_header("Passo 5: Distâncias Euclidianas D⁺ e D⁻")
    st.latex(r"D_i^+ = \sqrt{\sum_j (v_{ij} - A_j^+)^2},\quad D_i^- = \sqrt{\sum_j (v_{ij} - A_j^-)^2}")

    step_header("Passo 6: Coeficiente de Proximidade CC e Ranking")
    st.latex(r"CC_i = \frac{D_i^-}{D_i^+ + D_i^-} \in [0, 1]")

    rank = pd.Series(CC).rank(ascending=False, method='min').astype(int).values
    df_res = pd.DataFrame({"Alternativa": alts, "D⁺": D_plus, "D⁻": D_minus, "CC*": CC, "Ranking": rank})
    df_res = df_res.sort_values("Ranking")
    st.dataframe(df_res.style.format({"D⁺": "{:.4f}", "D⁻": "{:.4f}", "CC*": "{:.4f}"})
                  .background_gradient(cmap="RdYlGn", subset=["CC*"]),
                hide_index=True, use_container_width=True)

    best = df_res.iloc[0]["Alternativa"]
    st.success(f"🏆 Melhor alternativa segundo TOPSIS: **{best}** (CC* = {df_res.iloc[0]['CC*']:.4f})")

    # Sensibilidade
    def topsis_score_fn(w):
        cc, *_ = topsis_calculate(w)
        return cc
    render_sensitivity(topsis_score_fn, alts, crits, weights, higher_is_better=True, key_suffix="topsis")


# =============================================================================
# TAB 4: PROMETHEE II
# =============================================================================
with tabs[3]:
    st.header("📈 PROMETHEE II — Preference Ranking Organisation Method")
    theory_box(
        "Teoria condensada (Brans, 1985)",
        """
        <p>Método <b>não-compensatório</b> baseado em <b>fluxos de preferência par-a-par</b>.
        Para cada par (a, b) e critério j, calcula-se a <b>função de preferência</b> P<sub>j</sub>(d):</p>
        <ul>
            <li><b>Tipo I (Usual)</b>: P=1 se a > b, P=0 caso contrário</li>
            <li><b>Tipo V (Linear)</b>: rampa linear até p</li>
            <li><b>Tipo VI (Gaussiana)</b>: curva suave com σ</li>
        </ul>
        <p>Os fluxos <b>φ⁺(a)</b> (poder) e <b>φ⁻(a)</b> (fraqueza) agregam-se em <b>φ(a) = φ⁺ − φ⁻</b>,
        produzindo uma <b>pré-ordem completa</b>.</p>
        """
    )

    if not check_valid_input():
        st.stop()

    matrix, alts, crits, types = get_decision_matrix()
    weights = get_active_weights()
    show_active_weights_banner()
    m = len(alts)

    pref_type = st.radio("Função de preferência:", ["Tipo I (Usual)", "Tipo V (Linear)", "Tipo VI (Gaussiana)"],
                         horizontal=True, key="promethee_pref_type")

    def preference_function(d, ftype, p=1.0, sigma=1.0):
        if d <= 0:
            return 0.0
        if ftype.startswith("Tipo I"):
            return 1.0
        elif ftype.startswith("Tipo V"):
            return min(d / p, 1.0)
        else:  # Gaussiana
            return 1.0 - np.exp(-d**2 / (2 * sigma**2))

    def promethee_calculate(W):
        # Para tipos com parâmetros: p = 50% do intervalo, sigma = 30% do intervalo
        n_crit = len(crits)
        params = {"p": np.zeros(n_crit), "sigma": np.zeros(n_crit)}
        for j in range(n_crit):
            rng = matrix[:, j].max() - matrix[:, j].min()
            params["p"][j] = rng * 0.5 if rng > 0 else 1.0
            params["sigma"][j] = rng * 0.3 if rng > 0 else 1.0

        pi = np.zeros((m, m))
        for a in range(m):
            for b in range(m):
                if a == b: continue
                for j in range(n_crit):
                    d = matrix[a, j] - matrix[b, j] if types[j] == "max" else matrix[b, j] - matrix[a, j]
                    pi[a, b] += W[j] * preference_function(d, pref_type, params["p"][j], params["sigma"][j])
        phi_plus = pi.sum(axis=1) / (m - 1) if m > 1 else pi.sum(axis=1)
        phi_minus = pi.sum(axis=0) / (m - 1) if m > 1 else pi.sum(axis=0)
        phi_net = phi_plus - phi_minus
        return phi_net, pi, phi_plus, phi_minus

    phi_net, pi, phi_plus, phi_minus = promethee_calculate(weights)

    step_header("Passo 1: Função de Preferência por Critério")
    if pref_type.startswith("Tipo I"):
        st.latex(r"P_j(a,b) = \begin{cases} 1 & x_{aj} > x_{bj} \\ 0 & \text{caso contrário} \end{cases}")
    elif pref_type.startswith("Tipo V"):
        st.latex(r"P_j(a,b) = \begin{cases} 0 & d \le 0 \\ d/p & 0 < d < p \\ 1 & d \ge p \end{cases},\quad p = 50\% \text{ do intervalo}")
    else:
        st.latex(r"P_j(a,b) = 1 - e^{-d^2 / (2\sigma^2)},\quad \sigma = 30\% \text{ do intervalo}")

    step_header("Passo 2: Matriz de Preferência Agregada π(a, b)")
    st.latex(r"\pi(a,b) = \sum_j w_j \cdot P_j(a, b)")
    pi_df = pd.DataFrame(pi, index=alts, columns=alts)
    st.dataframe(pi_df.style.format("{:.4f}").background_gradient(cmap="Greens"),
                use_container_width=True)

    step_header("Passo 3: Fluxos φ⁺(a), φ⁻(a) e Fluxo Líquido φ(a)")
    st.latex(r"\phi^+(a) = \frac{1}{m-1}\sum_{b \ne a} \pi(a,b),\quad \phi^-(a) = \frac{1}{m-1}\sum_{b \ne a} \pi(b,a),\quad \phi(a) = \phi^+ - \phi^-")

    rank = pd.Series(phi_net).rank(ascending=False, method='min').astype(int).values
    df_res = pd.DataFrame({"Alternativa": alts, "φ⁺": phi_plus, "φ⁻": phi_minus, "φ líquido": phi_net, "Ranking": rank})
    df_res = df_res.sort_values("Ranking")
    st.dataframe(df_res.style.format({"φ⁺": "{:.4f}", "φ⁻": "{:.4f}", "φ líquido": "{:.4f}"})
                  .background_gradient(cmap="RdYlGn", subset=["φ líquido"]),
                hide_index=True, use_container_width=True)

    best = df_res.iloc[0]["Alternativa"]
    st.success(f"🏆 Melhor alternativa segundo PROMETHEE II: **{best}** (φ = {df_res.iloc[0]['φ líquido']:.4f})")

    def promethee_score_fn(w):
        phi, *_ = promethee_calculate(w)
        return phi
    render_sensitivity(promethee_score_fn, alts, crits, weights, higher_is_better=True, key_suffix="promethee")


# =============================================================================
# TAB 5: VIKOR
# =============================================================================
with tabs[4]:
    st.header("⚖️ VIKOR — VIseKriterijumska Optimizacija I Kompromisno Resenje")
    theory_box(
        "Teoria condensada (Opricovic & Tzeng, 2004)",
        """
        <p>Procura a <b>solução de compromisso</b> entre <b>utilidade do grupo (S)</b> e <b>arrependimento individual (R)</b>.</p>
        <ul>
            <li><b>S</b>: soma ponderada de desvios à solução ideal — proximidade global</li>
            <li><b>R</b>: maior desvio individual — pior critério para a alternativa</li>
            <li><b>Q</b>: índice de compromisso combinando S e R via parâmetro v ∈ [0,1]</li>
        </ul>
        <p>A melhor alternativa é a de <b>menor Q</b>. v = 0.5 = consenso padrão; v = 1 prioriza utilidade; v = 0 prioriza equidade.</p>
        """
    )

    if not check_valid_input():
        st.stop()

    matrix, alts, crits, types = get_decision_matrix()
    weights = get_active_weights()
    show_active_weights_banner()

    v_param = st.slider("Parâmetro v (estratégia):", 0.0, 1.0, 0.5, 0.05, key="vikor_v",
                        help="0.5 = consenso; 1 = utilidade pura; 0 = arrependimento puro")

    def vikor_calculate(W, v=0.5):
        n_crit = len(crits)
        f_best = np.array([matrix[:, j].max() if types[j] == "max" else matrix[:, j].min() for j in range(n_crit)])
        f_worst = np.array([matrix[:, j].min() if types[j] == "max" else matrix[:, j].max() for j in range(n_crit)])
        denom = f_best - f_worst
        denom = np.where(denom == 0, 1e-9, denom)
        # Cálculo dos termos individuais
        terms = np.zeros_like(matrix, dtype=float)
        for j in range(n_crit):
            terms[:, j] = W[j] * np.abs(f_best[j] - matrix[:, j]) / abs(denom[j])
        S = terms.sum(axis=1)
        R = terms.max(axis=1)
        S_star, S_minus = S.min(), S.max()
        R_star, R_minus = R.min(), R.max()
        Sr = (S_minus - S_star) if S_minus != S_star else 1e-9
        Rr = (R_minus - R_star) if R_minus != R_star else 1e-9
        Q = v * (S - S_star) / Sr + (1 - v) * (R - R_star) / Rr
        return Q, S, R, f_best, f_worst

    Q, S, R, f_best, f_worst = vikor_calculate(weights, v_param)

    step_header("Passo 1: Melhores e Piores Valores por Critério")
    st.latex(r"f_j^* = \max_i f_{ij}\text{ (max)};\quad f_j^- = \min_i f_{ij}\text{ (max)};\quad \text{inverso para min}")
    st.dataframe(pd.DataFrame({"Critério": crits, "Tipo": types, "f*": f_best, "f⁻": f_worst})
                  .style.format({"f*": "{:.4f}", "f⁻": "{:.4f}"}),
                hide_index=True, use_container_width=True)

    step_header("Passo 2: Índices S (utilidade) e R (arrependimento)")
    st.latex(r"S_i = \sum_j w_j \frac{f_j^* - f_{ij}}{f_j^* - f_j^-},\quad R_i = \max_j \left[ w_j \frac{f_j^* - f_{ij}}{f_j^* - f_j^-} \right]")

    step_header("Passo 3: Índice de Compromisso Q")
    st.latex(r"Q_i = v \cdot \frac{S_i - S^*}{S^- - S^*} + (1-v) \cdot \frac{R_i - R^*}{R^- - R^*}")

    rank = pd.Series(Q).rank(ascending=True, method='min').astype(int).values
    df_res = pd.DataFrame({"Alternativa": alts, "S": S, "R": R, "Q": Q, "Ranking (Q menor = melhor)": rank})
    df_res = df_res.sort_values("Ranking (Q menor = melhor)")
    st.dataframe(df_res.style.format({"S": "{:.4f}", "R": "{:.4f}", "Q": "{:.4f}"})
                  .background_gradient(cmap="RdYlGn_r", subset=["Q"]),
                hide_index=True, use_container_width=True)

    step_header("Passo 4: Condições de Aceitação")
    sorted_q = np.sort(Q)
    if len(sorted_q) >= 2:
        dq = sorted_q[1] - sorted_q[0]
        thresh = 1.0 / (len(alts) - 1) if len(alts) > 1 else 0
        c1_ok = dq >= thresh
        best_idx = int(np.argmin(Q))
        s_best_idx = int(np.argmin(S))
        r_best_idx = int(np.argmin(R))
        c2_ok = (best_idx == s_best_idx) or (best_idx == r_best_idx)
        c1_msg = "✅" if c1_ok else "❌"
        c2_msg = "✅" if c2_ok else "❌"
        st.markdown(f"""
        - **C1 (vantagem)**: Q(a'') − Q(a') ≥ 1/(J−1) → {dq:.4f} vs {thresh:.4f} {c1_msg}
        - **C2 (estabilidade)**: a' é o melhor em S ou R {c2_msg}
        """)
        if c1_ok and c2_ok:
            best = df_res.iloc[0]["Alternativa"]
            st.success(f"🏆 Solução de compromisso ÚNICA: **{best}** (Q = {df_res.iloc[0]['Q']:.4f})")
        else:
            st.warning("⚠️ Condições não totalmente verificadas — devolve-se um conjunto de compromissos aceitáveis.")

    def vikor_score_fn(w):
        q, *_ = vikor_calculate(w, v_param)
        return -q  # inverter porque higher_is_better=True na sensibilidade
    render_sensitivity(vikor_score_fn, alts, crits, weights, higher_is_better=True, key_suffix="vikor")


# =============================================================================
# TAB 6: COPRAS
# =============================================================================
with tabs[5]:
    st.header("📊 COPRAS — COmplex PRoportional ASsessment")
    theory_box(
        "Teoria condensada (Zavadskas & Kaklauskas, 1996)",
        """
        <p>Avalia alternativas como uma <b>função proporcional</b> entre benefícios e custos.</p>
        <ol>
            <li><b>Normalização por soma</b> (proporção por critério)</li>
            <li><b>Ponderação</b>: x̂<sub>ij</sub> = w<sub>j</sub> · x'<sub>ij</sub></li>
            <li>Separar <b>S⁺</b> (Benefícios) e <b>S⁻</b> (Custos)</li>
            <li><b>Índice Q<sub>i</sub></b> (fórmula oficial completa)</li>
            <li><b>Grau de utilidade U<sub>i</sub>(%)</b> = Q<sub>i</sub> / Q<sub>max</sub> × 100</li>
        </ol>
        """
    )

    if not check_valid_input():
        st.stop()

    matrix, alts, crits, types = get_decision_matrix()
    weights = get_active_weights()
    show_active_weights_banner()

    def copras_calculate(W):
        X_norm = normalize_sum(matrix)
        V = X_norm * W
        benefit_idx = [j for j, t in enumerate(types) if t == "max"]
        cost_idx = [j for j, t in enumerate(types) if t == "min"]
        S_plus = V[:, benefit_idx].sum(axis=1) if benefit_idx else np.zeros(len(alts))
        S_minus = V[:, cost_idx].sum(axis=1) if cost_idx else np.zeros(len(alts))
        if S_minus.sum() > 0 and (S_minus > 0).all():
            S_min_val = S_minus.min()
            sum_S_minus = S_minus.sum()
            sum_inv = (S_min_val / S_minus).sum()
            Q = S_plus + (S_min_val * sum_S_minus) / (S_minus * sum_inv) if sum_inv > 0 else S_plus
        else:
            Q = S_plus
        U = (Q / Q.max() * 100) if Q.max() > 0 else Q * 0
        return Q, U, X_norm, V, S_plus, S_minus

    Q, U, X_norm, V, S_plus, S_minus = copras_calculate(weights)

    step_header("Passo 1: Matriz Normalizada por Soma")
    st.latex(r"x'_{ij} = \frac{x_{ij}}{\sum_i x_{ij}}")
    st.dataframe(pd.DataFrame(X_norm, index=alts, columns=crits).style.format("{:.4f}"),
                use_container_width=True)

    step_header("Passo 2: Matriz Ponderada")
    st.latex(r"\hat{x}_{ij} = w_j \cdot x'_{ij}")
    st.dataframe(pd.DataFrame(V, index=alts, columns=crits).style.format("{:.4f}"),
                use_container_width=True)

    step_header("Passo 3: Somatórios S⁺ (Benefícios) e S⁻ (Custos)")
    st.latex(r"S_i^+ = \sum_{j \in MAX} \hat{x}_{ij},\quad S_i^- = \sum_{j \in MIN} \hat{x}_{ij}")
    st.dataframe(pd.DataFrame({"Alternativa": alts, "S⁺": S_plus, "S⁻": S_minus})
                  .style.format({"S⁺": "{:.4f}", "S⁻": "{:.4f}"}),
                hide_index=True, use_container_width=True)

    step_header("Passo 4: Índice Q_i (fórmula oficial)")
    st.latex(r"Q_i = S_i^+ + \frac{S_{\min}^- \cdot \sum_i S_i^-}{S_i^- \cdot \sum_i (S_{\min}^- / S_i^-)}")

    step_header("Passo 5: Grau de Utilidade U_i (%) e Ranking")
    st.latex(r"U_i = \frac{Q_i}{Q_{\max}} \times 100")

    rank = pd.Series(Q).rank(ascending=False, method='min').astype(int).values
    df_res = pd.DataFrame({"Alternativa": alts, "S⁺": S_plus, "S⁻": S_minus, "Q_i": Q, "U_i (%)": U, "Ranking": rank})
    df_res = df_res.sort_values("Ranking")
    st.dataframe(df_res.style.format({"S⁺": "{:.4f}", "S⁻": "{:.4f}", "Q_i": "{:.4f}", "U_i (%)": "{:.2f}"})
                  .background_gradient(cmap="RdYlGn", subset=["U_i (%)"]),
                hide_index=True, use_container_width=True)

    best = df_res.iloc[0]["Alternativa"]
    st.success(f"🏆 Melhor alternativa segundo COPRAS: **{best}** (U = {df_res.iloc[0]['U_i (%)']:.2f}%)")

    def copras_score_fn(w):
        q, u, *_ = copras_calculate(w)
        return q
    render_sensitivity(copras_score_fn, alts, crits, weights, higher_is_better=True, key_suffix="copras")


# =============================================================================
# TAB 7: AHP (como ranking — usando pesos AHP + utilidade)
# =============================================================================
with tabs[6]:
    st.header("🔍 AHP — Ranking de Alternativas")
    theory_box(
        "Teoria condensada (Saaty, 1980)",
        """
        <p>O AHP determina pesos via <b>comparação par-a-par</b> (ver aba <b>Motores de Pesos</b>).
        Para gerar um ranking das alternativas, aplica-se utilidade aditiva:</p>
        <p><b>S<sub>i</sub> = Σ w<sub>j</sub> · u<sub>j</sub>(x<sub>ij</sub>)</b>, onde u é a utilidade min-max
        (com inversão para critérios de custo).</p>
        <p>Os pesos AHP usados aqui vêm da matriz par-a-par definida na aba <b>Motores de Pesos > AHP</b>
        (ou usam-se os pesos manuais/injectados).</p>
        """
    )

    if not check_valid_input():
        st.stop()

    matrix, alts, crits, types = get_decision_matrix()
    weights = get_active_weights()
    show_active_weights_banner()

    def ahp_calculate(W):
        U = normalize_minmax(matrix, types)
        S = (U * W).sum(axis=1)
        return S, U

    S, U = ahp_calculate(weights)

    step_header("Passo 1: Utilidade Normalizada Min-Max (com inversão para Custos)")
    st.latex(r"u_j(x_{ij}) = \frac{x_{ij} - \min_i x_{ij}}{\max_i x_{ij} - \min_i x_{ij}} \text{ (max)};\quad u_j(x_{ij}) = \frac{\max - x_{ij}}{\max - \min} \text{ (min)}")
    st.dataframe(pd.DataFrame(U, index=alts, columns=crits).style.format("{:.4f}"),
                use_container_width=True)

    step_header("Passo 2: Score Global e Ranking")
    st.latex(r"S_i = \sum_{j=1}^n w_j \cdot u_j(x_{ij})")

    rank = pd.Series(S).rank(ascending=False, method='min').astype(int).values
    df_res = pd.DataFrame({"Alternativa": alts, "Score AHP": S, "% do máx": S / S.max() * 100 if S.max() > 0 else S, "Ranking": rank})
    df_res = df_res.sort_values("Ranking")
    st.dataframe(df_res.style.format({"Score AHP": "{:.4f}", "% do máx": "{:.1f}%"})
                  .background_gradient(cmap="RdYlGn", subset=["Score AHP"]),
                hide_index=True, use_container_width=True)

    best = df_res.iloc[0]["Alternativa"]
    st.success(f"🏆 Melhor alternativa segundo AHP: **{best}** (Score = {df_res.iloc[0]['Score AHP']:.4f})")

    def ahp_score_fn(w):
        s, _ = ahp_calculate(w)
        return s
    render_sensitivity(ahp_score_fn, alts, crits, weights, higher_is_better=True, key_suffix="ahp")


# =============================================================================
# TAB 8: ELECTRE III
# =============================================================================
with tabs[7]:
    st.header("🚫 ELECTRE III — ELimination Et Choix Traduisant la REalité")
    theory_box(
        "Teoria condensada (Roy, 1968+)",
        """
        <p>Método <b>não-compensatório</b> de <b>sobreclassificação (outranking)</b>.
        Para cada par (a, b), avalia se há <b>evidência suficiente</b> para dizer que a "supera" b.</p>
        <p>Usa <b>3 limiares</b> por critério:</p>
        <ul>
            <li><b>q</b> (indiferença): diferenças até q não significam preferência</li>
            <li><b>p</b> (preferência): diferenças ≥ p significam preferência total</li>
            <li><b>v</b> (veto): diferenças ≥ v anulam a sobreclassificação</li>
        </ul>
        <p>Resultado: <b>matriz de credibilidade S(a,b)</b> + destilação ascendente/descendente → ordenação parcial.</p>
        """
    )

    if not check_valid_input():
        st.stop()

    matrix, alts, crits, types = get_decision_matrix()
    weights = get_active_weights()
    show_active_weights_banner()

    n_crit = len(crits)
    m_alt = len(alts)

    st.markdown("**Limiares por critério (% do intervalo de cada critério):**")
    c1, c2, c3 = st.columns(3)
    q_pct = c1.slider("q (indiferença) %", 0, 30, 5, 1, key="electre_q")
    p_pct = c2.slider("p (preferência) %", 5, 50, 20, 1, key="electre_p")
    v_pct = c3.slider("v (veto) %", 30, 80, 50, 1, key="electre_v")

    def electre_calculate(W):
        # Limiares absolutos por critério
        q = np.zeros(n_crit); p = np.zeros(n_crit); v = np.zeros(n_crit)
        for j in range(n_crit):
            rng = matrix[:, j].max() - matrix[:, j].min()
            q[j] = rng * q_pct / 100
            p[j] = rng * p_pct / 100
            v[j] = rng * v_pct / 100
            if p[j] <= q[j]: p[j] = q[j] + 0.001
            if v[j] <= p[j]: v[j] = p[j] + 0.001

        # Concordância parcial c_j(a, b)
        c_partial = np.zeros((m_alt, m_alt, n_crit))
        for a in range(m_alt):
            for b in range(m_alt):
                if a == b: continue
                for j in range(n_crit):
                    if types[j] == "max":
                        g_a, g_b = matrix[a, j], matrix[b, j]
                    else:
                        g_a, g_b = -matrix[a, j], -matrix[b, j]  # inverter para tratar como max
                    diff = g_a + q[j] - g_b
                    diff2 = g_a + p[j] - g_b
                    if diff >= 0:
                        c_partial[a, b, j] = 1.0
                    elif diff2 <= 0:
                        c_partial[a, b, j] = 0.0
                    else:
                        c_partial[a, b, j] = (p[j] + g_a - g_b) / (p[j] - q[j])

        # Concordância global
        C = (c_partial * W).sum(axis=2)

        # Discordância parcial d_j(a, b)
        d_partial = np.zeros((m_alt, m_alt, n_crit))
        for a in range(m_alt):
            for b in range(m_alt):
                if a == b: continue
                for j in range(n_crit):
                    if types[j] == "max":
                        g_a, g_b = matrix[a, j], matrix[b, j]
                    else:
                        g_a, g_b = -matrix[a, j], -matrix[b, j]
                    diff = g_b - g_a - p[j]
                    diff2 = g_b - g_a - v[j]
                    if diff <= 0:
                        d_partial[a, b, j] = 0.0
                    elif diff2 >= 0:
                        d_partial[a, b, j] = 1.0
                    else:
                        d_partial[a, b, j] = (g_b - g_a - p[j]) / (v[j] - p[j])

        # Credibilidade S(a, b) com veto
        S_cred = C.copy()
        for a in range(m_alt):
            for b in range(m_alt):
                if a == b: continue
                for j in range(n_crit):
                    if d_partial[a, b, j] > C[a, b]:
                        S_cred[a, b] *= (1 - d_partial[a, b, j]) / (1 - C[a, b]) if C[a, b] < 1 else 0

        return C, d_partial, S_cred, c_partial, q, p, v

    C, d_partial, S_cred, c_partial, q_abs, p_abs, v_abs = electre_calculate(weights)

    step_header("Passo 1: Limiares q, p, v por critério (valores absolutos)")
    st.dataframe(pd.DataFrame({"Critério": crits, "q": q_abs, "p": p_abs, "v": v_abs})
                  .style.format({"q": "{:.4f}", "p": "{:.4f}", "v": "{:.4f}"}),
                hide_index=True, use_container_width=True)

    step_header("Passo 2: Concordância Global C(a, b)")
    st.latex(r"C(a, b) = \sum_j w_j \cdot c_j(a, b),\quad c_j \in [0, 1]")
    st.dataframe(pd.DataFrame(C, index=alts, columns=alts).style.format("{:.3f}").background_gradient(cmap="Greens"),
                use_container_width=True)

    step_header("Passo 3: Discordância máxima por par")
    d_max = d_partial.max(axis=2)
    st.latex(r"d_j(a, b) = \frac{g_b - g_a - p_j}{v_j - p_j} \text{ (rampa)};\quad d_{max}(a,b) = \max_j d_j")
    st.dataframe(pd.DataFrame(d_max, index=alts, columns=alts).style.format("{:.3f}").background_gradient(cmap="Reds"),
                use_container_width=True)

    step_header("Passo 4: Matriz de Credibilidade S(a, b)")
    st.latex(r"S(a, b) = C(a, b) \cdot \prod_{j: d_j > C} \frac{1 - d_j}{1 - C}")
    st.dataframe(pd.DataFrame(S_cred, index=alts, columns=alts).style.format("{:.3f}").background_gradient(cmap="RdYlGn"),
                use_container_width=True)

    step_header("Passo 5: Ranking por Dominância Líquida")
    cutoff = st.slider("Limiar de corte λ:", 0.5, 0.95, 0.7, 0.05, key="electre_lambda")
    outrank = (S_cred >= cutoff) & (np.eye(m_alt) == 0)
    net_dom = outrank.sum(axis=1) - outrank.sum(axis=0)
    rank = pd.Series(net_dom).rank(ascending=False, method='min').astype(int).values
    df_res = pd.DataFrame({"Alternativa": alts, "Sobreclassifica": outrank.sum(axis=1),
                           "Sobreclassificada por": outrank.sum(axis=0),
                           "Dominância líquida": net_dom, "Ranking": rank})
    df_res = df_res.sort_values("Ranking")
    st.dataframe(df_res.style.background_gradient(cmap="RdYlGn", subset=["Dominância líquida"]),
                hide_index=True, use_container_width=True)

    best = df_res.iloc[0]["Alternativa"]
    st.success(f"🏆 Melhor alternativa segundo ELECTRE III: **{best}** (dominância líquida = {df_res.iloc[0]['Dominância líquida']})")

    def electre_score_fn(w):
        _, _, S_c, *_ = electre_calculate(w)
        outr = (S_c >= cutoff) & (np.eye(m_alt) == 0)
        return outr.sum(axis=1) - outr.sum(axis=0)
    render_sensitivity(electre_score_fn, alts, crits, weights, higher_is_better=True, key_suffix="electre")


# =============================================================================
# TAB 9: MAUT
# =============================================================================
with tabs[8]:
    st.header("💡 MAUT — Multi-Attribute Utility Theory")
    theory_box(
        "Teoria condensada (Keeney & Raiffa, 1976)",
        """
        <p>Transforma cada valor numa <b>utilidade</b> u<sub>j</sub>(x) ∈ [0,1] e agrega ponderadamente:</p>
        <p style="text-align:center;font-size:18px;"><b>U<sub>i</sub> = Σ w<sub>j</sub> · u<sub>j</sub>(x<sub>ij</sub>)</b></p>
        <p>Funções de utilidade disponíveis:</p>
        <ul>
            <li><b>Linear</b>: proporcional (mais simples)</li>
            <li><b>Exponencial</b>: ganhos marginais decrescentes (côncava)</li>
            <li><b>Potência</b>: convexa ou côncava conforme expoente p</li>
        </ul>
        """
    )

    if not check_valid_input():
        st.stop()

    matrix, alts, crits, types = get_decision_matrix()
    weights = get_active_weights()
    show_active_weights_banner()

    func_type = st.radio("Função de utilidade:", ["Linear", "Exponencial (k=2)", "Potência (p=0.5)", "Potência (p=2)"],
                        horizontal=True, key="maut_func")

    def maut_calculate(W):
        U_lin = normalize_minmax(matrix, types)  # base linear
        if func_type == "Linear":
            U = U_lin
        elif func_type.startswith("Exponencial"):
            U = 1 - np.exp(-2 * U_lin)
            U = U / (1 - np.exp(-2))  # renormalizar para [0,1]
        elif func_type == "Potência (p=0.5)":
            U = U_lin ** 0.5
        else:  # Potência (p=2)
            U = U_lin ** 2
        S = (U * W).sum(axis=1)
        return S, U

    S, U = maut_calculate(weights)

    step_header("Passo 1: Utilidades Parciais u_j(x_ij)")
    if func_type == "Linear":
        st.latex(r"u_j(x) = \frac{x - x_{\min}}{x_{\max} - x_{\min}}")
    elif func_type.startswith("Exponencial"):
        st.latex(r"u_j(x) = \frac{1 - e^{-2 \cdot \tilde{x}}}{1 - e^{-2}},\quad \tilde{x} = \text{normalizado}")
    elif func_type == "Potência (p=0.5)":
        st.latex(r"u_j(x) = \tilde{x}^{0.5} \text{ (côncava — favorece ganhos pequenos)}")
    else:
        st.latex(r"u_j(x) = \tilde{x}^{2} \text{ (convexa — penaliza valores baixos)}")
    st.dataframe(pd.DataFrame(U, index=alts, columns=crits).style.format("{:.4f}"),
                use_container_width=True)

    step_header("Passo 2: Utilidade Global e Ranking")
    st.latex(r"U_i = \sum_{j=1}^n w_j \cdot u_j(x_{ij})")
    rank = pd.Series(S).rank(ascending=False, method='min').astype(int).values
    df_res = pd.DataFrame({"Alternativa": alts, "Utilidade U_i": S, "% do máx": S / S.max() * 100 if S.max() > 0 else S, "Ranking": rank})
    df_res = df_res.sort_values("Ranking")
    st.dataframe(df_res.style.format({"Utilidade U_i": "{:.4f}", "% do máx": "{:.1f}%"})
                  .background_gradient(cmap="RdYlGn", subset=["Utilidade U_i"]),
                hide_index=True, use_container_width=True)
    best = df_res.iloc[0]["Alternativa"]
    st.success(f"🏆 Melhor alternativa segundo MAUT: **{best}** (U = {df_res.iloc[0]['Utilidade U_i']:.4f})")

    def maut_score_fn(w):
        s, _ = maut_calculate(w)
        return s
    render_sensitivity(maut_score_fn, alts, crits, weights, higher_is_better=True, key_suffix="maut")


# =============================================================================
# TAB 10: DEMATEL
# =============================================================================
with tabs[9]:
    st.header("🌐 DEMATEL — Decision Making Trial and Evaluation Laboratory")
    theory_box(
        "Teoria condensada (Gabus & Fontela, 1972)",
        """
        <p>Modela <b>relações causa-efeito</b> entre critérios. Não ordena alternativas directamente
        — produz um diagrama causal para perceber a <b>estrutura do problema</b>.</p>
        <ul>
            <li>Matriz inicial Z (0-4): influência directa entre critérios</li>
            <li>Matriz total T = X(I−X)⁻¹: efeitos directos + indirectos</li>
            <li><b>R+C</b> (proeminência): importância global do critério</li>
            <li><b>R−C</b>: > 0 = critério-causa; < 0 = critério-efeito</li>
        </ul>
        <p>Aqui, na ausência de elicitação directa, estima-se Z via <b>correlações absolutas</b>
        entre critérios. A proeminência modula os pesos para gerar um ranking de alternativas.</p>
        """
    )

    if not check_valid_input():
        st.stop()

    matrix, alts, crits, types = get_decision_matrix()
    weights = get_active_weights()
    show_active_weights_banner()

    n_crit = len(crits)

    def dematel_calculate(W):
        # Z = correlação absoluta (proxy)
        try:
            Z = np.abs(np.corrcoef(matrix.T))
            Z = np.nan_to_num(Z)
            np.fill_diagonal(Z, 0)
        except Exception:
            Z = np.zeros((n_crit, n_crit))
        # Normalizar
        s = max(Z.sum(axis=1).max(), Z.sum(axis=0).max(), 1e-9)
        X = Z / s
        # T = X (I - X)^-1
        try:
            T = X @ np.linalg.inv(np.eye(n_crit) - X)
        except np.linalg.LinAlgError:
            T = np.eye(n_crit)
        R_vec = T.sum(axis=1)
        C_vec = T.sum(axis=0)
        prominence = R_vec + C_vec
        relation = R_vec - C_vec
        # Pesos ajustados pela proeminência
        if prominence.sum() > 0:
            W_adj = W * prominence
            W_adj = W_adj / W_adj.sum()
        else:
            W_adj = W
        U = normalize_minmax(matrix, types)
        S = (U * W_adj).sum(axis=1)
        return S, T, R_vec, C_vec, prominence, relation, W_adj

    S, T, R_vec, C_vec, prominence, relation, W_adj = dematel_calculate(weights)

    step_header("Passo 1: Matriz de Relação Total T = X(I - X)⁻¹")
    st.latex(r"X = Z / \max(\text{somas linhas, somas colunas}),\quad T = X (I - X)^{-1}")
    st.dataframe(pd.DataFrame(T, index=crits, columns=crits).style.format("{:.4f}")
                  .background_gradient(cmap="Blues"),
                use_container_width=True)

    step_header("Passo 2: Proeminência (R+C) e Relação Causa-Efeito (R−C)")
    st.latex(r"R_i = \sum_j t_{ij},\quad C_j = \sum_i t_{ij};\quad R+C \text{ (importância)};\quad R-C \text{ (causa/efeito)}")
    df_rc = pd.DataFrame({
        "Critério": crits, "R": R_vec, "C": C_vec, "R+C (Proeminência)": prominence,
        "R-C (Relação)": relation,
        "Tipo": ["🎯 Causa" if r > 0 else "📥 Efeito" for r in relation]
    })
    st.dataframe(df_rc.style.format({"R": "{:.4f}", "C": "{:.4f}", "R+C (Proeminência)": "{:.4f}", "R-C (Relação)": "{:.4f}"}),
                hide_index=True, use_container_width=True)

    step_header("Passo 3: Pesos Ajustados pela Proeminência")
    st.latex(r"w_j^{adj} = \frac{w_j \cdot (R_j + C_j)}{\sum_k w_k \cdot (R_k + C_k)}")
    st.dataframe(pd.DataFrame({"Critério": crits, "Peso Original": weights, "Peso Ajustado": W_adj})
                  .style.format({"Peso Original": "{:.4f}", "Peso Ajustado": "{:.4f}"}),
                hide_index=True, use_container_width=True)

    step_header("Passo 4: Ranking com Pesos Ajustados")
    rank = pd.Series(S).rank(ascending=False, method='min').astype(int).values
    df_res = pd.DataFrame({"Alternativa": alts, "Score DEMATEL": S, "Ranking": rank})
    df_res = df_res.sort_values("Ranking")
    st.dataframe(df_res.style.format({"Score DEMATEL": "{:.4f}"})
                  .background_gradient(cmap="RdYlGn", subset=["Score DEMATEL"]),
                hide_index=True, use_container_width=True)
    best = df_res.iloc[0]["Alternativa"]
    st.success(f"🏆 Melhor alternativa segundo DEMATEL: **{best}** (Score = {df_res.iloc[0]['Score DEMATEL']:.4f})")

    def dematel_score_fn(w):
        s, *_ = dematel_calculate(w)
        return s
    render_sensitivity(dematel_score_fn, alts, crits, weights, higher_is_better=True, key_suffix="dematel")


# =============================================================================
# TAB 11: FUZZY TOPSIS
# =============================================================================
with tabs[10]:
    st.header("🌫️ Fuzzy TOPSIS")
    theory_box(
        "Teoria condensada (Chen, 2000)",
        """
        <p>Estende o TOPSIS para lidar com <b>incerteza linguística</b> usando <b>Números Fuzzy Triangulares</b>
        (TFN) ã = (l, m, u).</p>
        <ul>
            <li>Cada valor x_ij é tratado como TFN: (val·(1-s), val, val·(1+s)) com spread s</li>
            <li>Normalização linear (não vectorial)</li>
            <li>Distância entre TFN pelo método do vértice</li>
            <li>FPIS = max dos u; FNIS = min dos l</li>
            <li>CC_i = D⁻ / (D⁺ + D⁻); ranking decrescente</li>
        </ul>
        <p>Útil quando os dados têm imprecisão inerente.</p>
        """
    )

    if not check_valid_input():
        st.stop()

    matrix, alts, crits, types = get_decision_matrix()
    weights = get_active_weights()
    show_active_weights_banner()

    spread = st.slider("Spread fuzzy (% do valor):", 5, 40, 15, 5, key="ftopsis_spread",
                       help="Cada valor x torna-se TFN: (x·(1-s), x, x·(1+s))")

    def ftopsis_calculate(W, s_pct=15):
        s = s_pct / 100.0
        # TFN para cada valor
        L = matrix * (1 - s)
        M = matrix.copy()
        U = matrix * (1 + s)
        n_crit = matrix.shape[1]
        # Normalização linear fuzzy
        Ln = np.zeros_like(L); Mn = np.zeros_like(M); Un = np.zeros_like(U)
        for j in range(n_crit):
            if types[j] == "max":
                denom = max(U[:, j].max(), 1e-9)
                Ln[:, j] = L[:, j] / denom
                Mn[:, j] = M[:, j] / denom
                Un[:, j] = U[:, j] / denom
            else:
                num = max(L[:, j].min(), 1e-9)
                Ln[:, j] = num / np.where(U[:, j] == 0, 1e-9, U[:, j])
                Mn[:, j] = num / np.where(M[:, j] == 0, 1e-9, M[:, j])
                Un[:, j] = num / np.where(L[:, j] == 0, 1e-9, L[:, j])
        # Ponderação
        Lw, Mw, Uw = Ln * W, Mn * W, Un * W
        # FPIS e FNIS
        fpis = np.array([Uw[:, j].max() for j in range(n_crit)])
        fnis = np.array([Lw[:, j].min() for j in range(n_crit)])
        # Distância pelo método do vértice
        def vd(al, am, au, b):
            return np.sqrt(((al - b)**2 + (am - b)**2 + (au - b)**2) / 3.0)
        D_plus = np.zeros(len(alts)); D_minus = np.zeros(len(alts))
        for i in range(len(alts)):
            for j in range(n_crit):
                D_plus[i] += vd(Lw[i, j], Mw[i, j], Uw[i, j], fpis[j])
                D_minus[i] += vd(Lw[i, j], Mw[i, j], Uw[i, j], fnis[j])
        denom = D_plus + D_minus
        denom = np.where(denom == 0, 1e-9, denom)
        CC = D_minus / denom
        return CC, D_plus, D_minus, Lw, Mw, Uw, fpis, fnis

    CC, D_plus, D_minus, Lw, Mw, Uw, fpis, fnis = ftopsis_calculate(weights, spread)

    step_header("Passo 1: Conversão para TFN e Normalização Linear Fuzzy")
    st.latex(r"\tilde{x}_{ij} = (x_{ij}(1-s),\, x_{ij},\, x_{ij}(1+s));\quad \tilde{r}_{ij} = \frac{\tilde{x}_{ij}}{\max u_{ij}} \text{ (max)}")

    step_header("Passo 2: Matriz Ponderada Fuzzy (mostrando valores médios m)")
    st.dataframe(pd.DataFrame(Mw, index=alts, columns=crits).style.format("{:.4f}"),
                use_container_width=True)

    step_header("Passo 3: FPIS (A⁺) e FNIS (A⁻)")
    st.dataframe(pd.DataFrame({"Critério": crits, "FPIS (A⁺)": fpis, "FNIS (A⁻)": fnis})
                  .style.format({"FPIS (A⁺)": "{:.4f}", "FNIS (A⁻)": "{:.4f}"}),
                hide_index=True, use_container_width=True)

    step_header("Passo 4: Distâncias Fuzzy e Coeficiente CC")
    st.latex(r"d(\tilde{a}, \tilde{b}) = \sqrt{\frac{1}{3}[(l_a-l_b)^2 + (m_a-m_b)^2 + (u_a-u_b)^2]};\quad CC_i = \frac{D_i^-}{D_i^+ + D_i^-}")
    rank = pd.Series(CC).rank(ascending=False, method='min').astype(int).values
    df_res = pd.DataFrame({"Alternativa": alts, "D⁺": D_plus, "D⁻": D_minus, "CC*": CC, "Ranking": rank})
    df_res = df_res.sort_values("Ranking")
    st.dataframe(df_res.style.format({"D⁺": "{:.4f}", "D⁻": "{:.4f}", "CC*": "{:.4f}"})
                  .background_gradient(cmap="RdYlGn", subset=["CC*"]),
                hide_index=True, use_container_width=True)
    best = df_res.iloc[0]["Alternativa"]
    st.success(f"🏆 Melhor alternativa segundo Fuzzy TOPSIS: **{best}** (CC* = {df_res.iloc[0]['CC*']:.4f})")

    def ftopsis_score_fn(w):
        cc, *_ = ftopsis_calculate(w, spread)
        return cc
    render_sensitivity(ftopsis_score_fn, alts, crits, weights, higher_is_better=True, key_suffix="ftopsis")


# =============================================================================
# TAB 12: FUZZY AHP
# =============================================================================
with tabs[11]:
    st.header("🧮 Fuzzy AHP")
    theory_box(
        "Teoria condensada (Chang, 1996)",
        """
        <p>Estende o AHP com <b>Números Fuzzy Triangulares (TFN)</b> nas comparações par-a-par,
        capturando incerteza nos julgamentos.</p>
        <p>Método do <b>centro de área</b> para defuzzificar:</p>
        <p style="text-align:center;font-size:18px;"><b>w<sub>crisp</sub> = (l + m + u) / 3</b></p>
        <p>Nesta implementação, os pesos crisp activos (manuais ou injectados) são expandidos em
        TFN com spread ±20% e usados para ranking via utilidade aditiva.</p>
        """
    )

    if not check_valid_input():
        st.stop()

    matrix, alts, crits, types = get_decision_matrix()
    weights = get_active_weights()
    show_active_weights_banner()

    fuzzy_spread = st.slider("Spread fuzzy dos pesos (% do valor):", 5, 40, 20, 5, key="fahp_spread")

    def fahp_calculate(W, s_pct=20):
        s = s_pct / 100.0
        L = W * (1 - s)
        M = W.copy()
        U = W * (1 + s)
        # Defuzzificação
        W_crisp = (L + M + U) / 3
        W_crisp = W_crisp / W_crisp.sum()
        # Ranking
        U_mat = normalize_minmax(matrix, types)
        S = (U_mat * W_crisp).sum(axis=1)
        return S, W_crisp, L, M, U

    S, W_crisp, L, M, U = fahp_calculate(weights, fuzzy_spread)

    step_header("Passo 1: Conversão de Pesos Crisp para TFN")
    st.latex(r"\tilde{w}_j = (w_j(1-s),\, w_j,\, w_j(1+s));\quad s = \text{spread}")
    df_fuzzy = pd.DataFrame({"Critério": crits, "l": L, "m": M, "u": U})
    st.dataframe(df_fuzzy.style.format({"l": "{:.4f}", "m": "{:.4f}", "u": "{:.4f}"}),
                hide_index=True, use_container_width=True)

    step_header("Passo 2: Defuzzificação por Centro de Área")
    st.latex(r"w_j^{crisp} = \frac{l_j + m_j + u_j}{3},\quad \text{depois normalizar: } w / \sum w")
    df_crisp = pd.DataFrame({"Critério": crits, "w fuzzy → crisp": W_crisp,
                             "%": [f"{x*100:.2f}%" for x in W_crisp]})
    st.dataframe(df_crisp.style.format({"w fuzzy → crisp": "{:.4f}"}),
                hide_index=True, use_container_width=True)

    step_header("Passo 3: Ranking por Utilidade Aditiva com Pesos Defuzzificados")
    st.latex(r"S_i = \sum_j w_j^{crisp} \cdot u_j(x_{ij})")
    rank = pd.Series(S).rank(ascending=False, method='min').astype(int).values
    df_res = pd.DataFrame({"Alternativa": alts, "Score Fuzzy AHP": S, "Ranking": rank})
    df_res = df_res.sort_values("Ranking")
    st.dataframe(df_res.style.format({"Score Fuzzy AHP": "{:.4f}"})
                  .background_gradient(cmap="RdYlGn", subset=["Score Fuzzy AHP"]),
                hide_index=True, use_container_width=True)
    best = df_res.iloc[0]["Alternativa"]
    st.success(f"🏆 Melhor alternativa segundo Fuzzy AHP: **{best}** (Score = {df_res.iloc[0]['Score Fuzzy AHP']:.4f})")

    def fahp_score_fn(w):
        s, *_ = fahp_calculate(w, fuzzy_spread)
        return s
    render_sensitivity(fahp_score_fn, alts, crits, weights, higher_is_better=True, key_suffix="fahp")


# =============================================================================
# TAB 13: DASHBOARD
# =============================================================================
with tabs[12]:
    st.header("🏆 Dashboard Consolidado")

    theory_box(
        "Como funciona",
        """
        <p>Para uma decisão <b>robusta</b>, aplicam-se múltiplos métodos MCDM e compara-se a sua convergência.
        Aqui consolidamos os rankings dos modelos que correram com sucesso usando <b>método de Borda invertido</b>
        (média de posições — menor = melhor).</p>
        <p>A alternativa com maior <b>convergência inter-modelo</b> (mesma posição em vários métodos)
        é a recomendação mais robusta.</p>
        """
    )

    if not check_valid_input():
        st.stop()

    matrix, alts, crits, types = get_decision_matrix()
    weights = get_active_weights()
    show_active_weights_banner()

    # Recalcular todos os modelos com os pesos activos
    all_rankings = {}
    all_scores = {}

    try:
        # TOPSIS
        R = normalize_vector(matrix); V = R * weights
        A_p = np.array([V[:, j].max() if types[j] == "max" else V[:, j].min() for j in range(len(crits))])
        A_n = np.array([V[:, j].min() if types[j] == "max" else V[:, j].max() for j in range(len(crits))])
        D_p = np.sqrt(((V - A_p) ** 2).sum(axis=1)); D_n = np.sqrt(((V - A_n) ** 2).sum(axis=1))
        denom = np.where(D_p + D_n == 0, 1e-9, D_p + D_n)
        topsis_cc = D_n / denom
        all_scores["TOPSIS"] = topsis_cc
        all_rankings["TOPSIS"] = pd.Series(topsis_cc).rank(ascending=False, method='min').astype(int).values
    except Exception as e:
        st.warning(f"TOPSIS falhou: {e}")

    try:
        # AHP / MAUT
        U_mm = normalize_minmax(matrix, types)
        ahp_s = (U_mm * weights).sum(axis=1)
        all_scores["AHP"] = ahp_s
        all_rankings["AHP"] = pd.Series(ahp_s).rank(ascending=False, method='min').astype(int).values
        all_scores["MAUT"] = ahp_s.copy()
        all_rankings["MAUT"] = all_rankings["AHP"].copy()
    except Exception as e:
        st.warning(f"AHP/MAUT falhou: {e}")

    try:
        # COPRAS
        Xn = normalize_sum(matrix); Vw = Xn * weights
        bi = [j for j, t in enumerate(types) if t == "max"]
        ci = [j for j, t in enumerate(types) if t == "min"]
        Sp = Vw[:, bi].sum(axis=1) if bi else np.zeros(len(alts))
        Sm = Vw[:, ci].sum(axis=1) if ci else np.zeros(len(alts))
        if Sm.sum() > 0 and (Sm > 0).all():
            sm_min = Sm.min(); sum_sm = Sm.sum(); sum_inv = (sm_min / Sm).sum()
            Q = Sp + (sm_min * sum_sm) / (Sm * sum_inv) if sum_inv > 0 else Sp
        else:
            Q = Sp
        all_scores["COPRAS"] = Q
        all_rankings["COPRAS"] = pd.Series(Q).rank(ascending=False, method='min').astype(int).values
    except Exception as e:
        st.warning(f"COPRAS falhou: {e}")

    try:
        # PROMETHEE (Tipo I usual)
        m_alt = len(alts)
        pi = np.zeros((m_alt, m_alt))
        for a in range(m_alt):
            for b in range(m_alt):
                if a == b: continue
                for j in range(len(crits)):
                    d = matrix[a, j] - matrix[b, j] if types[j] == "max" else matrix[b, j] - matrix[a, j]
                    if d > 0: pi[a, b] += weights[j]
        phi = (pi.sum(axis=1) - pi.sum(axis=0)) / max(m_alt - 1, 1)
        all_scores["PROMETHEE"] = phi
        all_rankings["PROMETHEE"] = pd.Series(phi).rank(ascending=False, method='min').astype(int).values
    except Exception as e:
        st.warning(f"PROMETHEE falhou: {e}")

    try:
        # VIKOR
        f_best = np.array([matrix[:, j].max() if types[j] == "max" else matrix[:, j].min() for j in range(len(crits))])
        f_worst = np.array([matrix[:, j].min() if types[j] == "max" else matrix[:, j].max() for j in range(len(crits))])
        den = np.where(f_best - f_worst == 0, 1e-9, f_best - f_worst)
        terms = np.zeros_like(matrix, dtype=float)
        for j in range(len(crits)):
            terms[:, j] = weights[j] * np.abs(f_best[j] - matrix[:, j]) / abs(den[j])
        Sv = terms.sum(axis=1); Rv = terms.max(axis=1)
        S_r = (Sv.max() - Sv.min()) if Sv.max() != Sv.min() else 1e-9
        R_r = (Rv.max() - Rv.min()) if Rv.max() != Rv.min() else 1e-9
        Qv = 0.5 * (Sv - Sv.min()) / S_r + 0.5 * (Rv - Rv.min()) / R_r
        all_scores["VIKOR"] = -Qv  # menor Q = melhor
        all_rankings["VIKOR"] = pd.Series(Qv).rank(ascending=True, method='min').astype(int).values
    except Exception as e:
        st.warning(f"VIKOR falhou: {e}")

    if not all_rankings:
        st.error("Nenhum modelo correu com sucesso. Verifique os dados de entrada.")
        st.stop()

    # Tabela consolidada
    st.subheader("Rankings por Modelo")
    methods = list(all_rankings.keys())
    df_dash = pd.DataFrame({"Alternativa": alts})
    for m_name in methods:
        df_dash[m_name] = all_rankings[m_name]
    df_dash["Posição Média"] = df_dash[methods].mean(axis=1).round(2)
    df_dash["Top-3 em N modelos"] = (df_dash[methods] <= 3).sum(axis=1)
    df_dash["Ranking Final"] = pd.Series(df_dash["Posição Média"]).rank(ascending=True, method='min').astype(int).values
    df_dash = df_dash.sort_values("Ranking Final")
    st.dataframe(df_dash.style.background_gradient(cmap="RdYlGn_r", subset=methods + ["Posição Média", "Ranking Final"]),
                hide_index=True, use_container_width=True)

    st.subheader("🥇 Top-3 Recomendado (por Posição Média)")
    top3 = df_dash.head(3)["Alternativa"].tolist()
    cols = st.columns(3)
    medals = ["🥇 1º lugar", "🥈 2º lugar", "🥉 3º lugar"]
    for k, (col, medal) in enumerate(zip(cols, medals)):
        if k < len(top3):
            col.metric(medal, top3[k])

    # Convergência
    total_top3 = sum(df_dash.head(3)["Top-3 em N modelos"].values)
    max_conv = 3 * len(methods)
    conv_pct = (total_top3 / max_conv * 100) if max_conv else 0
    st.info(f"**Convergência inter-modelo no Top-3**: {total_top3}/{max_conv} ({conv_pct:.0f}%)\n\n"
            f"Modelos: {', '.join(methods)}")

    # Heatmap
    st.subheader("Heatmap de Rankings por Modelo")
    heat = df_dash.set_index("Alternativa")[methods]
    st.dataframe(heat.style.background_gradient(cmap="RdYlGn_r").format("{:.0f}"),
                use_container_width=True)
