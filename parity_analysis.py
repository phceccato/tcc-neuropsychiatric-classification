import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

# ── Configuração visual ──────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': '#FAFAFA',
    'axes.facecolor': '#FAFAFA',
})

MEN_COLOR   = '#8B8FD4'   # azul-lavanda
WOMEN_COLOR = '#E88FA0'   # rosa-salmão
MEN_EDGE    = '#5C60B5'
WOMEN_EDGE  = '#C85070'
FILL_ALPHA  = 0.85
EMPTY_ALPHA = 0.5

# ── Leitura ──────────────────────────────────────────────────────────────────
df = pd.read_csv('EEG.machinelearing_data_BRMH.csv')

# Normaliza sexo
df['sex'] = df['sex'].str.strip().str.upper()

# ── Função: calcula métricas de parity por categoria ────────────────────────
def parity_metrics(df, group_col):
    """
    Para cada classe em group_col, trata "pertencer à classe" como C=1.
    Calcula P(C=1 | sex=M) e P(C=1 | sex=F).
    Retorna DataFrame com essas probabilidades e a diferença.
    """
    categories = sorted(df[group_col].dropna().unique())
    rows = []
    for cat in categories:
        df['label'] = (df[group_col] == cat).astype(int)
        p_m = df.loc[df['sex'] == 'M', 'label'].mean()
        p_f = df.loc[df['sex'] == 'F', 'label'].mean()
        gap = abs(p_m - p_f)
        rows.append({'category': cat, 'P_men': p_m, 'P_women': p_f, 'gap': gap})
    df.drop(columns=['label'], inplace=True)
    return pd.DataFrame(rows).sort_values('gap', ascending=False)

# ── Análises ─────────────────────────────────────────────────────────────────
main_parity = parity_metrics(df, 'main.disorder')
spec_parity = parity_metrics(df, 'specific.disorder')

print("=== PARIDADE – main.disorder ===")
print(main_parity.to_string(index=False))
print("\n=== PARIDADE – specific.disorder ===")
print(spec_parity.to_string(index=False))

# ── Counts para os dot-plots ─────────────────────────────────────────────────
def build_counts(df, group_col):
    return df.groupby([group_col, 'sex']).size().unstack(fill_value=0)

main_counts = build_counts(df, 'main.disorder')
spec_counts = build_counts(df, 'specific.disorder')

# ── Plot 1: Dot-plot estilo "equality of odds" – main.disorder ──────────────
def dot_plot(parity_df, counts_df, title, filename, col_w=3.5):
    categories = parity_df['category'].tolist()
    n = len(categories)
    fig_w = col_w * 2 + 1.5
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, max(5, n * 0.9 + 2)),
                              sharey=True, facecolor='#FAFAFA')
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)

    for ax_idx, (sex, color, edge, ax) in enumerate(zip(
            ['M', 'F'],
            [MEN_COLOR, WOMEN_COLOR],
            [MEN_EDGE, WOMEN_EDGE],
            axes)):

        sex_label = 'men' if sex == 'M' else 'women'
        ax.set_facecolor('#FAFAFA')
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, n - 0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(sex_label, fontsize=12, color=color, fontweight='bold')

        # linha de paridade (threshold visual)
        total_m = counts_df.get('M', pd.Series(dtype=float)).sum() if 'M' in counts_df.columns else 0
        total_f = counts_df.get('F', pd.Series(dtype=float)).sum() if 'F' in counts_df.columns else 0
        grand_total = total_m + total_f
        if grand_total > 0:
            baseline = total_f / grand_total if sex == 'F' else total_m / grand_total
        else:
            baseline = 0.5
        ax.axhline(y=n / 2 - 0.5, color='#AAAAAA', linewidth=1,
                   linestyle='--', alpha=0.7)

        for i, row in enumerate(parity_df.itertuples()):
            cat = row.category
            p_val = row.P_women if sex == 'F' else row.P_men

            # número de amostras nessa categoria e sexo
            try:
                cnt = counts_df.loc[cat, sex] if sex in counts_df.columns else 0
            except KeyError:
                cnt = 0

            # raio proporcional ao count (normalizado)
            max_cnt = counts_df.max().max() if len(counts_df) > 0 else 1
            radius = 0.18 + 0.25 * (cnt / max_cnt)

            # posição x ~ probabilidade
            x_pos = 0.2 + p_val * 1.1

            y_pos = n - 1 - i  # mais alto no topo

            # ponto preenchido = pertence (filled)
            circle_filled = plt.Circle((x_pos, y_pos), radius,
                                       color=color, alpha=FILL_ALPHA, zorder=3)
            ax.add_patch(circle_filled)

            # ponto vazio = não pertence
            x_empty = 0.2 + (1 - p_val) * 1.1
            circle_empty = plt.Circle((x_empty, y_pos - 0.3), radius * 0.8,
                                      fill=False, edgecolor=color,
                                      linewidth=1.5, alpha=EMPTY_ALPHA, zorder=3)
            ax.add_patch(circle_empty)

        # FPR / FNR annotation (aproximado)
        row_data = parity_df.iloc[0]
        p = row_data.P_women if sex == 'F' else row_data.P_men
        fpr_approx = round(p * 3) if round(p * 3) > 0 else 1
        fpr_denom  = round(1 / p) if p > 0 else 6
        fnr_approx = 1
        fnr_denom  = round(1 / (1 - p)) if (1 - p) > 0 else 4
        ax.text(0.5, 1.06, f'fpr = {fpr_approx}/{fpr_denom}\nfnr = {fnr_approx}/{fnr_denom}',
                transform=ax.transAxes, ha='center', va='top',
                fontsize=8.5, color='#444444')

        for spine in ax.spines.values():
            spine.set_visible(False)

    # ── Eixo Y com nomes das categorias (à esquerda) ──
    axes[0].set_yticks(range(n))
    axes[0].set_yticklabels(
        [c[:30] + ('…' if len(c) > 30 else '') for c in reversed(parity_df['category'].tolist())],
        fontsize=8.5
    )
    axes[0].yaxis.set_tick_params(length=0)

    # ── Título central "equality of odds" ──
    fig.text(0.5, 1.03, 'equality of odds', ha='center', fontsize=11, style='italic')

    # ── Legenda ──
    legend_elements = [
        mpatches.Patch(color=MEN_COLOR, alpha=FILL_ALPHA, label='pertence à classe'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
               markeredgecolor=MEN_COLOR, markersize=9, label='não pertence'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2,
               bbox_to_anchor=(0.5, -0.04), fontsize=9, frameon=False)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
    plt.close()
    print(f"Salvo: {filename}")


# ── Plot 2: Gráfico de barras – main + specific em dois subplots ──────────
def bar_parity_plot(main_df, spec_df, filename):
    from fractions import Fraction

    fig, axes = plt.subplots(2, 1, figsize=(max(9, len(spec_df) * 1.1), 11),
                              facecolor='#FAFAFA')
    fig.suptitle('P(C=1 | A=sexo) — Paridade Demográfica', fontsize=13,
                 fontweight='bold', y=1.01)

    for ax, parity_df, subtitle in [
        (axes[0], main_df,  'Main Disorder'),
        (axes[1], spec_df,  'Specific Disorder'),
    ]:
        cats = parity_df['category'].tolist()
        n = len(cats)
        x = np.arange(n)
        width = 0.35

        ax.set_facecolor('#FAFAFA')
        ax.bar(x - width/2, parity_df['P_men'],  width,
               color=MEN_COLOR,   alpha=0.85, label='Homens (M)',
               edgecolor=MEN_EDGE,   linewidth=0.8)
        ax.bar(x + width/2, parity_df['P_women'], width,
               color=WOMEN_COLOR, alpha=0.85, label='Mulheres (F)',
               edgecolor=WOMEN_EDGE, linewidth=0.8)

        ax.axhline(y=parity_df[['P_men', 'P_women']].values.mean(), color='#555',
                   linewidth=1, linestyle='--', alpha=0.5, label='Média geral')

        ax.set_xticks(x)
        ax.set_xticklabels(cats, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('P(C = 1 | A = sexo)', fontsize=10)
        ax.set_title(subtitle, fontsize=11, fontweight='bold', pad=8)
        ax.set_ylim(0, min(1, parity_df[['P_men', 'P_women']].max().max() * 1.25))
        ax.legend(fontsize=9, frameon=False)

        for i, row in enumerate(parity_df.itertuples()):
            ymax = max(row.P_men, row.P_women)
            ax.annotate(f'Δ={row.gap:.3f}', xy=(i, ymax + 0.012),
                        ha='center', fontsize=7.5, color='#333333')

        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
    plt.close()
    print(f"Salvo: {filename}")


# ── Plot 3: Heatmap – main + specific empilhados ──────────────────────────
def heatmap_parity(main_df, spec_df, filename):
    fig, axes = plt.subplots(2, 1,
                              figsize=(max(8, len(spec_df) * 1.1), 6.5),
                              facecolor='#FAFAFA')
    fig.suptitle('Heatmap de Paridade Demográfica por Sexo', fontsize=13,
                 fontweight='bold', y=1.01)

    for ax, parity_df, subtitle in [
        (axes[0], main_df,  'Main Disorder'),
        (axes[1], spec_df,  'Specific Disorder'),
    ]:
        data = parity_df[['P_men', 'P_women']].set_index(parity_df['category']).T
        data.index = ['Homens (M)', 'Mulheres (F)']

        ax.set_facecolor('#FAFAFA')
        im = ax.imshow(data.values, aspect='auto', cmap=plt.cm.RdYlGn,
                       vmin=0, vmax=data.values.max())

        ax.set_xticks(range(len(data.columns)))
        ax.set_xticklabels(data.columns, rotation=30, ha='right', fontsize=8.5)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(data.index, fontsize=10)
        ax.set_title(subtitle, fontsize=11, fontweight='bold', pad=8)

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data.values[i, j]
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=9, color='white' if val < 0.4 else '#222')

        plt.colorbar(im, ax=ax, shrink=0.8, label='P(C=1 | A=sexo)')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
    plt.close()
    print(f"Salvo: {filename}")


# ── Plot 1: Dot-plot – main + specific em duas linhas de painéis ──────────
def dot_plot_ref_style(main_parity, main_counts, spec_parity, spec_counts, filename):
    from fractions import Fraction

    n_main = len(main_parity)
    n_spec = len(spec_parity)

    height_main = max(4, n_main * 0.75 + 1.5)
    height_spec = max(4, n_spec * 0.75 + 1.5)

    fig = plt.figure(figsize=(8, height_main + height_spec + 1.5), facecolor='#FAFAFA')
    fig.suptitle('Paridade Demográfica por Sexo — Equality of Odds',
                 fontsize=12, fontweight='bold', y=1.01)
    fig.text(0.5, 0.99, 'equality of odds', ha='center', fontsize=10,
             style='italic', color='#555')

    gs = fig.add_gridspec(2, 2,
                          height_ratios=[height_main, height_spec],
                          hspace=0.35, wspace=0.05)

    panels = [
        (gs[0, 0], gs[0, 1], main_parity, main_counts, 'Main Disorder'),
        (gs[1, 0], gs[1, 1], spec_parity, spec_counts, 'Specific Disorder'),
    ]

    for gs_m, gs_f, parity_df, counts_df, subtitle in panels:
        cats = list(reversed(parity_df['category'].tolist()))
        n = len(cats)
        max_cnt = counts_df.values.max() if len(counts_df) > 0 else 1

        for gs_cell, sex, color, edge, label in [
            (gs_m, 'M', MEN_COLOR,   MEN_EDGE,   'men'),
            (gs_f, 'F', WOMEN_COLOR, WOMEN_EDGE, 'women'),
        ]:
            ax = fig.add_subplot(gs_cell)
            ax.set_facecolor('#FAFAFA')
            ax.set_xlim(-0.2, 1.2)
            ax.set_ylim(-0.7, n + 0.3)
            ax.set_xticks([])
            ax.set_yticks(range(n))
            ax.set_xlabel(label, fontsize=11, color=color, fontweight='bold', labelpad=6)

            ax.axhline(y=n / 2, color='#BBBBBB', linewidth=1, linestyle='--', alpha=0.8)

            row_top = parity_df.iloc[0]
            p_top = row_top.P_women if sex == 'F' else row_top.P_men
            frac = Fraction(p_top).limit_denominator(8)
            fnr_frac = Fraction(1 - p_top).limit_denominator(8) if (1 - p_top) > 0 else Fraction(0)
            ax.set_title(
                f'{subtitle}\nfpr={frac.numerator}/{frac.denominator}  '
                f'fnr={fnr_frac.numerator}/{fnr_frac.denominator}',
                fontsize=8, color='#444', pad=4
            )

            for i, cat in enumerate(cats):
                p = parity_df.loc[parity_df['category'] == cat,
                                   'P_women' if sex == 'F' else 'P_men'].values[0]
                try:
                    cnt = counts_df.loc[cat, sex] if sex in counts_df.columns else 0
                except KeyError:
                    cnt = 0

                s_fill  = 80  + 250 * (cnt / max_cnt)
                s_empty = 50  + 150 * (cnt / max_cnt)

                ax.scatter(0.5 + (p - 0.5) * 0.8, i + 0.15,
                           s=s_fill, color=color, alpha=0.85, zorder=4,
                           edgecolors=edge, linewidths=0.5)
                ax.scatter(0.5 + ((1 - p) - 0.5) * 0.8, i - 0.2,
                           s=s_empty, facecolors='none',
                           edgecolors=color, linewidths=1.5, alpha=0.65, zorder=4)

            for spine in ax.spines.values():
                spine.set_visible(False)

            if sex == 'M':
                ax.set_yticklabels(
                    [c[:28] + ('…' if len(c) > 28 else '') for c in cats],
                    fontsize=8)
                ax.yaxis.set_tick_params(length=0)
            else:
                ax.set_yticklabels([])

    legend_elements = [
        mpatches.Patch(color=MEN_COLOR,   alpha=0.85, label='pertence à classe'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
               markeredgecolor=MEN_COLOR, markersize=9, label='não pertence'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2,
               bbox_to_anchor=(0.5, -0.02), fontsize=9, frameon=False)

    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
    plt.close()
    print(f"Salvo: {filename}")


# ── Geração dos gráficos ──────────────────────────────────────────────────
print("\nGerando gráficos...\n")

dot_plot_ref_style(
    main_parity, main_counts,
    spec_parity, spec_counts,
    'parity_dotplot.png'
)

bar_parity_plot(
    main_parity, spec_parity,
    'parity_bars.png'
)

heatmap_parity(
    main_parity, spec_parity,
    'parity_heatmap.png'
)

print("\n✅ Todos os gráficos foram gerados com sucesso!")
