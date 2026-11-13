"""RESULTS.md: every benchmark result as readable dataset x method tables.

Pure formatting of the fit records (all_runs.csv); no fitting and no file access except the
final write. Every number shown here is checked again by verification.py.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from .config import PRIMARY, LC_METHODS, DATASETS, GROUPS, SELECTED_FOR_LC, LABEL_METRICS
from .statistics import proposed_versus_others_tests

SHORT = {'LC-FixedT': 'LC-T', 'LC-FixedHalfT': 'LC-T/2', 'LC-SoftFixedT': 'LC-2T', 'k-means++': 'KM++'}
NAMES = {'ari': 'ARI', 'nmi': 'NMI', 'pur': 'Purity', 'sil': 'Silhouette', 'pc': 'PC', 'pe': 'PE'}
OTHERS = [m for m in PRIMARY if m not in LC_METHODS]


def short(method):
    return SHORT.get(method, method)


def fmt(value):
    return '–' if pd.isna(value) else f'{value:.3f}'


def cell_means(frame, metric):
    """Dataset x method means over seeds; a cell with any undefined seed stays undefined."""
    grouped = frame.groupby(['dataset', 'method'])[metric]
    means = grouped.mean().where(grouped.count() == grouped.size())
    return means.unstack().reindex(index=DATASETS, columns=PRIMARY).round(10)


def winners(row, lower_is_better=False):
    best = row.min() if lower_is_better else row.max()
    return [m for m in PRIMARY if row[m] == best]


def header_row(extra=()):
    columns = ['Dataset'] + [short(m) for m in PRIMARY] + list(extra)
    return ['| ' + ' | '.join(columns) + ' |', '|' + '---|' * len(columns)]


def metric_table(table):
    lines = header_row(['Winner'])
    wins = dict.fromkeys(PRIMARY, 0)
    for dataset, row in table.iterrows():
        best = winners(row)
        for m in best:
            wins[m] += 1
        cells = [f'**{fmt(row[m])}**' if m in best else fmt(row[m]) for m in PRIMARY]
        lines.append(f'| {dataset} | ' + ' | '.join(cells) + ' | ' + ', '.join(short(m) for m in best) + ' |')
    complete = table.notna().all()
    mean = table.loc[:, complete].mean()
    lines.append('| **Mean** | ' + ' | '.join(
        ('–' if not complete[m] else f'**{mean[m]:.3f}**' if mean[m] == mean.max() else f'{mean[m]:.3f}')
        for m in PRIMARY) + f' | {short(mean.idxmax())} |')
    lines.append('| **Wins** | ' + ' | '.join(str(wins[m]) for m in PRIMARY) + ' | |')
    return lines


def ranking_table(tables, subset):
    rows = []
    for m in PRIMARY:
        row = {'method': m, 'wins': 0, 'ranks': []}
        for metric in LABEL_METRICS:
            t = tables[metric].loc[subset]
            row[metric] = t[m].mean()
            row[metric + '_wins'] = sum(m in winners(r) for _, r in t.iterrows())
            row['wins'] += row[metric + '_wins']
            row['ranks'].append(t.rank(axis=1, ascending=False, method='average')[m].mean())
        row['rank'] = float(np.mean(row['ranks']))
        rows.append(row)
    lines = ['| Place | Method | Mean ARI | Mean NMI | Mean Purity | ARI wins | NMI wins | Purity wins | Total wins | Mean rank ↓ |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for place, r in enumerate(sorted(rows, key=lambda r: r['rank']), 1):
        lines.append(f"| {place} | {short(r['method'])} | {r['ari']:.3f} | {r['nmi']:.3f} | {r['pur']:.3f} | "
                     f"{r['ari_wins']} | {r['nmi_wins']} | {r['pur_wins']} | **{r['wins']}** | {r['rank']:.2f} |")
    return lines


def significance_table(frame, subset, label):
    tests = proposed_versus_others_tests(frame, LC_METHODS, OTHERS, subset)
    lines = [f'**{label}** ({len(subset)} datasets). Cell = datasets where the LC variant is better / worse, '
             f'then the Holm-adjusted p-value; ✅ = significant (p < 0.05) after correcting all '
             f'{len(LC_METHODS) * len(OTHERS)} tests of that metric.', '',
             '| Metric | LC variant | ' + ' | '.join(short(o) for o in OTHERS) + ' |',
             '|---|---|' + '---|' * len(OTHERS)]
    for metric in LABEL_METRICS:
        for p in LC_METHODS:
            part = tests[(tests.metric == metric) & (tests.proposed == p)].set_index('other')
            cells = [f"{part.loc[o, 'proposed_better']}/{part.loc[o, 'other_better']} "
                     f"{'✅' if part.loc[o, 'significant_holm_0_05'] else ''}p={part.loc[o, 'p_holm']:.3f}" for o in OTHERS]
            lines.append(f'| {NAMES[metric]} | {short(p)} | ' + ' | '.join(cells) + ' |')
    return lines, tests


def sd_table(frame, metric):
    grouped = frame.groupby(['dataset', 'method'])[metric]
    mean = grouped.mean().where(grouped.count() == grouped.size()).unstack().reindex(index=DATASETS, columns=PRIMARY)
    sd = grouped.std(ddof=1).unstack().reindex(index=DATASETS, columns=PRIMARY)
    lines = header_row()
    for d in DATASETS:
        lines.append(f'| {d} | ' + ' | '.join('–' if pd.isna(mean.loc[d, m]) else
                                             f'{mean.loc[d, m]:.3f} ± {sd.loc[d, m]:.3f}' for m in PRIMARY) + ' |')
    return lines


def health_lines(frame):
    lines = ['| Method | Fits | Hit iteration cap | Fits with fewer clusters than K | Datasets where all seeds have fewer than K |',
             '|---|---|---|---|---|']
    details = []
    for m in PRIMARY:
        f = frame[frame.method == m]
        fewer = f.hard_clusters < f.classes
        always = f.assign(fewer=fewer).groupby('dataset').fewer.all()
        lines.append(f'| {short(m)} | {len(f)} | {int((~f.converged.astype(bool)).sum())} | {int(fewer.sum())} | {int(always.sum())} |')
        parts = [f'{d} {len(g)}/{f[f.dataset == d].seed.nunique()} (uses {int(g.hard_clusters.min())}–{int(g.hard_clusters.max())} of K={int(g.classes.iloc[0])})'
                 for d, g in f[fewer].groupby('dataset', sort=False)]
        if parts:
            details.append(f'- **{short(m)}:** ' + ', '.join(parts))
    return lines + [''] + details


def time_lines(frame):
    t = frame.groupby(['dataset', 'method']).total_time_s.mean().unstack().reindex(index=DATASETS, columns=PRIMARY)
    lines = header_row()
    for d in DATASETS:
        lines.append(f'| {d} | ' + ' | '.join(f'{t.loc[d, m]:.3f}' for m in PRIMARY) + ' |')
    return lines + ['| **Mean** | ' + ' | '.join(f'{t[m].mean():.3f}' for m in PRIMARY) + ' |']


def write_results(frame, output, figures, extra_sections=()):
    """Write output/RESULTS.md and output/lc_significance.csv; return the significance table."""
    output = Path(output)
    tables = {metric: cell_means(frame, metric) for metric in LABEL_METRICS}
    info = frame.drop_duplicates('dataset').set_index('dataset').reindex(DATASETS)
    development = [d for d in DATASETS if GROUPS[d] == 'development']
    later = [d for d in DATASETS if GROUPS[d] != 'development']
    not_selected = [d for d in DATASETS if d not in SELECTED_FOR_LC]
    seeds = sorted(frame.seed.unique())

    lines = [f'# Benchmark results: {len(PRIMARY)} methods, {len(DATASETS)} datasets, {len(seeds)} seeds', '',
             f'{len(frame):,} fresh fits ({len(DATASETS)} datasets × {len(PRIMARY)} methods × {len(seeds)} seeds, '
             f'seeds {seeds[0]}–{seeds[-1]}). Fits that hit the iteration cap: {int((~frame.converged.astype(bool)).sum())}. '
             'Every number below is recomputed from the saved predictions by the independent audit '
             '(verification.json).', '',
             'Each cell is the mean over the seeds. **Bold** = best on that dataset (ties at 10 decimals all count). '
             'Wins use only the label metrics ARI, NMI and Purity. Short names: LC-2T = LC-SoftFixedT, '
             'LC-T = LC-FixedT, LC-T/2 = LC-FixedHalfT, KM++ = k-means++.', '',
             '## Datasets', '',
             '| Dataset | Group | N | d | K |', '|---|---|---|---|---|'] + [
             f'| {d} | {GROUPS[d]} | {int(info.samples[d])} | {int(info.features[d])} | {int(info.classes[d])} |' for d in DATASETS] + [
             '', 'Groups: **development** = used while the methods were developed; **regime** = added to test where '
             'LC should win (optdigits, pendigits, satimage, mfeat-kar) or lose (new-thyroid); **held-out** = not looked '
             'at before their first full run.', '',
             f'## 1. Overall ranking, all {len(DATASETS)} datasets', ''] + ranking_table(tables, DATASETS) + [
             '', 'Mean rank = average of the per-dataset ranks on ARI, NMI and Purity (1 = best).', '',
             f'## 2. Overall ranking, {len(development)} development datasets', ''] + ranking_table(tables, development) + [
             '', f'## 3. Overall ranking, {len(later)} datasets added after development (regime + held-out)', ''] + \
            ranking_table(tables, later) + ['', f'## 4. Dataset × method tables', '']
    for metric in LABEL_METRICS:
        lines += [f'### {NAMES[metric]} (higher is better)', ''] + metric_table(tables[metric]) + ['']
    lines += ['## 5. Where the LC family comes first', '']
    for metric in LABEL_METRICS:
        first = [d for d in DATASETS if any(m in LC_METHODS for m in winners(tables[metric].loc[d]))]
        ties = [d for d in first if len(winners(tables[metric].loc[d])) == len(PRIMARY)]
        note = f' (includes {", ".join(ties)}, where all {len(PRIMARY)} methods tie, so it is not a real win)' if ties else ''
        lines.append(f'- **{NAMES[metric]}:** {len(first)} of {len(DATASETS)} — ' + (', '.join(first) or 'none') + note)
    lines += ['', '## 6. Is the difference significant? (Wilcoxon signed-rank over datasets, Holm-corrected)', '',
              'Seeds are not independent samples, so each dataset counts once (its mean over seeds).', '']
    all_tests = []
    for subset, label in [(DATASETS, f'All {len(DATASETS)} datasets'),
                          (not_selected, 'Without the 4 datasets selected for LC'),
                          (later, f'{len(later)} datasets added after development')]:
        block, tests = significance_table(frame, subset, label)
        lines += block + ['']
        all_tests.append(tests.assign(datasets=label))
    lines += ['## 7. Silhouette (descriptive only, not counted in wins)', '',
              'Silhouette uses no labels, so it measures compactness rather than agreement with the classes. '
              '"–" = undefined on at least one seed (the fit returned a single cluster).', ''] + \
             metric_table(cell_means(frame, 'sil')) + ['']
    lines += ['## 8. Run health', '',
              '"Fewer clusters than K" counts fits whose labels use fewer than the requested K clusters '
              '(an empty or merged cluster; PCM usually returns a single cluster).', ''] + health_lines(frame) + ['']
    lines += ['## 9. Run time: mean seconds per fit', '',
              'Fresh wall time of one fit with one BLAS thread, including the preprocessing each method performs; '
              'evaluation excluded. Depends on hardware; not part of any score.', ''] + time_lines(frame) + ['']
    lines += ['## 10. Appendix: mean ± sample SD over the seeds', '',
              'PC and PE are reported for completeness only: they measure membership sharpness, which a lower '
              'temperature increases by construction, and are undefined for hard (DPC, k-means++) and possibilistic '
              '(PCM) outputs.', '']
    for metric in ['ari', 'nmi', 'pur', 'sil', 'pc', 'pe']:
        lines += [f'### {NAMES[metric]}', ''] + sd_table(frame, metric) + ['']
    lines += ['## 11. Protocol and limits', '',
              '- All settings are fixed across datasets; features are standardized and exclude labels and IDs; '
              'the true class count supplies K. No method is tuned per dataset.',
              '- DPC uses the known-K top density-distance centers in place of the author\'s interactive selection. '
              'MC-FCM is an independently checked implementation of the published equations. k-means++ uses one '
              'start per seed (scikit-learn). See docs/ALGORITHMS.md and docs/MCFCM_PROVENANCE.md.',
              '- The 8 development datasets were seen while the methods were designed, and 4 regime datasets were '
              'chosen because LC was expected to do well; sections 3 and 6 report the subsets separately.',
              '- An earlier version of the study also compared a Gaussian mixture model (GMM); it is not one of the '
              'methods compared here. That run and every other experiment are kept in research/ (research/README.md).',
              '- Wins and ranks depend on the set of methods compared; numbers from other method sets must not be mixed.',
              '', '## 12. What can be claimed', '',
              '- Claim only what sections 1–6 show: means, win counts and the ✅ cells.',
              '- A cell without ✅ means "more wins, but not statistically shown"; do not call it significantly better.',
              '- Scores describe these datasets and this comparator set, not universal superiority.', '']
    lines += list(extra_sections)
    lines += ['## Output files', '',
              '- all_runs.csv: every fit (one row per dataset, method and seed); Predictions/: saved labels and memberships',
              '- summary.csv: mean and SD per dataset and method; ranking.csv: wins and ranks; lc_significance.csv: section 6',
              '- friedman.csv, nemenyi.csv, mean_ranks.csv, wilcoxon_holm.csv: omnibus and all-pairs tests',
              f'- Figures/: {figures} PNG figures at 600 DPI (docs/FIGURES.md); verification.json: the audit; '
              'experiment_manifest.json: code, data and library versions']
    (output / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    significance = pd.concat(all_tests, ignore_index=True)
    return significance
