"""Soft-teal benchmark plots saved only as high-resolution 600-DPI PNGs.

Every figure uses Matplotlib's constrained layout, so margins and gaps follow the text that is
actually drawn: font sizes (benchmark/figure_style.FONT) can change without re-tuning padding.
text_layout_problems() reports any text that overlaps other text or leaves the figure; the export
prints such problems and tests/test_figure_layout.py requires there are none.
"""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.text import Text
import numpy as np

from benchmark.config import PRIMARY, DATASETS, GROUPS, metric_methods
from benchmark.figure_style import DISPLAY, SHORT, COLORS, FONT
from algo._constants import BENCHMARK_METRICS
from benchmark.checkpoints import atomic_json
from benchmark.figure_data import (FOCUS_METHOD, BASELINE_METHOD,
    HEATMAP_NAMES, FIGURE_NAMES, relative_metric_changes, iteration_reductions,
    amortized_iteration_costs, fuzzifier_statistics)
from matplotlib.container import BarContainer
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Rectangle

METRICS = list(BENCHMARK_METRICS)

SOFT_TEAL = LinearSegmentedColormap.from_list('reference_soft_teal', ['#EDF7F5', '#2A9D8F'])
SOFT_TEAL.set_bad('#F4F4F4')
def style():
    """Matplotlib settings built from the current FONT table."""
    return {
    'font.family': 'DejaVu Sans', 'font.size': FONT['tick'], 'axes.titlesize': FONT['row'],
    'axes.labelsize': FONT['label'], 'xtick.labelsize': FONT['tick'], 'ytick.labelsize': FONT['tick'],
    'legend.fontsize': FONT['legend'], 'figure.titlesize': FONT['title'],
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.8,
    'axes.facecolor': 'white', 'figure.facecolor': 'white', 'savefig.facecolor': 'white',
    'grid.color': '#B0B0B0', 'grid.alpha': 0.25, 'grid.linewidth': 0.8,
    'figure.constrained_layout.h_pad': 0.08, 'figure.constrained_layout.w_pad': 0.08,
    'figure.constrained_layout.hspace': 0.03, 'figure.constrained_layout.wspace': 0.03,
    }


STYLE = style()
FULL_NAMES = {'pc':'Partition Coefficient','pe':'Partition Entropy',
              'sil':'Silhouette Coefficient','pur':'Purity',
              'nmi':'Normalized Mutual Information','ari':'Adjusted Rand Index'}
FOCUS_COLOR = '#6D597A'
WINNER_COLOR = '#C0392B'            # outline of the best method on each dataset in the heatmaps
NEGATIVE_COLOR = '#E76F51'
# Heatmaps grow with the number of datasets so their annotated cells stay legible.
SCALE = max(1.0, len(DATASETS) / 12)
# Bar charts draw one row per dataset group, so each row holds at most 8 datasets.
GROUP_TITLES = {'development': 'Development datasets', 'regime': 'Regime datasets',
                'held-out': 'Held-out datasets'}
DATASET_ROWS = [(group, [d for d in DATASETS if GROUPS[d] == group])
                for group in GROUP_TITLES if any(GROUPS[d] == group for d in DATASETS)]
ROW_SLOTS = max(len(names) for _, names in DATASET_ROWS)


def new_figure(width, height):
    return plt.figure(figsize=(width, height), layout='constrained')


def add_note(fig, text):
    """Footnote under the figure, wrapped to the figure width at the current note size."""
    import textwrap
    characters = max(20, int(fig.get_figwidth() * 72 / (0.6 * FONT['note'])))
    fig.supxlabel(textwrap.fill(text, characters), fontsize=FONT['note'])


def dataset_labels(datasets=DATASETS):
    return [SHORT.get(d, d) for d in datasets]


# ----------------------------------------------------------------------------- heatmaps

def metric_matrix(summary, metric, datasets=DATASETS, methods=PRIMARY):
    """Methods are rows; datasets are columns, as in the supplied heatmaps."""
    return summary.pivot(index='method', columns='dataset', values=metric + '_mean').reindex(
        index=methods, columns=datasets).to_numpy(dtype=float)


def heatmap_winners(matrix, metric):
    """[row, column] of the best value in every dataset column (PE: lowest); ties all count.

    Values are compared after rounding to ten decimals, as for the win counts.
    """
    winners = []
    for col in range(matrix.shape[1]):
        values = np.round(matrix[:, col], 10)
        finite = values[np.isfinite(values)]
        if len(finite):
            best = finite.min() if metric == 'pe' else finite.max()
            winners += [[int(row), col] for row in np.flatnonzero(values == best)]
    return winners


def draw_heatmap(fig, ax, matrix, metric, datasets, compact=False, methods=PRIMARY):
    finite = matrix[np.isfinite(matrix)]
    low, high = (float(finite.min()), float(finite.max())) if len(finite) else (0.0, 1.0)
    if high <= low:
        high = low + 1.0
    norm = Normalize(low, high)
    mesh = ax.pcolormesh(np.ma.masked_invalid(matrix), cmap=SOFT_TEAL, norm=norm,
                         edgecolors=(1, 1, 1, 0.28), linewidth=0.4, rasterized=False)
    tick, cell = (FONT['panel_tick'], FONT['panel_cell']) if compact else (FONT['tick'], FONT['cell'])
    ax.set_ylim(len(methods), 0)
    ax.set_xlim(0, len(datasets))
    ax.set_xticks(np.arange(len(datasets)) + 0.5, [SHORT.get(ds, ds) for ds in datasets],
                  rotation=35, ha='right', rotation_mode='anchor')
    ax.set_yticks(np.arange(len(methods)) + 0.5, methods)
    ax.tick_params(labelsize=tick, length=3)
    ax.set_title('Heatmap of ' + DISPLAY[metric], fontsize=FONT['panel_row'] if compact else FONT['title'], pad=8)
    for row in range(len(methods)):
        for col in range(len(datasets)):
            value = matrix[row, col]
            text = f'{value:.3f}' if np.isfinite(value) else '--'
            color = 'white' if np.isfinite(value) and norm(value) > 0.55 else '#333333'
            ax.text(col + 0.5, row + 0.5, text, ha='center', va='center', color=color, fontsize=cell)
    ax._cell_texts = list(ax.texts)
    # Outline the best method on every dataset.
    ax._winner_cells = heatmap_winners(matrix, metric)
    for row, col in ax._winner_cells:
        ax.add_patch(Rectangle((col + .05, row + .07), .9, .86, fill=False, edgecolor=WINNER_COLOR,
                               linewidth=2.6, joinstyle='miter', zorder=3))
    colorbar = fig.colorbar(mesh, ax=ax, fraction=0.03, pad=0.02, shrink=0.9)
    if colorbar.solids is not None:
        colorbar.solids.set_rasterized(False)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(labelsize=tick, length=3, color='#555555')
    return mesh


def fit_cell_text(fig, margin=0.9):
    """Shrink heatmap cell numbers only where they would not fit inside their cells."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        texts = getattr(ax, '_cell_texts', None)
        if not texts:
            continue
        (x0, y0), (x1, y1) = ax.transData.transform([(0, 0), (1, 1)])
        cell_width, cell_height = abs(x1 - x0), abs(y1 - y0)
        widest = max(t.get_window_extent(renderer).width for t in texts)
        tallest = max(t.get_window_extent(renderer).height for t in texts)
        factor = min(1., margin * cell_width / widest, margin * cell_height / tallest)
        if factor < 1.:
            for t in texts:
                t.set_fontsize(t.get_fontsize() * factor)
    return fig


def heatmap_figure(summary, metric, methods):
    fig = new_figure(15 * SCALE, 10.5)
    ax = fig.subplots()
    draw_heatmap(fig, ax, metric_matrix(summary, metric, DATASETS, methods), metric, DATASETS, methods=methods)
    add_note(fig, 'Outlined cell: best method on that dataset ' + ('(lowest PE)' if metric == 'pe' else '(highest value)')
             + '; ties are all outlined. "--" = undefined for that method.')
    return fit_cell_text(fig)


# ----------------------------------------------------------------------------- bar charts

def grouped_bars(ax, values, title=None, ylabel='Score', logarithmic=False, source=None,
                 datasets=DATASETS, slots=None, rotate=False):
    """Reference grouped-bar style; missing values remain gaps, never zero bars."""
    methods = list(values.columns)
    values = values.reindex(datasets)
    positions = np.arange(len(datasets))
    width = .80/max(1,len(methods))
    for index,method in enumerate(methods):
        color = FOCUS_COLOR if method==FOCUS_METHOD else COLORS[method]
        ax.bar(positions+(index-(len(methods)-1)/2)*width,values[method],
               width=width,label=method,color=color,edgecolor=(0,0,0,.40),linewidth=.45)
    if rotate:
        ax.set_xticks(positions,dataset_labels(datasets),rotation=30,ha='right',rotation_mode='anchor')
    else:
        ax.set_xticks(positions,dataset_labels(datasets))
    if slots:
        # Equal bar widths in every row; rotated names need room before the first bar.
        ax.set_xlim(-1.3 if rotate else -.6,slots-.4)
    if title:
        ax.set_title(title,pad=10)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True,alpha=.25)
    if logarithmic:
        ax.set_yscale('log')
    ax._plot_source = source
    ax._plot_datasets = list(datasets)


def grouped_bar_rows(container, values, ylabel='Score', logarithmic=False, source=None, compact=False):
    """One row of bars per dataset group in a figure or subfigure, all rows sharing the y axis."""
    axes = container.subplots(len(DATASET_ROWS), 1, sharey=True, squeeze=False)[:, 0]
    for ax, (group, names) in zip(axes, DATASET_ROWS):
        grouped_bars(ax, values, None, None, logarithmic, source, names, ROW_SLOTS, rotate=compact)
        ax.set_title(f'{GROUP_TITLES[group]} ({len(names)})', loc='left', pad=6,
                     fontsize=FONT['panel_row'] if compact else FONT['row'])
        ax.tick_params(labelsize=FONT['panel_tick'] if compact else FONT['tick'])
    container.supylabel(ylabel, fontsize=FONT['label'])
    return list(axes)


def method_legend(fig, axes, title=None):
    """Method colours above the plots; a figure title, if given, sits above them as one block."""
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside upper center', ncol=5, frameon=False, fontsize=FONT['legend'],
               title=title, title_fontsize=FONT['title'])


def metric_bars(summary, metric, methods):
    included = metric_methods(metric, methods)
    values = summary.pivot(index='dataset',columns='method',values=metric+'_mean')
    return values.reindex(index=DATASETS,columns=included)


def standalone_bar_figure(values,title,ylabel='Score',logarithmic=False,source=None,note=None):
    fig = new_figure(16, 14)
    axes = grouped_bar_rows(fig, values, ylabel, logarithmic, source)
    method_legend(fig, axes, title)
    if note:
        add_note(fig, note)
    return fig


def convergence_runtime_figure(summary,methods):
    fig = new_figure(20, 13)
    panels = fig.subfigures(1, 2)
    for panel, field, title, ylabel, scale, included in [
        (panels[0],'iterations_mean','(a) Iterations to convergence','Iterations',1.,[m for m in methods if m!='DPC']),
        (panels[1],'total_time_s_mean','(b) Total runtime in milliseconds','Runtime (ms)',1000.,methods)]:
        values=summary.pivot(index='dataset',columns='method',values=field).reindex(index=DATASETS,columns=included)*scale
        axes = grouped_bar_rows(panel, values, ylabel, True, {'kind':'summary','field':field,'scale':scale}, compact=True)
        panel.suptitle(title, fontsize=FONT['panel'])
        if field == 'total_time_s_mean':
            legend_axes = axes
    method_legend(fig, legend_axes)
    add_note(fig, 'DPC is non-iterative. Runtime includes preprocessing; evaluation is excluded.')
    return fig


def win_rank_figure(ranking,methods):
    ordered=ranking.set_index('method').reindex(methods)
    fig = new_figure(17, 8.5)
    axes = fig.subplots(1, 2)
    for ax,field,title,ylabel in [
        (axes[0],'wins','Win count: ARI, NMI, Purity','First-place credits'),
        (axes[1],'external_mean_rank','Class-agreement mean rank','Average rank (lower is better)')]:
        ax.bar(np.arange(len(methods)),ordered[field],color=[FOCUS_COLOR if m==FOCUS_METHOD else COLORS[m] for m in methods],
               edgecolor=(0,0,0,.4),linewidth=.5,label=field)
        ax.set_xticks(np.arange(len(methods)),methods,rotation=40,ha='right',rotation_mode='anchor')
        ax.set_ylabel(ylabel);ax.set_title(title,fontsize=FONT['panel']);ax.grid(True,alpha=.25)
        ax._plot_source={'kind':'ranking','field':field,'methods':methods}
    axes[1].set_ylim(float(ordered.external_mean_rank.max())*1.08,0)
    add_note(fig, 'Ties count. Wins and ranks use only ARI, NMI and Purity; PC, PE and Silhouette are not counted.')
    return fig


def signed_bar_figure(values,labels,title,ylabel,source,positive_color,note=None):
    fig = new_figure(max(14,.8*len(values)), 8.5)
    ax = fig.subplots()
    bars=ax.bar(np.arange(len(values)),values,color=[positive_color if v>=0 else NEGATIVE_COLOR for v in values],
                edgecolor=(0,0,0,.5),linewidth=.6,label='percent')
    rotation = {'rotation':35,'ha':'right','rotation_mode':'anchor'} if len(values)>8 else {}
    ax.set_xticks(np.arange(len(values)),labels,**rotation)
    ax.set_title(title,fontsize=FONT['title'])
    ax.set_ylabel(ylabel);ax.grid(True,alpha=.25);ax.axhline(0,color='black',linewidth=.9)
    ax.margins(y=.18)
    for bar,value in zip(bars,values):
        if np.isfinite(value):
            ax.annotate(f'{value:.1f}',(bar.get_x()+bar.get_width()/2,value),
                        xytext=(0,5 if value>=0 else -5),textcoords='offset points',
                        ha='center',va='bottom' if value>=0 else 'top',fontsize=FONT['value'])
    ax._plot_source=source
    if note:
        add_note(fig, note)
    return fig


def fuzzifier_figure(frame):
    stats=fuzzifier_statistics(frame)
    fig = new_figure(13*SCALE, 8.5)
    ax = fig.subplots(); x=np.arange(len(DATASETS))
    bars=ax.errorbar(x,stats.m_i_mean,yerr=np.vstack([stats.m_i_mean-stats.m_i_min,stats.m_i_max-stats.m_i_mean]),
                    fmt='o',color=COLORS['MC-FCM'],ecolor='#555555',elinewidth=1.5,capsize=5,markersize=8)
    ax.set_xticks(x,dataset_labels(),rotation=35,ha='right',rotation_mode='anchor')
    ax.set_title('MC-FCM per-point fuzzifier statistics',fontsize=FONT['title'])
    ax.set_ylabel('Per-point fuzzifier '+r'$m_i$');ax.grid(True,alpha=.25);ax.margins(y=.16)
    add_note(fig, 'Markers: dataset means. Ranges: mean per-point minimum and maximum across seeds.')
    ax._plot_source={'kind':'fuzzifier','method':'MC-FCM'}
    ax._range_artists=bars
    return fig


def temperature_controls_figure(summary,methods):
    controls=[m for m in ['LC-FixedT','LC-FixedHalfT','LC-SoftFixedT'] if m in methods]
    average=summary.groupby('method')[[m+'_mean' for m in METRICS]].mean()
    fig = new_figure(17, 11)
    for metric,ax in zip(METRICS,fig.subplots(2,3).ravel()):
        ax.bar(np.arange(len(controls)),average.loc[controls,metric+'_mean'],
               color=[FOCUS_COLOR if m==FOCUS_METHOD else COLORS[m] for m in controls],
               edgecolor=(0,0,0,.4),linewidth=.4,label=metric)
        ax.set_xticks(np.arange(len(controls)),[c.replace('LC-','LC-\n') for c in controls])
        ax.set_title(DISPLAY[metric]+(' (lower better)' if metric=='pe' else ' (higher better)'),fontsize=FONT['row'])
        ax.grid(axis='y',alpha=.2)
        ax._plot_source={'kind':'macro','field':metric+'_mean','methods':controls}
    fig.suptitle('Matched geometry: fixed-temperature controls\nEqual-weight dataset means',fontsize=FONT['title'])
    return fig


# ----------------------------------------------------------------------------- diagram

def pipeline_figure():
    """Actual LC-FixedHalfT flow, drawn as text and geometric shapes."""
    fig = new_figure(23, 9.5)
    ax = fig.subplots(); ax.set_xlim(-1,101); ax.set_ylim(-2,35); ax.axis('off')
    body, heading, formula = FONT['diagram'], FONT['diagram'] + 2, FONT['diagram'] + 3
    def box(x,y,w,h):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.22,rounding_size=.45',
                                   facecolor='white',edgecolor='black',linewidth=1.2))
    def text(x,y,s,size=body,bold=False):
        ax.text(x,y,s,ha='center',va='center',fontsize=size,fontweight='bold' if bold else 'normal')
    def arrow(start,end):
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=16,color='black',linewidth=1.1))
    for x,w in [(0,12),(14,14),(30,15)]:box(x,4,w,25)
    text(6,26,'1. Standardize\ndata',heading,True)
    text(6,18.5,'Exclude labels / IDs\n\nFeature mean = 0\nFeature std = 1')
    text(6,8,r'Output: $X$')
    text(21,26,'2. Local-mean\ncoordinates',heading,True)
    text(21,19.5,'Group duplicate rows\nwith their counts.\n\nAverage each row\nwith nearby rows.')
    text(21,12,r'$q_i=\frac{1}{2}(x_i+\ell_i)$',formula)
    text(21,7,'Expand duplicates;\nrecenter Q.')
    text(37.5,26,'3. Freeze\ntemperature',heading,True)
    text(37.5,20,'s = mean squared radius\nr = numerical rank')
    text(37.5,15,r'$t=T/2=\frac{s}{2(r+2)}$',formula)
    text(37.5,8,'Random positive U;\nnormalize each row.\nKeep t fixed.')
    box(48,4,31,25);text(63.5,27,'4. Alternate updates',heading,True)
    box(49,5,14,19);box(64.5,5,13.5,19)
    text(56,22,'4a. Update centres',body,True)
    text(56,16,r'$v_k=\frac{\sum_i u_{ik}q_i}{\sum_i u_{ik}}$',formula)
    text(56,9,'First-power\nmembership weights.')
    text(71.25,22,'4b. Memberships',body,True)
    text(71.25,16,r'$u_{ik}=\frac{e^{-\|q_i-v_k\|^2/t}}{\sum_j e^{-\|q_i-v_j\|^2/t}}$',body+1)
    text(71.25,9,'Stable softmax;\neach row sums to 1.')
    ax.add_patch(Polygon([[90,24],[100,17],[90,10],[80,17]],closed=True,fill=False,edgecolor='black',linewidth=1.2))
    text(90,20.2,'5. Stop?',heading,True)
    text(90,16,r'$\Delta U<\epsilon$ and $\Delta V_{rel}<\epsilon$'+'\nor 5,000 cycles?',body-2)
    box(82,0,17,6);text(90.5,4,'6. Output',heading,True)
    text(90.5,1.6,'U, V and convergence flag',body-2)
    for start,end in [((12.2,17),(13.8,17)),((28.2,17),(29.8,17)),((45.2,17),(47.8,17)),
                       ((63.2,15),(64.3,15)),((79.2,17),(79.8,17)),((90,10),(90,6.2))]:arrow(start,end)
    ax.plot([90,90,70],[24,32,32],color='black',linewidth=1.1);arrow((70,32),(70,29.2))
    text(85,33.3,'No',body);text(92.5,8,'Yes',body)
    fig.suptitle('LC-FixedHalfT pipeline',fontsize=FONT['title'])
    add_note(fig, 'Tolerance: 10⁻⁵. Reaching the iteration cap is reported as nonconvergence. All-constant data returns a flagged uniform solution.')
    return fig


# ----------------------------------------------------------------------------- layout check

def _drawn_texts(fig, renderer):
    """Visible, non-empty text that Matplotlib actually draws, with pixel boxes.

    Returns the (text, box) pairs and, for tick labels, the axis each label belongs to.
    """
    hidden, owner = set(), {}
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            drawn = {id(t) for tick in axis._update_ticks() for t in (tick.label1, tick.label2)} if ax.axison else set()
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                for t in (tick.label1, tick.label2):
                    owner[id(t)] = axis
                    if id(t) not in drawn:
                        hidden.add(id(t))
    items = []
    for text in fig.findobj(Text):
        if id(text) in hidden or not text.get_visible() or not text.get_text().strip():
            continue
        box = text.get_window_extent(renderer)
        if box.width > 0 and box.height > 0:
            items.append((text, box))
    return items, owner


def text_layout_problems(fig, tolerance=1.0):
    """Text that overlaps other text or extends past the figure edge, as readable messages."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    items, owner = _drawn_texts(fig, renderer)
    edge = fig.bbox
    problems = [f'outside the figure: {t.get_text()[:40]!r}' for t, b in items
                if b.x0 < edge.x0 - tolerance or b.x1 > edge.x1 + tolerance
                or b.y0 < edge.y0 - tolerance or b.y1 > edge.y1 + tolerance]
    boxes = np.array([[b.x0, b.y0, b.x1, b.y1] for _, b in items]) if items else np.zeros((0, 4))
    width = np.minimum(boxes[:, None, 2], boxes[None, :, 2]) - np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    height = np.minimum(boxes[:, None, 3], boxes[None, :, 3]) - np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    for i, j in zip(*np.nonzero(np.triu((width > tolerance) & (height > tolerance), 1))):
        a, b = items[i][0], items[j][0]
        same_axis = id(a) in owner and owner.get(id(a)) is owner.get(id(b))
        if same_axis and a.get_rotation() % 360 and a.get_rotation() == b.get_rotation():
            # Parallel rotated tick labels: their boxes overlap by design. The text lines do not
            # touch when the perpendicular gap between them exceeds one line of text.
            step = abs(items[i][1].x1 - items[j][1].x1)
            line = a.get_fontsize() * fig.dpi / 72
            if step * abs(np.sin(np.radians(a.get_rotation()))) >= line:
                continue
        problems.append(f'overlap: {a.get_text()[:40]!r} and {b.get_text()[:40]!r}')
    return problems


# ----------------------------------------------------------------------------- export

def capture_plot_data(fig):
    """Capture actual drawn bar heights/range markers, not a second input copy."""
    axes=[]
    def finite(values):return [float(v) if np.isfinite(v) else None for v in values]
    for ax in fig.axes:
        source=getattr(ax,'_plot_source',None)
        if source is None:continue
        item={'source':source,'series':[]}
        if hasattr(ax,'_plot_datasets'):item['datasets']=ax._plot_datasets
        for container in ax.containers:
            if isinstance(container,BarContainer):
                item['series'].append({'label':container.get_label(),
                                       'values':finite([bar.get_height() for bar in container])})
        if source['kind']=='fuzzifier':
            artists=ax._range_artists
            segments=artists.lines[2][0].get_segments()
            item['series']=[{'label':'m_i_mean','values':finite(artists.lines[0].get_ydata())},
                            {'label':'m_i_min','values':finite([segment[:,1].min() for segment in segments])},
                            {'label':'m_i_max','values':finite([segment[:,1].max() for segment in segments])}]
        axes.append(item)
    return {'datasets':DATASETS,'axes':axes}


def build_figures(summary, methods, ranking, frame):
    """Yield (name, figure) for all 22 figures, in a fixed order."""
    for metric in METRICS:
        yield 'heatmap_'+DISPLAY[metric], heatmap_figure(summary, metric, methods)
    for metric in METRICS:
        title=FULL_NAMES[metric]+(' (lower is better)' if metric=='pe' else ' (higher is better)')
        note='PC/PE are defined only for native fuzzy partitions.' if metric in ('pc','pe') else None
        yield DISPLAY[metric], standalone_bar_figure(metric_bars(summary,metric,methods),title,
            source={'kind':'summary','field':metric+'_mean','scale':1.},note=note)
    for name,field,title,ylabel,scale,included,note in [
        ('iteration','iterations_mean','Iterations to convergence','Iterations',1.,[m for m in methods if m!='DPC'],'DPC is non-iterative; no iteration count is assigned.'),
        ('time','total_time_s_mean','Total runtime','Runtime (ms)',1000.,methods,'Fresh whole-fit wall time, including preprocessing; evaluation excluded.')]:
        values=summary.pivot(index='dataset',columns='method',values=field).reindex(index=DATASETS,columns=included)*scale
        yield name, standalone_bar_figure(values,title,ylabel,True,{'kind':'summary','field':field,'scale':scale},note)
    yield 'convergence_and_runtime', convergence_runtime_figure(summary,methods)
    yield 'win_rank_summary', win_rank_figure(ranking,methods)
    changes=relative_metric_changes(summary)
    yield 'effect_size_lc_fixed_half_t_vs_fcm', signed_bar_figure(changes.percent.to_numpy(),[DISPLAY[m] for m in METRICS],
        FOCUS_METHOD+' effect relative to standard FCM','Mean direction-adjusted change vs FCM (%)',
        {'kind':'relative_change','focus':FOCUS_METHOD,'baseline':BASELINE_METHOD},FOCUS_COLOR,
        'Equal-weight dataset percentage changes; PE direction reversed. This is not a standardized statistical effect size.')
    yield 'iteration_reduction_lc_fixed_half_t_vs_fcm', signed_bar_figure(iteration_reductions(summary),dataset_labels(),
        FOCUS_METHOD+' convergence iterations relative to FCM','Iteration reduction vs FCM (%)',
        {'kind':'iteration_reduction','focus':FOCUS_METHOD,'baseline':BASELINE_METHOD},'#2A9D8F')
    yield 'per_iteration_cost_profile', standalone_bar_figure(amortized_iteration_costs(frame,methods),
        'Observed per-iteration cost profile','Amortized ms per outer iteration',True,{'kind':'iteration_cost'},
        'Mean of fit time / outer iterations. Includes preprocessing and PCM/PFCM initial FCM; this is not direct iteration profiling.')
    yield 'mcfcm_fuzzifier_statistics', fuzzifier_figure(frame)
    yield 'lc_fixed_half_t_pipeline', pipeline_figure()
    yield 'temperature_controls', temperature_controls_figure(summary,methods)


def export_figures(output,summary,methods,ranking,frame):
    figures=output/'Figures';figures.mkdir(exist_ok=True)
    previous=json.loads((figures/'manifest.json').read_text()) if (figures/'manifest.json').exists() else {'files':[]}
    manifest=[];layout={}
    with plt.rc_context(style()):
        for name,fig in build_figures(summary,methods,ranking,frame):
            path=figures/(name+'.png')
            problems=text_layout_problems(fig)
            if problems:
                layout[name]=problems
                print(f'WARNING: {name}.png has {len(problems)} text layout problem(s), e.g. {problems[0]}',flush=True)
            data=capture_plot_data(fig) if name not in HEATMAP_NAMES else None
            save_png(fig,path,heatmap=name in HEATMAP_NAMES,plot_data=data)
            manifest.append({'path':path.relative_to(output).as_posix(),
                             'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
            plt.close(fig)
            print('Figure generated:',path.name,flush=True)
    assert {p['path'].split('/')[-1][:-4] for p in manifest}==set(FIGURE_NAMES)
    source_root=Path(__file__).resolve().parent
    rendering_sources=[source_root/'figures.py',source_root/'benchmark/figure_data.py',source_root/'benchmark/figure_style.py']
    atomic_json(figures/'manifest.json',{'suite':'reference_complete_v3','style':'Grouped bars in dataset-group rows and diagrams; soft-teal heatmaps with the best method outlined; constrained layout. PNG only, 600 DPI.',
        'primary_methods':methods,'focus_method':FOCUS_METHOD,'fuzzifier_method':'MC-FCM',
        'text_layout_problems':layout,
        'rendering_source_sha256':{p.relative_to(source_root).as_posix():hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in rendering_sources},
        'files':manifest})
    # Prune only obsolete files previously generated by this program (runtime/wins).
    current={item['path'] for item in manifest}
    for item in previous['files']:
        path=output/item['path']
        if item['path'] not in current and path.parent.resolve()==figures.resolve() and path.suffix=='.png':
            path.unlink(missing_ok=True)
    return len(manifest)


def save_png(fig, path, heatmap=False, plot_data=None):
    """Save the displayed plot and its actual heatmap text for independent auditing."""
    metadata = {'Description': 'Reproducible clustering benchmark; 600 DPI.'}
    if heatmap:
        annotations = [{'text': label.get_text(), 'position': list(label.get_position())}
                       for label in fig.axes[0].texts]
        metadata['HeatmapAnnotations'] = json.dumps(annotations, separators=(',', ':'))
        metadata['HeatmapWinners'] = json.dumps(getattr(fig.axes[0], '_winner_cells', []), separators=(',', ':'))
    if plot_data is not None:metadata['PlotData']=json.dumps(plot_data,separators=(',',':'),allow_nan=False)
    if hasattr(path,'write'):
        fig.savefig(path,format='png',dpi=600,bbox_inches='tight',metadata=metadata)
        return
    destination=Path(path)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='wb',prefix='.'+destination.stem+'.',suffix='.tmp.png',
                                         dir=destination.parent,delete=False) as stream:
            temporary=Path(stream.name)
            fig.savefig(stream,format='png',dpi=600,bbox_inches='tight',metadata=metadata)
            stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,destination)
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)
