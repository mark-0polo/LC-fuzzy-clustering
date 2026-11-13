"""Figure labels, method colours and font sizes (points).

Every figure uses Matplotlib's constrained layout, which sizes margins and gaps from the text
actually drawn, so these sizes can be changed without re-tuning any padding. tests/
test_figure_layout.py checks that no text overlaps or leaves the figure.
"""
DISPLAY={'pc':'PC','pe':'PE','sil':'Sil','pur':'Pur','nmi':'NMI','ari':'ARI'}
SHORT={'Image Segmentation':'Image Seg.','Breast Cancer':'Breast Canc.','wall-following':'wall-follow.'}
COLORS={'FCM':'#264653','PCM':'#2A9D8F','PFCM':'#E9C46A','DPC':'#F4A261',
        'eFCM':'#E76F51','MC-FCM':'#577590',
        'LC-FixedT':'#9AA8A3','LC-FixedHalfT':'#466C82','LC-SoftFixedT':'#8C7853',
        'k-means++':'#B5838D'}

FONT = {
    'title': 24,         # figure title
    'panel': 20,         # caption of a panel in a combined figure, e.g. "(a) Adjusted Rand Index"
    'row': 17,           # dataset-group row title, e.g. "Held-out datasets (8)"
    'label': 17,         # axis labels
    'tick': 15,          # tick labels
    'legend': 16,        # method legend
    'note': 14,          # footnote under a figure
    'value': 14,         # numbers written on bars
    'cell': 13,          # numbers inside heatmap cells
    'panel_tick': 13,    # tick labels inside combined figures
    'panel_row': 15,     # row titles inside combined figures
    'panel_cell': 10,    # numbers inside the cells of the combined heatmap figure
    'diagram': 14,       # body text of the pipeline diagram
}
