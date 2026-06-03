# generate_fig1.py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 12)
ax.set_ylim(0, 10)
ax.axis('off')

# Colors
layer_color = '#e8f0fe'
box_color = '#4a90e2'
text_color = 'white'
arrow_color = '#333333'

# Helper to draw a rounded box
def draw_box(x, y, width, height, label, fontsize=10, facecolor=box_color, edgecolor='black'):
    rect = mpatches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.05",
                                   facecolor=facecolor, edgecolor=edgecolor, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, label, ha='center', va='center', fontsize=fontsize,
            color=text_color, weight='bold')

# Helper for arrow
def draw_arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=arrow_color, lw=1.5))

# Layer backgrounds
layers = [
    (0.5, 8.0, 11.0, 1.2, "Data Acquisition Layer"),
    (0.5, 5.8, 11.0, 1.2, "Processing & Feature Engineering"),
    (0.5, 3.0, 11.0, 1.8, "Machine Learning Layer"),
    (0.5, 0.5, 11.0, 1.2, "Decision & Deployment Layer")
]
for x, y, w, h, label in layers:
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                   facecolor=layer_color, edgecolor='gray', linewidth=1)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h - 0.3, label, ha='center', va='top', fontsize=12,
            color='black', weight='bold')

# Data Acquisition components
draw_box(1.0, 8.5, 2.0, 0.7, "IIoT Sensors", fontsize=9)
draw_box(4.0, 8.5, 2.0, 0.7, "PLCs / SCADA", fontsize=9)
draw_box(7.0, 8.5, 2.0, 0.7, "Legacy Systems", fontsize=9)
draw_box(3.5, 7.3, 4.0, 0.7, "Edge Gateway (Raspberry Pi)", fontsize=9)

# Arrows to edge gateway
draw_arrow(2.0, 8.5, 4.0, 7.7)
draw_arrow(5.0, 8.5, 5.5, 7.7)
draw_arrow(8.0, 8.5, 6.5, 7.7)

# Processing components
draw_box(1.0, 6.2, 2.5, 0.7, "Data Cleaning & Sync", fontsize=9)
draw_box(4.5, 6.2, 2.5, 0.7, "Feature Extraction", fontsize=9)
draw_box(8.0, 6.2, 2.5, 0.7, "Feature Store", fontsize=9)
draw_arrow(3.5, 6.55, 4.5, 6.55)
draw_arrow(7.0, 6.55, 8.0, 6.55)

# Arrow from edge to processing
draw_arrow(5.5, 7.3, 2.25, 6.55)

# ML components
draw_box(1.0, 3.5, 2.8, 0.8, "LSTM Autoencoder\n(Anomaly Detection)", fontsize=8)
draw_box(4.5, 3.5, 2.8, 0.8, "1D-CNN + Attention\n(Fault Classification)", fontsize=8)
draw_box(8.0, 3.5, 2.8, 0.8, "Attention-LSTM + XGBoost\n(RUL Prediction)", fontsize=8)
draw_arrow(3.8, 3.9, 4.5, 3.9)
draw_arrow(7.3, 3.9, 8.0, 3.9)

# Arrow from feature store to ML
draw_arrow(9.25, 6.2, 9.25, 4.3)  # down from feature store to middle of ML

# Decision components
draw_box(1.0, 1.0, 2.5, 0.7, "Maintenance Alerts", fontsize=9)
draw_box(4.5, 1.0, 2.5, 0.7, "Multi‑objective\nScheduler", fontsize=9)
draw_box(8.0, 1.0, 2.5, 0.7, "CMMS / MES\nIntegration", fontsize=9)
draw_arrow(3.5, 1.35, 4.5, 1.35)
draw_arrow(7.0, 1.35, 8.0, 1.35)

# Arrow from RUL to alerts
draw_arrow(9.25, 3.5, 9.25, 1.7)  # down from RUL to alerts

# Edge and cloud regions
# Edge: dashed box around gateway + processing layer
edge_rect = mpatches.FancyBboxPatch((0.3, 6.8), 4.0, 2.2, boxstyle="round,pad=0.02",
                                    facecolor='none', edgecolor='orange', linestyle='--', linewidth=2)
ax.add_patch(edge_rect)
ax.text(0.5, 9.0, "Edge", fontsize=10, weight='bold', color='orange', rotation=90)

# Cloud: dashed box around ML and decision layers
cloud_rect = mpatches.FancyBboxPatch((0.3, 0.2), 11.0, 6.6, boxstyle="round,pad=0.02",
                                     facecolor='none', edgecolor='green', linestyle='--', linewidth=2)
ax.add_patch(cloud_rect)
ax.text(0.5, 6.7, "Cloud", fontsize=10, weight='bold', color='green', rotation=90)

# Title
ax.text(6, 9.8, "IPMF – Intelligent Predictive Maintenance Framework",
        ha='center', va='center', fontsize=16, weight='bold')

plt.tight_layout()
plt.savefig('fig1.png', dpi=300, bbox_inches='tight')
print("Figure 1 saved as fig1.png")