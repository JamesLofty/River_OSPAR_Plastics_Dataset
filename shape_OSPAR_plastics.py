#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 20:50:11 2025

@author: jameslofty
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from collections import Counter
from matplotlib import gridspec
import re

data = pd.read_excel("synthetic_data/synthetic_data_top_25.xlsx")

# %%

unique_materials = np.array([
    'PO soft', 'PO hard', 'EPS', 'PS', 'Multilayer', 'Glass', 'PET', 
    'Metal', 'Textiles', 'Other plastics', 'Paper', 'Rubber', 'Wood'
], dtype=object)

palette = sns.color_palette('Set2', len(unique_materials))
material_colors = {material: palette[i] for i, material in enumerate(unique_materials)}
material_colors['Other plastics'] = 'tomato'
material_colors['Rubber'] = '#66bad9'

# %%

data_flat = data[
    # (data['Flatness_F'].between(0, 0.4)) &
    # (data['Elongation_E'].between(0, 1)) 
    # (data['long'].between(0, 0.025)) 
    # (data['intermediate'].between(0, 0.1)) 
    # (data['Diameter_d'].between(0, 0.1)) &
    # (data['Density_rho'].between(0, 1000))
    (data['flexibility'] == 'f')
]


plt.figure()
plt.scatter(data_flat['Flatness_F'], data_flat['Elongation_E'])
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()

percent = len(data_flat) / len(data) * 100
print(percent)

#%%
# data = data[data['common name'] == 'Cigarette filters']

# for i in data['common name'].unique():
#     df = data[data['common name'] == i]

#     fig, ax = plt.subplots(figsize=(3, 3))  # Create fig and ax

#     # Scatter plot
#     ax.scatter(df['Flatness_F'], df['Elongation_E'], alpha=0.3)

#     # KDE Contour over scatter plot
#     sns.kdeplot(
#         data=df,
#         x="Flatness_F",
#         y="Elongation_E",
#         fill=True,
#         cmap="Blues",
#         bw_adjust=2,
#         levels=10,
#         thresh=0.001,
#         alpha=0.8,
#         ax=ax  # Plot on the same Axes
#     )

#     # Axis limits and labels
#     ax.set_title(str(i), fontsize = 9)
#     ax.set_xlabel("$FL$ (-)")
#     ax.set_ylabel("$EL$ (-)")
#     ax.set_xlim(0, 1)
#     ax.set_ylim(0, 1)
#     # ax.legend(loc='upper right')

#     plt.tight_layout()

    
#     def sanitize_filename(name):
#         return re.sub(r'[<>:"/\\|?*]', '', name)

#     safe_i = sanitize_filename(i)
#     plt.savefig(f"python_figures/sup_kde/{safe_i}.png", format='png')

#     plt.show()
    
    # %%
fig = plt.figure(figsize=(5, 5))

gs = gridspec.GridSpec(2, 2, width_ratios=[5, 0.6], height_ratios=[0.6, 5],
                       wspace=0.05, hspace=0.05)

ax_scatter = plt.subplot(gs[1, 0])
ax_histx = plt.subplot(gs[0, 0], sharex=ax_scatter)
ax_histy = plt.subplot(gs[1, 1], sharey=ax_scatter)

# Scatter plot
# ax_scatter.scatter(data['Flatness_F'], data['Elongation_E'], alpha=0.3)

# KDE Contour over scatter plot
sns.kdeplot(
    data=data,
    x="Flatness_F",
    y="Elongation_E",
    fill=True,
    cmap="Blues",
    bw_adjust=2,
    levels=10,
    thresh=0.001,
    alpha=0.8,
    ax=ax_scatter
)

# # Marginal KDE plots
sns.kdeplot(data['Flatness_F'], ax=ax_histx, fill=True, bw = 0.2, alpha=0.5, linewidth=1)
sns.kdeplot(data['Elongation_E'], ax=ax_histy, fill=True, bw = 0.2, alpha=0.5, linewidth=1, vertical=True)

# # Hide marginal axis ticks and labels
ax_histx.axis('off')
ax_histy.axis('off')


markers = ['d', 's', 'D', '^', 'v', '<', '>', 'P', '*', 'X', 'H']
ordered_names = sorted(data['common name'].unique())
marker_map = {name: markers[i % len(markers)] for i, name in enumerate(ordered_names)}

# Loop over each (material, common name) group
for (material, common_name), df_group in data.groupby(['material', 'common name']):
    mean_elong = df_group['Elongation_E'].mean()
    mean_flat = df_group['Flatness_F'].mean()

    # Percentile-based error bars
    p10_elong = np.percentile(df_group['Elongation_E'], 25)
    p90_elong = np.percentile(df_group['Elongation_E'], 75)
    yerr = [[abs(mean_elong - p10_elong)], [abs(p90_elong - mean_elong)]]

    p10_flat = np.percentile(df_group['Flatness_F'], 25)
    p90_flat = np.percentile(df_group['Flatness_F'], 75)
    xerr = [[abs(mean_flat - p10_flat)], [abs(p90_flat - mean_flat)]]

    # Assign color by material, marker by common name
    color = material_colors.get(material, 'gray')
    marker = marker_map.get(common_name, 'o')

    if (df_group['flexibility'] == 'r').any():
        edge_color = 'blue'
    else:
        edge_color = 'red'

    ax_scatter.scatter(
        mean_flat, mean_elong,
        label=f'{common_name}',
        color=color,
        marker=marker,
        edgecolor = edge_color,
        lw = 1,
        s=100,
        zorder = 12
    )
    
    ax_scatter.errorbar(
        mean_flat, mean_elong,
        xerr=xerr, yerr=yerr,
        fmt=marker,
        markersize=10,
        color=color,
        alpha=0.5,
        zorder = 11
    )
    

ax_scatter.set_xlabel("$FL$ (-)")
ax_scatter.set_ylabel("$EL$ (-)")
ax_scatter.vlines(0.66, 0, 1, color='k', ls="--", zorder = 10)
ax_scatter.hlines(0.66, 0, 1, color='k', ls="--", zorder = 10)
ax_scatter.set_xlim(0, 1)
ax_scatter.set_ylim(0, 1)
ax_scatter.legend(loc='upper left', title = 'River-OSPAR category', bbox_to_anchor=(1.05, 1),   borderaxespad=0.)
plt.savefig(f"python_figures/EL vs FL.svg", format='svg')

plt.show()
