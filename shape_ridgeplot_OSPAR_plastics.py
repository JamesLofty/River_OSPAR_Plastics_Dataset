#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 22:02:35 2025

@author: jameslofty
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter, LogLocator, FormatStrFormatter
import glob

data = pd.read_excel("synthetic_data/synthetic_data_top25.xlsx")

data['CSF'] = data['L3 (cm)'] / np.sqrt(data['L1 (cm)'] * data['L2 (cm)'])

# %%
#######################################################################
unique_materials = np.array([
    'PO soft', 'PO hard', 'EPS', 'PS', 'Multilayer', 'Glass', 'PET', 
    'Metal', 'Textiles', 'Other plastics', 'Paper', 'Rubber', 'Wood'
], dtype=object)

palette = sns.color_palette('Set2', len(unique_materials))
material_colors = {material: palette[i] for i, material in enumerate(unique_materials)}
material_colors['Other plastics'] = 'tomato'
material_colors['Rubber'] = '#66bad9'

# %%

# If percentages are in 0–100 range, convert them to 0–1
data['river_prop'] = data['Average_Percentage_Rivers'] / 100
data['bank_prop'] = data['Average_Percentage_Riverbanks'] / 100

weight_data = data[['Common name', 'river_prop', 'bank_prop']].drop_duplicates()

mean_by_category = data.groupby('Common name')['L1 (cm)'].mean()
merged = weight_data.merge(mean_by_category, on='Common name')
merged.rename(columns={'L1 (cm)': 'mean_size'}, inplace=True)

merged['river_prop'] /= merged['river_prop'].sum()
merged['bank_prop'] /= merged['bank_prop'].sum()

# Reuse the weighted mean sizes
river_mean_size = np.dot(merged['river_prop'], merged['mean_size'])
bank_mean_size = np.dot(merged['bank_prop'], merged['mean_size'])

# Calculate weighted standard deviation
river_std = np.sqrt(np.dot(merged['river_prop'], (merged['mean_size'] - river_mean_size) ** 2))
bank_std = np.sqrt(np.dot(merged['bank_prop'], (merged['mean_size'] - bank_mean_size) ** 2))

print(f"Estimated mean size in rivers: {river_mean_size:.2f} cm ± {river_std:.2f} cm")
print(f"Estimated mean size on riverbanks: {bank_mean_size:.2f} cm ± {bank_std:.2f} cm")


# %%
mean_by_category = data.groupby('Common name')['L1 (cm)'].mean()

sorted_means = mean_by_category.sort_values(ascending=False)
print(sorted_means)

# %%
ordered_names = (
    data.groupby('Common name')
    .apply(lambda x: x['Average_Percentage_Rivers'].max())
    .sort_values(ascending=False)
    .index
)
# Ensure 'Common name' follows this order
data['Common name'] = pd.Categorical(data['Common name'], categories=ordered_names, ordered=True)
#######################################################################
# --- Assign Colors to Common Names ---
common_name_colors = {}
for name in data['Common name'].unique():
    material = data[data['Common name'] == name]['material'].iloc[0]
    common_name_colors[name] = material_colors.get(material, 'gray')  # fallback to gray if material not found

# %%


def ridgeplot(data, column, xlabel, x_min, x_max, log, scale, scale_markers):
    figsize = (1.3 , 7)
    sns.set_theme(rc={"axes.facecolor": (0, 0, 0, 0), "font.family": "DejaVu Sans"})
    g = sns.FacetGrid(
        data, 
        row="Common name", 
        hue="Common name", 
        palette=common_name_colors,
        aspect=5,  
        height=0.8,
        sharey=False,   # <-- IMPORTANT

    )
    
    g.fig.set_size_inches(figsize)  # Overall figure size
    
    g.map_dataframe(sns.kdeplot, x=column, fill=True, alpha=0.8, log_scale=log, bw_adjust=2)
    g.map_dataframe(sns.kdeplot, x=column, fill=False, alpha=0.8, log_scale=log, lw = 1, bw_adjust=2, color = "k")
    
    for ax, label in zip(g.axes.flat, ordered_names):
        ax.text(-0.1, 0.1, label, color='black', ha="right", va="center", transform=ax.transAxes)
    
    g.fig.subplots_adjust(hspace=-0.6)  
    g.set_titles("")
    g.set(ylabel='')
    # g.set(yscale='log')
    g.set(yticks=[])
    g.set(xlabel=xlabel)
    
    g.set(xlim=(x_min, x_max))

    # g.set(ylim=(0, 5))

    
    for ax in g.axes.flat:
        ymax = 0.0
    
        # Filled KDEs (PolyCollection)
        for coll in ax.collections:
            for path in coll.get_paths():
                v = path.vertices
                if v is not None and len(v):
                    ymax = max(ymax, np.max(v[:, 1]))
    
        # Outline KDEs (Line2D)
        for line in ax.lines:
            y = np.asarray(line.get_ydata())
            if y.size:
                ymax = max(ymax, np.max(y))
        
        ylim =3
        if ymax > ylim:
            ax.set_ylim(0, ymax + 0.6)
        else:
            ymax = ylim
            ax.set_ylim(0, ymax)
        
    for ax in g.axes.flat:
        ax.grid(False)
        ax.plot([0, 1], [0.02, 0.02],
                transform=ax.transAxes,
                color='black',
                alpha=0.5,
                linewidth=0.7,
                clip_on=False)
        
    for ax in g.axes.flat:
        ax.set_xticks(scale)
        ax.set_xticklabels(scale_markers, rotation=45)  # Rotated for readability
        ax.xaxis.set_visible(True)  

    tick_height = 0.20  # 20% of the axis height

    for ax in g.axes.flat:
        trans = ax.get_xaxis_transform()  
        for tick in scale:
            ax.plot([tick, tick], [0, tick_height],
                    transform=trans,
                    color='black', ls='--', alpha=0.3, linewidth=1,
                    clip_on=False)
                
    

    # plt.savefig(f"python_figures/{column}.svg", format='svg')

ridgeplot(data, 'L1 (cm)', '$L1$ (cm)', 0.08, 500, True, 
          [ 0.1, 1, 10, 100], 
          [ "0.1", "1", '10', '100'])

ridgeplot(data, 'L2 (cm)', '$L2$ (cm)', 0.008, 105, True, 
          [0.01, 0.1, 1, 10, 100], 
          ["0.01", "0.1", "1", "10", "100"])

ridgeplot(data, 'L3 (cm)', '$L3$ (cm)', 0.0008, 20, True,
          [0.001, 0.01, 0.1, 1, 10], 
          ["0.001", '0.01', "0.1", "1", "10" ])
# %%
# 

ridgeplot(data, 'CSF', 'CSF (-)', 0, 1, False,
          [0, 0.5, 1],
          ['0', '0.5', '1'])

ridgeplot(data, 'Elongation_E', '$EL$ (-)', 0, 1, False,
          [0, 0.5, 1],
          ['0', '0.5', '1'])

ridgeplot(data, 'Flatness_F', '$FL$ (-)', 0, 1, False,
          [0, 0.5, 1],
          ['0', '0.5', '1'])

ridgeplot(data, 'Mass (g)', 'Mass (g)', 0.00008, 500, True, 
          [0.0001, 0.01, 1, 100],
          ['0.0001',  '0.01', '1', '100'])

ridgeplot(data, 'Volume (cm^3)', 'Volume (cm$^3$)', 0.00008, 10000, True, 
          [0.0001, 0.01 , 1, 100, 10000],
          ['0.0001', '0.01', '1', '100', '10000'])

ridgeplot(data, 'Density (g/cm^3)', 'Density (g/cm$^3$)', 0, 10, True, 
          [0.01, 0.1, 1, 10],
          ['0.01', '0.1','1', '10'])

plt.show()
