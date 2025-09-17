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

data = pd.read_excel("synthetic_data/synthetic_data_top_25.xlsx")

data['CSF'] = data['short'] / np.sqrt(data['long'] * data['intermediate'])

data['short'] = data['short']*100
data['intermediate'] = data['intermediate']*100
data['long'] = data['long']*100

data['Volume_V'] = data['Volume_V']*1e+6
data['Density_rho'] = data['Density_rho']/1000
data['Diameter_d'] = data['Diameter_d']*100 
data['Mass_M'] = data['Mass_M'] * 1000
data['Density_rho'] = data['Mass_M'] / data['Volume_V']
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

weight_data = data[['common name', 'river_prop', 'bank_prop']].drop_duplicates()

mean_by_category = data.groupby('common name')['Diameter_d'].mean()
merged = weight_data.merge(mean_by_category, on='common name')
merged.rename(columns={'Diameter_d': 'mean_size'}, inplace=True)

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
mean_by_category = data.groupby('common name')['long'].mean()

sorted_means = mean_by_category.sort_values(ascending=False)
print(sorted_means)

#%%

plt.figure()

# Loop through each unique Average_Percentage_Rivers group
for i in data['Average_Percentage_Rivers'].unique():
    df = data[data['Average_Percentage_Rivers'] == i]
    
    x = np.mean(df['Average_Percentage_Rivers'])  # x-axis: mean of the group
    y = np.mean(df['Diameter_d'])              # y-axis: median diameter
    yerr = np.std(df['Diameter_d'])              # error bar: std deviation of diameter
    
    plt.errorbar(x, y, yerr=yerr, fmt='o', label=f"{i}%")  # fmt='o' makes it a dot

plt.xlabel("Average Percentage Rivers")
plt.ylabel("Median Diameter")
plt.title("Median Diameter vs. River Percentage (with Std Dev)")
plt.grid(True)
plt.show()



# %%
ordered_names = (
    data.groupby('common name')
    .apply(lambda x: x['Average_Percentage_Rivers'].max())
    .sort_values(ascending=False)
    .index
)
# Ensure 'Common name' follows this order
data['common name'] = pd.Categorical(data['common name'], categories=ordered_names, ordered=True)
#######################################################################
# --- Assign Colors to Common Names ---
common_name_colors = {}
for name in data['common name'].unique():
    material = data[data['common name'] == name]['material'].iloc[0]
    common_name_colors[name] = material_colors.get(material, 'gray')  # fallback to gray if material not found

# %%


def ridgeplot(data, column, xlabel, x_min, x_max, log, scale, scale_markers):
    figsize = (1.3 , 7)
    sns.set_theme(rc={"axes.facecolor": (0, 0, 0, 0), "font.family": "DejaVu Sans"})
    g = sns.FacetGrid(
        data, 
        row="common name", 
        hue="common name", 
        palette=common_name_colors,
        aspect=5,  
        height=0.8  
    )
    
    g.fig.set_size_inches(figsize)  # Overall figure size
    
    g.map_dataframe(sns.kdeplot, x=column, fill=True, alpha=0.8, log_scale=log, bw_adjust=1.5)
    g.map_dataframe(sns.kdeplot, x=column, fill=False, alpha=0.8, log_scale=log, lw = 1.5, bw_adjust=1, color = "k")
    
    for ax, label in zip(g.axes.flat, ordered_names):
        ax.text(-0.1, 0.1, label, color='black', ha="right", va="center", transform=ax.transAxes)
    
    g.fig.subplots_adjust(hspace=-0.6)  
    g.set_titles("")
    g.set(ylabel='')
    g.set(yscale='log')
    g.set(yticks=[])
    g.set(xlabel=xlabel)
    
    g.set(xlim=(x_min, x_max), ylim=(0.1, 50))
    for ax in g.axes.flat:
        ax.grid(False)
        ax.axhline(0.12, color='black', alpha=0.5, linewidth=0.5)
        
    for ax in g.axes.flat:
        ax.set_xticks(scale)
        ax.set_xticklabels(scale_markers, rotation=45)  # Rotated for readability
        ax.xaxis.set_visible(True)  

        for tick in scale:
            ax.plot([tick, tick], [0.01, 0.5], color='black', ls='--', alpha=0.3, linewidth=1)

    plt.savefig(f"python_figures/{column}.svg", format='svg')
        
        #%%

ridgeplot(data, 'long', '$L1$ (cm)', 0.1, 500, True, 
          [ 0.1, 1, 10, 100], 
          [ "0.1", "1", '10', '100'])

ridgeplot(data, 'intermediate', '$L2$ (cm)', 0.01, 105, True, 
          [0.01, 0.1, 1, 10, 100], 
          ["0.01", "0.1", "1", "10", "100"])

ridgeplot(data, 'short', '$L3$ (cm)', 0.001, 20, True,
          [0.001, 0.01, 0.1, 1, 10], 
          ["0.001", '0.01', "0.1", "1", "10" ])

ridgeplot(data, 'CSF', 'CSF (-)', -0.1, 1.1, False,
          [0, 0.5, 1],
          ['0', '0.5', '1'])

ridgeplot(data, 'Elongation_E', '$EL$ (-)', -0.1, 1.1, False,
          [0, 0.5, 1],
          ['0', '0.5', '1'])

ridgeplot(data, 'Flatness_F', '$FL$ (-)', -0.1, 1.1, False,
          [0, 0.5, 1],
          ['0', '0.5', '1'])


ridgeplot(data, 'Diameter_d', '$D_n$ (cm)', 0.05, 50, True,
          [0.1, 1, 10],
          ['0.1', '1', '10'])

ridgeplot(data, 'Mass_M', 'Mass (g)', 0.0001, 500, True, 
          [0.0001, 0.01, 1, 100],
          ['0.0001',  '0.01', '1', '100'])

ridgeplot(data, 'Volume_V', 'Volume (cm$^3$)', 0.0001, 10000, True, 
          [0.0001, 0.01 , 1, 100, 10000],
          ['0.0001', '0.01', '1', '100', '10000'])

ridgeplot(data, 'Density_rho', 'Density (g/cm$^3$)', 0, 10, True, 
          [0.01, 0.1, 1, 10],
          ['0.01', '0.1','1', '10'])



plt.show()
