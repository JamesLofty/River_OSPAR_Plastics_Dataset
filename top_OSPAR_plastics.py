import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from collections import Counter

# Load Data
data = pd.read_excel("OSPAR_meta_analysis.xlsx")
data["Common name 2"] = data["Common name"]
data = data.set_index('Common name')

ospar_ids = data['Ospar ID']
Material = data['Material']
Common_name = data['Common name 2']
Flexibility = data['flexibility']
Density_max = data['Density_max']
Density_min = data['Density_min']

# Extract Data for Rivers and Riverbanks
data_riverbanks = data.loc[:, data.columns.str.contains('riverbank', case=False)]
data_riverbanks['OSPAR ID'] = ospar_ids.loc[data_riverbanks.index]
data_riverbanks['Material'] = Material.loc[data_riverbanks.index]
data_riverbanks['Common name'] = Common_name.loc[data_riverbanks.index]
data_riverbanks['Density_max'] = Density_max.loc[data_riverbanks.index]
data_riverbanks['Density_min'] = Density_min.loc[data_riverbanks.index]

data_rivers = data.loc[:, data.columns.str.contains('watercolumn', case=False)]
data_rivers['OSPAR ID'] = ospar_ids.loc[data_rivers.index]
data_rivers['Material'] = Material.loc[data_rivers.index]
data_rivers['Common name'] = Common_name.loc[data_rivers.index]
data_rivers['Flexibility'] = Flexibility.loc[data_rivers.index]
data_rivers['Density_max'] = Density_max.loc[data_rivers.index]
data_rivers['Density_min'] = Density_min.loc[data_rivers.index]

data_ALL = pd.merge(data_riverbanks, data_rivers, left_index=True, right_index=True)

# Calculate the average percentage for each dataset
data_riverbanks['Average_Percentage_Riverbanks'] = data_riverbanks.filter(like='percentage').mean(axis=1)
data_rivers['Average_Percentage_Rivers'] = data_rivers.filter(like='percentage').mean(axis=1)
data_ALL['Average_Percentage'] = data_ALL.filter(like='percentage').mean(axis=1)

# Merge Data
average_data = pd.DataFrame({
    'Average_Percentage_Riverbanks': data_riverbanks['Average_Percentage_Riverbanks'],
    'Average_Percentage_Rivers': data_rivers['Average_Percentage_Rivers']
}).dropna()

#%%
# Get top 20 plastics
average_data_ALL = data_ALL.sort_values(by='Average_Percentage', ascending=False).head(25)

average_data = pd.merge(average_data_ALL, average_data, left_index=True, right_index=True)

average_data = average_data.sort_values(by='Average_Percentage', ascending=False).head(25)

average_data['osparID'] = average_data['OSPAR ID_y']

average_data.to_excel("synthetic_data/top_25_plastics.xlsx", index=False)  # Set index=False to avoid saving the index column

#%%
# Colour palette for legend
unique_materials = np.array(['PO soft', 'PO hard', 'EPS', 'PS', 'Multilayer', 'Glass', 'PET', 'Metal', 'Textiles',
                             'Other plastics', 'Paper', 'Rubber', 'Wood'], dtype=object)

palette = sns.color_palette('Set2', len(unique_materials))
material_colors = {material: palette[i] for i, material in enumerate(unique_materials)}
material_colors['Other plastics'] = 'tomato'
material_colors['Rubber'] = '#66bad9'

# Ensure that the 'Material' column is available in the filtered data
average_data['Material_Riverbanks'] = data.loc[average_data.index, 'Material'].values
average_data['Material_Rivers'] = data.loc[average_data.index, 'Material'].values

# Assign custom colors for each material based on the material_colors dictionary
average_data['Colour_Riverbanks'] = average_data['Material_Riverbanks'].map(material_colors)
average_data['Colour_Rivers'] = average_data['Material_Rivers'].map(material_colors)

# Plot
average_data = average_data.sort_values(by='Average_Percentage_Rivers', ascending=False)

#%%
total_amount_count = 258024

average_data['total_amount_per_item'] = (average_data['Average_Percentage']/100) * total_amount_count

top_20_SUM = np.sum(average_data['total_amount_per_item'])

percent_of_total = percent_of_total = (top_20_SUM / total_amount_count) * 100

print('top 25 items represent=', percent_of_total, '% of all itmes')

#%%
# Plot
# Count occurrences of each material in both categories
material_counts = Counter(average_data['Material_Rivers']) + Counter(average_data['Material_Riverbanks'])

# Sort materials by most frequently observed
sorted_materials = sorted(material_counts.keys(), key=lambda m: material_counts[m], reverse=True)

# Create legend only for materials in the sorted order
material_patches = [mpatches.Patch(color=material_colors[material], label=f"{material}") 
                    for material in sorted_materials if material in material_colors]

# Plot
fig, ax = plt.subplots(figsize=(5, 6))
x = np.arange(len(average_data))  # the label locations

# Separate the data into two series: one for Rivers and one for Riverbanks
rivers_data = average_data['Average_Percentage_Rivers']
riverbanks_data = average_data['Average_Percentage_Riverbanks']

# Create diverging bars: negative for Rivers and positive for Riverbanks
bars1 = ax.barh(x, rivers_data,
                label='Rivers', color=average_data['Colour_Rivers'], 
                edgecolor='black')

# bars2 = ax.barh(x, riverbanks_data,
#                 label='Riverbanks', color=average_data['Colour_Riverbanks'], 
#                 edgecolor='black', alpha=0.5)

# Adjust axis limits to accommodate both positive and negative bars
# ax.set_xlim(-25, 25)

# Labels and Titles
ax.set_xlabel('Percentage (%)')
ax.set_ylabel('OSPAR category')
ax.set_title('Top 25 Most Common OSPAR Indexed Items \n in Rivers and on Riverbanks')
ax.set_yticks(x)
ax.set_yticklabels(average_data.index)
ax.legend(loc='lower right')  # First legend for Rivers & Riverbanks
ax.invert_yaxis()
ax.grid(axis='x', linestyle='--', alpha=0.7)
ax.set_xlim(0,25)

plt.tight_layout()

# Add second legend for material colors
material_legend = ax.legend(handles=material_patches, title="Material",
                            loc='center', bbox_to_anchor=(1, 0.35))
ax.add_artist(material_legend)  # Add this legend separately

# plt.legend(loc='center', bbox_to_anchor=(0.75, 0.1))
sns.despine(top=True, right=True, left=False, bottom=False)
plt.savefig("python_figures/OSPAR_river_trial.svg", format="svg")

plt.show()

#%%

fig, ax = plt.subplots(figsize=(5, 6))
x = np.arange(len(average_data))  # the label locations

# Separate the data into two series: one for Rivers and one for Riverbanks
rivers_data = average_data['Average_Percentage_Rivers']
riverbanks_data = average_data['Average_Percentage_Riverbanks']

# Create diverging bars: negative for Rivers and positive for Riverbanks
# bars1 = ax.barh(x, rivers_data,
#                 label='Rivers', color=average_data['Colour_Rivers'], 
#                 edgecolor='black')

bars2 = ax.barh(x, riverbanks_data,
                label='Riverbanks', color=average_data['Colour_Riverbanks'], 
                edgecolor='black',  hatch = "/")

# Adjust axis limits to accommodate both positive and negative bars
# ax.set_xlim(-25, 25)

# Labels and Titles
ax.set_xlabel('Percentage (%)')
ax.set_ylabel('OSPAR category')
ax.set_title('Top 25 Most Abundant OSPAR indexed items \n in Rivers and on Riverbanks')
ax.set_yticks(x)
ax.set_yticklabels(average_data.index)
ax.legend(loc='lower right')  # First legend for Rivers & Riverbanks
ax.invert_yaxis()
ax.grid(axis='x', linestyle='--', alpha=0.7)
ax.set_xlim(0,25)

plt.tight_layout()

# Add second legend for material colors
# material_legend = ax.legend(handles=material_patches, title="Material Colour",
#                             loc='center', bbox_to_anchor=(1, 0.35))
# ax.add_artist(material_legend)  # Add this legend separately

# plt.legend(loc='center', bbox_to_anchor=(0.75, 0.1))
sns.despine(top=True, right=True, left=False, bottom=False)

plt.savefig("python_figures/OSPAR_riverbank_trial.svg", format="svg")

plt.show()


