import pandas as pd
import numpy as np
from bokeh.plotting import figure, show, save
from bokeh.io import output_file
from bokeh.models import ColumnDataSource, HoverTool, Legend
from bokeh.palettes import Category10
from bokeh.layouts import column
from bokeh.transform import factor_cmap

# Read the metrics files
cnnlstm_spec = pd.read_csv('models/CNNLSTM/spec/CNNLSTM_20250425_020424_metrics.csv')
vit = pd.read_csv('models/ViT/spec/ViT_20250425_045614_metrics.csv')
heartgpt = pd.read_csv('models/HeartGPT/token/HeartGPT_20250425_092041_metrics.csv')

# Create a list of models and their data
models = {
    'CNNLSTM Spec': cnnlstm_spec,
    'ViT': vit,
    'HeartGPT': heartgpt
}

# Create a color palette
colors = Category10[3]

# 1. Training Loss Comparison
p1 = figure(title='Training Loss Comparison Across Models',
           x_axis_label='Epoch',
           y_axis_label='Training Loss',
           width=800, height=400)

for i, (model_name, data) in enumerate(models.items()):
    source = ColumnDataSource(data={
        'epoch': data['epoch'],
        'train_loss': data['train_loss'],
        'model_name': [model_name] * len(data)
    })
    p1.line('epoch', 'train_loss',
            line_width=2, color=colors[i],
            legend_label=model_name,
            source=source)
    p1.scatter('epoch', 'train_loss',
             size=6, color=colors[i],
             fill_alpha=0.6,
             source=source)

p1.legend.location = 'top_right'
p1.legend.click_policy = 'hide'
p1.add_tools(HoverTool(tooltips=[
    ('Model', '@model_name'),
    ('Epoch', '@epoch'),
    ('Loss', '@train_loss{0.000}')
]))

# 2. Validation Accuracy Comparison
p2 = figure(title='Validation Accuracy Comparison Across Models',
           x_axis_label='Epoch',
           y_axis_label='Validation Accuracy',
           width=800, height=400)

for i, (model_name, data) in enumerate(models.items()):
    source = ColumnDataSource(data={
        'epoch': data['epoch'],
        'val_accuracy': data['val_accuracy'],
        'model_name': [model_name] * len(data)
    })
    p2.line('epoch', 'val_accuracy',
            line_width=2, color=colors[i],
            legend_label=model_name,
            source=source)
    p2.scatter('epoch', 'val_accuracy',
             size=6, color=colors[i],
             fill_alpha=0.6,
             source=source)

p2.legend.location = 'bottom_right'
p2.legend.click_policy = 'hide'
p2.add_tools(HoverTool(tooltips=[
    ('Model', '@model_name'),
    ('Epoch', '@epoch'),
    ('Accuracy', '@val_accuracy{0.000}')
]))

# 3. F1 Score Comparison
p3 = figure(title='F1 Score Comparison Across Models',
           x_axis_label='Epoch',
           y_axis_label='F1 Score',
           width=800, height=400)

for i, (model_name, data) in enumerate(models.items()):
    source = ColumnDataSource(data={
        'epoch': data['epoch'],
        'val_f1': data['val_f1'],
        'model_name': [model_name] * len(data)
    })
    p3.line('epoch', 'val_f1',
            line_width=2, color=colors[i],
            legend_label=model_name,
            source=source)
    p3.scatter('epoch', 'val_f1',
             size=6, color=colors[i],
             fill_alpha=0.6,
             source=source)

p3.legend.location = 'bottom_right'
p3.legend.click_policy = 'hide'
p3.add_tools(HoverTool(tooltips=[
    ('Model', '@model_name'),
    ('Epoch', '@epoch'),
    ('F1 Score', '@val_f1{0.000}')
]))

# 4. Precision-Recall Trade-off
p4 = figure(title='Precision-Recall Trade-off',
           x_axis_label='Precision',
           y_axis_label='Recall',
           width=800, height=400)

for i, (model_name, data) in enumerate(models.items()):
    source = ColumnDataSource(data={
        'val_precision': data['val_precision'],
        'val_recall': data['val_recall'],
        'model_name': [model_name] * len(data)
    })
    p4.scatter('val_precision', 'val_recall',
              size=8, color=colors[i],
              fill_alpha=0.6,
              legend_label=model_name,
              source=source)

p4.legend.location = 'bottom_right'
p4.legend.click_policy = 'hide'
p4.add_tools(HoverTool(tooltips=[
    ('Model', '@model_name'),
    ('Precision', '@val_precision{0.000}'),
    ('Recall', '@val_recall{0.000}')
]))

# 5. ROC AUC Comparison
p5 = figure(title='ROC AUC Comparison Across Models',
           x_axis_label='Epoch',
           y_axis_label='ROC AUC',
           width=800, height=400)

for i, (model_name, data) in enumerate(models.items()):
    source = ColumnDataSource(data={
        'epoch': data['epoch'],
        'val_auc_roc': data['val_auc_roc'],
        'model_name': [model_name] * len(data)
    })
    p5.line('epoch', 'val_auc_roc',
            line_width=2, color=colors[i],
            legend_label=model_name,
            source=source)
    p5.scatter('epoch', 'val_auc_roc',
             size=6, color=colors[i],
             fill_alpha=0.6,
             source=source)

p5.legend.location = 'bottom_right'
p5.legend.click_policy = 'hide'
p5.add_tools(HoverTool(tooltips=[
    ('Model', '@model_name'),
    ('Epoch', '@epoch'),
    ('ROC AUC', '@val_auc_roc{0.000}')
]))

# 6. Best Performance Metrics Bar Chart
best_metrics = {
    'CNNLSTM Spec': {
        'Accuracy': cnnlstm_spec['val_accuracy'].max(),
        'F1 Score': cnnlstm_spec['val_f1'].max(),
        'ROC AUC': cnnlstm_spec['val_auc_roc'].max()
    },
    'ViT': {
        'Accuracy': vit['val_accuracy'].max(),
        'F1 Score': vit['val_f1'].max(),
        'ROC AUC': vit['val_auc_roc'].max()
    },
    'HeartGPT': {
        'Accuracy': heartgpt['val_accuracy'].max(),
        'F1 Score': heartgpt['val_f1'].max(),
        'ROC AUC': heartgpt['val_auc_roc'].max()
    }
}

metrics_df = pd.DataFrame(best_metrics).T
metrics_df = metrics_df.reset_index()
metrics_df = metrics_df.rename(columns={'index': 'Model'})
metrics_df = metrics_df.melt(id_vars=['Model'], 
                           value_vars=['Accuracy', 'F1 Score', 'ROC AUC'],
                           var_name='Metric', value_name='Score')

p6 = figure(title='Best Performance Metrics Comparison',
           x_range=metrics_df['Model'].unique(),
           width=800, height=400)

source = ColumnDataSource(metrics_df)
p6.vbar(x='Model', top='Score', width=0.8,
        source=source,
        fill_color=factor_cmap('Metric', palette=Category10[3], factors=metrics_df['Metric'].unique()))

p6.xaxis.major_label_orientation = 0.8
p6.add_tools(HoverTool(tooltips=[
    ('Model', '@Model'),
    ('Metric', '@Metric'),
    ('Score', '@Score{0.000}')
]))

# Create layout and save
layout = column(p1, p2, p3, p4, p5, p6)
output_file('model_comparison.html')
save(layout) 