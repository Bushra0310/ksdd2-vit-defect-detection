# ksdd2-vit-defect-detection
Fine-tuning Vision Transformers for rare industrial defect detection on KolektorSDD2 (MAAI901)
# Fine-Tuning Vision Transformers for Rare Industrial Defect Detection

MAAI901 Deep Learning for Visual Data, University of Wollongong in Dubai.
Applied Vision Transformer Report.

## What this project does
This project compares a ResNet-50 baseline with three Vision Transformer (ViT-B/16)
setups (frozen, fine-tuned, and fine-tuned with class-weighted loss) for detecting
rare surface defects. The focus is on why accuracy is misleading when defects are rare.

## Dataset
KolektorSDD2 (ViCoS Lab, University of Ljubljana):
https://www.vicos.si/resources/kolektorsdd2
Licence: CC BY-NC-SA 4.0 (non-commercial). The dataset is NOT included in this
repository. Please download it from the link above.

## Notebook
KSDD2_ViT_defect_detection.ipynb contains the full pipeline: data loading, labelling
from masks, train/validation split, pre-processing, training of all models (3 seeds),
evaluation, threshold analysis, a higher-resolution experiment and error analysis.

## How to run
1. Open the notebook in Google Colab with a T4 GPU.
2. Put the dataset in your Google Drive and update the paths at the top of the notebook.
3. Run the cells in order. Results are saved to the results folder set in the notebook.

## Notes
- Results can change slightly between runs, even with fixed seeds.
- Model checkpoints are not included because of their size.
