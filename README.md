# ASE_project

## Description
This repository contains my work on developing a defect image recognition system for microchip drill hole positions using Neural Network and OCR(Optical Character Recognition) in an academia-industry collaboration project sponsored by the ASE Holdings. 

## Data
- Distribution map of pink spot defects
- Distribution map of yellow base defects
- Summary statistics of the distribution maps

## Model
- OCR model to read the text embedded in the images
- Multimodal model: Use ResNet to extract visual features and DNN to process numerical data. Finally, I implemented a late fusion strategy to leverage predictions from both models and make the final prediction.
