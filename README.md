# Ulcerative Colitis Detection System

## Overview

This project implements an AI-powered system for detecting Ulcerative Colitis (UC) from colonoscopy images using deep learning. The system consists of a ResNet-18 based classifier trained on colonoscopy images and a Flask web application for easy image analysis. This is used as a demonstrative project in collaboration with a physical tool developed for a health exhibition.

## Model Architecture

The classification model is built on ResNet-18 with transfer learning:
- Pre-trained on ImageNet weights
- Fine-tuned on colonoscopy images with layers 3 and 4 unfrozen
- Binary classification: Normal vs Ulcerative Colitis
- Uses data augmentation and weighted sampling for class imbalance handling

## Web Application

The Flask web interface provides:
- Multi-image upload support
- Real-time AI analysis of colonoscopy images
- Comprehensive medical report generation
- Risk assessment with visual progress indicators
- Responsive design optimized for medical use

## Setup and Installation

Below are the instructions for setting up the local development environment should anyone be interested in modifying the code.

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended)

### Step-by-Step Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/linusc9516/UC_Classifier
   cd UC_Classifier
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download and prepare dataset**
   - Download the [colonoscopy images dataset](https://zenodo.org/records/5827695#.ZF-92OzMJqs) (zip file named `patient_based_classified_images.zip`)
   - Extract the images to the project directory
   - Run the data preparation script:
     ```bash
     split_directories.bat
     ```

4. **Train the model**
   - Modify training parameters in [`model.py`](model.py) if needed
   - Run the training script:
     ```bash
     python model.py
     ```
   - The trained model will be saved as `2classifier_rn18_layer3_4_unfreeze.pt`, or whatever you wish to rename it to.

5. **Launch the web application**
   ```bash
   python app.py
   ```
   - Open your browser and navigate to `http://localhost:5000`
   - Upload colonoscopy images for analysis

## Usage

1. Access the web interface at `http://localhost:5000`
2. Upload one or multiple colonoscopy images
3. Click "Analyze Images" to process the images
4. Review the AI-generated medical report with risk assessment

## File Structure

- [`app.py`](app.py) - Flask web application
- [`model.py`](model.py) - Model training and architecture
- [`templates/`](templates/) - HTML templates for the web interface
- [`static/`](static/) - CSS styles and static assets
- [`2class/`](2class/) - Training dataset directory
- [`test/`](test/) - Sample test images

## Disclaimer

This AI system is for research and educational purposes only. It should not replace professional medical diagnosis or consultation with qualified practicioners.
Credit also goes to the owner of the dataset used in this project.