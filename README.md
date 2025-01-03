# ReefCheck Data Analysis and COTS Prediction

🌊 Machine learning-powered early warning system for predicting Crown-of-Thorns Starfish (COTS) outbreaks on coral reefs, achieving 77% accuracy on real-world data.

## About

This project combines statistical analysis of 15+ ecological indicators with advanced machine learning to predict and prevent devastating COTS outbreaks before they occur. Key features:

- 🎯 **Predictive Power**: Identifies COTS population changes with 77% accuracy
- 📊 **Scientific Rigor**: Validated against extensive ReefCheck survey data
- 🔍 **Early Warning**: Detects outbreak precursor patterns up to a year in advance
- 🖥️ **Interactive UI**: Web interface for real-time predictions from survey data
- 🧬 **Ecosystem Insights**: Reveals complex relationships between marine species

Perfect for marine biologists, reef managers, and conservation researchers working to protect coral reef ecosystems.

## Project Structure

- `database_tools/`: Scripts for importing and managing ReefCheck survey data
- `model_training/`: Scripts for training the COTS prediction model
- `model_inference/`: Scripts for running predictions using the trained model
- `predict_ui/`: Web interface for running predictions
- `statistical_analysis/`: Statistical analysis of COTS population factors
- `models/`: Trained model files (single source of truth for model artifacts)
- `reference_docs/`: Documentation and reference materials

## Prerequisites

1. Python 3.x
2. Required Python packages:
```bash
pip install -r requirements.txt
```

## Training the Model

1. Train the model:
```bash
python model_training/predict_cots_changes.py
```
This will create a trained model file in the `models/` directory.

## Running the Prediction UI

The project includes a web interface for testing COTS predictions:

2. Start the prediction server:
```bash
cd predict_ui
python predict_server.py
```
The server will run on port 5001.

3. Open `predict_ui/cots_predictor.html` in your web browser

4. Enter survey data values and click "Predict" to get COTS population change predictions

## Model Features

The model uses the following key indicators from ReefCheck survey data:

### Fish Population Indicators
- Grouper counts (30-40cm, 40-50cm, >60cm, Total)
- Humphead Wrasse presence
- Parrotfish presence

### Invertebrate Indicators
- Giant Clam measurements (40-50cm, >50cm, Total)
- Tripneustes presence

### Coral Health Indicators
- Bleaching (% of Population and Colony)
- Coral Damage

### Derived Features
The model also uses derived binary indicators (presence/absence) for:
- Bleaching
- Coral damage
- Parrotfish
- Humphead wrasse
- Tripneustes
- Total giant clam count

## Data Requirements

For reliable predictions, surveys should include data for at least 5 of the key indicators listed above. The model uses averages across survey segments where available, requiring at least 2 valid segments per indicator.

## Research Findings

### Statistical Analysis Results

The analysis identified several significant patterns preceding COTS population changes:

#### Common Indicators for Both Increases and Decreases
- **Grouper Populations**: Multiple size classes (30-40cm, 40-50cm, >60cm) showed significant correlations
  - Before increases: Effect sizes 1.01-1.57, present in 4.8-23.8% of surveys
  - Before decreases: Effect sizes 0.93-1.00, present in 14.3-42.9% of surveys
- **Giant Clams (40-50cm)**: Strong indicator for both changes
  - Before increases: Effect size 1.26, 900% higher than control
  - Before decreases: Effect size 1.53, 1100% higher than control

#### Unique Patterns Before COTS Increases
1. Giant Clams >50cm (Effect size: 0.90, 589.4% higher than control)
2. Bleaching indicators:
   - Population percentage (Effect size: 0.55, 119.8% higher)
   - Colony percentage (Effect size: 0.54, 52% higher)
3. Coral Damage (Effect size: 0.54, 47.4% higher)

#### Unique Patterns Before COTS Decreases
1. Humphead Wrasse (Effect size: 1.04, 640.4% higher than control)
2. Parrotfish (Effect size: 0.96, 209.2% higher)
3. Tripneustes (Effect size: 0.78, 631.4% higher)

### Predictive Model Performance

The COTS prediction model demonstrates strong predictive capabilities:

- **Overall Accuracy**: 77.1% on known events
- **Feature Importance** (Top 5):
  1. Total Giant Clam Count (20.97%)
  2. Coral Damage Other (13.35%)
  3. Bleaching Population % (11.89%)
  4. Bleaching Colony % (10.68%)
  5. Parrotfish Presence (5.93%)

The model shows particularly strong performance in predicting COTS increases, with many predictions showing confidence levels above 80%. This suggests the model is especially valuable for early warning of potential COTS outbreaks.
