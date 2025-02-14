# ML Model Explorer

## Overview
ML Model Explorer is an interactive web application built with Streamlit that allows users to experiment with different machine learning classifiers and understand their performance characteristics. The app provides a user-friendly interface for exploring popular datasets, testing various classification algorithms, and visualizing their results through multiple performance metrics.

Try it out live at [ml-model-explorer.streamlit.app](https://ml-model-explorer.streamlit.app)

## Features

### Dataset Selection
- Choose from classic machine learning datasets:
  - Iris Dataset
  - Breast Cancer Dataset
  - Wine Dataset

### Supported Classifiers
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machines (SVM)
- Decision Trees
- Random Forest
- Gradient Boosting
- Naive Bayes

### Interactive Model Tuning
- Real-time hyperparameter adjustment via intuitive sliders
- Classifier-specific parameter controls:
  - Logistic Regression: C parameter
  - KNN: number of neighbors (K)
  - SVM: C parameter
  - Decision Tree: maximum depth
  - Random Forest: number of estimators and maximum depth
  - Gradient Boosting: number of estimators and maximum depth

### Performance Analytics
- Comprehensive model evaluation metrics:
  - Accuracy Score
  - Precision Score
  - Recall Score
  - F1 Score
- Visual performance analysis:
  - Interactive Confusion Matrix
  - Detailed Classification Report
  - ROC Curve (for binary classification)

## Quick Start

### Using the Live App
Visit [ml-model-explorer.streamlit.app](https://ml-model-explorer.streamlit.app) to try the application instantly in your browser.

### Running Locally

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ml-model-explorer.git
cd ml-model-explorer
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Launch the application:
```bash
streamlit run src/main.py
```

## Dependencies
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- streamlit

## Usage

1. Select a dataset from the sidebar dropdown menu
2. Choose a classifier type
3. Adjust the hyperparameters using the interactive sliders
4. Click the "Predict" button to see the results
5. Explore the various performance metrics and visualizations

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with Streamlit
- Uses scikit-learn's built-in datasets and classifiers
- Visualization powered by matplotlib and seaborn
