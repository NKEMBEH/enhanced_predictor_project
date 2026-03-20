# Enhanced Substance Dependency Predictor

An advanced AI-powered Django web application for predicting substance dependency based on behavioral and demographic factors using machine learning.

## Features

- **Single Prediction**: Get instant predictions for individuals with multi-step forms
- **Batch Prediction**: Upload and analyze multiple records at once with Excel support
- **LIME Explainability**: Understand model predictions with feature importance analysis
- **Visualizations**: Interactive charts and analytics for batch predictions
- **Secure & Professional UI**: Modern, responsive design with gradient aesthetics

## Technology Stack

- **Backend**: Django 5.2.4, Python 3.11
- **ML Model**: AdaBoost Classifier with scikit-learn
- **Interpretability**: LIME (Local Interpretable Model-agnostic Explanations)
- **Frontend**: Bootstrap 5.3.3, Font Awesome Icons
- **Database**: SQLite3 (easily switched to PostgreSQL for production)
- **Data Processing**: Pandas, NumPy, Matplotlib

## Installation

### Prerequisites
- Python 3.11+
- pip
- Virtual environment (recommended)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/NKEMBEH/enhanced_predictor_project.git
cd enhanced_predictor_project
```

2. **Create and activate virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run migrations**
```bash
python manage.py migrate
```

5. **Start the development server**
```bash
python manage.py runserver
```

Visit `http://localhost:8000` in your browser.

## Usage

### Single Prediction
1. Navigate to "Single Prediction"
2. Fill out the multi-step form with demographic and behavioral data
3. Click "Get Prediction" to receive results
4. View LIME explanation for model interpretability

### Batch Prediction
1. Navigate to "Batch Prediction"
2. Upload an Excel file (.xls or .xlsx)
3. View visualizations and prediction results
4. Download full results as Excel

## Deployment

### Deploy to Render (Free)

1. Push to GitHub
2. Sign up at [render.com](https://render.com)
3. Create new "Web Service" from your GitHub repository
4. Configure:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn main_project.wsgi:application`
5. Set environment variables:
   - `DJANGO_SECRET_KEY`: Your Django secret key
   - `DEBUG`: false

## Project Structure

```
enhanced_predictor_project/
├── main_project/          # Django project settings
├── predictor/             # Main Django app
│   ├── models.py
│   ├── views.py
│   ├── forms.py
│   ├── urls.py
│   ├── templates/
│   └── ml_model/          # ML assets and training data
├── templates/             # Base templates
├── manage.py
├── requirements.txt
└── db.sqlite3
```

## Model Details

- **Algorithm**: Tuned AdaBoost Classifier
- **Input Features**: 35 categorical features (values 1-9)
- **Output Classes**: 10 substance dependency categories
- **Accuracy**: 85%+

## License

This project is provided as-is for educational and research purposes.

## Author

NKEMBEH BENJAMIN MEZINDAH

---

For questions or issues, please open an issue on GitHub.
