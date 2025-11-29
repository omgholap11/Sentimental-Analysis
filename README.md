# 🎭 Emotion Analyzer - Sentiment Analysis Project

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A modern, real-time emotion analysis web application powered by Machine Learning. This project uses Natural Language Processing (NLP) techniques to detect and classify emotions in text into six categories: **Sadness, Anger, Love, Surprise, Fear, and Joy**.

## 📸 Screenshots

### Main Interface - Sadness Detection
![Sadness Detection](screenshots/Screenshot%202025-11-30%20020726.png)

### Joy Detection Example
![Joy Detection](screenshots/image-1764448812662.png)

## ✨ Features

- 🎯 **Real-time Emotion Detection** - Instant analysis of text input
- 📊 **Visual Confidence Distribution** - Interactive charts showing emotion probabilities
- 🎨 **Modern UI/UX** - Dark-themed, gradient-based design with smooth animations
- 📈 **Statistics Tracking** - Monitor total analyses and most common emotions
- 💡 **Example Prompts** - Pre-built examples to test the model
- 📜 **Analysis History** - Track recent emotion detections
- 🚀 **Fast & Responsive** - Optimized for quick predictions

## 🎭 Supported Emotions

The model can classify text into the following six emotions:

| Emotion | Emoji | Description |
|---------|-------|-------------|
| **Sadness** | 😢 | Feeling down or melancholic |
| **Anger** | 😠 | Expressing frustration or rage |
| **Love** | ❤️ | Showing affection and warmth |
| **Surprise** | 😲 | Unexpected or shocking |
| **Fear** | 😨 | Anxious or frightened |
| **Joy** | 😊 | Happy and delighted |

## 🏗️ Project Structure

```
Sentimental-Analysis/
│
├── app.py                                    # Main Streamlit application
├── test.py                                   # Testing script for model predictions
├── Senti_analysis_logical_reggresor.pkl      # Trained Logistic Regression model
├── vectorizer_tfidf.pkl                      # TF-IDF vectorizer for text transformation
├── screenshots/                              # UI screenshots
│   ├── Screenshot 2025-11-30 020726.png
│   └── image-1764448812662.png
├── README.md                                 # Project documentation
└── .git/                                     # Git version control
```

## 🧠 Machine Learning Model

### Model Architecture
- **Algorithm**: Logistic Regression
- **Text Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Dataset**: [Kaggle Emotions Dataset](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp)
- **Classes**: 6 emotions (sadness, anger, love, surprise, fear, joy)

### How It Works

1. **Text Preprocessing**: Input text is cleaned and preprocessed
2. **Vectorization**: Text is converted to numerical features using TF-IDF
3. **Prediction**: Logistic Regression model predicts the emotion class
4. **Visualization**: Results are displayed with confidence scores

## 🛠️ Technologies Used

- **Python 3.8+** - Core programming language
- **Streamlit** - Web application framework
- **scikit-learn** - Machine learning library
- **pandas** - Data manipulation
- **joblib** - Model serialization
- **Plotly** - Interactive visualizations
- **TF-IDF** - Text vectorization technique

## 📦 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/omgholap11/Sentimental-Analysis.git
cd Sentimental-Analysis
```

### Step 2: Install Dependencies
```bash
pip install streamlit pandas joblib plotly scikit-learn
```

### Step 3: Run the Application
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## 🚀 Usage

### Using the Web Interface

1. **Enter Text**: Type or paste your text in the input area
2. **Analyze**: Click the "Analyze Emotion" button
3. **View Results**: See the detected emotion with confidence distribution
4. **Try Examples**: Use pre-built examples from the sidebar
5. **Track Statistics**: Monitor your analysis history

### Using the Test Script

```python
python test.py
```

This script demonstrates how to use the model programmatically:

```python
import joblib

# Load model and vectorizer
model = joblib.load("Senti_analysis_logical_reggresor.pkl")
vectorizer = joblib.load("vectorizer_tfidf.pkl")

# Predict emotion
text = "I am so happy today!"
text_vector = vectorizer.transform([text])
prediction = model.predict(text_vector)[0]

emotions = ['sadness', 'anger', 'love', 'surprise', 'fear', 'joy']
print(f"Predicted Emotion: {emotions[prediction]}")
```

## 📊 Dataset Information

The model was trained on the **Emotions Dataset for NLP** from Kaggle, which contains:
- Text samples labeled with one of six emotions
- Diverse range of emotional expressions
- Pre-processed and cleaned text data

**Dataset Source**: [Kaggle Emotions Dataset](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp)

## 🎨 UI Components

### Main Features

1. **Header Section**: Gradient-styled title with modern typography
2. **Input Area**: Large text area with custom styling
3. **Emotion Card**: Dynamic result display with color-coded emotions
4. **Confidence Chart**: Bar chart showing emotion distribution
5. **Sidebar**: Statistics, emotion list, and history controls
6. **Example Buttons**: Quick-test functionality
7. **History Timeline**: Visual representation of recent analyses

### Design Highlights

- Dark theme with gradient accents
- Smooth animations and transitions
- Responsive layout for all screen sizes
- Color-coded emotion representation
- Interactive charts and visualizations

## 🔧 Configuration

### Customizing Emotions

Edit the `EMOTION_CONFIG` dictionary in `app.py`:

```python
EMOTION_CONFIG = {
    'emotion_name': {
        'emoji': '😊',
        'color': '#HEX_COLOR',
        'desc': 'Description text'
    }
}
```

### Styling Customization

Modify the CSS in the `st.markdown()` section for custom themes, colors, and layouts.

## 📈 Future Enhancements

- [ ] Add more emotion categories
- [ ] Implement deep learning models (LSTM, BERT)
- [ ] Multi-language support
- [ ] Export analysis results to CSV
- [ ] User authentication and personal dashboards
- [ ] API endpoint for external integration
- [ ] Mobile app version
- [ ] Batch text analysis
- [ ] Sentiment trend visualization over time

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the MIT License.

## 👨‍💻 Developer

**Om Gholap**

- GitHub: [@omgholap11](https://github.com/omgholap11)
- Email: omgholap051@gmail.com

---

<div align="center">

**If you found this project helpful, please give it a ⭐!**

</div>
