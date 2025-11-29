import pandas as pd
import joblib

model = joblib.load("Senti_analysis_logical_reggresor.pkl")
vectorizer = joblib.load("vectorizer_tfidf.pkl")


def predict_emotion(text):

    text_vector = vectorizer.transform([text])
    
    prediction = model.predict(text_vector)[0]
    
    return prediction

if __name__ == "__main__":
    text = "Where is that stupid."
    result = predict_emotion(text)
    print("Input Text:", text)
    print("Predicted Emotion Number:", result)

    emotions_numbers = ['sadness', 'anger', 'love', 'surprise', 'fear', 'joy']

    print("Predicted Emotions: " , emotions_numbers[result])
   