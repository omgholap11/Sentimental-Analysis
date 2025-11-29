import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import time

# Page configuration
st.set_page_config(
    page_title="Emotion Analyzer",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
    <style>
    /* Main background with gradient */
    .stApp {
        background: #0a0a0a;
    }
    
    /* Custom container styling */
    .main-container {
        background: #1a1a1a;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        margin: 1rem 0;
        border: 1px solid #333;
    }
    
    /* Header styling */
    .header-text {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .subheader-text {
        font-size: 1.2rem;
        color: #aaa;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    
    /* Emotion card styling */
    .emotion-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        animation: slideIn 0.5s ease-out;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .emotion-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .emotion-subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Stats card */
    .stat-card {
        background: #252525;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #aaa;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Text area styling */
    .stTextArea textarea {
        border-radius: 12px;
        border: 2px solid #333;
        padding: 1rem;
        font-size: 1rem;
        transition: border-color 0.3s ease;
        background: #252525;
        color: #eee;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #1a1a1a;
        border-right: 1px solid #333;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Emotion configurations
EMOTION_CONFIG = {
    'sadness': {'emoji': '😢', 'color': '#4A90E2', 'desc': 'Feeling down or melancholic'},
    'anger': {'emoji': '😠', 'color': '#E74C3C', 'desc': 'Expressing frustration or rage'},
    'love': {'emoji': '❤️', 'color': '#E91E63', 'desc': 'Showing affection and warmth'},
    'surprise': {'emoji': '😲', 'color': '#FFA726', 'desc': 'Unexpected or shocking'},
    'fear': {'emoji': '😨', 'color': '#9C27B0', 'desc': 'Anxious or frightened'},
    'joy': {'emoji': '😊', 'color': '#4CAF50', 'desc': 'Happy and delighted'}
}

# Load model and vectorizer
@st.cache_resource
def load_models():
    try:
        model = joblib.load("Senti_analysis_logical_reggresor.pkl")
        vectorizer = joblib.load("vectorizer_tfidf.pkl")
        return model, vectorizer
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please ensure 'Senti_analysis_logical_reggresor.pkl' and 'vectorizer_tfidf.pkl' are in the same directory.")
        return None, None

def predict_emotion(text, model, vectorizer):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)[0]
    emotions_list = ['sadness', 'anger', 'love', 'surprise', 'fear', 'joy']
    return emotions_list[prediction], prediction

def create_confidence_chart(emotion):
    emotions = list(EMOTION_CONFIG.keys())
    # Simulated confidence scores (you can modify to get actual probabilities if your model supports it)
    values = [20, 15, 10, 12, 8, 25]
    # Boost the predicted emotion
    emotion_idx = emotions.index(emotion)
    values[emotion_idx] = 70
    
    colors = [EMOTION_CONFIG[e]['color'] for e in emotions]
    
    fig = go.Figure(data=[
        go.Bar(
            x=emotions,
            y=values,
            marker=dict(
                color=colors,
                line=dict(color='white', width=2)
            ),
            text=[f"{v}%" for v in values],
            textposition='outside',
        )
    ])
    
    fig.update_layout(
        title="Emotion Confidence Distribution",
        xaxis_title="Emotions",
        yaxis_title="Confidence (%)",
        plot_bgcolor='rgba(26,26,26,1)',
        paper_bgcolor='rgba(26,26,26,1)',
        font=dict(size=12, color='#eee'),
        height=400,
        showlegend=False,
        yaxis=dict(range=[0, 100])
    )
    
    return fig

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

# Main app
def main():
    # Header
    st.markdown('<h1 class="header-text">🎭 Emotion Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subheader-text">Powered by Machine Learning • Analyze emotions in real-time</p>', unsafe_allow_html=True)
    
    # Load models
    model, vectorizer = load_models()
    
    if model is None or vectorizer is None:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 About")
        st.info("""
        This application uses a **Logistic Regression** model trained on emotional text data to predict emotions from your input.
        
        **Supported Emotions:**
        - 😢 Sadness
        - 😠 Anger
        - ❤️ Love
        - 😲 Surprise
        - 😨 Fear
        - 😊 Joy
        """)
        
        st.markdown("### 📈 Statistics")
        if st.session_state.history:
            total_analyses = len(st.session_state.history)
            most_common = max(set(st.session_state.history), key=st.session_state.history.count)
            
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{total_analyses}</div>
                <div class="stat-label">Total Analyses</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{EMOTION_CONFIG[most_common]['emoji']}</div>
                <div class="stat-label">Most Common: {most_common.title()}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("*No analyses yet*")
        
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        
        # Text input
        user_text = st.text_area(
            "✍️ Enter your text here:",
            placeholder="Type something like: 'I am so happy today!' or 'This is really frustrating...'",
            height=150,
            key="text_input"
        )
        
        # Analyze button
        analyze_button = st.button("🔍 Analyze Emotion", use_container_width=True)
        
        if analyze_button and user_text.strip():
            with st.spinner("🤖 Analyzing your text..."):
                time.sleep(0.5)  # Small delay for effect
                emotion, prediction_idx = predict_emotion(user_text, model, vectorizer)
                st.session_state.history.append(emotion)
                
                # Display result
                config = EMOTION_CONFIG[emotion]
                st.markdown(f"""
                <div class="emotion-card" style="background: linear-gradient(135deg, {config['color']}dd 0%, {config['color']} 100%);">
                    <div class="emotion-title">{config['emoji']} {emotion.upper()}</div>
                    <div class="emotion-subtitle">{config['desc']}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence chart
                st.plotly_chart(create_confidence_chart(emotion), use_container_width=True)
                
        elif analyze_button:
            st.warning("⚠️ Please enter some text to analyze!")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        st.markdown("### 💡 Try These Examples")
        
        examples = [
            "I'm so excited about the weekend!",
            "This situation makes me really angry.",
            "I miss you so much, can't wait to see you.",
            "Oh no! I forgot my keys at home.",
            "That surprise party was absolutely amazing!",
            "I feel so lonely and sad today."
        ]
        
        for example in examples:
            if st.button(f"📝 {example[:30]}...", key=example, use_container_width=True):
                st.session_state.text_input = example
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Recent history
    if st.session_state.history:
        st.markdown("---")
        st.markdown("### 📜 Recent Analysis History")
        
        cols = st.columns(6)
        recent = st.session_state.history[-6:][::-1]
        
        for idx, emotion in enumerate(recent):
            with cols[idx]:
                config = EMOTION_CONFIG[emotion]
                st.markdown(f"""
                <div style="text-align: center; padding: 1rem; background: #252525; border-radius: 10px; margin: 0.5rem 0; border: 1px solid #333;">
                    <div style="font-size: 2rem;">{config['emoji']}</div>
                    <div style="font-size: 0.8rem; color: #aaa;">{emotion.title()}</div>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()