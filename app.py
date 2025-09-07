#pip install streamlit tensorflow "sentence-transformers>=2.2.0" torch
# Streamlit app that works with any input - title, abstract, or both
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer, util
import pickle
import subprocess
import json
import tempfile
import os

# Force CPU usage and disable GPU warnings at the very beginning
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'

# Load recommendation models only
@st.cache_resource
def load_recommendation_models():
    """Load recommendation components."""
    try:
        embeddings = pickle.load(open('models/embeddings.pkl','rb'))
        sentences = pickle.load(open('models/sentences.pkl','rb'))
        rec_model = SentenceTransformer('models/recommendation_model')
        return embeddings, sentences, rec_model
    except Exception as e:
        st.error(f"Error loading recommendation models: {e}")
        return None, None, None

# Recommendation function
def get_recommendations(input_text, embeddings, sentences, rec_model, num_recommendations=5):
    """Find similar papers based on input text (title, abstract, or both)."""
    if not input_text.strip():
        return []
    
    input_embedding = rec_model.encode(input_text)
    cosine_scores = util.cos_sim(embeddings, input_embedding)
    top_similar_papers = torch.topk(cosine_scores, dim=0, k=num_recommendations, sorted=True)
    
    return [sentences[i.item()] for i in top_similar_papers.indices]

# Create a modified version of your working script for programmatic use
def create_prediction_script():
    """Create a version of your working script that accepts text input and returns JSON."""
    script_content = '''
import os
# Set environment variables BEFORE importing tensorflow
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import sys
import json

def load_models():
    try:
        model_path = "models/model.keras"
        text_vocab_path = "models/text_vocabulary.pkl"
        label_vocab_path = "models/label_vocabulary.pkl"
        config_path = "models/vectorizer_config.pkl"
        
        # Load the model without its optimizer state for compatibility
        classification_model = keras.models.load_model(model_path, compile=False)
        
        with open(text_vocab_path, "rb") as f:
            text_vocabulary = pickle.load(f)
        with open(label_vocab_path, "rb") as f:
            label_vocab = pickle.load(f)
        with open(config_path, "rb") as f:
            vectorizer_config = pickle.load(f)
        
        text_vectorizer = layers.TextVectorization(
            max_tokens=vectorizer_config['max_tokens'],
            ngrams=vectorizer_config['ngrams'],
            output_mode="count"
        )
        text_vectorizer.set_vocabulary(text_vocabulary)
        
        return classification_model, text_vectorizer, label_vocab
    except Exception as e:
        print(json.dumps({"error": f"Model loading failed: {str(e)}"}))
        return None, None, None

def predict_subjects(text, model, vectorizer, label_vocab):
    try:
        preprocessed_text = vectorizer([text])
        predictions = model.predict(preprocessed_text, verbose=0)
        binary_predictions = np.round(predictions).astype(int)[0]
        hot_indices = np.argwhere(binary_predictions == 1.0).flatten()
        predicted_labels = np.take(label_vocab, hot_indices)
        return list(predicted_labels)
    except Exception as e:
        print(json.dumps({"error": f"Prediction failed: {str(e)}"}))
        return []

if __name__ == "__main__":
    try:
        text = sys.argv[1] if len(sys.argv) > 1 else ""
        if not text:
            print(json.dumps({"error": "No text provided"}))
            sys.exit(1)
        
        model, vectorizer, vocab = load_models()
        if model is None:
            print(json.dumps({"error": "Failed to load models"}))
            sys.exit(1)
        
        subjects = predict_subjects(text, model, vectorizer, vocab)
        print(json.dumps({"subjects": subjects}))
    except Exception as e:
        print(json.dumps({"error": f"Script error: {str(e)}"}))
'''
    
    script_path = 'predict_script.py'
    with open(script_path, 'w') as f:
        f.write(script_content)
    return script_path

def predict_subjects_via_script(text):
    """Use the working command-line script to predict subjects."""
    if not text.strip():
        return []
    
    try:
        # Create the prediction script
        script_path = create_prediction_script()
        
        # Enhanced environment variables to force CPU usage
        env = os.environ.copy()
        env.update({
            'TF_CPP_MIN_LOG_LEVEL': '2',
            'TF_ENABLE_ONEDNN_OPTS': '0',
            'CUDA_VISIBLE_DEVICES': '',
            'TF_FORCE_GPU_ALLOW_GROWTH': 'false',
            'PYTHONIOENCODING': 'utf-8'
        })
        
        # Run the script with proper error handling
        result = subprocess.run([
            'python3', script_path, text
        ], capture_output=True, text=True, timeout=90, env=env)
        
        # Clean up the script file
        try:
            os.remove(script_path)
        except:
            pass
        
        if result.returncode == 0:
            # Parse the JSON output from stdout
            stdout_lines = result.stdout.strip().split('\n')
            
            # Find the JSON line (should be the last meaningful line)
            for line in reversed(stdout_lines):
                line = line.strip()
                if line.startswith('{') and line.endswith('}'):
                    try:
                        response = json.loads(line)
                        if 'error' in response:
                            st.error(f"Model error: {response['error']}")
                            return []
                        return response.get('subjects', [])
                    except json.JSONDecodeError as e:
                        continue
            
            # If no JSON found, there might be an issue
            st.warning("No valid JSON response from prediction script")
            return []
            
        else:
            # Handle errors - show only relevant error messages
            stderr_lines = result.stderr.split('\n')
            stdout_lines = result.stdout.split('\n')
            
            # Look for error messages in stdout (our JSON errors)
            for line in stdout_lines:
                if line.strip().startswith('{"error"'):
                    try:
                        error_data = json.loads(line.strip())
                        if 'error' in error_data:
                            st.error(f"Model loading error: {error_data['error']}")
                            return []
                    except:
                        pass
            
            # Look for meaningful errors in stderr (excluding CUDA warnings)
            actual_errors = []
            for line in stderr_lines:
                if any(keyword in line.lower() for keyword in ['error', 'exception', 'failed']):
                    if 'cuda' not in line.lower():  # Skip CUDA errors since we're using CPU
                        actual_errors.append(line)
            
            if actual_errors:
                st.error(f"Execution error: {actual_errors[0]}")
            else:
                st.warning("Prediction failed - check model files and TensorFlow version")
            return []
            
    except subprocess.TimeoutExpired:
        st.error("Prediction timeout - please try again")
        return []
    except Exception as e:
        st.error(f"Script execution error: {e}")
        return []

def combine_inputs(title, abstract):
    """Combine title and abstract intelligently for processing."""
    parts = []
    if title.strip():
        parts.append(f"Title: {title.strip()}")
    if abstract.strip():
        parts.append(f"Abstract: {abstract.strip()}")
    return " ".join(parts)

# Main app
def main():
    st.title("Research Paper Analysis App")
    st.write("Enter paper information to get recommendations and subject predictions. Works with title, abstract, or both!")
    
    # Load recommendation models
    embeddings, sentences, rec_model = load_recommendation_models()
    
    if not all([embeddings is not None, sentences is not None, rec_model is not None]):
        st.error("Failed to load recommendation models. Please check your files.")
        return
    
    # Input fields
    st.subheader("Input")
    paper_title = st.text_input(
        "Paper Title (optional)", 
        placeholder="e.g., Attention Is All You Need",
        help="Enter a title for better recommendations"
    )
    
    paper_abstract = st.text_area(
        "Paper Abstract (optional)", 
        placeholder="Enter the abstract here for more accurate analysis...",
        height=150,
        help="Enter an abstract for better subject prediction and recommendations"
    )
    
    # Analysis options
    col1, col2 = st.columns(2)
    
    with col1:
        analyze_btn = st.button("Analyze Paper", type="primary")
    
    with col2:
        num_recommendations = st.slider("Number of recommendations", 3, 10, 5)
    
    if analyze_btn:
        # Validate input - need at least something
        if not paper_title.strip() and not paper_abstract.strip():
            st.error("Please enter at least a title or abstract to analyze.")
            return
        
        # Prepare text for analysis
        combined_text = combine_inputs(paper_title, paper_abstract)
        
        # Show what we're analyzing
        st.subheader("Analysis Results")
        
        if paper_title.strip() and paper_abstract.strip():
            st.info("📄 Analyzing both title and abstract")
        elif paper_title.strip():
            st.info("📝 Analyzing title only")
        else:
            st.info("📃 Analyzing abstract only")
        
        # Create two columns for results
        results_col1, results_col2 = st.columns(2)
        
        with results_col1:
            # Similar Papers
            st.markdown("### 🔍 Similar Papers")
            with st.spinner("Finding similar papers..."):
                # Use whatever input we have for recommendations
                search_text = paper_title if paper_title.strip() else paper_abstract
                recommendations = get_recommendations(
                    search_text, embeddings, sentences, rec_model, num_recommendations
                )
                
            if recommendations:
                for i, paper in enumerate(recommendations, 1):
                    st.write(f"**{i}.** {paper}")
            else:
                st.warning("No similar papers found.")
        
        with results_col2:
            # Subject Prediction
            st.markdown("### 🎯 Predicted Subjects")
            with st.spinner("Predicting subject areas..."):
                # Use whatever text we have for subject prediction
                predict_text = paper_abstract if paper_abstract.strip() else paper_title
                predicted_subjects = predict_subjects_via_script(predict_text)
                
            if predicted_subjects:
                # Display as colored badges
                for subject in predicted_subjects:
                    st.markdown(f"🏷️ **{subject}**")
                
                st.success(f"✅ Found {len(predicted_subjects)} subject area(s)")
            else:
                # Only show this if we actually tried to predict
                if paper_title.strip() or paper_abstract.strip():
                    st.info("No specific subject areas detected")
        
        # Additional analysis info
        st.markdown("---")
        st.subheader("📊 Analysis Summary")
        
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        
        with summary_col1:
            st.metric("Input Type", 
                     "Title + Abstract" if paper_title.strip() and paper_abstract.strip() 
                     else "Title Only" if paper_title.strip() 
                     else "Abstract Only")
        
        with summary_col2:
            st.metric("Similar Papers Found", len(recommendations) if 'recommendations' in locals() else 0)
        
        with summary_col3:
            st.metric("Subject Areas", len(predicted_subjects) if 'predicted_subjects' in locals() else 0)
    
    # Help section
    with st.expander("💡 How to Use This App"):
        st.markdown("""
        **Flexible Input Options:**
        - **Title Only**: Get recommendations based on title similarity + subject prediction from title
        - **Abstract Only**: Get recommendations based on content similarity + subject prediction from abstract  
        - **Title + Abstract**: Best results! Uses title for recommendations, abstract for subject prediction
        
        **What Each Feature Does:**
        - **Similar Papers**: Finds papers with similar content using semantic similarity
        - **Subject Prediction**: Identifies research areas/topics using your trained model
        
        **Tips for Better Results:**
        - Titles work great for finding similar papers by topic
        - Abstracts give more accurate subject predictions
        - Combine both for comprehensive analysis
        """)
    
    # Model info sidebar
    with st.sidebar:
        st.header("📈 Model Status")
        
        if embeddings is not None:
            st.metric("Papers in Database", len(sentences))
            st.success("✅ Recommendation model loaded")
        else:
            st.error("❌ Recommendation model failed")
        
        st.markdown("---")
        st.subheader("🛠️ Features")
        st.markdown("""
        • **Smart Input Handling**  
        • **Semantic Similarity Search**  
        • **Multi-label Subject Prediction**  
        • **Flexible Analysis Options**
        • **CPU-Only Processing**
        """)
        
        st.markdown("---")
        st.caption("💡 Works with any combination of title/abstract!")

if __name__ == "__main__":
    main()
