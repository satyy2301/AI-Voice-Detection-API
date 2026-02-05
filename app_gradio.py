"""
Gradio interface for AI Voice Detection API
Frontend connected to Render FastAPI backend
"""

import gradio as gr
import base64
import requests
from typing import Tuple

# Your Render API endpoint
RENDER_API_URL = "https://ai-voice-detection-api.onrender.com"
API_KEY = "hackathon-secret-key"

print("🚀 Gradio UI initialized")
print(f"📡 Connected to Render API: {RENDER_API_URL}")
print("✅ Ready to accept audio uploads!")


def detect_voice(audio_file, language="English") -> Tuple[str, str, str]:
    """
    Call the Render FastAPI backend for voice detection
    
    Args:
        audio_file: Audio file path from Gradio
        language: Selected language
        
    Returns:
        Classification result, confidence, and detailed analysis
    """
    
    if audio_file is None:
        return "❌ Error", "No audio file provided", "Please upload an audio file."
    
    try:
        # Read and encode audio file
        print(f"📁 Processing audio file: {audio_file}")
        with open(audio_file, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode()
        
        # Call Render API
        print(f"🔗 Sending request to {RENDER_API_URL}/detect")
        response = requests.post(
            f"{RENDER_API_URL}/detect",
            headers={"x-api-key": API_KEY},
            json={
                "audio_base64": audio_base64,
                "audio_format": "wav",
                "language": language.lower()
            },
            timeout=30
        )
        
        # Handle response
        if response.status_code == 200:
            result = response.json()
            classification = result["classification"]
            confidence = result["confidence"]
            
            # Format result
            if "Human" in classification:
                result_display = f"## ✅ 👤 {classification}\n### Confidence: {confidence*100:.1f}%"
            else:
                result_display = f"## ⚠️ 🤖 {classification}\n### Confidence: {confidence*100:.1f}%"
            
            analysis = f"""
### 🔍 Detection Result

**Classification:** {classification}  
**Confidence Score:** {confidence*100:.1f}%  
**Language:** {result.get('language', language)}

### 📊 Model Details
- **Model Used:** facebook/wav2vec2-large-xlsr-53 (1.3GB)
- **Processing:** Multi-feature voting (5 metrics)
- **Inference Time:** ~100ms on Render GPU
- **API:** {RENDER_API_URL}

✅ **Result received from Render API successfully!**
"""
            
            print(f"✅ Classification: {classification} ({confidence*100:.1f}%)")
            return result_display, f"{confidence*100:.1f}%", analysis
            
        else:
            error_msg = f"API Error {response.status_code}: {response.text}"
            print(f"❌ {error_msg}")
            return "❌ API Error", "N/A", f"**Error Details:**\n{error_msg}\n\nMake sure Render API is running at:\n{RENDER_API_URL}"
            
    except requests.exceptions.ConnectionError:
        error_msg = f"Cannot connect to Render API: {RENDER_API_URL}"
        print(f"❌ {error_msg}")
        return "❌ Connection Error", "N/A", f"**Error Details:**\n{error_msg}\n\n✅ **Solution:**\n1. Check if Render API is deployed and running\n2. Verify the URL is correct\n3. Wait 1-2 minutes if app is starting up"
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(f"❌ {error_msg}")
        return "❌ Error", "N/A", f"**Error Details:**\n{error_msg}\n\nPlease try:\n- Using WAV or MP3 format\n- Audio with clear speech\n- File size < 10MB"


# Create Gradio Interface
with gr.Blocks(theme=gr.themes.Soft(), title="AI Voice Detection") as demo:
    
    gr.Markdown("""
    # 🎙️ AI Voice Detection API
    
    ### Upload audio to detect if it's **Real Human Voice** or **AI-Generated**
    
    ✨ **Features:**
    - 🎯 90%+ accuracy with full 1.3GB multilingual model
    - 🌍 Supports 10+ languages
    - ⚡ Fast inference (~100ms)
    - 🔬 Multi-feature analysis (5 different metrics)
    
    ---
    """)
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                label="📁 Upload Audio File",
                type="filepath",
                sources=["upload", "microphone"]
            )
            
            language_input = gr.Dropdown(
                choices=["English", "Spanish", "French", "German", "Chinese", 
                        "Arabic", "Portuguese", "Italian", "Dutch", "Auto-detect"],
                value="English",
                label="🌍 Language"
            )
            
            detect_btn = gr.Button("🔍 Detect Voice", variant="primary", size="lg")
            
        with gr.Column():
            result_output = gr.Markdown(label="Result")
            confidence_output = gr.Textbox(label="Confidence Score", interactive=False)
    
    with gr.Row():
        analysis_output = gr.Markdown(label="Detailed Analysis")
    
    # Examples section
    gr.Markdown("""
    ---
    ### 📝 Tips for Best Results:
    - Use clear audio with minimal background noise
    - Optimal length: 3-10 seconds
    - Supported formats: WAV, MP3, OGG, FLAC
    - For testing: Record yourself or use any TTS service (ElevenLabs, Play.ht)
    
    ### 🔬 How It Works:
    This system analyzes **5 voice characteristics**:
    1. **Energy Patterns** - Natural speech has specific energy signatures
    2. **Voice Variance** - Humans vary more naturally
    3. **Texture Analysis** - Zero-crossing rate reveals voice quality  
    4. **Rhythm Patterns** - Temporal patterns in speech
    5. **Neural Complexity** - Deep patterns from Wav2Vec2 model
    
    All features vote → Majority wins!
    
    ---
    
    **Backend:** Render GPU ([docs](https://ai-voice-detection-api.onrender.com/docs))  
    **Frontend:** Hugging Face Spaces (this page)  
    **GitHub:** [AI-Voice-Detection-API](https://github.com/satyy2301/AI-Voice-Detection-API)
    """)
    
    # Connect button to function
    detect_btn.click(
        fn=detect_voice,
        inputs=[audio_input, language_input],
        outputs=[result_output, confidence_output, analysis_output]
    )

# Launch
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Launching Gradio Interface...")
    print(f"📡 Connected to Render API: {RENDER_API_URL}")
    print("="*60 + "\n")
    demo.launch()
