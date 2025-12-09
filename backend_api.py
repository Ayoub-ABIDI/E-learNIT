# backend_api.py 
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
from PIL import Image
import numpy as np
from Multimodal_rag import EnhancedSignLanguageRAG
from dotenv import load_dotenv
import traceback

load_dotenv()

app = Flask(__name__)
CORS(app)

print("🚀 Initializing Enhanced RAG system...")
rag_system = EnhancedSignLanguageRAG(llm_provider="groq")
print(f"✅ RAG System ready! Database: {rag_system.collection.count()} embeddings")

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "E-learNIT Sign Language API",
        "version": "2.1 - Improved Phrase Construction",
        "status": "running",
        "database_size": rag_system.collection.count(),
        "endpoints": {
            "GET /": "API information",
            "GET /health": "Health check",
            "POST /recognize": "Recognize sign from image",
            "POST /construct_phrase": "Construct phrase from signs (IMPROVED)",
            "POST /text_to_speech": "Text-to-speech preparation"
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "database_size": rag_system.collection.count(),
        "augmentation": "enabled"
    })

@app.route('/recognize', methods=['POST'])
def recognize_sign():
    try:
        print("📸 Received recognition request")
        data = request.json
        
        if not data or 'image' not in data:
            print("❌ No image in request")
            return jsonify({
                "success": False,
                "error": "No image provided"
            }), 400
        
        # Decode image
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        print(f"✅ Image decoded: {image.size}")
        
        # Recognize with RAG
        result = rag_system.recognize_with_llm(image)
        
        if result:
            print(f"🎯 Recognized: {result['sign']} ({result['confidence']:.2%})")
            print(f"   Reasoning: {result.get('reasoning', 'N/A')[:50]}...")
            
            return jsonify({
                "success": True,
                "sign": result['sign'],
                "confidence": float(result['confidence']),
                "category": result['category'],
                "reasoning": result.get('reasoning', ''),
                "alternatives": [
                    {
                        "sign": alt.get('sign', ''),
                        "confidence": float(alt.get('confidence', 0))
                    }
                    for alt in result.get('all_candidates', [])[:3]
                ]
            })
        else:
            print("⚠️ No sign detected")
            return jsonify({
                "success": False,
                "message": "No sign detected"
            })
    
    except Exception as e:
        print(f"❌ Recognition error: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/construct_phrase', methods=['POST'])
def construct_phrase():
    """
    IMPROVED: Construct natural phrases in Tunisian, French, and English
    """
    try:
        print("\n" + "="*70)
        print("📝 PHRASE CONSTRUCTION REQUEST")
        print("="*70)
        
        data = request.json
        signs_list = data.get('signs', [])
        
        if len(signs_list) < 2:
            print("❌ Not enough signs")
            return jsonify({
                "success": False,
                "message": "Need at least 2 signs to construct phrase"
            })
        
        print(f"🔤 Input signs ({len(signs_list)}): {signs_list}")
        
        # IMPROVED PROMPT for LLM
        prompt = f"""You are an expert in Tunisian Sign Language (TSL) and Tunisian Arabic (Darija).

TASK: Create a natural, meaningful phrase from these sign language gestures.

DETECTED SIGNS: {' → '.join(signs_list)}

TUNISIAN ARABIC VOCABULARY REFERENCE:
- aaweni = "help me" / "I need help"
- behi = "ok" / "good" / "fine" / "alright"
- ca va = "how are you" / "are you okay"
- de rien = "you're welcome" / "excuse me" / "sorry"
- ena njweb = "can I reply" / "may I answer"
- enti tjwen = "you reply" / "you answer"
- fhmet = "I understood" / "I get it"
- ma fhmtch = "I don't understand" / "I didn't get it"
- madrsa = "school"
- manaarefch = "I don't know"
- naaref = "I know"
- naawnek = "can I help you" / "how can I assist"
- note khyba = "bad grade" / "poor mark"
- soueel = "question"
- yaq9ra = "to read" / "to study"
- yekteb = "to write"

INSTRUCTIONS:
1. Look at the sequence of signs
2. Remove any obvious duplicates (if same sign appears multiple times consecutively)
3. Construct a natural, conversational phrase that makes sense in an educational context
4. Provide translations in Tunisian Arabic (Darija), French, and English
5. Add context explaining what the person is trying to communicate

RESPOND ONLY IN THIS JSON FORMAT (no markdown, no extra text):
{{
    "tunisian_phrase": "the complete natural phrase in Tunisian Arabic/Darija",
    "french_phrase": "la phrase complète et naturelle en français",
    "english_phrase": "the complete natural phrase in English",
    "explanation": "brief explanation of what the person is communicating in this educational context"
}}

EXAMPLE:
If signs are: ["naawnek", "soueel", "fhmet"]
Output:
{{
    "tunisian_phrase": "Naawnek? 3andi soueel. Fhmet.",
    "french_phrase": "Puis-je vous aider? J'ai une question. J'ai compris.",
    "english_phrase": "Can I help you? I have a question. I understood.",
    "explanation": "A student is asking if they can help, mentioning they have a question, and confirming they understood something."
}}

NOW CONSTRUCT THE PHRASE:"""

        print("\n📤 Sending to LLM...")
        print(f"Prompt length: {len(prompt)} chars")
        
        # Query LLM
        if not rag_system.llm_client:
            print("⚠️ No LLM client available")
            return jsonify({
                "success": False,
                "message": "LLM not configured"
            }), 500
        
        response = rag_system.llm_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,  # Slightly more creative
            max_tokens=800
        )
        
        llm_response = response.choices[0].message.content
        print("\n📥 LLM Response:")
        print(llm_response)
        
        # Parse JSON response
        import json
        import re
        
        # Extract JSON from response
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', llm_response, re.DOTALL)
        
        if json_match:
            phrase_data = json.loads(json_match.group())
            
            print("\n✅ PARSED PHRASE:")
            print(f"🇹🇳 Tunisian: {phrase_data.get('tunisian_phrase', 'N/A')}")
            print(f"🇫🇷 French: {phrase_data.get('french_phrase', 'N/A')}")
            print(f"🇬🇧 English: {phrase_data.get('english_phrase', 'N/A')}")
            print(f"💡 Context: {phrase_data.get('explanation', 'N/A')}")
            print("="*70 + "\n")
            
            return jsonify({
                "success": True,
                "phrase_tunisian": phrase_data.get('tunisian_phrase', ''),
                "phrase_french": phrase_data.get('french_phrase', ''),
                "phrase_english": phrase_data.get('english_phrase', ''),
                "context": phrase_data.get('explanation', ''),
                "raw_signs": signs_list,
                "sign_count": len(signs_list)
            })
        else:
            print("⚠️ Could not parse JSON from LLM response")
            print("="*70 + "\n")
            
            # Fallback: basic concatenation
            return jsonify({
                "success": True,
                "phrase_tunisian": ' '.join(signs_list),
                "phrase_french": ' '.join(signs_list),
                "phrase_english": ' '.join(signs_list),
                "context": "Direct sign-by-sign translation (LLM parsing failed)",
                "raw_signs": signs_list,
                "sign_count": len(signs_list)
            })
    
    except Exception as e:
        print(f"\n❌ PHRASE CONSTRUCTION ERROR: {str(e)}")
        traceback.print_exc()
        print("="*70 + "\n")
        
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/text_to_speech', methods=['POST'])
def text_to_speech():
    try:
        print("🔊 Received text-to-speech request")
        data = request.json
        
        text = data.get('text', '')
        language = data.get('language', 'fr')
        
        if not text:
            return jsonify({
                "success": False,
                "error": "No text provided"
            }), 400
        
        print(f"🗣️ Text: {text}")
        print(f"🌍 Language: {language}")
        
        return jsonify({
            "success": True,
            "text": text,
            "language": language,
            "message": "Text ready for speech synthesis"
        })
    
    except Exception as e:
        print(f"❌ Text-to-speech error: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == '__main__':
    PORT = 5001
    print("\n" + "="*70)
    print("🚀 E-learNIT Sign Language API Server v2.1")
    print("   IMPROVED: Better phrase construction with context")
    print("="*70)
    print(f"📡 Server running at: http://localhost:{PORT}")
    print(f"🗄️  Database size: {rag_system.collection.count()} embeddings")
    print(f"🤖 LLM: Groq (llama-3.3-70b-versatile)")
    print(f"🎯 Features: Deduplication, context extraction, natural phrases")
    print("="*70 + "\n")
    
    app.run(host='0.0.0.0', port=PORT, debug=True, use_reloader=False)