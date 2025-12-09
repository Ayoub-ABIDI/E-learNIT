# E-learNIT: AI-Powered Tunisian Sign Language Translator

<div align="center">

![E-learNIT Logo](logo.png)

**Real-time Tunisian Sign Language Translation for Online Education**

[![Herotopia Challenge](https://img.shields.io/badge/Herotopia-AI_Challenge-blue?style=for-the-badge)](https://herotopia.ai)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg?style=for-the-badge)](LICENSE)

[ Demo Video](#demo) • [ Documentation](#documentation) • [ Quick Start](#quick-start) • [ Architecture](#architecture)

</div>

---

## Table of Contents

- [Problem Statement](#-problem-statement)
- [Our Solution](#-our-solution)
- [Innovation: RAG vs Classical CV](#-innovation-rag-vs-classical-cv)
- [Technical Architecture](#-technical-architecture)
- [System Workflow](#-system-workflow)
- [Installation & Setup](#-installation--setup)
- [Usage Guide](#-usage-guide)
- [Supported Vocabulary](#-supported-vocabulary)
- [Performance Metrics](#-performance-metrics)
- [Future Roadmap](#-future-roadmap)
- [Team & Acknowledgments](#-team--acknowledgments)

---

## Problem Statement

### The Challenge

In Tunisia, **over 250,000 deaf individuals** face significant barriers in accessing online education:

| Challenge | Impact |
|-----------|--------|
|  **No real-time translation** | Deaf students cannot participate in live discussions |
|  **Limited accessibility tools** | Existing tools don't support Tunisian Sign Language (TSL) |
|  **Communication barriers** | Unable to ask questions or contribute during virtual classes |
|  **Language isolation** | Tunisian Sign Language differs from international standards |
|  **Educational inequality** | 70% lower graduation rates compared to hearing peers |

### Our Target Users

- **Deaf and hard-of-hearing students** in Tunisia
- **Teachers** with deaf students in their online classes
- **Educational institutions** seeking inclusive platforms
- **Online meeting participants** who use Tunisian Sign Language

---

## 💡 Our Solution

**E-learNIT** is a Chrome extension that provides **real-time Tunisian Sign Language translation** for online education platforms, enabling deaf students to:

### ✅ Core Features

```
┌────────────────────────────────────────────────────────────┐
│  🎥 WEBCAM → 🤟 SIGN → 📝 TEXT → 🔊 SPEECH → 💬 CHAT      │
└────────────────────────────────────────────────────────────┘
```

1. **🎥 Real-time Sign Recognition**
   - Captures signs from webcam every 3-4 seconds
   - Works seamlessly during video calls
   - No manual intervention needed

2. **📝 Multi-language Translation**
   - **Tunisian Arabic** (Darija) - Native language
   - **French** - Common educational language
   - **English** - International communication

3. **🔊 Text-to-Speech Synthesis**
   - Converts translated text to natural speech
   - Helps hearing participants understand
   - Customizable voice and speed

4. **💬 Direct Chat Integration**
   - One-click message sending
   - Works with Google Meet, Teams, Zoom
   - Preserves conversation flow

5. **🤖 Context-Aware Phrases**
   - LLM constructs natural sentences
   - Understands educational context
   - Explains intent behind signs

---

## Innovation: RAG vs Classical CV

### Why Traditional Computer Vision Falls Short

Most sign language recognition systems use **classical computer vision** approaches with CNNs/RNNs. However, these have critical limitations:

| Classical CV Approach | ❌ Limitations |
|----------------------|---------------|
| **CNN/RNN Models** | Requires extensive retraining for new signs |
| **Transfer Learning** | High computational cost (GPU required) |
| **Fixed Vocabulary** | Cannot adapt without model updates |
| **No Context Understanding** | Translates signs word-by-word only |
| **Energy Consumption** | 10-50W per inference (not scalable) |

### Our Innovation: Multimodal RAG System

We introduce a **Retrieval-Augmented Generation (RAG)** approach that combines:

```
┌────────────────────────────────────────────────────────────┐
│            MULTIMODAL RAG ARCHITECTURE                      │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  1️⃣ VISUAL ENCODING (CLIP)                                │
│     • Converts images to 512-dim embeddings               │
│     • Pre-trained on 400M image-text pairs                │
│     • Zero-shot learning capability                       │
│                                                            │
│  2️⃣ VECTOR DATABASE (ChromaDB)                            │
│     • Stores sign embeddings with metadata                │
│     • Cosine similarity search (<50ms)                    │
│     • Augmented dataset (6x larger)                       │
│                                                            │
│  3️⃣ LLM REASONING (LLaMA 3.3 70B)                         │
│     • Analyzes top candidates with context                │
│     • Constructs natural phrases                          │
│     • Explains communicative intent                       │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### Advantages Over Classical Models

| Feature | Classical CV | **Our RAG System** ✅ |
|---------|-------------|---------------------|
| **Adding New Signs** | Retrain entire model (hours) | Add to database (seconds) |
| **Computational Cost** | High (GPU required) | Low (CPU sufficient) |
| **Energy per Inference** | 10-50W | <2W |
| **Context Understanding** | None | Yes (via LLM) |
| **Vocabulary Size** | Fixed at training | Infinitely expandable |
| **Phrase Construction** | Not supported | Natural language output |
| **Accuracy** | 75-80% | **85%+** |

---

## Technical Architecture

### System Components

```
┌───────────────────────────────────────────────────────────────┐
│                     CLIENT SIDE                               │
│                                                               │
│  ┌─────────────────┐        ┌─────────────────┐              │
│  │  Chrome         │        │  Content        │              │
│  │  Extension      │◄──────►│  Script         │              │
│  │  (Popup UI)     │        │  (Overlay)      │              │
│  └─────────────────┘        └────────┬────────┘              │
│                                       │                       │
│                                       │ Captures Video        │
│                                       ▼                       │
│                              ┌─────────────────┐              │
│                              │   Webcam        │              │
│                              │   Feed          │              │
│                              └─────────────────┘              │
│                                                               │
└───────────────────────────────────────────────────────────────┘
                                    │
                                    │ Base64 Image
                                    │ (HTTP POST)
                                    ▼
┌───────────────────────────────────────────────────────────────┐
│                     BACKEND (Flask API)                       │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  1. Image Preprocessing                                 │ │
│  │     • Decode Base64                                     │ │
│  │     • Resize to 224x224                                 │ │
│  │     • Normalize pixel values                            │ │
│  └─────────────────────────────────────────────────────────┘ │
│                           │                                   │
│                           ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  2. CLIP Encoding                                       │ │
│  │     • OpenAI CLIP ViT-B/32                              │ │
│  │     • Generates 512-dim embedding                       │ │
│  │     • Inference time: ~100ms                            │ │
│  └─────────────────────────────────────────────────────────┘ │
│                           │                                   │
│                           ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  3. ChromaDB Vector Search                              │ │
│  │     • Cosine similarity query                           │ │
│  │     • Returns top 7 candidates                          │ │
│  │     • Search time: ~50ms                                │ │
│  │     • Database: 6000+ embeddings                        │ │
│  └─────────────────────────────────────────────────────────┘ │
│                           │                                   │
│                           ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  4. LLM Reasoning (Groq + LLaMA 3.3 70B)                │ │
│  │     • Analyzes candidates with vocabulary context       │ │
│  │     • Selects best match with confidence                │ │
│  │     • Constructs natural phrases (if multiple signs)    │ │
│  │     • Inference time: ~2s (Groq acceleration)           │ │
│  └─────────────────────────────────────────────────────────┘ │
│                           │                                   │
└───────────────────────────┼───────────────────────────────────┘
                            │
                            │ JSON Response
                            │ {sign, confidence, context, phrases}
                            ▼
┌───────────────────────────────────────────────────────────────┐
│                      OUTPUT LAYER                             │
│                                                               │
│  📝 Text Display (3 languages)                                │
│  🔊 Speech Synthesis (Web Speech API)                         │
│  💬 Chat Injection (Platform-specific)                        │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Frontend** | JavaScript (ES6+) | Chrome extension logic |
| **UI Framework** | HTML5 + CSS3 | Overlay and popup interface |
| **Backend** | Flask (Python) | REST API server |
| **Visual Encoding** | OpenAI CLIP ViT-B/32 | Image → embedding conversion |
| **Vector Database** | ChromaDB | Embedding storage & similarity search |
| **LLM Inference** | Groq API (LLaMA 3.3 70B) | Context reasoning & phrase construction |
| **Speech Synthesis** | Web Speech API | Text-to-speech conversion |
| **Computer Vision** | OpenCV + PIL | Image preprocessing |

---

##  Dataset

Due to size constraints, the Tunisian Sign Language dataset is hosted separately.

**Download**: 

After downloading:
1. Extract to project root
2. Ensure folder structure: `dataset/`
3. Run backend: `python backend_api.py`

## System Workflow

### End-to-End Process

```
┌──────────────────────────────────────────────────────────────┐
│  PHASE 1: SIGN DETECTION                                     │
└──────────────────────────────────────────────────────────────┘

1. User shows sign to webcam during Google Meet/Teams/Zoom call
   │
   ├─► Content script captures video frame every 4 seconds
   │
   └─► Converts frame to Base64 JPEG (640x480)


┌──────────────────────────────────────────────────────────────┐
│  PHASE 2: VISUAL ENCODING                                    │
└──────────────────────────────────────────────────────────────┘

2. Backend receives image via POST /recognize
   │
   ├─► Decodes Base64 → PIL Image → RGB
   │
   ├─► Resizes to 224x224 (CLIP input size)
   │
   └─► CLIP model generates 512-dimensional embedding
       (Normalized L2 vector for cosine similarity)


┌──────────────────────────────────────────────────────────────┐
│  PHASE 3: SIMILARITY SEARCH                                  │
└──────────────────────────────────────────────────────────────┘

3. Query embedding compared with ChromaDB
   │
   ├─► Cosine similarity: score = dot(query, db_vector)
   │
   ├─► Returns top 7 candidates with distances
   │
   └─► Example: [("behi", 0.12), ("ca va", 0.25), ...]
       (Lower distance = higher similarity)


┌──────────────────────────────────────────────────────────────┐
│  PHASE 4: LLM REASONING                                      │
└──────────────────────────────────────────────────────────────┘

4. LLaMA 3.3 70B analyzes candidates
   │
   ├─► Prompt includes:
   │   • Top 7 candidates with similarity scores
   │   • Tunisian vocabulary reference
   │   • Educational context
   │
   ├─► LLM selects most likely sign
   │
   └─► Returns: {sign, confidence, reasoning}


┌──────────────────────────────────────────────────────────────┐
│  PHASE 5: DEDUPLICATION                                      │
└──────────────────────────────────────────────────────────────┘

5. Content script checks for duplicates
   │
   ├─► If same sign within 3 seconds → SKIP
   │
   ├─► If different sign OR 3+ seconds passed → ADD
   │
   └─► Updates sign sequence display


┌──────────────────────────────────────────────────────────────┐
│  PHASE 6: PHRASE CONSTRUCTION (User triggered)               │
└──────────────────────────────────────────────────────────────┘

6. User clicks "Construct Phrase" button
   │
   ├─► Cleans sign sequence (removes consecutive duplicates)
   │   Example: [behi, behi, soueel, fhmet]
   │             → [behi, soueel, fhmet]
   │
   ├─► Sends to POST /construct_phrase
   │
   ├─► LLM constructs natural phrase with context:
   │
   │   Input: ["naawnek", "soueel", "fhmet"]
   │
   │   Output:
   │   🇹🇳 Tunisian: "Naawnek? 3andi soueel. Fhmet."
   │   🇫🇷 French: "Puis-je vous aider? J'ai une question. J'ai compris."
   │   🇬🇧 English: "Can I help you? I have a question. I understood."
   │   💡 Context: "Student politely asks for help, mentions having a
   │               question, and confirms understanding."
   │
   └─► Displays in overlay


┌──────────────────────────────────────────────────────────────┐
│  PHASE 7: OUTPUT                                             │
└──────────────────────────────────────────────────────────────┘

7A. SPEECH SYNTHESIS (User clicks "Speak Aloud")
    │
    ├─► Web Speech API synthesizes selected language
    │
    └─► Audio plays through browser


7B. CHAT INJECTION (User clicks "Send to Chat")
    │
    ├─► Platform detection (Meet/Teams/Zoom)
    │
    ├─► Finds platform-specific chat input selector
    │
    ├─► Injects text via DOM manipulation
    │
    └─► Clicks send button (or simulates Enter key)
```

### Key Innovations in Workflow

1. **Deduplication Layer**: Prevents same sign from being detected multiple times when user holds position

2. **Lazy Phrase Construction**: Only triggers when user explicitly requests (avoids premature/wrong phrases)

3. **Platform-Agnostic**: Detects Google Meet, Teams, or Zoom and adapts chat injection accordingly

4. **Context Preservation**: LLM explains *why* signs were chosen and what user is communicating

---

## Installation & Setup

**Extract the Vector Database**
Extract the file:

chroma_db_augmented.tar.gz

### Prerequisites

```bash
# Check Python version (must be 3.8+)
python --version  # or python3 --version

# Check pip
pip --version  # or pip3 --version
```

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/E-learNIT.git
cd E-learNIT
```

### Step 2: Get Groq API Key (FREE)

1. Visit: **https://console.groq.com/keys**
2. Sign up (free account)
3. Create new API key
4. Copy the key (starts with `gsk_...`)

### Step 3: Configure Environment

Create `.env` file in project root:

```bash
# .env
GROQ_API_KEY=gsk_your_actual_api_key_here
```

**⚠️ Important**: Replace `gsk_your_actual_api_key_here` with your real key!

### Step 4: Install Python Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
flask==3.0.0
flask-cors==4.0.0
chromadb==0.4.22
transformers==4.36.0
torch==2.1.0
torchvision==0.16.0
opencv-python==4.8.1
Pillow==10.1.0
python-dotenv==1.0.0
groq==0.4.2
numpy==1.24.3
```

### Step 5: Prepare Dataset

Organize your Tunisian Sign Language dataset:

```
dataset/
├── Alphabet/
│   ├── A/
│   │   ├── sign_a_001.jpg
│   │   ├── sign_a_002.jpg
│   ├── B/
├── Numbers/
│   ├── 1/
│   ├── 2/
├── Words/
│   ├── naawnek/
│   │   ├── gesture_001.jpg
│   │   ├── gesture_002.avi
│   ├── behi/
│   ├── soueel/
│   ├── fhmet/
│   └── ...
```

### Step 6: Start Backend Server

```bash
python backend_api.py
```

**✅ Expected Output:**
```
 E-learNIT Sign Language API Server v2.1
   IMPROVED: Better phrase construction with context
======================================================================
📡 Server running at: http://localhost:5001
🗄️  Database size: 6243 embeddings
🤖 LLM: Groq (llama-3.3-70b-versatile)
🎯 Features: Deduplication, context extraction, natural phrases
======================================================================

 * Serving Flask app 'backend_api'
 * Running on http://127.0.0.1:5001
```

### Step 7: Install Chrome Extension

**1. Open Chrome browser and navigate to extensions**
   - Type `chrome://extensions/` in the address bar

<img width="1910" alt="1-chrome-extensions-page" src="https://github.com/user-attachments/assets/982bae15-0e53-4829-8811-ec841b48beb9" />

**2. Enable Developer mode**
   - Toggle the **"Developer mode"** switch in the top-right corner

<img width="496" alt="2-enable-developer-mode" src="https://github.com/user-attachments/assets/048bc286-ceb0-40d0-8f90-912c17ad80b3" />

**3. Click "Load unpacked"**
   - Click the **"Load unpacked"** button that appears

<img width="461" alt="3-load-unpacked-button" src="https://github.com/user-attachments/assets/2c03b475-a646-4b4f-b26c-a0e865e76a12" />

**4. Select the extension folder**
   - Navigate to your E-learNIT project folder
   - Select the `extension` folder (contains `manifest.json`)

<img width="797" alt="4-select-extension-folder" src="https://github.com/user-attachments/assets/658a48b9-997f-4470-b422-22d153b00b9f" />

**5. Extension loaded successfully**
   - E-learNIT extension appears in your extensions list

<img width="764" alt="5-extension-loaded" src="https://github.com/user-attachments/assets/44fca104-c354-455d-a7bc-9d48fac11eab" />

**6. Pin extension to toolbar**
   - Click the puzzle icon in Chrome toolbar
   - Pin E-learNIT for easy access

<img width="1152" alt="6-pin-extension-toolbar" src="https://github.com/user-attachments/assets/8caa03c2-5688-457f-87ed-89277dabdc16" />

**7. Extension icon appears in toolbar ✅**
   - Click the E-learNIT icon to start using it

<img width="461" alt="7-extension-ready" src="https://github.com/user-attachments/assets/1c87a734-1aef-4ba8-83d2-a07b635d6439" />



---

## Usage Guide

### Quick Start

1. **Navigate** to Google Meet, Microsoft Teams, or Zoom
2. **Join or start** a meeting
3. **Open chat panel** (💬 icon)
4. **Click E-learNIT icon** in browser toolbar
5. **Click "Start Recognition"**

### Interface Overview

```
┌─────────────────────────────────────────────┐
│  E-learNIT                             [✕]  │
├─────────────────────────────────────────────┤
│  ⚫ Recognizing...                           │
│                                             │
│  Current Sign                               │
│  ┌───────────────────────────────────────┐  │
│  │         behi                          │  │
│  │         85% confidence                │  │
│  └───────────────────────────────────────┘  │
│                                             │
│  Sign Sequence (5 signs)                    │
│  naawnek → soueel → fhmet → behi → yekteb   │
│                                             │
│  [📝 Construct Phrase]  [⏸️ Pause]  [🗑️]   │
│                                             │
│  📝 Constructed Phrase                      │
│  🇹🇳 Naawnek? 3andi soueel. Fhmet. Behi.   │
│  🇫🇷 Puis-je vous aider? J'ai une          │
│      question. J'ai compris. D'accord.     │
│  🇬🇧 Can I help you? I have a question.    │
│      I understood. Okay.                   │
│  💡 Context: Student asks for help,         │
│     mentions having a question, confirms   │
│     understanding, and agrees.             │
│                                             │
│  [🔊 Speak]  [💬 Send to Chat]  [🇹🇳▼]     │
└─────────────────────────────────────────────┘
```

### Workflow Example

**Scenario**: Student needs help understanding a concept

1. **Show signs**:
   - Sign: `naawnek` (Can I help you?) → Detected ✅
   - Sign: `soueel` (Question) → Detected ✅
   - Sign: `ma fhmtch` (I don't understand) → Detected ✅

2. **Pause recognition**: Click ⏸️ button

3. **Construct phrase**: Click "📝 Construct Phrase"

4. **System generates**:
   ```
   🇹🇳 Tunisian: "Naawnek? 3andi soueel. Ma fhmtch."
   🇫🇷 French: "Puis-je vous aider? J'ai une question. Je ne comprends pas."
   🇬🇧 English: "Can I help you? I have a question. I don't understand."
   💡 Context: "The student is politely asking for assistance,
               indicating they have a question and are struggling
               to understand the current topic."
   ```

5. **Choose output**:
   - 🔊 Click "Speak Aloud" → Browser speaks French phrase
   - 💬 Click "Send to Chat" → Message appears in Google Meet chat

---

## 📚 Supported Vocabulary

### Tunisian-French-English Mapping

| Tunisian (Darija) | French | English |
|-------------------|--------|---------|
| naawnek | Puis-je vous aider? | Can I help you? |
| behi | D'accord / Bien | Okay / Good |
| ca va | Ça va? | How are you? |
| soueel | Question | Question |
| fhmet | J'ai compris | I understood |
| ma fhmtch | Je ne comprends pas | I don't understand |
| yekteb | Écrire | To write |
| yaq9ra | Lire / Étudier | To read / To study |
| madrsa | École | School |
| naaref | Je sais | I know |
| manaarefch | Je ne sais pas | I don't know |
| ena njweb | Puis-je répondre? | Can I reply? |
| enti tjwen | Tu réponds | You reply |
| note khyba | Mauvaise note | Bad grade |

---



## 🎓 Technical Challenges & Solutions

### Challenge 1: Sign Duplication

**Problem**: Same sign detected multiple times when user holds position

**Solution**: 
```javascript
// Deduplication logic with 3-second window
if (result.sign !== lastDetectedSign || 
    timeSinceLastSign > 3000) {
  signSequence.push(result.sign);
}
```

### Challenge 2: Context Loss

**Problem**: Word-by-word translation lacks meaning

**Solution**: LLM-powered phrase construction with educational context

**Example**:
```
Signs: [naawnek, soueel, fhmet]

❌ Without LLM: "help question understood"
✅ With LLM: "Can I help you? I have a question. I understood."
   + Context explaining student intent
```

### Challenge 3: Platform Compatibility

**Problem**: Different chat input types across platforms

**Solution**: Platform-specific DOM selectors

```javascript
// Google Meet: textarea
// Teams: contentEditable div (CKEditor)
// Zoom: chat-box textarea

if (platform === 'teams') {
  chatInput.innerHTML = '';
  const paragraph = document.createElement('p');
  paragraph.textContent = text;
  chatInput.appendChild(paragraph);
}
```

### Challenge 4: Dataset Scarcity

**Problem**: Limited Tunisian Sign Language data

**Solution**: Data augmentation (rotation, brightness, contrast, flip, zoom, blur)

**Result**: 6x more training data without manual collection

---

## 🔮 Future Roadmap

### Short-term (3-6 months)

- [ ] **Expand vocabulary** to 200+ signs
- [ ] **Mobile app** version (React Native)
- [ ] **Offline mode** with local LLM (Llama.cpp)
- [ ] **Chrome Web Store** publication

### Medium-term (6-12 months)

- [ ] **Multi-user recognition** (detect multiple signers)
- [ ] **Sign language learning module** (interactive tutorials)
- [ ] **Integration** with Google Classroom
- [ ] **Support** for other Arabic dialects

### Long-term (1-2 years)

- [ ] **Real-time sign animation** (reverse translation: text → sign)
- [ ] **AI avatar** demonstrating signs
- [ ] **National deployment** in Tunisian schools
- [ ] **International expansion** (Moroccan, Algerian sign languages)

---



---

## 📜 License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 E-learNIT Team from IEEE ENIT Student Branch

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## 📞 Contact & Links

- **📧 Email**: ayoub.abdi@ieee.org

---

## 📄 Citations

If you use this project in your research, please cite:



---

<div align="center">

## 🌟 Making Online Education Accessible for All

**E-learNIT** • Built with ❤️ for the Tunisian deaf community

*Herotopia Challenge 2025*


</div>
