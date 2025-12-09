# multimodal_rag.py - COMPLETE SYSTEM
import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import chromadb
from transformers import CLIPProcessor, CLIPModel
import torch
import json
import re
from datetime import datetime
from collections import deque
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


class EnhancedSignLanguageRAG:
    def __init__(self, llm_provider="groq"):
        print("Initializing Enhanced RAG with Data Augmentation and LLM...")
        
        # Load CLIP model
        print("📥 Loading CLIP model...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)

        # Setup ChromaDB with augmented data
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db_augmented")
        self.collection = self.setup_database()

        # Setup LLM
        self.llm_provider = llm_provider
        self.setup_llm()
        
        # Sign sequence buffer (for phrase construction)
        self.sign_sequence = deque(maxlen=10)  # Last 10 signs
        self.last_sign_time = datetime.now()

        print(f"✅ System ready! Database: {self.collection.count()} embeddings")

    def setup_llm(self):
        """Setup LLM based on provider choice"""
        print(f"🤖 Setting up LLM provider: {self.llm_provider}")
        
        if self.llm_provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                print("⚠️ GROQ_API_KEY not found in .env file!")
                print("Get free key at: https://console.groq.com/keys")
                self.llm_client = None
            else:
                self.llm_client = Groq(api_key=api_key)
                print("✅ Groq LLM initialized (using llama-3.3-70b)")

    def setup_database(self):
        """Load or create the database."""
        try:
            collection = self.chroma_client.get_collection("tunisian_sign_language_aug")
            print("✅ Loaded existing database")
            return collection
        except:
            print("🔄 Creating new database with augmentation...")
            return self.create_augmented_database()

    def augment_image(self, image, augmentation_type):
        """Apply data augmentation to an image."""
        img = image.copy()

        if augmentation_type == 'rotate_left':
            img = img.rotate(-15, expand=False)
        elif augmentation_type == 'rotate_right':
            img = img.rotate(15, expand=False)
        elif augmentation_type == 'brightness_up':
            img = ImageEnhance.Brightness(img).enhance(1.3)
        elif augmentation_type == 'brightness_down':
            img = ImageEnhance.Brightness(img).enhance(0.7)
        elif augmentation_type == 'contrast':
            img = ImageEnhance.Contrast(img).enhance(1.5)
        elif augmentation_type == 'flip':
            img = ImageOps.mirror(img)
        elif augmentation_type == 'zoom':
            w, h = img.size
            img = img.crop((w*0.1, h*0.1, w*0.9, h*0.9)).resize((w, h))
        elif augmentation_type == 'blur':
            img_array = np.array(img)
            img_array = cv2.GaussianBlur(img_array, (5, 5), 0)
            img = Image.fromarray(img_array)

        return img

    def extract_frames_from_video(self, video_path, num_frames=5):
        """Extract frames from video files."""
        frames = []
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Cannot open video: {video_path}")
            return frames

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            return frames

        frame_indices = np.linspace(0, total_frames-1, min(num_frames, total_frames), dtype=int)

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

        cap.release()
        print(f"✅ Extracted {len(frames)} frames from video")
        return frames

    def create_augmented_database(self):
        """Create ChromaDB collection with augmented images in batches."""
        collection = self.chroma_client.create_collection(
            name="tunisian_sign_language_aug",
            metadata={"hnsw:space": "cosine"}
        )

        dataset_path = "./dataset"
        batch_size = 1000  # Process in batches of 1000
        current_batch = []

        if not os.path.exists(dataset_path):
            print("❌ Dataset folder not found!")
            return collection

        # Reduced augmentations to avoid exceeding batch limits
        augmentations = [
            'original', 'rotate_left', 'rotate_right',
            'brightness_up', 'brightness_down', 'contrast'
        ]

        print(f"\n📊 Will create {len(augmentations)} versions of each image\n")

        def add_batch_to_db(batch_data):
            """Add a batch of embeddings to the database."""
            if not batch_data:
                return
            try:
                collection.add(
                    embeddings=[d['embedding'] for d in batch_data],
                    metadatas=[d['metadata'] for d in batch_data],
                    ids=[d['id'] for d in batch_data]
                )
                print(f"✅ Added batch of {len(batch_data)} embeddings to database")
                return True
            except Exception as e:
                print(f"⚠️ Batch error: {e}")
                return False

        for category in os.listdir(dataset_path):
            category_path = os.path.join(dataset_path, category)
            if not os.path.isdir(category_path):
                continue

            print(f"📁 Processing category: {category}")

            for item in os.listdir(category_path):
                item_path = os.path.join(category_path, item)

                if os.path.isdir(item_path):
                    sign_name = item
                    for file in os.listdir(item_path):
                        file_path = os.path.join(item_path, file)
                        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            self.process_image_with_augmentation(
                                file_path, category, sign_name, augmentations, current_batch
                            )
                        elif file.lower().endswith(('.avi', '.mp4')):
                            self.process_video(
                                file_path, category, sign_name, current_batch
                            )
                        
                        # Check if batch is full
                        if len(current_batch) >= batch_size:
                            add_batch_to_db(current_batch)
                            current_batch = []

                elif item.lower().endswith(('.jpg', '.jpeg', '.png')):
                    sign_name = os.path.splitext(item)[0]
                    self.process_image_with_augmentation(
                        item_path, category, sign_name, augmentations, current_batch
                    )
                elif item.lower().endswith(('.avi', '.mp4')):
                    sign_name = os.path.splitext(item)[0]
                    self.process_video(
                        item_path, category, sign_name, current_batch
                    )
                
                # Check if batch is full
                if len(current_batch) >= batch_size:
                    add_batch_to_db(current_batch)
                    current_batch = []

        # Add any remaining data
        if current_batch:
            add_batch_to_db(current_batch)

        print(f"\n🎉 Database created! Total embeddings: {collection.count()}")
        return collection

    def process_image_with_augmentation(self, file_path, category, sign_name, augmentations, batch_data):
        """Process an image with multiple augmentations and add to batch."""
        try:
            original_image = Image.open(file_path).convert('RGB')
            for aug_type in augmentations:
                img = original_image if aug_type == 'original' else self.augment_image(original_image, aug_type)
                embedding = self.generate_embedding(img)
                if embedding:
                    batch_data.append({
                        "id": f"{category}_{sign_name}_{os.path.basename(file_path)}_{aug_type}_{len(batch_data)}",
                        "embedding": embedding,
                        "metadata": {
                            "sign": sign_name,
                            "category": category,
                            "file_path": file_path,
                            "augmentation": aug_type,
                            "french_meaning": sign_name
                        }
                    })
            print(f"✅ {sign_name} → {len(augmentations)} versions created")
        except Exception as e:
            print(f"⚠️ Error with {file_path}: {e}")

    def process_video(self, file_path, category, sign_name, batch_data):
        """Extract frames from video and add embeddings to batch."""
        try:
            frames = self.extract_frames_from_video(file_path, num_frames=3)  # Reduced to 3 frames
            if not frames:
                return
            for i, frame in enumerate(frames):
                embedding = self.generate_embedding(frame)
                if embedding:
                    batch_data.append({
                        "id": f"{category}_{sign_name}_{os.path.basename(file_path)}_frame{i}_{len(batch_data)}",
                        "embedding": embedding,
                        "metadata": {
                            "sign": sign_name,
                            "category": category,
                            "file_path": file_path,
                            "frame_number": i,
                            "french_meaning": sign_name,
                            "source_type": "video"
                        }
                    })
            print(f"✅ {sign_name} (video) → {len(frames)} frames extracted")
        except Exception as e:
            print(f"⚠️ Error with video {file_path}: {e}")

    def generate_embedding(self, image):
        """Generate CLIP embedding for an image."""
        try:
            inputs = self.clip_processor(images=image, return_tensors="pt", padding=True)
            with torch.no_grad():
                features = self.clip_model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
            return features[0].cpu().numpy().tolist()
        except Exception as e:
            print(f"⚠️ Embedding generation error: {e}")
            return None

    def recognize_with_llm(self, image):
        """
        RAG Pipeline with LLM reasoning
        1. Get top candidates from vector DB
        2. Ask LLM to reason about which sign it is
        """
        try:
            # STEP 1: Vector similarity search
            query_embedding = self.generate_embedding(image)
            if query_embedding is None:
                return None
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=7,  # Get top 7 candidates
                include=['metadatas', 'distances']
            )
            
            if not results['metadatas'] or not results['metadatas'][0]:
                return None
            
            candidates = results['metadatas'][0]
            distances = results['distances'][0]
            
            # STEP 2: Prepare LLM prompt
            prompt = self.create_llm_prompt(candidates, distances)
            
            # STEP 3: Get LLM reasoning
            llm_response = self.query_llm(prompt)
            
            # STEP 4: Parse LLM response
            result = self.parse_llm_response(llm_response, candidates, distances)
            
            return result
            
        except Exception as e:
            print(f"Recognition error: {e}")
            return None

    def create_llm_prompt(self, candidates, distances):
        """Create prompt for LLM with Tunisian vocabulary"""
        prompt = """You are an expert in Tunisian Sign Language recognition.

TUNISIAN VOCABULARY REFERENCE (Tunisian Arabic/Darija):
- aaweni = "help me"
- behi = "ok" / "good" / "fine"
- ca va = "good" / "how are you"
- de rien = "excuse me" / "sorry" / "you're welcome"
- ena njweb = "can I reply"
- enti tjwen = "you reply"
- fhmet = "understood" / "I understand"
- ma fhmtch = "I don't understand"
- madrsa = "school"
- manaarefch = "I don't know"
- naaref = "I know"
- naawnek = "can I help you"
- note khyba = "bad mark" / "bad grade"
- soueel = "question"
- yaq9ra = "read" / "study"
- yekteb = "write"

I detected a sign gesture and found these possible matches from my database:

"""
        for i, (cand, dist) in enumerate(zip(candidates, distances), 1):
            confidence = 1 / (1 + dist)
            prompt += f"{i}. Sign: '{cand['sign']}' | Category: {cand['category']} | Similarity: {confidence:.1%}\n"
        
        prompt += """
Based on the similarity scores, context, and the Tunisian vocabulary above, which sign is most likely correct?

Respond in JSON format:
{
    "recognized_sign": "sign_name",
    "confidence": 0.95,
    "reasoning": "brief explanation why this sign was chosen"
}
"""
        return prompt

    def query_llm(self, prompt):
        """Query the LLM"""
        if self.llm_client is None:
            print("⚠️ LLM not configured, using fallback")
            return None
        
        try:
            if self.llm_provider == "groq":
                response = self.llm_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=500
                )
                return response.choices[0].message.content
        
        except Exception as e:
            print(f"LLM error: {e}")
            return None

    def parse_llm_response(self, llm_response, candidates, distances):
        """Parse LLM JSON response"""
        if llm_response is None:
            # Fallback: use highest similarity
            confidence_scores = [1 / (1 + d) for d in distances]
            best_idx = np.argmax(confidence_scores)
            
            return {
                "sign": candidates[best_idx]['sign'],
                "confidence": float(confidence_scores[best_idx]),
                "category": candidates[best_idx]['category'],
                "reasoning": "Fallback: highest similarity score",
                "all_candidates": [
                    {"sign": c['sign'], "confidence": float(conf)}
                    for c, conf in zip(candidates, confidence_scores)
                ]
            }
        
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                llm_data = json.loads(json_match.group())
                
                # Find matching candidate
                recognized = llm_data['recognized_sign']
                for cand in candidates:
                    if cand['sign'].lower() == recognized.lower():
                        return {
                            "sign": cand['sign'],
                            "confidence": llm_data.get('confidence', 0.9),
                            "category": cand['category'],
                            "reasoning": llm_data.get('reasoning', ''),
                            "all_candidates": [
                                {"sign": c['sign'], "confidence": 1/(1+d)}
                                for c, d in zip(candidates, distances)
                            ]
                        }
        except:
            pass
        
        # Final fallback
        confidence_scores = [1 / (1 + d) for d in distances]
        best_idx = np.argmax(confidence_scores)
        
        return {
            "sign": candidates[best_idx]['sign'],
            "confidence": float(confidence_scores[best_idx]),
            "category": candidates[best_idx]['category'],
            "reasoning": "LLM parsing failed, using similarity",
            "all_candidates": [
                {"sign": c['sign'], "confidence": float(conf)}
                for c, conf in zip(candidates, confidence_scores)
            ]
        }

    def construct_phrase(self):
        """Use LLM to construct meaningful phrase from sign sequence"""
        if len(self.sign_sequence) < 2 or self.llm_client is None:
            return None
        
        signs_list = list(self.sign_sequence)
        
        prompt = f"""Given this sequence of Tunisian sign language gestures:
{' → '.join([s['sign'] for s in signs_list])}

TUNISIAN ARABIC VOCABULARY:
- aaweni = help me
- behi = ok / good / fine
- ca va = good / how are you
- de rien = excuse me / sorry / you're welcome
- ena njweb = can I reply
- enti tjwen = you reply
- fhmet = understood / I understand
- ma fhmtch = I don't understand
- madrsa = school
- manaarefch = I don't know
- naaref = I know
- naawnek = can I help you
- note khyba = bad mark / bad grade
- soueel = question
- yaq9ra = read / study
- yekteb = write

Construct what the person is saying in THREE languages:
1. The original Tunisian Arabic phrase (using Tunisian words)
2. French translation
3. English translation

Respond in JSON:
{{
    "tunisian_phrase": "the complete phrase in Tunisian Arabic",
    "french_phrase": "la phrase complète en français",
    "english_phrase": "the complete phrase in English",
    "explanation": "brief context about what they're communicating"
}}
"""
        
        try:
            response = self.query_llm(prompt)
            if response:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
        except Exception as e:
            print(f"Phrase construction error: {e}")
        
        return None

    def start_realtime_camera(self):
        """Real-time camera with LLM reasoning"""
        print("\n📷 Starting Enhanced Real-Time Recognition")
        print("🎯 Show signs to camera")
        print("Commands:")
        print("  'q' - Quit")
        print("  's' - Save frame")
        print("  'p' - Construct phrase from sequence")
        print("  SPACE - Pause/Resume")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot access camera")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        active = True
        last_result = None
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # UI Header
            cv2.putText(frame, "Tunisian Sign Language - Enhanced RAG with Augmentation", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Database: {self.collection.count()} embeddings (augmented)", 
                       (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(frame, "q=quit | s=save | p=phrase | SPACE=pause", 
                       (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            if active:
                # Process every 15 frames
                if frame_count % 15 == 0:
                    try:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(frame_rgb)
                        
                        # Recognize with LLM
                        last_result = self.recognize_with_llm(pil_image)
                        
                        if last_result:
                            # Add to sequence
                            self.sign_sequence.append(last_result)
                            self.last_sign_time = datetime.now()
                    
                    except Exception as e:
                        print(f"Error: {e}")
                
                # Display results
                if last_result:
                    sign = last_result['sign']
                    conf = last_result['confidence']
                    reasoning = last_result.get('reasoning', '')
                    
                    # Color coding
                    color = (0, 255, 0) if conf > 0.7 else (0, 255, 255) if conf > 0.5 else (0, 0, 255)
                    
                    cv2.putText(frame, f"Sign: {sign}", (10, 110), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
                    cv2.putText(frame, f"Confidence: {conf:.1%}", (10, 145), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Show LLM reasoning
                    if reasoning:
                        cv2.putText(frame, f"Reasoning: {reasoning[:40]}", (10, 175), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
                    
                    # Show sequence
                    seq_text = f"Sequence: {' → '.join([s['sign'] for s in list(self.sign_sequence)[-5:]])}"
                    cv2.putText(frame, seq_text[:60], (10, 205), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
                    
                    # Alternatives
                    for i, alt in enumerate(last_result['all_candidates'][1:3]):
                        cv2.putText(frame, f"Alt: {alt['sign']} ({alt['confidence']:.1%})", 
                                   (10, 235 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
                else:
                    cv2.putText(frame, "Show sign to camera...", (10, 110), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            else:
                cv2.putText(frame, "PAUSED", (10, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            
            cv2.imshow('Enhanced Sign Language Recognition with Augmentation', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(filename, frame)
                print(f"💾 Saved: {filename}")
            elif key == ord('p'):
                print("\n🔤 Constructing phrase from sequence...")
                phrase_result = self.construct_phrase()
                if phrase_result:
                    print(f"🇹🇳 Tunisian: {phrase_result.get('tunisian_phrase', 'N/A')}")
                    print(f"🇫🇷 French: {phrase_result.get('french_phrase', 'N/A')}")
                    print(f"🇬🇧 English: {phrase_result.get('english_phrase', 'N/A')}")
                    print(f"💡 Context: {phrase_result.get('explanation', 'N/A')}\n")
                else:
                    print("⚠️ Not enough signs for phrase construction\n")
            elif key == ord(' '):
                active = not active
                print(f"🔄 Recognition {'ACTIVE' if active else 'PAUSED'}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("👋 Camera stopped")


# ====================================================================
# MAIN EXECUTION
# ====================================================================
if __name__ == "__main__":
    # Initialize system with Groq LLM
    rag = EnhancedSignLanguageRAG(llm_provider="groq")
    
    # Start camera
    rag.start_realtime_camera()