# dataset_collector.py
import cv2
import os
from datetime import datetime

class DatasetCollector:
    def __init__(self, output_dir="./dataset"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.cap = cv2.VideoCapture(0)
        
    def record_sign(self, word, num_samples=5):
        """
        Record multiple samples of one sign
        
        Args:
            word: The word (e.g., "teacher", "book")
            num_samples: Number of video samples to record
        """
        word_dir = os.path.join(self.output_dir, word)
        os.makedirs(word_dir, exist_ok=True)
        
        print(f"\n{'='*50}")
        print(f"Recording sign for: {word}")
        print(f"We'll record {num_samples} samples")
        print(f"{'='*50}\n")
        
        for sample_num in range(1, num_samples + 1):
            print(f"\nSample {sample_num}/{num_samples}")
            print("Press SPACE when ready to record (3 seconds)")
            print("Press 'q' to skip this sample")
            
            # Wait for user to be ready
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                    
                # Show live feed
                cv2.putText(frame, f"Sample {sample_num}/{num_samples} - Press SPACE to record", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Word: {word}", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.imshow('Dataset Collector', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    break
                elif key == ord('q'):
                    return
            
            # Countdown
            for i in range(3, 0, -1):
                ret, frame = self.cap.read()
                cv2.putText(frame, str(i), (250, 250), 
                           cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 10)
                cv2.imshow('Dataset Collector', frame)
                cv2.waitKey(1000)
            
            # Record video
            print("RECORDING...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = os.path.join(word_dir, f"{word}_{sample_num}_{timestamp}.avi")
            
            # Video writer
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(video_path, fourcc, 20.0, (640, 480))
            
            # Record for 3 seconds
            frames_recorded = 0
            while frames_recorded < 60:  # 3 seconds at 20 fps
                ret, frame = self.cap.read()
                if ret:
                    out.write(frame)
                    cv2.putText(frame, "RECORDING...", (200, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    cv2.imshow('Dataset Collector', frame)
                    frames_recorded += 1
                    cv2.waitKey(50)
            
            out.release()
            print(f"✓ Saved: {video_path}")
            
        print(f"\n✓ Completed recording all samples for: {word}\n")
    
    def extract_key_frames(self, video_path, num_frames=5):
        """
        Extract key frames from a video
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame intervals
        frame_indices = [int(i * total_frames / (num_frames + 1)) 
                        for i in range(1, num_frames + 1)]
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        
        cap.release()
        return frames
    
    def process_all_videos(self):
        """
        Process all recorded videos and extract frames
        """
        processed_dir = os.path.join(self.output_dir, "processed")
        os.makedirs(processed_dir, exist_ok=True)
        
        # Go through each word folder
        for word in os.listdir(self.output_dir):
            word_path = os.path.join(self.output_dir, word)
            if not os.path.isdir(word_path) or word == "processed":
                continue
            
            print(f"\nProcessing videos for: {word}")
            word_processed_dir = os.path.join(processed_dir, word)
            os.makedirs(word_processed_dir, exist_ok=True)
            
            # Process each video
            video_files = [f for f in os.listdir(word_path) if f.endswith('.avi')]
            for video_file in video_files:
                video_path = os.path.join(word_path, video_file)
                frames = self.extract_key_frames(video_path, num_frames=3)
                
                # Save frames
                base_name = os.path.splitext(video_file)[0]
                for i, frame in enumerate(frames):
                    frame_path = os.path.join(word_processed_dir, f"{base_name}_frame{i}.jpg")
                    cv2.imwrite(frame_path, frame)
                    print(f"  ✓ Saved frame: {frame_path}")
    
    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()


# Usage script
if __name__ == "__main__":
    collector = DatasetCollector(output_dir="./tunisian_sign_dataset")
    
    # List of education words to collect
    education_words = [
        "entendant",      # معلم
        "sourd",      # تلميذ
        "behi",         # كتاب
        "cv va",          # قلم
        "Merci",     # كراس
        "de rien",       # مدرسة
        "s'ill vous plait",        # يكتب
        "fhemt",         # يقرأ
        "ma fhemtch",        # يتعلم
        "naaref", 
        "mana3rafch",
        "Note khyba",
        "aaweni",
        "Naawnek",  
        "Madrsa",
        "examen",
        "ya9ra",
        "ykteb",                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 # سؤال
        "ena njewb",
        "enti tjeweb",   
    ]
    

    
    print("="*60)
    print("TUNISIAN SIGN LANGUAGE DATASET COLLECTOR")
    print("="*60)
    print("\nInstructions:")
    print("1. For each word, we'll record 5 samples")
    print("2. Perform the sign clearly in front of the camera")
    print("3. Try different angles and speeds")
    print("4. Make sure lighting is good")
    print("\n")
    
    for word in education_words:
        record = input(f"\nReady to record '{word}'? (y/n): ")
        if record.lower() == 'y':
            collector.record_sign(word, num_samples=5)
        else:
            print(f"Skipped: {word}")
    
    # Extract frames from all videos
    print("\n" + "="*60)
    print("EXTRACTING KEY FRAMES FROM VIDEOS...")
    print("="*60)
    collector.process_all_videos()
    
    collector.release()
    print("\n✓ DATASET COLLECTION COMPLETE!")
    print(f"Videos saved in: ./tunisian_sign_dataset/")
    print(f"Frames saved in: ./tunisian_sign_dataset/processed/")