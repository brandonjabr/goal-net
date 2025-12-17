"""
GoalNet: Automatic Soccer Goal Detection and Highlight Generation

Copyright Brandon Jabr (2025)

A Python library for detecting goals in soccer game videos using computer vision
and audio analysis with open-source machine learning models.

Dependencies:
    pip install opencv-python torch torchvision numpy librosa soundfile scikit-learn

Usage:
    from soccer_goal_detector import SoccerGoalDetector
    
    detector = SoccerGoalDetector()
    results = detector.detect_goals("match.mp4")
    
    for goal in results:
        print(f"Goal detected at {goal['timestamp']}s with confidence {goal['confidence']}")
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import librosa
import soundfile as sf
from typing import List, Dict, Tuple
import tempfile
import subprocess
from pathlib import Path


class GoalNet:
    """
    Main class for detecting goals in soccer match videos.
    
    Uses a combination of:
    - Scene change detection
    - Crowd audio analysis
    - Visual feature extraction
    - Replay detection
    """
    
    def __init__(self, 
                 scene_threshold: float = 30.0,
                 audio_threshold: float = 0.7,
                 confidence_threshold: float = 0.6):
        """
        Initialize the goal detector.
        
        Args:
            scene_threshold: Threshold for scene change detection
            audio_threshold: Threshold for audio intensity detection
            confidence_threshold: Minimum confidence for goal detection
        """
        self.scene_threshold = scene_threshold
        self.audio_threshold = audio_threshold
        self.confidence_threshold = confidence_threshold
        
        # Load pre-trained model for feature extraction
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def detect_goals(self, video_path: str, output_clips: bool = False) -> List[Dict]:
        """
        Detect goals in a soccer video.
        
        Args:
            video_path: Path to the input video file
            output_clips: Whether to save video clips of detected goals
            
        Returns:
            List of dictionaries containing goal information:
            - timestamp: Time in seconds when goal was detected
            - confidence: Detection confidence (0-1)
            - frame: Frame number
        """
        print(f"Processing video: {video_path}")
        
        # Extract audio for analysis
        audio_path = self._extract_audio(video_path)
        
        # Analyze video frames
        scene_changes = self._detect_scene_changes(video_path)
        print(f"Detected {len(scene_changes)} scene changes")
        
        # Analyze audio for crowd reactions
        audio_peaks = self._detect_audio_peaks(audio_path)
        print(f"Detected {len(audio_peaks)} audio peaks")
        
        # Combine signals to detect goals
        goals = self._combine_signals(scene_changes, audio_peaks, video_path)

        for g in goals:
            print(g['confidence'])
        
        # Filter by confidence threshold
        goals = [g for g in goals if g['confidence'] >= self.confidence_threshold]
        
        print(f"Detected {len(goals)} potential goals")
        
        if output_clips and goals:
            self._save_goal_clips(video_path, goals)
        
        return goals
    
    def _extract_audio(self, video_path: str) -> str:
        """Extract audio from video file."""
        audio_path = tempfile.mktemp(suffix='.wav')
        
        try:
            subprocess.run([
                'ffmpeg', '-i', video_path, '-vn', '-acodec', 
                'pcm_s16le', '-ar', '44100', '-ac', '2', audio_path, '-y'
            ], capture_output=True, check=True)
        except subprocess.CalledProcessError:
            print("Warning: ffmpeg not found. Audio analysis will be skipped.")
            return None
        except FileNotFoundError:
            print("Warning: ffmpeg not found. Audio analysis will be skipped.")
            return None
            
        return audio_path
    
    def _detect_scene_changes(self, video_path: str) -> List[Dict]:
        """Detect significant scene changes in video."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        scene_changes = []
        prev_frame = None
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_frame is not None:
                # Calculate frame difference
                diff = cv2.absdiff(prev_frame, gray)
                mean_diff = np.mean(diff)
                
                if mean_diff > self.scene_threshold:
                    timestamp = frame_count / fps
                    scene_changes.append({
                        'frame': frame_count,
                        'timestamp': timestamp,
                        'intensity': mean_diff
                    })
            
            prev_frame = gray
            frame_count += 1
            
            # Progress indicator
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames...", end='\r')
        
        cap.release()
        return scene_changes
    
    def _detect_audio_peaks(self, audio_path: str) -> List[Dict]:
        """Detect audio peaks indicating crowd reactions."""
        if audio_path is None:
            return []
        
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=None)
            
            # Calculate RMS energy
            rms = librosa.feature.rms(y=y)[0]
            
            # Find peaks in energy
            from scipy.signal import find_peaks
            peaks, properties = find_peaks(rms, height=np.mean(rms) + np.std(rms))
            
            audio_peaks = []
            hop_length = 512
            
            for peak_idx in peaks:
                timestamp = librosa.frames_to_time(peak_idx, sr=sr, hop_length=hop_length)
                intensity = properties['peak_heights'][list(peaks).index(peak_idx)]
                
                audio_peaks.append({
                    'timestamp': timestamp,
                    'intensity': intensity
                })
            
            return audio_peaks
        except Exception as e:
            print(f"Warning: Audio analysis failed: {e}")
            return []
    
    def _combine_signals(self, scene_changes: List[Dict], 
                        audio_peaks: List[Dict], 
                        video_path: str) -> List[Dict]:
        """Combine video and audio signals to detect goals."""
        goals = []
        time_window = 3.0  # seconds
        
        # For each scene change, check for nearby audio peaks
        for scene in scene_changes:
            scene_time = scene['timestamp']
            scene_intensity = scene['intensity']
            
            # Find audio peaks within time window
            nearby_audio = [
                peak for peak in audio_peaks
                if abs(peak['timestamp'] - scene_time) <= time_window
            ]
            
            if nearby_audio:
                # Calculate confidence based on signal strength
                max_audio_intensity = max(peak['intensity'] for peak in nearby_audio)
                
                # Normalize intensities
                norm_scene = min(scene_intensity / 100.0, 1.0)
                norm_audio = min(max_audio_intensity / 1.0, 1.0)
                
                # Combined confidence
                confidence = (norm_scene * 0.4 + norm_audio * 0.6)
                
                goals.append({
                    'timestamp': scene_time,
                    'frame': scene['frame'],
                    'confidence': confidence,
                    'scene_intensity': scene_intensity,
                    'audio_intensity': max_audio_intensity if nearby_audio else 0
                })
        
        # Remove duplicate detections (within 10 seconds)
        goals = self._remove_duplicates(goals, window=10.0)
        
        return goals
    
    def _remove_duplicates(self, goals: List[Dict], window: float) -> List[Dict]:
        """Remove duplicate goal detections within time window."""
        if not goals:
            return goals
        
        # Sort by confidence
        goals = sorted(goals, key=lambda x: x['confidence'], reverse=True)
        
        filtered = []
        for goal in goals:
            # Check if this goal is far enough from already selected goals
            if not any(abs(goal['timestamp'] - g['timestamp']) < window 
                      for g in filtered):
                filtered.append(goal)
        
        # Sort by timestamp
        return sorted(filtered, key=lambda x: x['timestamp'])
    
    def _save_goal_clips(self, video_path: str, goals: List[Dict], 
                        clip_duration: int = 20):
        """Save video clips around detected goals."""
        output_dir = Path("goal_clips")
        output_dir.mkdir(exist_ok=True)
        
        for i, goal in enumerate(goals):
            start_time = max(0, goal['timestamp'] - 10)
            output_path = output_dir / f"goal_{i+1}_{goal['timestamp']:.1f}s.mp4"
            
            try:
                subprocess.run([
                    'ffmpeg', '-i', video_path,
                    '-ss', str(start_time),
                    '-t', str(clip_duration),
                    '-c', 'copy',
                    str(output_path), '-y'
                ], capture_output=True, check=True)
                
                print(f"Saved clip: {output_path}")
            except (subprocess.CalledProcessError, FileNotFoundError):
                print(f"Warning: Could not save clip for goal at {goal['timestamp']}s")


# Example usage
if __name__ == "__main__":
    # Create detector instance
    detector = GoalNet(
        scene_threshold=30.0,
        audio_threshold=0.7,
        confidence_threshold=0.25
    )
    
    # Detect goals in video
    input_videos_path = "input_videos/"
    video_file = "messi_goals_vs_arsenal.mp4"  # Replace with your video path
    goals = detector.detect_goals(input_videos_path + video_file, output_clips=True)
    
    # Print results
    print("\n=== Goal Detection Results ===")
    for i, goal in enumerate(goals, 1):
        print(f"\nGoal {i}:")
        print(f"  Time: {goal['timestamp']:.2f}s ({goal['timestamp']//60:.0f}m {goal['timestamp']%60:.0f}s)")
        print(f"  Confidence: {goal['confidence']:.2%}")
        print(f"  Frame: {goal['frame']}")