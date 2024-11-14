import cv2
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Load the BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image):
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

def extract_frames_and_caption(video_path, interval=5):
    # Open video file
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval)  # Number of frames to skip
    
    captions = []
    frame_id = 0

    while True:
        success, frame = video.read()
        if not success:
            break
        # Process only at specified intervals
        if frame_id % frame_interval == 0:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            caption = generate_caption(pil_image)
            captions.append((frame_id / fps, caption))  # Save time in seconds and caption
            print(f"Time {frame_id / fps:.2f}s: {caption}")
        
        frame_id += 1

    video.release()
    return captions

# Example usage
video_path = "C:/Users/karth/OneDrive/Desktop/I age_captioning/te.mp4"  # Replace with your video file path
captions = extract_frames_and_caption(video_path, interval=5)  # Captions every 5 seconds

# Optional: Save captions as an SRT file
with open("captions.srt", "w") as f:
    for idx, (time_sec, caption) in enumerate(captions, 1):
        start_time = f"{int(time_sec // 3600):02}:{int((time_sec % 3600) // 60):02}:{int(time_sec % 60):02},000"
        end_time = f"{int((time_sec + 1) // 3600):02}:{int(((time_sec + 1) % 3600) // 60):02}:{int((time_sec + 1) % 60):02},000"
        f.write(f"{idx}\n{start_time} --> {end_time}\n{caption}\n\n")

print("Captions saved to captions.srt")
