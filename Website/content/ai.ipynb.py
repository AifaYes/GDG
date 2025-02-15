#!pip install torch numpy logfire pandas imageio pillow openai supabase langgraph pydantic-ai diffusers moviepy
import os
import torch
import numpy as np
import pandas as pd
import imageio
import cv2
import ffmpeg
from dotenv import load_dotenv
from PIL import Image  
from openai import AsyncOpenAI
from supabase import create_client, Client
from diffusers import StableDiffusionPipeline, DiffusionPipeline, DPMSolverMultistepScheduler

# Load environment variables
load_dotenv()

# Load API keys
base_url = os.getenv('BASE_URL', 'https://api.openai.com/v1')
api_key = os.getenv('LLM_API_KEY')

if not api_key:
    raise ValueError("LLM_API_KEY is missing. Please set it in the environment variables.")

openai_client = AsyncOpenAI(api_key=api_key)

supabase_url = os.getenv("SUPABASE_URL")
supabase_service_key = os.getenv("SUPABASE_SERVICE_KEY")

if not supabase_url or not supabase_service_key:
    raise ValueError("Supabase credentials are missing.")

supabase: Client = create_client(supabase_url, supabase_service_key)

# **🔵 Read Data & Generate Prompt**
def generate_prompt_from_data(file_path):
    try:
        df = pd.read_csv(file_path)  # Load data
        print("Dataset loaded successfully. Sample row:")
        sample_row = df.sample(1).to_dict(orient="records")[0]  # Select a random row
        print(sample_row)  # Log the sample row
        prompt = f"Generate an image and video inspired by the following dataset: {sample_row}"
        return prompt
    except Exception as e:
        print(f"Error reading data: {e}")
        return "Generate a generic futuristic AI design."

# **🔵 Optimized Image Generation**
def generate_image(prompt: str):
    try:
        pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
        pipe.to("cuda" if torch.cuda.is_available() else "cpu")  
        image = pipe(prompt).images[0]
        image_path = "generated_image.png"
        image.save(image_path)
        print(f"Image saved as {image_path}")
    except Exception as e:
        print(f"Error generating image: {e}")

# **🟢 Optimized Video Generation**
def generate_video(prompt: str):
    try:
        pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.to("cuda" if torch.cuda.is_available() else "cpu")

        # Generate video frames
        video_frames = pipe(prompt, num_inference_steps=25).frames
        video_frames = [(frame * 255).astype(np.uint8) for frame in video_frames]  # Scale to 0-255 and convert to uint8
        video_frames = [frame[..., :3] for frame in video_frames]  # Ensure 3 channels (RGB)

        # Save frames as a video
        video_path = "generated_video.mp4"
        imageio.mimsave(video_path, video_frames, fps=24)
        print(f"Video saved at {video_path}")

        # **Overlay text using OpenCV**
        final_video_path = "final_video.mp4"
        overlay_text_on_video(video_path, final_video_path, prompt)

        # **Validate the generated video**
        validate_video(final_video_path)

    except Exception as e:
        print(f"Error generating video: {e}")

# **Overlay Text Using OpenCV**
def overlay_text_on_video(video_path, output_path, text):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        out.write(frame)

    cap.release()
    out.release()
    print(f"Final video with text saved as {output_path}")

# **Validate Video Using FFmpeg**
def validate_video(video_path: str):
    try:
        probe = ffmpeg.probe(video_path)
        print(f"Video validation successful. Details:")
        print(f"- Duration: {probe['format']['duration']} seconds")
        print(f"- Resolution: {probe['streams'][0]['width']}x{probe['streams'][0]['height']}")
        print(f"- Codec: {probe['streams'][0]['codec_name']}")
    except Exception as e:
        print(f"Error validating video: {e}")

# **🔹 Example Usage**
if __name__ == "__main__":
    data_file = "tiktok_trends_cleaned_.csv"  # Change to your actual data file
    prompt = generate_prompt_from_data(data_file)
    
    generate_image(prompt)
    generate_video(prompt)
