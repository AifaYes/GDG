import os
import torch
import numpy as np
import logfire
import pandas as pd
import imageio
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv
from PIL import Image  
from openai import AsyncOpenAI
from supabase import create_client, Client
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.config import get_stream_writer
from langgraph.types import interrupt
import requests
from moviepy.editor import VideoFileClip

# Import Pydantic AI components
try:
    from pydantic_ai.models.openai import OpenAIModel
    from pydantic_ai import Agent, RunContext
    from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter
except ImportError:
    raise ImportError("pydantic_ai is missing. Install it using 'pip install pydantic-ai'.")

# Load environment variables
load_dotenv()

# Configure logfire to suppress warnings (optional)
logfire.configure(send_to_logfire='never')

# Load API keys
base_url = os.getenv('BASE_URL', 'https://api.openai.com/v1')
api_key = os.getenv('LLM_API_KEY')

if not api_key:
    raise ValueError("LLM_API_KEY is missing. Please set it in the environment variables.")

openai_client = AsyncOpenAI(api_key=api_key)

supabase_url = os.getenv("SUPABASE_URL")
supabase_service_key = os.getenv("SUPABASE_SERVICE_KEY")

if not supabase_url or not supabase_service_key:
    raise ValueError("Supabase credentials are missing. Please set SUPABASE_URL and SUPABASE_SERVICE_KEY.")

supabase: Client = create_client(supabase_url, supabase_service_key)

# Initialize Agents
reasoner = Agent(
    OpenAIModel("gpt-3.5-turbo", base_url=base_url, api_key=api_key),
    system_prompt="You are an AI expert in designing AI agents with Pydantic AI."
)

router_agent = Agent(
    OpenAIModel("gpt-4", base_url=base_url, api_key=api_key),
    system_prompt="Your job is to determine if the user wants to continue coding the AI agent or finish the conversation."
)

end_conversation_agent = Agent(
    OpenAIModel("gpt-4", base_url=base_url, api_key=api_key),
    system_prompt="Your job is to end the conversation and provide final execution instructions."
)

# Define AgentState
class AgentState(TypedDict):
    latest_user_message: str
    messages: Annotated[List[bytes], lambda x, y: x + y]
    scope: str

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
    from diffusers import StableDiffusionPipeline

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
    from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
    from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip

    try:
        # Load the text-to-video pipeline
        pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.to("cuda" if torch.cuda.is_available() else "cpu")

        # Generate video frames
        video_frames = pipe(prompt, num_inference_steps=25).frames

        # Convert frames to uint8 and ensure they have 3 channels (RGB)
        video_frames = [(frame * 255).astype(np.uint8) for frame in video_frames]  # Scale to 0-255 and convert to uint8
        video_frames = [frame[..., :3] for frame in video_frames]  # Ensure 3 channels (RGB)

        # Save frames as a video using imageio
        video_path = "generated_video.mp4"
        imageio.mimsave(video_path, video_frames, fps=24)
        print(f"Video saved at {video_path}")

        # Overlay Text on Video
        video = VideoFileClip(video_path)
        text_clip = TextClip(prompt, fontsize=50, color="white", size=video.size)
        text_clip = text_clip.set_position("center").set_duration(video.duration)
        final_video = CompositeVideoClip([video, text_clip])

        final_video.write_videofile("final_video.mp4", fps=24)
        print(f"Final video saved at final_video.mp4")

        # **Validate the generated video**
        validate_video("final_video.mp4")

    except Exception as e:
        print(f"Error generating video: {e}")

# **Validate Video**
def validate_video(video_path: str):
    try:
        # Load the video using moviepy
        video = VideoFileClip(video_path)
        print(f"Video validation successful. Details:")
        print(f"- Duration: {video.duration} seconds")
        print(f"- Resolution: {video.size}")
        print(f"- FPS: {video.fps}")
    except Exception as e:
        print(f"Error validating video: {e}")

# **Build Workflow**
builder = StateGraph(AgentState)
builder.add_node("define_scope_with_reasoner", define_scope_with_reasoner)
builder.add_node("coder_agent", coder_agent)
builder.add_node("get_next_user_message", get_next_user_message)
builder.add_node("finish_conversation", finish_conversation)

builder.add_edge(START, "define_scope_with_reasoner")
builder.add_edge("define_scope_with_reasoner", "coder_agent")
builder.add_edge("coder_agent", "get_next_user_message")
builder.add_conditional_edges("get_next_user_message", route_user_message, {"coder_agent": "coder_agent", "finish_conversation": "finish_conversation"})
builder.add_edge("finish_conversation", END)

# **Configure Memory Persistence**
memory = MemorySaver()
agentic_flow = builder.compile(checkpointer=memory)

# **🔹 Example Usage**
if __name__ == "__main__":
    data_file = "data.csv"  # Change to your actual data file
    prompt = generate_prompt_from_data(data_file)
    
    generate_image(prompt)
    generate_video(prompt)  