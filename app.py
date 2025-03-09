import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image 
import numpy as np 
import os 
import tempfile
import spaces
import gradio as gr

import subprocess
import sys

def install_flash_attn_wheel():
    flash_attn_wheel_url = "https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
    try:
        # Call pip to install the wheel file
        subprocess.check_call([sys.executable, "-m", "pip", "install", flash_attn_wheel_url])
        print("Wheel installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install the flash attnetion wheel. Error: {e}")

install_flash_attn_wheel()

import cv2
try:
    from mmengine.visualization import Visualizer
except ImportError:
    Visualizer = None
    print("Warning: mmengine is not installed, visualization is disabled.")
    
# Load the model and tokenizer 
model_path = "ByteDance/Sa2VA-4B"
 
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="cuda:0",
    trust_remote_code=True,
).eval().cuda()

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code = True,
)

from third_parts import VideoReader
def read_video(video_path, video_interval):
    vid_frames = VideoReader(video_path)[::video_interval]
    
    temp_dir = tempfile.mkdtemp()
    os.makedirs(temp_dir, exist_ok=True)
    image_paths = []  # List to store paths of saved images
    
    for frame_idx in range(len(vid_frames)):
        frame_image = vid_frames[frame_idx]
        frame_image = frame_image[..., ::-1]  # BGR (opencv system) to RGB (numpy system)
        frame_image = Image.fromarray(frame_image)
        vid_frames[frame_idx] = frame_image

        # Save the frame as a .jpg file in the temporary folder
        image_path = os.path.join(temp_dir, f"frame_{frame_idx:04d}.jpg")
        frame_image.save(image_path, format="JPEG")

        # Append the image path to the list
        image_paths.append(image_path)
    return vid_frames, image_paths

def visualize(pred_mask, image_path, work_dir):
    visualizer = Visualizer()
    img = cv2.imread(image_path)
    visualizer.set_image(img)
    visualizer.draw_binary_masks(pred_mask, colors='g', alphas=0.4)
    visual_result = visualizer.get_image()

    output_path = os.path.join(work_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, visual_result)
    return output_path

@spaces.GPU
def image_vision(image_input_path, prompt):
    image_path = image_input_path
    text_prompts = f"<image>{prompt}"
    image = Image.open(image_path).convert('RGB')
    input_dict = {
        'image': image,
        'text': text_prompts,
        'past_text': '',
        'mask_prompts': None,
        'tokenizer': tokenizer,
    }
    return_dict = model.predict_forward(**input_dict)
    print(return_dict)
    answer = return_dict["prediction"] # the text format answer
    
    seg_image = return_dict["prediction_masks"]
    
    if '[SEG]' in answer and Visualizer is not None:
        pred_masks = seg_image[0]
        temp_dir = tempfile.mkdtemp()
        pred_mask = pred_masks
        os.makedirs(temp_dir, exist_ok=True)
        seg_result = visualize(pred_mask, image_input_path, temp_dir)
        return answer, seg_result
    else:
        return answer, None

@spaces.GPU(duration=80)
def video_vision(video_input_path, prompt, video_interval):
    # Open the original video
    cap = cv2.VideoCapture(video_input_path)

    # Get original video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)

    frame_skip_factor = video_interval
    new_fps = None
    # Calculate new FPS
    if video_interval == 1:
        new_fps = original_fps
    else:
        new_fps = original_fps / frame_skip_factor

    vid_frames, image_paths = read_video(video_input_path, video_interval)
    # create a question (<image> is a placeholder for the video frames)
    question = f"<image>{prompt}"
    result = model.predict_forward(
        video=vid_frames,
        text=question,
        tokenizer=tokenizer,
    )
    prediction = result['prediction']
    print(prediction)

    if '[SEG]' in prediction and Visualizer is not None:
        _seg_idx = 0
        pred_masks = result['prediction_masks'][_seg_idx]
        seg_frames = []
        masked_only_frames = []  # New list for masked-only frames

        for frame_idx in range(len(vid_frames)):
            pred_mask = pred_masks[frame_idx]
            temp_dir = tempfile.mkdtemp()
            os.makedirs(temp_dir, exist_ok=True)

            # Create visualized frame with segmentation overlay
            seg_frame = visualize(pred_mask, image_paths[frame_idx], temp_dir)
            seg_frames.append(seg_frame)

            # Create a binary mask image (white mask on black background)
            binary_mask = (pred_mask.astype('uint8') * 255)  # Convert mask to 0/255
            binary_mask_path = os.path.join(temp_dir, f"binary_mask_{frame_idx}.png")
            cv2.imwrite(binary_mask_path, binary_mask)
            masked_only_frames.append(binary_mask_path)

        output_video = "output_video.mp4"
        masked_video = "masked_only_video.mp4"  # New video file for masked areas only

        # Read the first image to get the size (resolution)
        frame = cv2.imread(seg_frames[0])
        height, width, layers = frame.shape

        # Define the video codec and create VideoWriter objects
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
        video = cv2.VideoWriter(output_video, fourcc, new_fps, (width, height))
        masked_video_writer = cv2.VideoWriter(masked_video, fourcc, new_fps, (width, height), isColor=False)

        # Write frames to the videos
        for idx, (seg_frame_path, mask_frame_path) in enumerate(zip(seg_frames, masked_only_frames)):
            seg_frame = cv2.imread(seg_frame_path)
            mask_frame = cv2.imread(mask_frame_path, cv2.IMREAD_GRAYSCALE)  # Read the binary mask in grayscale

            video.write(seg_frame)
            masked_video_writer.write(mask_frame)

        # Release the video writers
        video.release()
        masked_video_writer.release()

        print(f"Video created successfully at {output_video}")
        print(f"Masked-only video created successfully at {masked_video}")

        return result['prediction'], output_video, masked_video

            
    else:
        return result['prediction'], None, None
    


# Gradio UI

with gr.Blocks(analytics_enabled=False) as demo:
    with gr.Column():
        gr.Markdown("# Sa2VA: Marrying SAM2 with LLaVA for Dense Grounded Understanding of Images and Videos")
        gr.HTML("""
        <div style="display:flex;column-gap:4px;">
            <a href="https://github.com/magic-research/Sa2VA">
                <img src='https://img.shields.io/badge/GitHub-Repo-blue'>
            </a> 
            <a href="https://arxiv.org/abs/2501.04001">
                <img src='https://img.shields.io/badge/ArXiv-Paper-red'>
            </a>
            <a href="https://huggingface.co/spaces/fffiloni/Sa2VA-simple-demo?duplicate=true">
                <img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/duplicate-this-space-sm.svg" alt="Duplicate this Space">
            </a>
            <a href="https://huggingface.co/fffiloni">
                <img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/follow-me-on-HF-sm-dark.svg" alt="Follow me on HF">
            </a>
        </div>
        """)
        with gr.Tab("Single Image"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(label="Image IN", type="filepath")
                    with gr.Row():
                        instruction = gr.Textbox(label="Instruction", scale=4)
                        submit_image_btn = gr.Button("Submit", scale=1)
                with gr.Column():
                    output_res = gr.Textbox(label="Response")
                    output_image = gr.Image(label="Segmentation", type="numpy")
    
            submit_image_btn.click(
                fn = image_vision,
                inputs = [image_input, instruction],
                outputs = [output_res, output_image]
            )
        with gr.Tab("Video"):
            with gr.Row():
                with gr.Column():
                    video_input = gr.Video(label="Video IN")
                    frame_interval = gr.Slider(label="Frame interval", step=1, minimum=1, maximum=12, value=6)
                    with gr.Row():
                        vid_instruction = gr.Textbox(label="Instruction", scale=4)
                        submit_video_btn = gr.Button("Submit", scale=1)
                with gr.Column():
                    vid_output_res = gr.Textbox(label="Response")
                    output_video = gr.Video(label="Segmentation")
                    masked_output = gr.Video(label="Masked video")
            
            submit_video_btn.click(
                fn = video_vision,
                inputs = [video_input, vid_instruction, frame_interval],
                outputs = [vid_output_res, output_video, masked_output]
            )

demo.queue().launch(show_api=False, show_error=True)

