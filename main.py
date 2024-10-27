#working with audio in the end.
import subprocess
import openai
import os
import fal_client
import requests
from datetime import datetime
from moviepy.editor import ImageSequenceClip, concatenate_videoclips
import json

mp3_file_path = "output.mp3"  # Replace with the path of the mp3 file
gptkey = ""  # Replace with your OpenAI key
fal_api_key = ""  # Replace with fal API key
output_folder_path = "output_files"  # Folder for storing output

# Ensure FAL API key is set
os.environ['FAL_KEY'] = fal_api_key

# Function to convert mp3 to srt using OpenAI Whisper
def mp3_to_srt(mp3_file_path):
    print(f"Converting {mp3_file_path} to SRT...")
    client = openai.OpenAI(api_key=gptkey)
    with open(mp3_file_path, 'rb') as audio_file:
        transcript_response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="srt"
        )
    srt_file_path = mp3_file_path.rsplit('.', 1)[0] + '.srt'
    with open(srt_file_path, 'w', encoding='utf-8') as srt_file:
        srt_file.write(transcript_response)
    print(f"SRT file saved at {srt_file_path}")
    return srt_file_path

# Function to generate image prompts and timings from the LLM
def generate_image_prompts_and_timings(srt_file_path):
    print(f"Generating image prompts and timings from {srt_file_path}...")
    with open(srt_file_path, 'r', encoding='utf-8') as srt_file:
        srt_content = srt_file.read()

    # Send SRT content to GPT with a specified output format
    client = openai.OpenAI(api_key=gptkey)
    system_prompt = """
    You are a creative director. Analyze the provided transcript and generate:
    1. Image prompts for each scene (include setting, vibe, lighting).
    2. Timings for each image, using start and end times from the transcript.

    Output a JSON list with "prompt", "start_time", and "end_time" for each scene.

    Example:
    {
        "scenes": [
            {
                "prompt": "A beautiful sunset over a desert.",
                "start_time": "00:00:00,000",
                "end_time": "00:00:05,000"
            },
            {
                "prompt": "A bustling stock exchange.",
                "start_time": "00:00:05,000",
                "end_time": "00:00:10,000"
            }
        ]
    }
    """

    user_prompt = f"""Transcript: {srt_content}"""
    
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model="gpt-4o",  # Adjust the model if necessary
        response_format={"type": "json_object"},
    )

    # Access the content from the first choice correctly
    content = chat_completion.choices[0].message.content
    print(f"LLM response:\n{content}")
    return json.loads(content)  # Convert JSON string to Python dictionary

# Function to create images based on LLM prompts
def generate_and_save_images(scenes, folder_path):
    print("Generating images from LLM prompts...")
    images = []
    timings = []
    for scene in scenes:
        prompt = scene['prompt']
        print(f"Generating image for prompt: {prompt}")
        try:
            # Make request to generate an image
            handler = fal_client.submit(
                "fal-ai/flux/schnell",
                arguments={
                    "prompt": prompt,
                    "image_size": "portrait_16_9",
                    "num_inference_steps": 4,
                    "num_images": 1,
                    "enable_safety_checker": True
                }
            )
            result = handler.get()

            if 'images' in result and len(result['images']) > 0:
                image_url = result['images'][0]['url']
                image_response = requests.get(image_url)
                if image_response.status_code == 200:
                    image_filename = f"{prompt[:50].replace(' ', '_')}.jpg"
                    image_path = os.path.join(folder_path, image_filename)
                    with open(image_path, 'wb') as img_file:
                        img_file.write(image_response.content)
                    images.append(image_path)
                    timings.append((scene['start_time'], scene['end_time']))
                    print(f"Image saved at: {image_path}")
                else:
                    print(f"Failed to download image for prompt: {prompt}, Status Code: {image_response.status_code}")
            else:
                print(f"No image generated for prompt: {prompt}")
        except Exception as e:
            print(f"Error generating image for prompt '{prompt}': {e}")
    return images, timings

# Function to create a video based on the LLM timings and generated images
def create_video_with_timings(images, timings, output_video):
    print(f"Creating video with {len(images)} images...")
    clips = []
    for image_path, (start_time, end_time) in zip(images, timings):
        duration = (datetime.strptime(end_time, "%H:%M:%S,%f") - datetime.strptime(start_time, "%H:%M:%S,%f")).total_seconds()
        clip = ImageSequenceClip([image_path], durations=[duration])
        clips.append(clip)
    
    final_clip = concatenate_videoclips(clips)
    final_clip.write_videofile(output_video, fps=24)
    print(f"Video saved at: {output_video}")

# Function to add subtitles and audio to the video
# Function to add subtitles and audio to the video
def add_subtitles_and_audio(input_video_path, output_video_path, srt_path, audio_path):
    print(f"Adding subtitles and audio to video {input_video_path}...")
    ffmpeg_command = [
        "ffmpeg",
        "-i", input_video_path,  # Input video
        "-i", audio_path,  # Input audio
        "-vf", f"subtitles={srt_path}",  # Add subtitles from SRT file
        "-c:v", "libx264",  # Re-encode the video using the H.264 codec
        "-c:a", "aac",  # Re-encode audio using AAC codec
        "-b:a", "192k",  # Set audio bitrate
        "-strict", "experimental",  # Allow experimental AAC usage
        output_video_path  # Output video path
    ]
    
    subprocess.run(ffmpeg_command, shell=False)
    print(f"Final video with subtitles and audio saved at: {output_video_path}")


# Main function to manage the end-to-end process
def main():
    # Create unique directory based on date-time
    dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    full_output_folder = os.path.join(output_folder_path, dir_name)
    os.makedirs(full_output_folder, exist_ok=True)
    print(f"Output folder created: {full_output_folder}")

    # Step 1: Convert mp3 to srt
    srt_file_path = mp3_to_srt(mp3_file_path)

    # Step 2: Generate image prompts and timings from the srt content
    llm_output = generate_image_prompts_and_timings(srt_file_path)
    scenes = llm_output.get('scenes', [])

    # Step 3: Generate and save images based on the LLM prompts and timings
    generated_images, timings = generate_and_save_images(scenes, full_output_folder)

    # Step 4: Create a video from generated images based on LLM timings
    output_video_path = os.path.join(full_output_folder, "output_video.mp4")
    create_video_with_timings(generated_images, timings, output_video_path)

    # Step 5: Add subtitles and audio to the video
    final_video_path = os.path.join(full_output_folder, "final_video.mp4")
    add_subtitles_and_audio(output_video_path, final_video_path, srt_file_path, mp3_file_path)

    print(f"Video created successfully at {final_video_path}")

if __name__ == "__main__":
    main()
