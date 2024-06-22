import argparse
import csv
import os
import openai
import base64

import requests
import tqdm

from .utils import extract_frames, prompts, read_video_list


# def get_caption(frame, prompt, api_key):
#     headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
#     payload = {
#         "model": "gpt-4-vision-preview",
#         "messages": [
#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "text",
#                         "text": prompt,
#                     },
#                     {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame[0]}"}},
#                     {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame[1]}"}},
#                     {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame[2]}"}},
#                 ],
#             }
#         ],
#         "max_tokens": 300,
#     }
#     response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=60)
#     caption = response.json()["choices"][0]["message"]["content"]
#     caption = caption.replace("\n", " ")
#     return caption

# From Yue
openai.api_type = "azure"
openai.api_base = "https://trustllm-gpt4v.openai.azure.com/"
openai.api_version = "2023-08-01-preview"
openai.api_key = '5476c2b68337408cb343e195044521a3'
max_tokens = 4096

def encode_image(img):
    # with open(image_path, "rb") as image_file:
    return base64.b64encode(img.read()).decode('utf-8')


def get_res(frame, prompt):
    img1 = (frame[0])
    img2 = (frame[1])
    img3 = (frame[2])
    chat_completion = openai.ChatCompletion.create(
        engine="TrustLLM-GPT4v",
        max_tokens=max_tokens,
        messages=[
            {
                "role": "system", 
                "content": "You need to describe the visual content and context of the image without making a medical diagnosis."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img1}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img2}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img3}"}},
                    
                ]
            }
        ],
        temperature=1,
    )
    print(chat_completion.choices[0].message.content)
    caption = chat_completion.replace("\n", " ")
    return caption



def main(args):
    # ======================================================
    # 1. read video list
    # ======================================================
    videos = read_video_list(args.video_folder, args.output_file)
    f = open(args.output_file, "a")
    writer = csv.writer(f)

    # ======================================================
    # 2. generate captions
    # ======================================================
    for video in tqdm.tqdm(videos):
        video_path = os.path.join(args.video_folder, video)
        frame, length = extract_frames(video_path, base_64=True)
        if len(frame) < 3:
            continue

        prompt = prompts[args.prompt]
        caption = get_res(frame, prompt)

        writer.writerow((video, caption, length))
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_folder", type=str)
    parser.add_argument("output_file", type=str)
    parser.add_argument("--prompt", type=str, default="three_frames")
    args = parser.parse_args()

    main(args)
