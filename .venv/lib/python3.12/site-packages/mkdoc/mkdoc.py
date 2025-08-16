import os
import sys
import argparse
from pathlib import Path
import time
from functools import wraps
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

RATE_LIMIT = int(os.environ.get("RATE_LIMIT", 3))

client = OpenAI(
    api_key=os.environ.get("MOONSHOT_API_KEY"),
    base_url="https://api.moonshot.cn/v1",
)

def rate_limit(limit_per_minute):
    min_interval = 60.0 / limit_per_minute
    last_called = [0.0]

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret
        return wrapper
    return decorator

def upload_image(image_path):
    file_object = client.files.create(file=Path(image_path), purpose="file-extract")
    return client.files.content(file_id=file_object.id).text

@rate_limit(limit_per_minute=RATE_LIMIT)
def generate_content(image_contents, base_path, directory_name, additional_prompt=""):
    combined_content = "\n\n".join([f"Image content: {content}" for content in image_contents])
    
    messages = [
        {
            "role": "system",
            "content": "帮我分析一系列屏幕截图，生成给用户的使用手册",
        },
        {
            "role": "system",
            "content": combined_content,
        },
        {
            "role": "user",
            "content": f"图片嵌入时的基础路径为：{base_path}",
        },
        {
            "role": "user",
            "content": f"总标题参考 '{directory_name}'，生成适当的标题。",
        },
        {
            "role": "user",
            "content": "帮我分析这些屏幕截图，生成给用户的markdown使用手册。请不要输出手册之外的任何描述内容，请将图片嵌入到markdown中。",
        },
    ]

    if additional_prompt:
        messages.append({
            "role": "user",
            "content": additional_prompt,
        })

    completion = client.chat.completions.create(
        model="moonshot-v1-32k",
        messages=messages,
        temperature=0.3,
    )

    return completion.choices[0].message.content

def generate_markdown(directory):
    markdown_path = f"{directory}.md"
    directory_name = os.path.basename(directory)
    
    if os.path.exists(markdown_path):
        markdown_mtime = os.path.getmtime(markdown_path)
    else:
        markdown_mtime = 0

    need_update = False
    image_files = []

    for root, _, files in os.walk(directory):
        for file in sorted(files):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                image_path = os.path.join(root, file)
                image_files.append((file, image_path))
                if os.path.getmtime(image_path) > markdown_mtime:
                    need_update = True

    if need_update:
        print(f"Updating: {markdown_path}")
        image_contents = []
        for file, image_path in image_files:
            print(f"Uploading: {image_path}")
            image_content = upload_image(image_path)
            image_contents.append(image_content)

        print("Generating content...")
        base_path = os.path.relpath(directory, os.path.dirname(markdown_path))
        print(f"Base path: {base_path}")

        # 读取 prompt.txt 文件（如果存在）
        prompt_file = os.path.join(directory, "prompt.txt")
        additional_prompt = ""
        if os.path.exists(prompt_file):
            with open(prompt_file, 'r', encoding='utf-8') as f:
                additional_prompt = f.read().strip()

        content = generate_content(image_contents, base_path, directory_name, additional_prompt)

        with open(markdown_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Generated/Updated: {markdown_path}")
    else:
        print(f"No updates needed for: {markdown_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate Markdown documentation from image directories.")
    parser.add_argument("path", help="Path to the directory containing image subdirectories")
    args = parser.parse_args()

    if not os.path.isdir(args.path):
        print(f"Error: {args.path} is not a valid directory")
        sys.exit(1)

    for item in os.listdir(args.path):
        item_path = os.path.join(args.path, item)
        if os.path.isdir(item_path):
            generate_markdown(item_path)

if __name__ == "__main__":
    main()