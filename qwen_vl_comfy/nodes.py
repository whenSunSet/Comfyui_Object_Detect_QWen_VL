import os
import ast
import json
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

import folder_paths


def parse_json(json_output: str) -> str:
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "```json":
            json_output = "\n".join(lines[i + 1:])
            json_output = json_output.split("```")[0]
            break
    return json_output


def parse_boxes(text: str, img_width: int, img_height: int, input_w: int, input_h: int) -> List[List[int]]:
    text = parse_json(text)
    try:
        data = ast.literal_eval(text)
    except Exception:
        end_idx = text.rfind('"}') + len('"}')
        truncated = text[:end_idx] + "]"
        data = ast.literal_eval(truncated)
    boxes = []
    for item in data:
        box = item.get("bbox_2d") or item.get("bbox") or item
        y1, x1, y2, x2 = box[1], box[0], box[3], box[2]
        abs_y1 = int(y1 / input_h * img_height)
        abs_x1 = int(x1 / input_w * img_width)
        abs_y2 = int(y2 / input_h * img_height)
        abs_x2 = int(x2 / input_w * img_width)
        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1
        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1
        boxes.append([abs_x1, abs_y1, abs_x2, abs_y2])
    return boxes


@dataclass
class QwenModel:
    model: Any
    processor: Any


class DownloadAndLoadQwenModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ([
                    "Qwen/Qwen2.5-VL-3B-Instruct",
                    "Qwen/Qwen2.5-VL-7B-Instruct",
                    "Qwen/Qwen2.5-VL-32B-Instruct",
                    "Qwen/Qwen2.5-VL-72B-Instruct",
                ], ),
            }
        }

    RETURN_TYPES = ("QWEN_MODEL",)
    RETURN_NAMES = ("qwen_model",)
    FUNCTION = "load"
    CATEGORY = "Qwen2.5-VL"

    def load(self, model_name: str):
        model_dir = os.path.join(folder_paths.models_dir, "Qwen", model_name.replace("/", "_"))
        if not os.path.exists(model_dir):
            snapshot_download(repo_id=model_name, local_dir=model_dir, local_dir_use_symlinks=False)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        processor = AutoProcessor.from_pretrained(model_dir)
        return (QwenModel(model=model, processor=processor),)


class QwenVLDetection:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "qwen_model": ("QWEN_MODEL",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "Locate the object and output bbox in JSON"}),
            },
        }

    RETURN_TYPES = ("JSON", "BBOX")
    RETURN_NAMES = ("text", "bboxes")
    FUNCTION = "detect"
    CATEGORY = "Qwen2.5-VL"

    def detect(self, qwen_model: QwenModel, image, prompt: str):
        model = qwen_model.model
        processor = qwen_model.processor
        device = next(model.parameters()).device

        if isinstance(image, torch.Tensor):
            image = (image.squeeze().clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise ValueError("Unsupported image type")

        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": [{"type": "text", "text": prompt}, {"image": image}]},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True).to(device)
        output_ids = model.generate(**inputs, max_new_tokens=1024)
        gen_ids = [output_ids[len(inp):] for inp, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = processor.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        input_h = inputs['image_grid_thw'][0][1] * 14
        input_w = inputs['image_grid_thw'][0][2] * 14
        boxes = parse_boxes(output_text, image.width, image.height, input_w, input_h)
        return (output_text, boxes)


NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadQwenModel": DownloadAndLoadQwenModel,
    "QwenVLDetection": QwenVLDetection,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadQwenModel": "Download and Load Qwen2.5-VL Model",
    "QwenVLDetection": "Qwen2.5-VL Object Detection",
}
