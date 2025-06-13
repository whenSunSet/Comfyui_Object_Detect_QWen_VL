# ComfyUI Qwen2.5-VL Object Detection Node

This repository provides a custom [ComfyUI](https://github.com/comfyanonymous/ComfyUI) node for running object detection with the [Qwen 2.5 VL](https://github.com/QwenLM/Qwen2.5-VL) model. The node downloads the selected model on demand, runs a detection prompt and outputs bounding boxes that can be used with segmentation nodes such as [SAM2](https://github.com/kijai/ComfyUI-segment-anything-2).

## Nodes

### `DownloadAndLoadQwenModel`
Downloads a chosen Qwen 2.5-VL model into `models/Qwen` and returns the loaded model and processor.

### `QwenVLDetection`
Runs a detection prompt on an input image using the loaded model. The node outputs the raw text response and the bounding box coordinates.

The bounding boxes are converted to absolute pixel coordinates so that they can be directly fed into the SAM2 nodes.

## Usage
1. Place this repository inside your `ComfyUI/custom_nodes` directory.
2. From the **Download and Load Qwen2.5-VL Model** node, select the model you want to use. The snapshot download will resume automatically if a previous attempt was interrupted.
3. Connect the output model to **Qwen2.5-VL Object Detection**, provide an image and a prompt like `Locate the cat and output its bbox coordinates in JSON format`.
4. Feed the resulting bounding boxes into the SAM2 workflow for further processing.
