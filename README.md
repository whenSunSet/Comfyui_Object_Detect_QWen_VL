# ComfyUI Qwen2.5-VL Object Detection Node

This repository provides a custom [ComfyUI](https://github.com/comfyanonymous/ComfyUI) node for running object detection with the [Qwen 2.5 VL](https://github.com/QwenLM/Qwen2.5-VL) model. The node downloads the selected model on demand, runs a detection prompt and outputs bounding boxes that can be used with segmentation nodes such as [SAM2](https://github.com/kijai/ComfyUI-segment-anything-2).

## Nodes

### `DownloadAndLoadQwenModel`
Downloads a chosen Qwen 2.5-VL model into `models/Qwen` and returns the loaded model and processor. You can also choose which device to load the model onto (e.g. `cuda:1` if you have multiple GPUs).

### `QwenVLDetection`
Runs a detection prompt on an input image using the loaded model. The node outputs the raw text response and the bounding box coordinates. You can specify which boxes to return using the **bbox_selection** parameter and filter them with **score_threshold**:

- `all` – return all boxes (default)
- `merge` – merge all boxes into a single bounding box
- Comma-separated indices such as `0`, `1,2` or `0,2` – return only the selected boxes, sorted by detection confidence
- `score_threshold` – discard boxes whose confidence score is below this value (default `0`)
- If no boxes remain after filtering, the highest scoring box is returned so downstream nodes like SAM2 do not fail

The bounding boxes are converted to absolute pixel coordinates so that they can be directly fed into the SAM2 nodes.

## Usage
1. Place this repository inside your `ComfyUI/custom_nodes` directory.
2. From the **Download and Load Qwen2.5-VL Model** node, select the model you want to use and, if necessary, choose the device (such as `cuda:1`) where it should be loaded. The snapshot download will resume automatically if a previous attempt was interrupted.
3. Connect the output model to **Qwen2.5-VL Object Detection**, provide an image and the object you want to locate (e.g. `cat`). Optionally set **bbox_selection** to `merge` or specific indices like `0,2` if you only want certain boxes, and adjust **score_threshold** to filter out low-confidence results. The node will automatically build the detection prompt and return the selected bounding boxes.
4. Feed the resulting bounding boxes into the SAM2 workflow for further processing.
