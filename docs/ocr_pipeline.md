# PaddleOCR 3.x Module Reference

Input image / PDF page 
↓
[1] Optional Document Preprocessing 
├─ Document Orientation Classification 
│ └─ PP-LCNet_x1_0_doc_ori_infer 
│ 
└─ Document Image Unwarping 
└─ UVDoc_infer 
↓
[2] Layout Detection 
└─ PP-DocLayout_plus-L_infer 
- detect: text, formula, title, image, table... 
↓
[3] Formula Recognition
└─ PP-FormulaNet_plus-S_infer 
- get crop from bbox label=formula 
- export LaTeX/formula text 
↓
[4] Formula Masking
└─ No model needed 
- whiten bbox formula before OCR text 
↓
[5] General Text OCR
├─ Text Detection 
│ └─ PP-OCRv5_mobile_det 
│ 
├─ Optional Text Line Orientation Classification 
│ └─ PP-LCNet_x0_25_textline_ori_infer 
│ 
└─ Text Recognition 
└─ PP-OCRv5_mobile_rec 
↓
[6] Postprocessing
└─ No model needed 
- merge OCR text + LaTeX formula + image placeholders 
- sort by bbox / reading order
- export Markdown/JSON/visualization
- drop single-character noise rows after tab/space stripping, except for `\n`, `a`-`e`, `A`-`E`, and LaTeX wrapper characters
- emit formula blocks as display math so downstream markdown renderers can keep LaTeX rendering enabled without `$$`

This document summarizes the main PaddleOCR 3.x modules used in document OCR, layout analysis, formula recognition, and text recognition.

Sources:
- Document Image Orientation Classification: https://www.paddleocr.ai/main/en/version3.x/module_usage/doc_img_orientation_classification.html
- Text Image Rectification / Unwarping: https://www.paddleocr.ai/main/en/version3.x/module_usage/text_image_unwarping.html
- Layout Detection: https://www.paddleocr.ai/main/en/version3.x/module_usage/layout_detection.html
- Formula Recognition Pipeline: https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/formula_recognition.html
- Text Detection: https://www.paddleocr.ai/main/en/version3.x/module_usage/text_detection.html
- Text Line Orientation Classification: https://www.paddleocr.ai/main/en/version3.x/module_usage/textline_orientation_classification.html
- Text Recognition: https://www.paddleocr.ai/main/en/version3.x/module_usage/text_recognition.html


---

## 1. Document Image Orientation Classification

### Purpose

Detects the global orientation of a document image and helps rotate it back to the correct direction before OCR.

Typical model:
- `PP-LCNet_x1_0_doc_ori`

Place:
- `/home/viettran_orin/models/PP-LCNet_x1_0_doc_ori_infer/`

Supported classes:

- `0°`
- `90°`
- `180°`
- `270°`

### Input

Supported input types:

- `numpy.ndarray` image data
- Local image file path
- Local PDF file path
- Image/PDF URL
- Local directory containing images
- List of the above input types

### Output

Main output fields:

| Field | Meaning |
|---|---|
| `input_path` | Input image/PDF path |
| `page_index` | PDF page index, or `None` for image input |
| `class_ids` | Predicted class IDs |
| `scores` | Confidence scores |
| `label_names` | Predicted orientation labels, such as `0`, `90`, `180`, `270` |

### Configuration Parameters

| Parameter | Type | Default | Description |
|---|---:|---:|---|
| `model_name` | `str | None` | `None` | Model name. If `None`, uses `PP-LCNet_x1_0_doc_ori`. |
| `model_dir` | `str | None` | `None` | Local model directory. |
| `device` | `str | None` | `None` | Inference device, e.g. `cpu`, `gpu`, `gpu:0`, `gpu:0,1`, `npu`. |
| `engine` | `str | None` | `None` | Inference engine: `paddle`, `paddle_static`, `paddle_dynamic`, `transformers`, or `None`. |
| `engine_config` | `dict | None` | `None` | Extra inference-engine configuration. |
| `enable_hpi` | `bool` | `False` | Enables high-performance inference. |
| `use_tensorrt` | `bool` | `False` | Enables TensorRT subgraph acceleration when supported. |
| `precision` | `str` | `fp32` | TensorRT precision, usually `fp32` or `fp16`. |
| `enable_mkldnn` | `bool` | `True` | Enables MKL-DNN acceleration on CPU. |
| `mkldnn_cache_capacity` | `int` | `10` | MKL-DNN cache capacity. |
| `cpu_threads` | `int` | `10` | Number of CPU inference threads. |
| `input` | `Python Var | str | list` | required | Input data for `predict()`. |
| `batch_size` | `int` | `1` | Batch size for prediction. |

---

## 2. Text Image Rectification / Unwarping

### Purpose

Corrects document image distortion, including skew, perspective deformation, and curved/warped text regions, before OCR.

Typical model:
- `UVDoc`

Place:
- `/home/viettran_orin/models/UVDoc_infer`

### Input

Supported input types:

- `numpy.ndarray` image data
- Local image file path
- Local PDF file path
- Image/PDF URL
- Local directory containing images
- List of the above input types

### Output

Main output fields:

| Field | Meaning |
|---|---|
| `input_path` | Path of the input image |
| `page_index` | PDF page index, or `None` for image input |
| `doctr_img` | Rectified/unwarped image result |

### Configuration Parameters

| Parameter | Type | Default | Description |
|---|---:|---:|---|
| `model_name` | `str` | `None` | Model name, commonly `UVDoc`. |
| `model_dir` | `str` | `None` | Local model directory. |
| `device` | `str` | `None` | Inference device, e.g. `cpu`, `gpu`, `gpu:0`, `gpu:0,1`, `npu`. |
| `engine` | `str | None` | `None` | Inference engine: `paddle`, `paddle_static`, `paddle_dynamic`, `transformers`, or `None`. |
| `engine_config` | `dict | None` | `None` | Extra inference-engine configuration. |
| `enable_hpi` | `bool` | `False` | Enables high-performance inference. |
| `use_tensorrt` | `bool` | `False` | Enables TensorRT acceleration when supported. |
| `precision` | `str` | `fp32` | TensorRT precision, such as `fp32` or `fp16`. |
| `enable_mkldnn` | `bool` | `True` | Enables MKL-DNN acceleration on CPU. |
| `mkldnn_cache_capacity` | `int` | `10` | MKL-DNN cache capacity. |
| `cpu_threads` | `int` | `10` | Number of CPU inference threads. |
| `input` | `Python Var | str | list` | required | Input data for `predict()`. |
| `batch_size` | `int` | `1` | Batch size for prediction. |

---

## 3. Layout Detection

### Purpose

Detects and classifies layout regions in a document image, such as text, title, table, formula, image, chart, header, footer, footnote, and other document blocks.

Model:
- `PP-DocLayout_plus-L`
- `PP-DocBlockLayout`

Place:
- `/home/viettran_orin/models/PP_DocLayout-M_infer/`
- `/home/viettran_orin/models/PP-DocBlockLayout_infer/`

### Input

Supported input types:

- `numpy.ndarray` image data
- Local image file path
- Local PDF file path
- Image/PDF URL
- Local directory containing images
- List of the above input types

### Output

Main output fields:

| Field | Meaning |
|---|---|
| `input_path` | Input image/PDF path |
| `page_index` | PDF page index, or `None` for image input |
| `boxes` | List of detected layout regions |
| `boxes[].cls_id` | Class ID |
| `boxes[].label` | Class label, e.g. `text`, `table`, `formula` |
| `boxes[].score` | Confidence score |
| `boxes[].coordinate` | Bounding box in `[xmin, ymin, xmax, ymax]` format |

### Configuration Parameters

| Parameter | Type | Default | Description |
|---|---:|---:|---|
| `model_name` | `str | None` | `None` | Model name. If `None`, uses the default layout model. |
| `model_dir` | `str | None` | `None` | Local model directory. |
| `device` | `str | None` | `None` | Inference device, e.g. `cpu`, `gpu`, `gpu:0`, `gpu:0,1`, `npu`. |
| `engine` | `str | None` | `None` | Inference engine: `paddle`, `paddle_static`, `paddle_dynamic`, `transformers`, or `None`. |
| `engine_config` | `dict | None` | `None` | Extra inference-engine configuration. |
| `enable_hpi` | `bool` | `False` | Enables high-performance inference. |
| `use_tensorrt` | `bool` | `False` | Enables TensorRT acceleration when supported. |
| `precision` | `str` | `fp32` | TensorRT precision, usually `fp32` or `fp16`. |
| `enable_mkldnn` | `bool` | `True` | Enables MKL-DNN acceleration on CPU. |
| `mkldnn_cache_capacity` | `int` | `10` | MKL-DNN cache capacity. |
| `cpu_threads` | `int` | `10` | Number of CPU inference threads. |
| `input` | `Python Var | str | list` | required | Input data for `predict()`. |
| `batch_size` | `int` | `1` | Batch size for prediction. |
| `layout_nms` | `bool` | model default | Enables NMS post-processing for overlapping layout boxes. |

---

## 4. Formula Recognition Pipeline

### Purpose

Recognizes mathematical formulas from document images and outputs editable LaTeX source code. The pipeline can combine formula recognition with optional document orientation correction, image unwarping, and layout detection.

Main modules inside the pipeline:

- Formula Recognition Module
- Layout Detection Module, optional
- Document Image Orientation Classification Module, optional
- Text Image Rectification / Unwarping Module, optional

Typical formula models include:
- `PP-FormulaNet_plus-S`

Place:
- `/home/viettran_orin/models/PP-FormulaNet_plus-S_infer/`

### Input

Supported input types:

- `numpy.ndarray` image data
- Local image file path
- Local PDF file path
- Image/PDF URL
- Local directory containing images
- List of the above input types

### Output

Main output fields:

| Field | Meaning |
|---|---|
| `input_path` | Input image/PDF path |
| `page_index` | PDF page index, or `None` for image input |
| `model_settings` | Enabled/disabled pipeline modules |
| `doc_preprocessor_res` | Document orientation/unwarping result if enabled |
| `layout_det_res` | Layout detection result if enabled |
| `formula_res_list` | List of recognized formulas |
| `formula_res_list[].rec_formula` | Recognized formula in LaTeX |
| `formula_res_list[].formula_region_id` | Formula region ID |
| `formula_res_list[].dt_polys` | Formula region bounding box/polygon |

### Configuration Parameters

| Parameter | Type | Default | Description |
|---|---:|---:|---|
| `doc_orientation_classify_model_name` | `str | None` | `None` | Document orientation model name. |
| `doc_orientation_classify_model_dir` | `str | None` | `None` | Document orientation model directory. |
| `doc_orientation_classify_batch_size` | `int | None` | `None` | Batch size for document orientation model. Defaults to `1`. |
| `doc_unwarping_model_name` | `str | None` | `None` | Text image unwarping model name. |
| `doc_unwarping_model_dir` | `str | None` | `None` | Text image unwarping model directory. |
| `doc_unwarping_batch_size` | `int | None` | `None` | Batch size for unwarping model. Defaults to `1`. |
| `use_doc_orientation_classify` | `bool | None` | `None` | Whether to use document orientation classification. Pipeline default is `True`. |
| `use_doc_unwarping` | `bool | None` | `None` | Whether to use text image unwarping. Pipeline default is `True`. |
| `layout_detection_model_name` | `str | None` | `None` | Layout detection model name. |
| `layout_detection_model_dir` | `str | None` | `None` | Layout detection model directory. |
| `layout_threshold` | `float | dict | None` | `None` | Layout score threshold. Default is usually `0.5`. Can be global or per class ID. |
| `layout_nms` | `bool | None` | `None` | Whether to use NMS for layout detection. Pipeline default is `True`. |
| `layout_unclip_ratio` | `float | tuple | dict | None` | `None` | Expansion ratio for detected layout boxes. Pipeline default is `1.0`. |
| `layout_merge_bboxes_mode` | `str | dict | None` | `None` | Overlapping-box filtering mode: `large`, `small`, or `union`. |
| `layout_detection_batch_size` | `int | None` | `None` | Batch size for layout detection. Defaults to `1`. |
| `use_layout_detection` | `bool | None` | `None` | Whether to use layout detection. Pipeline default is `True`. |
| `formula_recognition_model_name` | `str | None` | `None` | Formula recognition model name. |
| `formula_recognition_model_dir` | `str | None` | `None` | Formula recognition model directory. |
| `formula_recognition_batch_size` | `int | None` | `None` | Batch size for formula recognition. Defaults to `1`. |
| `device` | `str | None` | `None` | Inference device, e.g. `cpu`, `gpu:0`, `npu:0`, `xpu:0`. |
| `engine` | `str | None` | `None` | Inference engine: `paddle`, `paddle_static`, `paddle_dynamic`, `transformers`, or `None`. |
| `engine_config` | `dict | None` | `None` | Extra inference-engine configuration. |
| `enable_hpi` | `bool | None` | `None` | Enables high-performance inference. |
| `use_tensorrt` | `bool` | `False` | Enables TensorRT acceleration when supported. |
| `precision` | `str` | `fp32` | Inference precision, e.g. `fp32` or `fp16`. |
| `enable_mkldnn` | `bool` | `True` | Enables MKL-DNN acceleration on CPU. |
| `mkldnn_cache_capacity` | `int` | `10` | MKL-DNN cache capacity. |
| `cpu_threads` | `int` | `10` | Number of CPU inference threads. |
| `paddlex_config` | `str | None` | `None` | Path to PaddleX pipeline configuration file. |

### `predict()` Parameters

| Parameter | Type | Default | Description |
|---|---:|---:|---|
| `input` | `Python Var | str | list` | required | Input image/PDF/path/URL/directory/list. |
| `use_layout_detection` | `bool | None` | `None` | Override whether to use layout detection during inference. |
| `use_doc_orientation_classify` | `bool | None` | `None` | Override whether to use document orientation classification during inference. |
| `use_doc_unwarping` | `bool | None` | `None` | Override whether to use text image unwarping during inference. |
| `layout_threshold` | `float | dict | None` | `None` | Override layout threshold. |
| `layout_nms` | `bool | None` | `None` | Override layout NMS. |
| `layout_unclip_ratio` | `float | tuple | dict | None` | `None` | Override layout box expansion ratio. |
| `layout_merge_bboxes_mode` | `str | None` | `None` | Override layout box merge/filter mode. |

---

## 5. Text Detection

### Purpose

Locates text regions in an image and outputs text bounding polygons. These regions are usually cropped and passed to a text recognition model.

Model:
- `PP-OCRv5_mobile_det`

PLace:
- `/home/viettran_orin/models/PP-OCRv5_mobile_det/`

### Input

Supported input types:

- `numpy.ndarray` image data
- Local image file path
- Local PDF file path
- Image/PDF URL
- Local directory containing images
- List of the above input types

### Output

Main output fields:

| Field | Meaning |
|---|---|
| `input_path` | Input image/PDF path |
| `page_index` | PDF page index, or `None` for image input |
| `dt_polys` | Detected text polygons, each with 4 vertices |
| `dt_scores` | Confidence scores for detected text regions |

### Configuration Parameters

| Parameter | Type | Default | Description |
|---|---:|---:|---|
| `model_name` | `str | None` | `None` | Model name. If `None`, uses `PP-OCRv5_server_det`. |
| `model_dir` | `str | None` | `None` | Local model directory. |
| `device` | `str | None` | `None` | Inference device, e.g. `cpu`, `gpu`, `gpu:0`, `gpu:0,1`, `npu`. |
| `engine` | `str | None` | `None` | Inference engine: `paddle`, `paddle_static`, `paddle_dynamic`, `transformers`, or `None`. |
| `engine_config` | `dict | None` | `None` | Extra inference-engine configuration. |
| `enable_hpi` | `bool` | `False` | Enables high-performance inference. |
| `use_tensorrt` | `bool` | `False` | Enables TensorRT acceleration when supported. |
| `precision` | `str` | `fp32` | TensorRT precision, usually `fp32` or `fp16`. |
| `enable_mkldnn` | `bool` | `True` | Enables MKL-DNN acceleration on CPU. |
| `mkldnn_cache_capacity` | `int` | `10` | MKL-DNN cache capacity. |
| `cpu_threads` | `int` | `10` | Number of CPU inference threads. |
| `limit_side_len` | `int | None` | `None` | Side-length limit for input image. Uses model default if `None`. |
| `limit_type` | `str | None` | `None` | Side-length limit mode: `min` or `max`. |
| `max_side_limit` | `int | None` | `None` | Maximum side length limit. |
| `thresh` | `float | None` | `None` | Pixel score threshold for text probability map. |
| `box_thresh` | `float | None` | `None` | Average box score threshold. |
| `unclip_ratio` | `float | None` | `None` | Expansion ratio for text regions. |
| `input_shape` | `tuple | None` | `None` | Model input shape in `(C, H, W)` format. |
| `input` | `Python Var | str | list` | required | Input data for `predict()`. |
| `batch_size` | `int` | `1` | Batch size for prediction. |

---

## 6. Text Line Orientation Classification

### Purpose

Detects whether a cropped text line is upright or upside down, then helps correct it before text recognition.

Model:
- `PP-LCNet_x0_25_textline_ori`

Place:
- `/home/viettran_orin/models/PP-LCNet_x0_25_textline_ori_infer/`

Supported classes:

- `0°`
- `180°`

### Input

Supported input types:

- `numpy.ndarray` image data
- Local image file path
- Local PDF file path
- Image/PDF URL
- Local directory containing images
- List of the above input types

In a full OCR pipeline, the input is usually a cropped text-line image from text detection.

### Output

Main output fields:

| Field | Meaning |
|---|---|
| `input_path` | Input image/PDF path |
| `page_index` | PDF page index, or `None` for image input |
| `class_ids` | Predicted class IDs |
| `scores` | Confidence scores |
| `label_names` | Predicted orientation labels, e.g. `0_degree`, `180_degree` |

### Configuration Parameters

| Parameter | Type | Default | Description |
|---|---:|---:|---|
| `model_name` | `str | None` | `None` | Model name. If `None`, uses `PP-LCNet_x0_25_textline_ori`. |
| `model_dir` | `str | None` | `None` | Local model directory. |
| `device` | `str | None` | `None` | Inference device, e.g. `cpu`, `gpu`, `gpu:0`, `gpu:0,1`, `npu`. |
| `engine` | `str | None` | `None` | Inference engine: `paddle`, `paddle_static`, `paddle_dynamic`, `transformers`, or `None`. |
| `engine_config` | `dict | None` | `None` | Extra inference-engine configuration. |
| `enable_hpi` | `bool` | `False` | Enables high-performance inference. |
| `use_tensorrt` | `bool` | `False` | Enables TensorRT acceleration when supported. |
| `precision` | `str` | `fp32` | TensorRT precision, usually `fp32` or `fp16`. |
| `enable_mkldnn` | `bool` | `True` | Enables MKL-DNN acceleration on CPU. |
| `mkldnn_cache_capacity` | `int` | `10` | MKL-DNN cache capacity. |
| `cpu_threads` | `int` | `10` | Number of CPU inference threads. |
| `input` | `Python Var | str | list` | required | Input data for `predict()`. |
| `batch_size` | `int` | `1` | Batch size for prediction. |

---

## 7. Text Recognition

### Purpose

Recognizes text content from cropped text-line images and converts it into editable/searchable text.

Model:
- `PP-OCRv5_mobile_rec`

Place:
- `/home/viettran_orin/models/PP-OCRv5_mobile_rec/`

### Input

Supported input types:

- `numpy.ndarray` image data
- Local image file path
- Local PDF file path
- Image/PDF URL
- Local directory containing images
- List of the above input types

In a full OCR pipeline, the input is usually a cropped text region generated by the text detection module.

### Output

Main output fields:

| Field | Meaning |
|---|---|
| `input_path` | Input text-line image path |
| `page_index` | PDF page index, or `None` for image input |
| `rec_text` | Recognized text |
| `rec_score` | Recognition confidence score |

### Configuration Parameters

| Parameter | Type | Default | Description |
|---|---:|---:|---|
| `model_name` | `str | None` | `None` | Model name. If `None`, uses `PP-OCRv5_server_rec`. |
| `model_dir` | `str | None` | `None` | Local model directory. |
| `device` | `str | None` | `None` | Inference device, e.g. `cpu`, `gpu`, `gpu:0`, `gpu:0,1`, `npu`. |
| `engine` | `str | None` | `None` | Inference engine: `paddle`, `paddle_static`, `paddle_dynamic`, `transformers`, or `None`. |
| `engine_config` | `dict | None` | `None` | Extra inference-engine configuration. |
| `enable_hpi` | `bool` | `False` | Enables high-performance inference. |
| `use_tensorrt` | `bool` | `False` | Enables TensorRT acceleration when supported. |
| `precision` | `str` | `fp32` | TensorRT precision, usually `fp32` or `fp16`. |
| `enable_mkldnn` | `bool` | `True` | Enables MKL-DNN acceleration on CPU. |
| `mkldnn_cache_capacity` | `int` | `10` | MKL-DNN cache capacity. |
| `cpu_threads` | `int` | `10` | Number of CPU inference threads. |
| `input` | `Python Var | str | list` | required | Input data for `predict()`. |
| `batch_size` | `int` | `1` | Batch size for prediction. |

---
