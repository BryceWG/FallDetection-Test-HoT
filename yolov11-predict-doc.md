Title: Predict

URL Source: https://docs.ultralytics.com/modes/predict

Markdown Content:
[](https://github.com/ultralytics/ultralytics/tree/main/docs/en/modes/predict.md "Edit this page")

Model Prediction with Ultralytics YOLO
--------------------------------------

![Image 1: Ultralytics YOLO ecosystem and integrations](https://github.com/ultralytics/docs/releases/download/0/ultralytics-yolov8-ecosystem-integrations.avif)

Introduction
------------

In the world of [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) and [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv), the process of making sense out of visual data is called 'inference' or 'prediction'. Ultralytics YOLO11 offers a powerful feature known as **predict mode** that is tailored for high-performance, real-time inference on a wide range of data sources.

**Watch:** How to Extract the Outputs from Ultralytics YOLO Model for Custom Projects.

Real-world Applications
-----------------------

| Manufacturing | Sports | Safety |
| --- | --- | --- |
| ![Image 2: Vehicle Spare Parts Detection](https://github.com/RizwanMunawar/ultralytics/assets/62513924/a0f802a8-0776-44cf-8f17-93974a4a28a1) | ![Image 3: Football Player Detection](https://github.com/RizwanMunawar/ultralytics/assets/62513924/7d320e1f-fc57-4d7f-a691-78ee579c3442) | ![Image 4: People Fall Detection](https://github.com/RizwanMunawar/ultralytics/assets/62513924/86437c4a-3227-4eee-90ef-9efb697bdb43) |
| Vehicle Spare Parts Detection | Football Player Detection | People Fall Detection |

Why Use Ultralytics YOLO for Inference?
---------------------------------------

Here's why you should consider YOLO11's predict mode for your various inference needs:

*   **Versatility:** Capable of making inferences on images, videos, and even live streams.
*   **Performance:** Engineered for real-time, high-speed processing without sacrificing [accuracy](https://www.ultralytics.com/glossary/accuracy).
*   **Ease of Use:** Intuitive Python and CLI interfaces for rapid deployment and testing.
*   **Highly Customizable:** Various settings and parameters to tune the model's inference behavior according to your specific requirements.

### Key Features of Predict Mode

YOLO11's predict mode is designed to be robust and versatile, featuring:

*   **Multiple Data Source Compatibility:** Whether your data is in the form of individual images, a collection of images, video files, or real-time video streams, predict mode has you covered.
*   **Streaming Mode:** Use the streaming feature to generate a memory-efficient generator of `Results` objects. Enable this by setting `stream=True` in the predictor's call method.
*   **Batch Processing:** The ability to process multiple images or video frames in a single batch, further speeding up inference time.
*   **Integration Friendly:** Easily integrate with existing data pipelines and other software components, thanks to its flexible API.

Ultralytics YOLO models return either a Python list of `Results` objects, or a memory-efficient Python generator of `Results` objects when `stream=True` is passed to the model during inference:

Predict

Return a list with `stream=False`Return a generator with `stream=True`



fromultralyticsimport YOLO

# Load a model
model = YOLO(“yolo11n.pt”) # pretrained YOLO11n model

# Run batched inference on a list of images
results = model([“image1.jpg”, “image2.jpg”]) # return a list of Results objects

# Process results list
for result in results:
 boxes = result.boxes # Boxes object for bounding box outputs
 masks = result.masks # Masks object for segmentation masks outputs
 keypoints = result.keypoints # Keypoints object for pose outputs
 probs = result.probs # Probs object for classification outputs
 obb = result.obb # Oriented boxes object for OBB outputs
 result.show() # display to screen
 result.save(filename=“result.jpg”) # save to disk




fromultralyticsimport YOLO

# Load a model
model = YOLO(“yolo11n.pt”) # pretrained YOLO11n model

# Run batched inference on a list of images
results = model([“image1.jpg”, “image2.jpg”], stream=True) # return a generator of Results objects

# Process results generator
for result in results:
 boxes = result.boxes # Boxes object for bounding box outputs
 masks = result.masks # Masks object for segmentation masks outputs
 keypoints = result.keypoints # Keypoints object for pose outputs
 probs = result.probs # Probs object for classification outputs
 obb = result.obb # Oriented boxes object for OBB outputs
 result.show() # display to screen
 result.save(filename=“result.jpg”) # save to disk


Inference Sources
-----------------

YOLO11 can process different types of input sources for inference, as shown in the table below. The sources include static images, video streams, and various data formats. The table also indicates whether each source can be used in streaming mode with the argument `stream=True` ✅. Streaming mode is beneficial for processing videos or live streams as it creates a generator of results instead of loading all frames into memory.

Tip

Use `stream=True` for processing long videos or large datasets to efficiently manage memory. When `stream=False`, the results for all frames or data points are stored in memory, which can quickly add up and cause out-of-memory errors for large inputs. In contrast, `stream=True` utilizes a generator, which only keeps the results of the current frame or data point in memory, significantly reducing memory consumption and preventing out-of-memory issues.

| Source | Example | Type | Notes |
| --- | --- | --- | --- |
| image | `'image.jpg'` | `str` or `Path` | Single image file. |
| URL | `'https://ultralytics.com/images/bus.jpg'` | `str` | URL to an image. |
| screenshot | `'screen'` | `str` | Capture a screenshot. |
| PIL | `Image.open('image.jpg')` | `PIL.Image` | HWC format with RGB channels. |
| [OpenCV](https://www.ultralytics.com/glossary/opencv) | `cv2.imread('image.jpg')` | `np.ndarray` | HWC format with BGR channels `uint8 (0-255)`. |
| numpy | `np.zeros((640,1280,3))` | `np.ndarray` | HWC format with BGR channels `uint8 (0-255)`. |
| torch | `torch.zeros(16,3,320,640)` | `torch.Tensor` | BCHW format with RGB channels `float32 (0.0-1.0)`. |
| CSV | `'sources.csv'` | `str` or `Path` | CSV file containing paths to images, videos, or directories. |
| video ✅ | `'video.mp4'` | `str` or `Path` | Video file in formats like MP4, AVI, etc. |
| directory ✅ | `'path/'` | `str` or `Path` | Path to a directory containing images or videos. |
| glob ✅ | `'path/*.jpg'` | `str` | Glob pattern to match multiple files. Use the `*` character as a wildcard. |
| YouTube ✅ | `'https://youtu.be/LNwODJXcvt4'` | `str` | URL to a YouTube video. |
| stream ✅ | `'rtsp://example.com/media.mp4'` | `str` | URL for streaming protocols such as RTSP, RTMP, TCP, or an IP address. |
| multi-stream ✅ | `'list.streams'` | `str` or `Path` | `*.streams` text file with one stream URL per row, i.e. 8 streams will run at batch-size 8. |
| webcam ✅ | `0` | `int` | Index of the connected camera device to run inference on. |

Below are code examples for using each source type:

Prediction sources

imagescreenshotURLPILOpenCVnumpytorchCSVvideodirectoryglobYouTubeStreamMulti-StreamWebcam

Run inference on an image file.



fromultralyticsimport YOLO

# Load a pretrained YOLO11n model
model = YOLO(“yolo11n.pt”)

# Define path to the image file
source = “path/to/image.jpg”

# Run inference on the source
results = model(source) # list of Results objects


Run inference on the current screen content as a screenshot.



fromultralyticsimport YOLO

# Load a pretrained YOLO11n model
model = YOLO(“yolo11n.pt”)

# Define current screenshot as source
source = “screen”

# Run inference on the source
results = model(source) # list of Results objects


Run inference on an image or video hosted remotely via URL.



fromultralyticsimport YOLO

# Load a pretrained YOLO11n model
model = YOLO(“yolo11n.pt”)

# Define remote image or video URL
source = “https://ultralytics.com/images/bus.jpg”

# Run inference on the source
results = model(source) # list of Results objects


Run inference on an image opened with Python Imaging Library (PIL).



fromPILimport Image

fromultralyticsimport YOLO

# Load a pretrained YOLO11n model
model = YOLO(“yolo11n.pt”)

# Open an image using PIL
source = Image.open(“path/to/image.jpg”)

# Run inference on the source
results = model(source) # list of Results objects


Run inference on an image read with OpenCV.



importcv2

fromultralyticsimport YOLO

# Load a pretrained YOLO11n model
model = YOLO(“yolo11n.pt”)

# Read an image using OpenCV
source = cv2.imread(“path/to/image.jpg”)

# Run inference on the source
results = model(source) # list of Results objects


Run inference on an image represented as a numpy array.



importnumpyasnp

fromultralyticsimport YOLO

# Load a pretrained YOLO11n model
model = YOLO(“yolo11n.pt”)

# Create a random numpy array of HWC shape (640, 640, 3) with values in range [0, 255] and type uint8
source = np.random.randint(low=0, high=255, size=(640, 640, 3), dtype=“uint8”)

# Run inference on the source
results = model(source) # list of Results objects


Run inference on an image represented as a [PyTorch](https://www.ultralytics.com/glossary/pytorch) tensor.



importtorch

fromultralyticsimport YOLO

# Load a pretrained YOLO11n model
model = YOLO(“yolo11n.pt”)

# Create a random torch tensor of BCHW shape (1, 3, 640, 640) with values in range [0, 1] and type float32
source = torch.rand(1, 3, 640, 640, dtype=torch.float32)

# Run inference on the source
results = model(source) # list of Results objects


Run inference on a collection of images, URLs, videos and directories listed in a CSV file.



fromultralyticsimport YOLO

# Load a pretrained YOLO11n model
model = YOLO(“yolo11n.pt”)

# Define a path to a CSV file with images, URLs, videos and directories
source = “path/to/file.csv”

# Run inference on the source
results = model(source) # list of Results objects


Run inference on a video file. By using `stream=True`, you can create a generator of Results objects to reduce memory usage.



fromultralyticsimport YOLO

# Load a pretrained YOLO11n model
model = YOLO(“yolo11n.pt”)

# Define path to video file
source = “path/to/video.mp4”

# Run inference on the source
results = model(source, stream=True) # generator of Results objects


Run inference on all images and videos in a directory. To also capture images and videos in subdirectories use a glob pattern, i.e. `path/to/dir/**/*`.



fromultralyticsimport YOLO

# Load a pretrained YOLO11n model
model = YOLO(“yolo11n.pt”)

# Define path to directory containing images and videos for inference
source = “path/to/dir”

# Run inference on the source
results = model(source, stream=True) # generator of Results objects


Run inference on all images and videos that match a glob expression with `*` characters.



fromultralyticsimport YOLO

# Load a pretrained YOLO11n model
model = YOLO(“yolo11n.pt”)

# Define a glob search for all JPG files in a directory
source = “path/to/dir/.jpg"

# OR define a recursive glob search for all JPG files including subdirectories
source = "path/to/dir/**/.jpg”

# Run inference on the source
results = model(source, stream=True) # generator of Results objects


Run inference on a YouTube video. By using `stream=True`, you can create a generator of Results objects to reduce memory usage for long videos.



fromultralyticsimport YOLO

# Load a pretrained YOLO11n model
model = YOLO(“yolo11n.pt”)

# Define source as YouTube video URL
source = “https://youtu.be/LNwODJXcvt4”

# Run inference on the source
results = model(source, stream=True) # generator of Results objects


Use the stream mode to run inference on live video streams using RTSP, RTMP, TCP, or IP address protocols. If a single stream is provided, the model runs inference with a [batch size](https://www.ultralytics.com/glossary/batch-size) of 1. For multiple streams, a `.streams` text file can be used to perform batched inference, where the batch size is determined by the number of streams provided (e.g., batch-size 8 for 8 streams).



fromultralyticsimport YOLO

# Load a pretrained YOLO11n model
model = YOLO(“yolo11n.pt”)

# Single stream with batch-size 1 inference
source = “rtsp://example.com/media.mp4” # RTSP, RTMP, TCP, or IP streaming address

# Run inference on the source
results = model(source, stream=True) # generator of Results objects


For single stream usage, the batch size is set to 1 by default, allowing efficient real-time processing of the video feed.

To handle multiple video streams simultaneously, use a `.streams` text file containing the streaming sources. The model will run batched inference where the batch size equals the number of streams. This setup enables efficient processing of multiple feeds concurrently.



fromultralyticsimport YOLO

# Load a pretrained YOLO11n model
model = YOLO(“yolo11n.pt”)

# Multiple streams with batched inference (e.g., batch-size 8 for 8 streams)
source = “path/to/list.streams” # *.streams text file with one streaming address per line

# Run inference on the source
results = model(source, stream=True) # generator of Results objects


Example `.streams` text file:



rtsp://example.com/media1.mp4
rtsp://example.com/media2.mp4
rtmp://example2.com/live
tcp://192.168.1.100:554
…


Each row in the file represents a streaming source, allowing you to monitor and perform inference on several video streams at once.

You can run inference on a connected camera device by passing the index of that particular camera to `source`.



fromultralyticsimport YOLO

# Load a pretrained YOLO11n model
model = YOLO(“yolo11n.pt”)

# Run inference on the source
results = model(source=0, stream=True) # generator of Results objects


Inference Arguments
-------------------

`model.predict()` accepts multiple arguments that can be passed at inference time to override defaults:

Example



fromultralyticsimport YOLO

# Load a pretrained YOLO11n model
model = YOLO(“yolo11n.pt”)

# Run inference on ‘bus.jpg’ with arguments
model.predict(“https://ultralytics.com/images/bus.jpg”, save=True, imgsz=320, conf=0.5)


Inference arguments:

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `source` | `str` | `'ultralytics/assets'` | Specifies the data source for inference. Can be an image path, video file, directory, URL, or device ID for live feeds. Supports a wide range of formats and sources, enabling flexible application across [different types of input](https://docs.ultralytics.com/modes/predict/#inference-sources). |
| `conf` | `float` | `0.25` | Sets the minimum confidence threshold for detections. Objects detected with confidence below this threshold will be disregarded. Adjusting this value can help reduce false positives. |
| `iou` | `float` | `0.7` | [Intersection Over Union](https://www.ultralytics.com/glossary/intersection-over-union-iou) (IoU) threshold for Non-Maximum Suppression (NMS). Lower values result in fewer detections by eliminating overlapping boxes, useful for reducing duplicates. |
| `imgsz` | `int` or `tuple` | `640` | Defines the image size for inference. Can be a single integer `640` for square resizing or a (height, width) tuple. Proper sizing can improve detection [accuracy](https://www.ultralytics.com/glossary/accuracy) and processing speed. |
| `half` | `bool` | `False` | Enables half-[precision](https://www.ultralytics.com/glossary/precision) (FP16) inference, which can speed up model inference on supported GPUs with minimal impact on accuracy. |
| `device` | `str` | `None` | Specifies the device for inference (e.g., `cpu`, `cuda:0` or `0`). Allows users to select between CPU, a specific GPU, or other compute devices for model execution. |
| `batch` | `int` | `1` | Specifies the batch size for inference (only works when the source is [a directory, video file or `.txt` file](https://docs.ultralytics.com/modes/predict/#inference-sources)). A larger batch size can provide higher throughput, shortening the total amount of time required for inference. |
| `max_det` | `int` | `300` | Maximum number of detections allowed per image. Limits the total number of objects the model can detect in a single inference, preventing excessive outputs in dense scenes. |
| `vid_stride` | `int` | `1` | Frame stride for video inputs. Allows skipping frames in videos to speed up processing at the cost of temporal resolution. A value of 1 processes every frame, higher values skip frames. |
| `stream_buffer` | `bool` | `False` | Determines whether to queue incoming frames for video streams. If `False`, old frames get dropped to accommodate new frames (optimized for real-time applications). If \`True', queues new frames in a buffer, ensuring no frames get skipped, but will cause latency if inference FPS is lower than stream FPS. |
| `visualize` | `bool` | `False` | Activates visualization of model features during inference, providing insights into what the model is "seeing". Useful for debugging and model interpretation. |
| `augment` | `bool` | `False` | Enables test-time augmentation (TTA) for predictions, potentially improving detection robustness at the cost of inference speed. |
| `agnostic_nms` | `bool` | `False` | Enables class-agnostic Non-Maximum Suppression (NMS), which merges overlapping boxes of different classes. Useful in multi-class detection scenarios where class overlap is common. |
| `classes` | `list[int]` | `None` | Filters predictions to a set of class IDs. Only detections belonging to the specified classes will be returned. Useful for focusing on relevant objects in multi-class detection tasks. |
| `retina_masks` | `bool` | `False` | Returns high-resolution segmentation masks. The returned masks (`masks.data`) will match the original image size if enabled. If disabled, they have the image size used during inference. |
| `embed` | `list[int]` | `None` | Specifies the layers from which to extract feature vectors or [embeddings](https://www.ultralytics.com/glossary/embeddings). Useful for downstream tasks like clustering or similarity search. |
| `project` | `str` | `None` | Name of the project directory where prediction outputs are saved if `save` is enabled. |
| `name` | `str` | `None` | Name of the prediction run. Used for creating a subdirectory within the project folder, where prediction outputs are stored if `save` is enabled. |

Visualization arguments:

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `show` | `bool` | `False` | If `True`, displays the annotated images or videos in a window. Useful for immediate visual feedback during development or testing. |
| `save` | `bool` | `False` or `True` | Enables saving of the annotated images or videos to file. Useful for documentation, further analysis, or sharing results. Defaults to True when using CLI & False when used in Python. |
| `save_frames` | `bool` | `False` | When processing videos, saves individual frames as images. Useful for extracting specific frames or for detailed frame-by-frame analysis. |
| `save_txt` | `bool` | `False` | Saves detection results in a text file, following the format `[class] [x_center] [y_center] [width] [height] [confidence]`. Useful for integration with other analysis tools. |
| `save_conf` | `bool` | `False` | Includes confidence scores in the saved text files. Enhances the detail available for post-processing and analysis. |
| `save_crop` | `bool` | `False` | Saves cropped images of detections. Useful for dataset augmentation, analysis, or creating focused datasets for specific objects. |
| `show_labels` | `bool` | `True` | Displays labels for each detection in the visual output. Provides immediate understanding of detected objects. |
| `show_conf` | `bool` | `True` | Displays the confidence score for each detection alongside the label. Gives insight into the model's certainty for each detection. |
| `show_boxes` | `bool` | `True` | Draws bounding boxes around detected objects. Essential for visual identification and location of objects in images or video frames. |
| `line_width` | `None` or `int` | `None` | Specifies the line width of bounding boxes. If `None`, the line width is automatically adjusted based on the image size. Provides visual customization for clarity. |

Image and Video Formats
-----------------------

YOLO11 supports various image and video formats, as specified in [ultralytics/data/utils.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/utils.py). See the tables below for the valid suffixes and example predict commands.

### Images

The below table contains valid Ultralytics image formats.

Note

HEIC images are supported for inference only, not for training.

| Image Suffixes | Example Predict Command | Reference |
| --- | --- | --- |
| `.bmp` | `yolo predict source=image.bmp` | [Microsoft BMP File Format](https://en.wikipedia.org/wiki/BMP_file_format) |
| `.dng` | `yolo predict source=image.dng` | [Adobe DNG](https://en.wikipedia.org/wiki/Digital_Negative) |
| `.jpeg` | `yolo predict source=image.jpeg` | [JPEG](https://en.wikipedia.org/wiki/JPEG) |
| `.jpg` | `yolo predict source=image.jpg` | [JPEG](https://en.wikipedia.org/wiki/JPEG) |
| `.mpo` | `yolo predict source=image.mpo` | [Multi Picture Object](https://fileinfo.com/extension/mpo) |
| `.png` | `yolo predict source=image.png` | [Portable Network Graphics](https://en.wikipedia.org/wiki/PNG) |
| `.tif` | `yolo predict source=image.tif` | [Tag Image File Format](https://en.wikipedia.org/wiki/TIFF) |
| `.tiff` | `yolo predict source=image.tiff` | [Tag Image File Format](https://en.wikipedia.org/wiki/TIFF) |
| `.webp` | `yolo predict source=image.webp` | [WebP](https://en.wikipedia.org/wiki/WebP) |
| `.pfm` | `yolo predict source=image.pfm` | [Portable FloatMap](https://en.wikipedia.org/wiki/Netpbm#File_formats) |
| `.HEIC` | `yolo predict source=image.HEIC` | [High Efficiency Image Format](https://en.wikipedia.org/wiki/HEIF) |

### Videos

The below table contains valid Ultralytics video formats.

| Video Suffixes | Example Predict Command | Reference |
| --- | --- | --- |
| `.asf` | `yolo predict source=video.asf` | [Advanced Systems Format](https://en.wikipedia.org/wiki/Advanced_Systems_Format) |
| `.avi` | `yolo predict source=video.avi` | [Audio Video Interleave](https://en.wikipedia.org/wiki/Audio_Video_Interleave) |
| `.gif` | `yolo predict source=video.gif` | [Graphics Interchange Format](https://en.wikipedia.org/wiki/GIF) |
| `.m4v` | `yolo predict source=video.m4v` | [MPEG-4 Part 14](https://en.wikipedia.org/wiki/M4V) |
| `.mkv` | `yolo predict source=video.mkv` | [Matroska](https://en.wikipedia.org/wiki/Matroska) |
| `.mov` | `yolo predict source=video.mov` | [QuickTime File Format](https://en.wikipedia.org/wiki/QuickTime_File_Format) |
| `.mp4` | `yolo predict source=video.mp4` | [MPEG-4 Part 14 - Wikipedia](https://en.wikipedia.org/wiki/MPEG-4_Part_14) |
| `.mpeg` | `yolo predict source=video.mpeg` | [MPEG-1 Part 2](https://en.wikipedia.org/wiki/MPEG-1) |
| `.mpg` | `yolo predict source=video.mpg` | [MPEG-1 Part 2](https://en.wikipedia.org/wiki/MPEG-1) |
| `.ts` | `yolo predict source=video.ts` | [MPEG Transport Stream](https://en.wikipedia.org/wiki/MPEG_transport_stream) |
| `.wmv` | `yolo predict source=video.wmv` | [Windows Media Video](https://en.wikipedia.org/wiki/Windows_Media_Video) |
| `.webm` | `yolo predict source=video.webm` | [WebM Project](https://en.wikipedia.org/wiki/WebM) |

Working with Results
--------------------

All Ultralytics `predict()` calls will return a list of `Results` objects:

Results



fromultralyticsimport YOLO

# Load a pretrained YOLO11n model
model = YOLO(“yolo11n.pt”)

# Run inference on an image
results = model(“https://ultralytics.com/images/bus.jpg”)
results = model(
 [
 “https://ultralytics.com/images/bus.jpg”,
 “https://ultralytics.com/images/zidane.jpg”,
 ]
) # batch inference


`Results` objects have the following attributes:

| Attribute | Type | Description |
| --- | --- | --- |
| `orig_img` | `numpy.ndarray` | The original image as a numpy array. |
| `orig_shape` | `tuple` | The original image shape in (height, width) format. |
| `boxes` | `Boxes, optional` | A Boxes object containing the detection bounding boxes. |
| `masks` | `Masks, optional` | A Masks object containing the detection masks. |
| `probs` | `Probs, optional` | A Probs object containing probabilities of each class for classification task. |
| `keypoints` | `Keypoints, optional` | A Keypoints object containing detected keypoints for each object. |
| `obb` | `OBB, optional` | An OBB object containing oriented bounding boxes. |
| `speed` | `dict` | A dictionary of preprocess, inference, and postprocess speeds in milliseconds per image. |
| `names` | `dict` | A dictionary mapping class indices to class names. |
| `path` | `str` | The path to the image file. |
| `save_dir` | `str, optional` | Directory to save results. |

`Results` objects have the following methods:

| Method | Return Type | Description |
| --- | --- | --- |
| `update()` | `None` | Updates the Results object with new detection data (boxes, masks, probs, obb, keypoints). |
| `cpu()` | `Results` | Returns a copy of the Results object with all tensors moved to CPU memory. |
| `numpy()` | `Results` | Returns a copy of the Results object with all tensors converted to numpy arrays. |
| `cuda()` | `Results` | Returns a copy of the Results object with all tensors moved to GPU memory. |
| `to()` | `Results` | Returns a copy of the Results object with tensors moved to specified device and dtype. |
| `new()` | `Results` | Creates a new Results object with the same image, path, names, and speed attributes. |
| `plot()` | `np.ndarray` | Plots detection results on an input RGB image and returns the annotated image. |
| `show()` | `None` | Displays the image with annotated inference results. |
| `save()` | `str` | Saves annotated inference results image to file and returns the filename. |
| `verbose()` | `str` | Returns a log string for each task, detailing detection and classification outcomes. |
| `save_txt()` | `str` | Saves detection results to a text file and returns the path to the saved file. |
| `save_crop()` | `None` | Saves cropped detection images to specified directory. |
| `summary()` | `List[Dict]` | Converts inference results to a summarized dictionary with optional normalization. |
| `to_df()` | `DataFrame` | Converts detection results to a Pandas DataFrame. |
| `to_csv()` | `str` | Converts detection results to CSV format. |
| `to_xml()` | `str` | Converts detection results to XML format. |
| `to_html()` | `str` | Converts detection results to HTML format. |
| `to_json()` | `str` | Converts detection results to JSON format. |
| `to_sql()` | `None` | Converts detection results to SQL-compatible format and saves to database. |

For more details see the [`Results` class documentation](https://docs.ultralytics.com/reference/engine/results/).

### Boxes

`Boxes` object can be used to index, manipulate, and convert bounding boxes to different formats.

Boxes



fromultralyticsimport YOLO

# Load a pretrained YOLO11n model
model = YOLO(“yolo11n.pt”)

# Run inference on an image
results = model(“https://ultralytics.com/images/bus.jpg”) # results list

# View results
for r in results:
 print(r.boxes) # print the Boxes object containing the detection bounding boxes


Here is a table for the `Boxes` class methods and properties, including their name, type, and description:

| Name | Type | Description |
| --- | --- | --- |
| `cpu()` | Method | Move the object to CPU memory. |
| `numpy()` | Method | Convert the object to a numpy array. |
| `cuda()` | Method | Move the object to CUDA memory. |
| `to()` | Method | Move the object to the specified device. |
| `xyxy` | Property (`torch.Tensor`) | Return the boxes in xyxy format. |
| `conf` | Property (`torch.Tensor`) | Return the confidence values of the boxes. |
| `cls` | Property (`torch.Tensor`) | Return the class values of the boxes. |
| `id` | Property (`torch.Tensor`) | Return the track IDs of the boxes (if available). |
| `xywh` | Property (`torch.Tensor`) | Return the boxes in xywh format. |
| `xyxyn` | Property (`torch.Tensor`) | Return the boxes in xyxy format normalized by original image size. |
| `xywhn` | Property (`torch.Tensor`) | Return the boxes in xywh format normalized by original image size. |

For more details see the [`Boxes` class documentation](https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Boxes).

### Masks

`Masks` object can be used index, manipulate and convert masks to segments.

Masks



fromultralyticsimport YOLO

# Load a pretrained YOLO11n-seg Segment model
model = YOLO(“yolo11n-seg.pt”)

# Run inference on an image
results = model(“https://ultralytics.com/images/bus.jpg”) # results list

# View results
for r in results:
 print(r.masks) # print the Masks object containing the detected instance masks


Here is a table for the `Masks` class methods and properties, including their name, type, and description:

| Name | Type | Description |
| --- | --- | --- |
| `cpu()` | Method | Returns the masks tensor on CPU memory. |
| `numpy()` | Method | Returns the masks tensor as a numpy array. |
| `cuda()` | Method | Returns the masks tensor on GPU memory. |
| `to()` | Method | Returns the masks tensor with the specified device and dtype. |
| `xyn` | Property (`torch.Tensor`) | A list of normalized segments represented as tensors. |
| `xy` | Property (`torch.Tensor`) | A list of segments in pixel coordinates represented as tensors. |

For more details see the [`Masks` class documentation](https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Masks).

### Keypoints

`Keypoints` object can be used index, manipulate and normalize coordinates.

Keypoints



fromultralyticsimport YOLO

# Load a pretrained YOLO11n-pose Pose model
model = YOLO(“yolo11n-pose.pt”)

# Run inference on an image
results = model(“https://ultralytics.com/images/bus.jpg”) # results list

# View results
for r in results:
 print(r.keypoints) # print the Keypoints object containing the detected keypoints


Here is a table for the `Keypoints` class methods and properties, including their name, type, and description:

| Name | Type | Description |
| --- | --- | --- |
| `cpu()` | Method | Returns the keypoints tensor on CPU memory. |
| `numpy()` | Method | Returns the keypoints tensor as a numpy array. |
| `cuda()` | Method | Returns the keypoints tensor on GPU memory. |
| `to()` | Method | Returns the keypoints tensor with the specified device and dtype. |
| `xyn` | Property (`torch.Tensor`) | A list of normalized keypoints represented as tensors. |
| `xy` | Property (`torch.Tensor`) | A list of keypoints in pixel coordinates represented as tensors. |
| `conf` | Property (`torch.Tensor`) | Returns confidence values of keypoints if available, else None. |

For more details see the [`Keypoints` class documentation](https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Keypoints).

### Probs

`Probs` object can be used index, get `top1` and `top5` indices and scores of classification.

Probs



fromultralyticsimport YOLO

# Load a pretrained YOLO11n-cls Classify model
model = YOLO(“yolo11n-cls.pt”)

# Run inference on an image
results = model(“https://ultralytics.com/images/bus.jpg”) # results list

# View results
for r in results:
 print(r.probs) # print the Probs object containing the detected class probabilities


Here's a table summarizing the methods and properties for the `Probs` class:

| Name | Type | Description |
| --- | --- | --- |
| `cpu()` | Method | Returns a copy of the probs tensor on CPU memory. |
| `numpy()` | Method | Returns a copy of the probs tensor as a numpy array. |
| `cuda()` | Method | Returns a copy of the probs tensor on GPU memory. |
| `to()` | Method | Returns a copy of the probs tensor with the specified device and dtype. |
| `top1` | Property (`int`) | Index of the top 1 class. |
| `top5` | Property (`list[int]`) | Indices of the top 5 classes. |
| `top1conf` | Property (`torch.Tensor`) | Confidence of the top 1 class. |
| `top5conf` | Property (`torch.Tensor`) | Confidences of the top 5 classes. |

For more details see the [`Probs` class documentation](https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Probs).

### OBB

`OBB` object can be used to index, manipulate, and convert oriented bounding boxes to different formats.

OBB



fromultralyticsimport YOLO

# Load a pretrained YOLO11n model
model = YOLO(“yolo11n-obb.pt”)

# Run inference on an image
results = model(“https://ultralytics.com/images/boats.jpg”) # results list

# View results
for r in results:
 print(r.obb) # print the OBB object containing the oriented detection bounding boxes


Here is a table for the `OBB` class methods and properties, including their name, type, and description:

| Name | Type | Description |
| --- | --- | --- |
| `cpu()` | Method | Move the object to CPU memory. |
| `numpy()` | Method | Convert the object to a numpy array. |
| `cuda()` | Method | Move the object to CUDA memory. |
| `to()` | Method | Move the object to the specified device. |
| `conf` | Property (`torch.Tensor`) | Return the confidence values of the boxes. |
| `cls` | Property (`torch.Tensor`) | Return the class values of the boxes. |
| `id` | Property (`torch.Tensor`) | Return the track IDs of the boxes (if available). |
| `xyxy` | Property (`torch.Tensor`) | Return the horizontal boxes in xyxy format. |
| `xywhr` | Property (`torch.Tensor`) | Return the rotated boxes in xywhr format. |
| `xyxyxyxy` | Property (`torch.Tensor`) | Return the rotated boxes in xyxyxyxy format. |
| `xyxyxyxyn` | Property (`torch.Tensor`) | Return the rotated boxes in xyxyxyxy format normalized by image size. |

For more details see the [`OBB` class documentation](https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.OBB).

Plotting Results
----------------

The `plot()` method in `Results` objects facilitates visualization of predictions by overlaying detected objects (such as bounding boxes, masks, keypoints, and probabilities) onto the original image. This method returns the annotated image as a NumPy array, allowing for easy display or saving.

Plotting



fromPILimport Image

fromultralyticsimport YOLO

# Load a pretrained YOLO11n model
model = YOLO(“yolo11n.pt”)

# Run inference on ‘bus.jpg’
results = model([“https://ultralytics.com/images/bus.jpg”, “https://ultralytics.com/images/zidane.jpg”]) # results list

# Visualize the results
for i, r in enumerate(results):
 # Plot results image
 im_bgr = r.plot() # BGR-order numpy array
 im_rgb = Image.fromarray(im_bgr[…, ::-1]) # RGB-order PIL image

 # Show results to screen (in supported environments)
 r.show()

 # Save results to disk
 r.save(filename=f"results{i}.jpg")


### `plot()` Method Parameters

The `plot()` method supports various arguments to customize the output:

| Argument | Type | Description | Default |
| --- | --- | --- | --- |
| `conf` | `bool` | Include detection confidence scores. | `True` |
| `line_width` | `float` | Line width of bounding boxes. Scales with image size if `None`. | `None` |
| `font_size` | `float` | Text font size. Scales with image size if `None`. | `None` |
| `font` | `str` | Font name for text annotations. | `'Arial.ttf'` |
| `pil` | `bool` | Return image as a PIL Image object. | `False` |
| `img` | `numpy.ndarray` | Alternative image for plotting. Uses the original image if `None`. | `None` |
| `im_gpu` | `torch.Tensor` | GPU-accelerated image for faster mask plotting. Shape: (1, 3, 640, 640). | `None` |
| `kpt_radius` | `int` | Radius for drawn keypoints. | `5` |
| `kpt_line` | `bool` | Connect keypoints with lines. | `True` |
| `labels` | `bool` | Include class labels in annotations. | `True` |
| `boxes` | `bool` | Overlay bounding boxes on the image. | `True` |
| `masks` | `bool` | Overlay masks on the image. | `True` |
| `probs` | `bool` | Include classification probabilities. | `True` |
| `show` | `bool` | Display the annotated image directly using the default image viewer. | `False` |
| `save` | `bool` | Save the annotated image to a file specified by `filename`. | `False` |
| `filename` | `str` | Path and name of the file to save the annotated image if `save` is `True`. | `None` |
| `color_mode` | `str` | Specify the color mode, e.g., 'instance' or 'class'. | `'class'` |

Thread-Safe Inference
---------------------

Ensuring thread safety during inference is crucial when you are running multiple YOLO models in parallel across different threads. Thread-safe inference guarantees that each thread's predictions are isolated and do not interfere with one another, avoiding race conditions and ensuring consistent and reliable outputs.

When using YOLO models in a multi-threaded application, it's important to instantiate separate model objects for each thread or employ thread-local storage to prevent conflicts:

Thread-Safe Inference

Instantiate a single model inside each thread for thread-safe inference:



fromthreadingimport Thread

fromultralyticsimport YOLO


defthread_safe_predict(model, image_path):
“”“Performs thread-safe prediction on an image using a locally instantiated YOLO model.”“”
 model = YOLO(model)
 results = model.predict(image_path)
 # Process results


# Starting threads that each have their own model instance
Thread(target=thread_safe_predict, args=(“yolo11n.pt”, “image1.jpg”)).start()
Thread(target=thread_safe_predict, args=(“yolo11n.pt”, “image2.jpg”)).start()


For an in-depth look at thread-safe inference with YOLO models and step-by-step instructions, please refer to our [YOLO Thread-Safe Inference Guide](https://docs.ultralytics.com/guides/yolo-thread-safe-inference/). This guide will provide you with all the necessary information to avoid common pitfalls and ensure that your multi-threaded inference runs smoothly.

Streaming Source `for`\-loop
----------------------------

Here's a Python script using OpenCV (`cv2`) and YOLO to run inference on video frames. This script assumes you have already installed the necessary packages (`opencv-python` and `ultralytics`).

Streaming for-loop



importcv2

fromultralyticsimport YOLO

# Load the YOLO model
model = YOLO(“yolo11n.pt”)

# Open the video file
video_path = “path/to/your/video/file.mp4”
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
 # Read a frame from the video
 success, frame = cap.read()

 if success:
 # Run YOLO inference on the frame
 results = model(frame)

 # Visualize the results on the frame
 annotated_frame = results[0].plot()

 # Display the annotated frame
 cv2.imshow(“YOLO Inference”, annotated_frame)

 # Break the loop if ‘q’ is pressed
 if cv2.waitKey(1) & 0xFF == ord(“q”):
 break
 else:
 # Break the loop if the end of the video is reached
 break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()


This script will run predictions on each frame of the video, visualize the results, and display them in a window. The loop can be exited by pressing 'q'.

FAQ
---

### What is Ultralytics YOLO and its predict mode for real-time inference?

Ultralytics YOLO is a state-of-the-art model for real-time [object detection](https://www.ultralytics.com/glossary/object-detection), segmentation, and classification. Its **predict mode** allows users to perform high-speed inference on various data sources such as images, videos, and live streams. Designed for performance and versatility, it also offers batch processing and streaming modes. For more details on its features, check out the [Ultralytics YOLO predict mode](https://docs.ultralytics.com/modes/predict/#key-features-of-predict-mode).

### How can I run inference using Ultralytics YOLO on different data sources?

Ultralytics YOLO can process a wide range of data sources, including individual images, videos, directories, URLs, and streams. You can specify the data source in the `model.predict()` call. For example, use `'image.jpg'` for a local image or `'https://ultralytics.com/images/bus.jpg'` for a URL. Check out the detailed examples for various [inference sources](https://docs.ultralytics.com/modes/predict/#inference-sources) in the documentation.

### How do I optimize YOLO inference speed and memory usage?

To optimize inference speed and manage memory efficiently, you can use the streaming mode by setting `stream=True` in the predictor's call method. The streaming mode generates a memory-efficient generator of `Results` objects instead of loading all frames into memory. For processing long videos or large datasets, streaming mode is particularly useful. Learn more about [streaming mode](https://docs.ultralytics.com/modes/predict/#key-features-of-predict-mode).

### What inference arguments does Ultralytics YOLO support?

The `model.predict()` method in YOLO supports various arguments such as `conf`, `iou`, `imgsz`, `device`, and more. These arguments allow you to customize the inference process, setting parameters like confidence thresholds, image size, and the device used for computation. Detailed descriptions of these arguments can be found in the [inference arguments](https://docs.ultralytics.com/modes/predict/#inference-arguments) section.

### How can I visualize and save the results of YOLO predictions?

After running inference with YOLO, the `Results` objects contain methods for displaying and saving annotated images. You can use methods like `result.show()` and `result.save(filename="result.jpg")` to visualize and save the results. For a comprehensive list of these methods, refer to the [working with results](https://docs.ultralytics.com/modes/predict/#working-with-results) section.