## <div align="center">Scoreboard Watcher</div>

<p align="center">
 Extraction of the player information occurs by a series of detection followed by text recognition.  Utilized both PyTesseract and CRNN for player information extraction. <a href="https://developer.nvidia.com/tensorrt">TensorRT</a> has been utilized for detection.TensorRt Version of Scoreboard Watcher spikes up the performance 2x times. The results are post-processed before dispatching them to the evaluator. </p>

## <div align="center">System Architecture</div>
 <p>
   <img  src="https://github.com/StanlyHardy/score_watch/blob/experimental/assets/graphics/sys_arch.png"></a>
</p>

## <div align="center">How it works?</div>
- There can be either an Image session or a Video Playback session. Both sessions are bound to session context.
- The Score Manager retrieves the frames from the session context and dispatches them to detection.
- After an initial warm-up epoch, the model is ready for inference via ONNXRuntime over the incoming frames and passes the result to OCRRoot. 
- The OCRRoot uses either the TessarOCR or CRNN for Player information extraction. The recognized noisy labels get sanitized before passing them to the coordinator. Renderer could be utilized to draw results.
- The Result Coordinator coordinates the result for further evaluation during the image playback session. 

## <div align="center">Features</div>

- [x] Recognize the Player Names.
- [x] Determine the scores.
- [x] Find the current serving player.
- [x] Evaluate the average correct match.
## <div align="center">Demo</div>
 <p>
   <img  src="https://github.com/StanlyHardy/score_watch/blob/scoreboard_dev/assets/demo/1.jpg">
</p>

## <div>Install</div>
#### <div>Requirements</div>
- Linux
- CUDA>= 10.0
- Python >= 3.6

#### Steps
1. Create a virtual conda environment and activate it.

```bash
conda create -n scorewatch python=3.9 -y
conda activate scorewatch
```
2. Install Pytorch
```bash
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

4. Install TensorRT
```bash
pip install -U nvidia-tensorrt --index-url https://pypi.ngc.nvidia.com
```
5. Clone the repository
```bash
git clone https://github.com/StanlyHardy/score_watch # clone
cd score_watch
```
6. Install other requirements
```bash
pip install -r requirements.txt # install
```
7. The provided `.engine` file is platform specifc. So, export `detector.pt` within `assets/models` to TensorRT engine using the official <a href="https://github.com/ultralytics/yolov5/blob/master/export.py">exporter </a>. 

## <div >Inference</div>
Inference could run either on Video or Image streams.  The configuration could be changed via `assets/config/app_config.yaml`. If the `evaluation` is set to true, the inference occurs in validatation dataset and performs evaluation to determine the Average scores for correct Player names, Scores and Serving Player. Please change the input paths of `video` or `images`.
```
python app.py 
```


## <div >App Configuration</div>
<details>
 <summary>App configuration(click to expand)</summary>
<table>
 <tr>
  <td rowspan="6">&nbsp; Paths </td>
  <td>&nbsp; <code>video_path</code></td>
  <td>&nbsp;Path of the video on which the evaluation needs to be done.</td>
 </tr>
 <tr>
  <td>&nbsp;<code>img_path</code></td>
  <td>&nbsp;Directory containing the test images. Ground truth needs to be available for evaluation with image set.</td>
 </tr>
  <tr>
  <td>&nbsp;<code>players_path</code></td>
  <td>&nbsp;Path containing player informations</td>
 </tr>
 <tr>
  <td>&nbsp;<code>groundtruth_path</code></td>
  <td>&nbsp;Ground truth data which is in json format that has got the player information.</td>
 </tr>
 <tr>
  <td>&nbsp;<code>output_video_path</code></td>
  <td>&nbsp;The path to save the video if the output needs to be saved and visualized later.</td>
 </tr>
 <tr>
  <td>&nbsp;<code>logs_path</code></td>
  <td>&nbsp;Path where the output log will be saved.</td>
 </tr>
 <tr>
  <td rowspan="5">&nbsp; Streamer </td>
  <td>&nbsp; <code>should_draw'</code></td>
  <td>&nbsp;Draws over the frames for visualization , if enabled.</td>
 </tr>
 <tr>
  <td>&nbsp;<code>view_imshow</code></td>
  <td>&nbsp;The output visualization shall be turned on/off with this parameter.</td>
 </tr>
  <tr>
  <td>&nbsp;<code>save_stream</code></td>
  <td>&nbsp;Turning on this field enables the video output to be saved in the path defined in <code>output_video_path</code></td>
 </tr>
 <tr>
  <td>&nbsp;<code>debug</code></td>
  <td>&nbsp;Displays debug logs if enabled</td>
 </tr>
 <tr>
  <td>&nbsp;<code>evaluation</code></td>
  <td>&nbsp;Turn on if the evaluation has to be done over the image set. Both image set and the annotations are required in this case.</td>
 </tr>
 <td rowspan="5">&nbsp; Models </td>
  <td>&nbsp; <code>score_det_model'</code></td>
  <td>&nbsp; Path of the score detector model.</td>
 </tr>
 <tr>
  <td>&nbsp;<code>detector_config</code></td>
  <td>&nbsp; Path of the config file for the score detector. </td>
 </tr>
  <tr>
  <td>&nbsp;<code>text_rec_model</code></td>
  <td>&nbsp;CRNN Model path responsible for Player information recognition. </td>
 </tr>
 <tr>
  <td>&nbsp;<code>text_rec_config</code></td>
  <td>&nbsp;Path to the configuration for the CRNN model</td>
 </tr>
 <tr>
  <td>&nbsp;<code>ocr_engine</code></td>
  <td>&nbsp;Choose between <code>CRNN</code> or <code>PyTesseract</code>. </td>
 </tr>
</table>
</details>
<details>
 <summary>Detector Configuration(click to expand)</summary>
<table>
 <tr>
  <td rowspan="5">&nbsp; YOLOv5 </td>
  <td>&nbsp; <code>execution_env</code></td>
  <td>&nbsp;ONNX Runtime provides support for CUDA, CPU and TensorRT. By default, CUDA is chosen. ONNX Runtime falls back to cpu if CUDA is unavailable.</td>
 </tr>
 <tr>
  <td>&nbsp;<code>conf_thresh</code></td>
  <td>&nbsp;Detection confidence</td>
 </tr>
  <tr>
  <td>&nbsp;<code>iou_thres</code></td>
  <td>&nbsp;IOU threshold to gauge the overlap.</td>
 </tr>
 <tr>
  <td>&nbsp;<code>warm_up</code></td>
  <td>&nbsp;Number of samples to be used during the warm up phase.</td>
 </tr>
 <tr>
  <td>&nbsp;<code>class_labels</code></td>
  <td>&nbsp;Class labels</td>
 </tr>
 <tr>
</table>
</details>

## <div>Roadmap</div>
- [ ] Quantize the model
- [ ] Multi Threaded Inference
- [ ] Train CRNN with wide set of Data.
- [ ] Compute Precision/Recall for evaluation.
- [ ] Gauge Attention OCR / Transformer architectures

## <div >Acknowledgements</div>
* [ONNX Runtime](https://onnxruntime.ai/docs/install/)&nbsp; 
* [YOLOv5](https://github.com/ultralytics/yolov5)&nbsp;
* [TesserOCR](https://github.com/sirfz/tesserocr)&nbsp; 
* [CRNN](https://www.kaggle.com/alizahidraja/custom-ocr-crnn)&nbsp;
* [TrOCR](https://huggingface.co/docs/transformers/model_doc/trocr)&nbsp;
