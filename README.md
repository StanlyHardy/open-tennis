# Score Watcher
<p>
Detect and recognize the Player information from the scoreboard. The Scoreboard detection is done via <a href="https://github.com/ultralytics/yolov5">Yolov5</a>. The inference for the detection is done via <a href="https://github.com/microsoft/onnxruntime">ONNX Runtime</a> . The Player information is extracted via CRNN or TessarOCR.
</p>

## <div align="center">System Architecture</div>

![alt text](https://github.com/StanlyHardy/score_watch/blob/scoreboard_dev/assets/graphics/system_arch.png)

### <div>Features</div>

- [x] Recognize the Player Names.
- [x] Determine the scores.
- [x] Find the current serving player.
- [x] Evaluate the average correct match.

### <div>Install</div>

```bash
git clone https://github.com/StanlyHardy/score_watch # clone
cv score_watch
pip install -r requirements.txt # install
```

### <div>Inference</div>
Inference could run either on Video or Image streams.  The configuration could be changed via `assets/config/app_config.yaml`. If the `evaluation` is set to true, the inference occurs in validatation dataset and performs evaluation to determine the Average scores for correct Player names, Scores and Serving Player. Please change the input paths of `video` or `images`.
```
python app.py 
```
### <div>Demo</div>
![alt text](https://github.com/StanlyHardy/score_watch/blob/scoreboard_dev/assets/demo/1.jpg)

### Future work

- [ ] Train CRNN with wide set of Data.
- [ ] Gauge Attention OCR / Transformer architectures
- [ ] Multi Threaded Inference
