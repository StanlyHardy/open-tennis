# Score Watcher
<p>
Detect and recognize the Player information from the scoreboard. The Scoreboard detection is done via <a href="https://github.com/ultralytics/yolov5">Yolov5</a>. The inference for the detection is done via <a href="https://github.com/microsoft/onnxruntime">ONNX Runtime</a> . The Player information is extracted via CRNN or TessarOCR.
</p>

## <div align="center">System Architecture</div>

![alt text](https://github.com/StanlyHardy/score_watch/blob/scoreboard_dev/assets/graphics/system_architecture.png)

### <div>Features</div>

- [x] Recognize Player Names
- [x] Determine score
- [x] Current Serving Player Indicator

### <div>Install</div>

```bash
git clone https://github.com/StanlyHardy/score_watch # clone
cv score_watch
pip install -r requirements.txt # install
```

### <div>Inference</div>
Inference could run either on Video or Image streams. The configuration could be changed via `assets/config/app_config.yaml`. If the `evaluation` is set to true, the inference occurs in validatation dataset and performs evaluation to determine the Average scores for correct Player names, Scores and Serving Player.
```
python app.py # Runs inference
```

### Future work

- [ ] Train CRNN with wide set of Data.
- [ ] Adapt Attention OCR or Transformer architectures
- [ ] Multi Threaded Inference
