# Score Watcher
<p>
Detect and recognize the Player information from the scoreboard. The Scoreboard detection is done via [Yolov5](https://github.com/ultralytics/yolov5). The inference for the detection is done via [ONNX Runtime](https://github.com/microsoft/onnxruntime) . The Player information is extracted via CRNN or TessarOCR.
</p>

## <div align="center">System Architecture</div>

![alt text](https://github.com/StanlyHardy/score_watch/blob/scoreboard_dev/assets/graphics/system_architecture.png)

## <div align="center">Features</div>

- [x] Recognize Player Names
- [x] Determine score
- [x] Current Serving Player Indicator

### <div align="center">Install</div>

```bash
git clone https://github.com/StanlyHardy/score_watch # clone
cv score_watch
pip install -r requirements.txt # install
```

### <div align="center">Inference</div>
```
TODO
```

### Future work

- [ ] Train CRNN with wide set of Data.
- [ ] Adapt Attention OCR or Transformer architectures
- [ ] Multithreaded Inference
