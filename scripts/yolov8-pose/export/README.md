```bash
python3 -m onnxruntime.tools.make_dynamic_shape_fixed --dim_param unk__250 --dim_value 10 best.onnx best.fixed.onnx
```

```bash
/usr/src/tensorrt/bin/trtexec --onnx=best.onnx --fp16 --separateProfileRun --useSpinWait --saveEngine=best.engine
```