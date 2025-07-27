## 📥 Tải dữ liệu

Bộ dữ liệu được sử dụng trong dự án này có thể tải từ Kaggle tại đường dẫn sau:

👉 [Interior Design Styles - Kaggle](https://www.kaggle.com/datasets/stepanyarullin/interior-design-styles)

### Lưu ý:
chỉ tải bộ test, KHÔNG TẢI BỘ TRAIN VỀ

### Cách dùng:
Chạy file eval, move 1 trọng số bất kỳ từ folder models ra, sau đó chỉnh tên model đó vào dòng 29:

Custom CNN:
```python
session = ort.InferenceSession("custom_cnn.onnx", providers=["CPUExecutionProvider"])
```

YOLOv11:
```python
session = ort.InferenceSession("yolov11.onnx", providers=["CPUExecutionProvider"])
```

YOLOv8:
```python
session = ort.InferenceSession("yolov8n-cls.onnx", providers=["CPUExecutionProvider"])
```

### Lưu ý 2 dòng cuối (126, 127):

Nhớ đổi tên file để có thể dễ dàng đánh giá và xem kết quả các mô hình
