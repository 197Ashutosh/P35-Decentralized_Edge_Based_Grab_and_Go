import cv2
import torch
from ultralytics import YOLOWorld

_orig_load = torch.load
def _patch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _orig_load(*args, **kwargs)
torch.load = _patch_load

def get_color(name):
    h = abs(hash(name))
    r = (h & 0xFF) % 200 + 40
    g = ((h >> 8) & 0xFF) % 200 + 40
    b = ((h >> 16) & 0xFF) % 200 + 40
    return (b, g, r)

def main():
    model = YOLOWorld("yolov8s-world.pt")
    target_classes = [
        "cell phone",
        "smartphone",
        "chips packet",
        "potato chips",
        "chocolate",
        "chocolate bar",
        "candy",
        "cold drink",
        "bottle",
        "soda can",
        "biscuit packet",
        "cookies",
        "snack packet",
        "packaged food",
        "instant noodles",
        "juice box",
        "apple",
        "banana",
        "orange",
        "fruit",
        "vegetable",
        "cereal box",
        "soap",
        "box",
        "cup",
        "mug",
        "book",
        "wallet",
        "pen",
        "watch",
        "grocery item",
        "human hand",
        "person"
    ]
    model.set_classes(target_classes)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not accessible!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=0.12, iou=0.45, verbose=False)
        boxes_data = results[0].boxes

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 45), (25, 25, 25), -1)
        cv2.putText(frame, "UNIVERSAL RETAIL SCANNER (GROCERIES, PHONES, SNACKS) | 'q' to Exit", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        if boxes_data is not None and len(boxes_data) > 0:
            for box in boxes_data:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                class_name = model.names[cls_id]

                color = get_color(class_name)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

                label = f"{class_name.upper()} {int(conf * 100)}%"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, max(0, y1 - 26)), (x1 + w + 10, y1), color, -1)
                cv2.putText(frame, label, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        cv2.imshow("Live Universal Retail Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
