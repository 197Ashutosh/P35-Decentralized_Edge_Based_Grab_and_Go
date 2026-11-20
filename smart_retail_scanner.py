import cv2
import torch
import numpy as np
from ultralytics import YOLO

_orig_load = torch.load
def _patch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _orig_load(*args, **kwargs)
torch.load = _patch_load

BARCODE_CATALOG = {
    "8901719101038": {"brand": "Parle", "item": "Hide & Seek Chocolate Chip", "price": 30},
    "8901719114137": {"brand": "Parle", "item": "Hide & Seek Fab Vanilla", "price": 30},
    "8901491101837": {"brand": "PepsiCo", "item": "Lays India's Magic Masala", "price": 20},
    "8901491001007": {"brand": "PepsiCo", "item": "Kurkure Masala Munch", "price": 20},
    "8901764012228": {"brand": "Coca-Cola", "item": "Coca-Cola 500ml", "price": 40},
    "8901764022227": {"brand": "Coca-Cola", "item": "Thums Up 500ml", "price": 40},
    "7622201755104": {"brand": "Cadbury", "item": "Oreo Vanilla Creme", "price": 35},
    "7622201140023": {"brand": "Cadbury", "item": "Dairy Milk Silk", "price": 80},
    "8901058852370": {"brand": "Nestle", "item": "Maggi 2-Minute Noodles", "price": 14}
}

COCO_RETAIL_MAP = {
    "cell phone": {"brand": "Mobile", "item": "Smartphone Device", "price": 14999, "type": "Electronics"},
    "bottle": {"brand": "Beverage", "item": "Cold Drink / Water Bottle", "price": 40, "type": "Drink"},
    "cup": {"brand": "Instant Food", "item": "Cup Noodles / Hot Beverage", "price": 50, "type": "Food"},
    "banana": {"brand": "Fresh Farm", "item": "Fresh Banana Bunch", "price": 30, "type": "Fruit"},
    "apple": {"brand": "Fresh Farm", "item": "Fresh Washington Apple", "price": 25, "type": "Fruit"},
    "orange": {"brand": "Fresh Farm", "item": "Fresh Orange", "price": 20, "type": "Fruit"},
    "sandwich": {"brand": "Bakery", "item": "Fresh Deli Sandwich", "price": 60, "type": "Bakery"},
    "book": {"brand": "Parle / Packaged", "item": "Hide & Seek / Biscuit Pack", "price": 30, "type": "Biscuits"}
}

def main():
    model = YOLO("yolov8s.pt")
    barcode_detector = cv2.barcode.BarcodeDetector()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not available!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cart = []
    scanned_barcodes = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=0.30, verbose=False)
        boxes_data = results[0].boxes

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (20, 20, 20), -1)
        cv2.putText(frame, "AI RETAIL BRAND & BARCODE SCANNER | Press 'q' to Exit | 'c' to Clear Cart", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        try:
            res = barcode_detector.detectAndDecode(frame)
            if len(res) == 3:
                decoded_info, decoded_type, points = res
                if points is not None and len(points) > 0 and decoded_info:
                    b_code = decoded_info[0] if isinstance(decoded_info, (list, tuple)) else decoded_info
                    b_code = str(b_code).strip()
                    if b_code:
                        pts = points[0].astype(int)
                        cv2.polylines(frame, [pts], True, (0, 255, 0), 4)
                        product_info = BARCODE_CATALOG.get(b_code, {"brand": "Packaged Retail", "item": f"Barcode {b_code}", "price": 25})
                        tag = f"{product_info['brand'].upper()}: {product_info['item']} - Rs.{product_info['price']}"
                        cv2.putText(frame, tag, (pts[0][0], max(30, pts[0][1] - 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        if b_code not in scanned_barcodes:
                            scanned_barcodes.add(b_code)
                            cart.append(product_info)
        except Exception:
            pass

        if boxes_data is not None and len(boxes_data) > 0:
            for box in boxes_data:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                name = model.names[cls_id]

                if name in COCO_RETAIL_MAP:
                    item_data = COCO_RETAIL_MAP[name]
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                    color = (0, 200, 255) if name == "cell phone" else (255, 120, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

                    header = f"[{item_data['brand']}] {item_data['item']} (Rs.{item_data['price']})"
                    (tw, th), _ = cv2.getTextSize(header, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    cv2.rectangle(frame, (x1, max(0, y1 - 25)), (x1 + tw + 10, y1), color, -1)
                    cv2.putText(frame, header, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        cv2.rectangle(frame, (frame.shape[1] - 320, 60), (frame.shape[1] - 10, 260), (35, 35, 35), -1)
        cv2.rectangle(frame, (frame.shape[1] - 320, 60), (frame.shape[1] - 10, 260), (0, 255, 255), 2)
        cv2.putText(frame, "CART (SCANNED ITEMS):", (frame.shape[1] - 305, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        y_offset = 115
        total_bill = 0
        for idx, item in enumerate(cart[-4:]):
            entry = f"{idx+1}. {item['item'][:18]} Rs.{item['price']}"
            cv2.putText(frame, entry, (frame.shape[1] - 305, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
            total_bill += item['price']

        cv2.line(frame, (frame.shape[1] - 310, 220), (frame.shape[1] - 20, 220), (100, 100, 100), 1)
        cv2.putText(frame, f"TOTAL: Rs. {total_bill}", (frame.shape[1] - 305, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        cv2.imshow("Smart Retail Scanner", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            cart.clear()
            scanned_barcodes.clear()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
