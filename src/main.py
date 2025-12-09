from ultralytics import YOLO
import cv2
import numpy as np
import util
from sort import *
from util import get_car, read_license_plate, write_csv


results = {}

mot_tracker = Sort()

# load models
coco_model = YOLO('models/yolov8n.pt')
license_plate_detector = YOLO('models/license_plate_detector.pt')

# load video
cap = cv2.VideoCapture('data/input_videos/video2.mp4')

if not cap.isOpened():
    print(f"Error: Tidak bisa membuka video di path: {cap}")
    exit()

vehicles = [2, 3, 5, 7]

# read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}
        # detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))

        # detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:

                # crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                # process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # read license plate number
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                  'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': license_plate_text,
                                                                    'bbox_score': score,
                                                                    'text_score': license_plate_text_score}}

        # write results
        for car_id, data in results[frame_nmr].items():
            # Ambil data
            car_bbox = data['car']['bbox']
            lp_bbox = data['license_plate']['bbox']
            text = data['license_plate']['text']

            # Gambar kotak untuk mobil (Hijau)
            cv2.rectangle(frame, (int(car_bbox[0]), int(car_bbox[1])), (int(car_bbox[2]), int(car_bbox[3])), (0, 255, 0), 2)

            # Gambar kotak untuk plat nomor (Biru)
            cv2.rectangle(frame, (int(lp_bbox[0]), int(lp_bbox[1])), (int(lp_bbox[2]), int(lp_bbox[3])), (255, 0, 0), 2)

            # Tulis teks plat nomor (Merah)
            cv2.putText(frame, text, (int(lp_bbox[0]), int(lp_bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Tampilkan frame yang sudah digambar di jendela video
        cv2.imshow('Deteksi Plat Nomor - Tekan "q" untuk keluar', frame)

        # dan untuk mendeteksi jika tombol 'q' ditekan
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# write results
write_csv(results, 'data/outputs/hasil.csv')
print("Video selesai diproses. Hasil disimpan di folder outputs")