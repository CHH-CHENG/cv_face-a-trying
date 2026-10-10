import cv2
import face_recognition
import os
import numpy as np

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)


    DB_DIR = "face_db"
    known_face_encodings = []
    known_face_names = []

    if not os.path.exists(DB_DIR):
        print(f"找不到 '{DB_DIR}' 文件夹")
        return

    for filename in os.listdir(DB_DIR):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            # 提取名字，支持格式如 chengh.jpg 或 chengh_1.jpg
            raw_name = os.path.splitext(filename)[0]
            clean_name = raw_name.split("_")[0] if "_" in raw_name else raw_name

            image_path = os.path.join(DB_DIR, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)

            if len(encodings) > 0:
                known_face_encodings.append(encodings[0])
                known_face_names.append(clean_name)
                print(f"{filename} -> [{clean_name}]")

    print(f"识别{set(known_face_names)}")

    # 2. 打开摄像头
    video_capture = cv2.VideoCapture(0, cv2.CAP_V4L2)
    print("\n按 'Q' 键退出")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("无法获取摄像头画面。")
            break
        #CPU 优化核心：为了让 CPU 跑得丝滑不卡顿，将检测画面缩小一半（0.5）
        SCALE = 0.5
        small_frame = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        #纯 CPU 最速 HOG 检测模型
        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = "Unknown"

            if len(known_face_encodings) > 0:
                # 黄金容错率 0.53，侧头也能认出来，且不容易误判
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.55)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

            # 坐标还原 2 倍放大到大图上画框
            inv_scale = int(1 / SCALE)
            top *= inv_scale
            right *= inv_scale
            bottom *= inv_scale
            left *= inv_scale

            # 根据是否识别成功选择颜色
            if name == "Unknown":
                color = (0, 0, 255)
                text_color = (255, 255, 255)
            else:
                color = (0, 255, 0)
                text_color = (0, 0, 0)

            # 绘制人脸方框
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            # 绘制名字背景条
            cv2.rectangle(frame, (left, bottom - 30), (right, bottom), color, cv2.FILLED)
            # 绘制名字文本
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.7, text_color, 1)

        # 显示最终画面
        cv2.imshow("Lightweight Face Recognition (CPU)", frame)

        # 按 Q 退出
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()