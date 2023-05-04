import cv2
import mediapipe as mp
import time
import math


class HandDetector:
    def __init__(self, mode=False, max_hands=2, model_complexity=1, detection_con=0.5, track_con=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.model_complexity = model_complexity
        self.detection_con = detection_con
        self.track_con = track_con
        self.results = None
        self.pos_list = None
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands, self.model_complexity, self.detection_con,
                                         self.track_con)
        self.mp_draw = mp.solutions.drawing_utils
        self.fingertips = [4, 8, 12, 16, 20]

    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for landmark in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, landmark, self.mp_hands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, hand_no=0, draw_lm=True, draw_box=True, box_offset=0, draw_box_offset=20):
        x_list = []
        y_list = []
        z_list = []
        bounding_box = []

        self.pos_list = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_no]
            for lm_id, lm in enumerate(my_hand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy, cz = int(lm.x * w), int(lm.y * h), round(lm.z * 200)
                x_list.append(cx)
                y_list.append(cy)
                z_list.append(cz)
                # print(id, cx, cy)
                self.pos_list.append([lm_id, cx, cy, cz])
                if draw_lm:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
            x_min, x_max = min(x_list)-box_offset, max(x_list)+box_offset
            y_min, y_max = min(y_list)-box_offset, max(y_list)+box_offset
            bounding_box = [(x_min, y_min), (x_max, y_max)]
            if draw_box:
                cv2.rectangle(img, (bounding_box[0][0]-draw_box_offset, bounding_box[0][1]-draw_box_offset),
                              (bounding_box[1][0]+draw_box_offset, bounding_box[1][1]+draw_box_offset), (0, 255, 0), 2)

        return self.pos_list, bounding_box

    def fingers_up(self):
        fingers = []

        # Thumb
        if (self.pos_list[self.fingertips[0]][1] > self.pos_list[self.fingertips[0] - 1][1] and self.pos_list[5][1] >
            self.pos_list[17][1]) or (self.pos_list[self.fingertips[0]][1] < self.pos_list[self.fingertips[0] - 1][1]
                                      and self.pos_list[5][1] < self.pos_list[17][1]):
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 Fingers
        for lm_id in range(1, 5):
            if self.pos_list[self.fingertips[lm_id]][2] < self.pos_list[self.fingertips[lm_id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def find_distance(self, img, p1, p2, draw=True, r=10, t=3):
        x1, y1 = self.pos_list[p1][1:]
        x2, y2 = self.pos_list[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = 0

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
            length = math.hypot(x2 - x1, y2 - y1)

        return length, [x1, y1, x2, y2, cx, cy]


def main():
    previous_time = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    while True:
        success, img = cap.read()
        img = detector.find_hands(img)
        lm_list, bounding_box = detector.find_position(img, draw_lm=False)
        if len(lm_list) != 0:
            print(lm_list[8])

        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
    