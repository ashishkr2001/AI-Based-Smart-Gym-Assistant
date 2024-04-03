import cv2
import mediapipe as mp
import numpy as np
import time
import os
import playsound
 
from utils import calculate_angle
from playsound import playsound


from pathlib import Path
import os
BASE_DIR = Path(__file__).resolve().parent
print(BASE_DIR)
voices = os.path.join(BASE_DIR,'voices')
playsound(voices +'/welcome.mp3')
time.sleep(2)
playsound(voices +'/select_exercise.mp3')
# import threading


language = 'en'



mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

for lndmrk in mp_pose.PoseLandmark:
    print(lndmrk)


cap = cv2.VideoCapture(700)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,640)

editable = True
selected = ""
pTime = 0

biceps_curl_counter = 0
biceps_curl_stage = None

overhead_press_counter = 0
overhead_press_stage = None

tricep_counter = 0
tricep_stage = None

leg_squat_stage = None
leg_squat_counter = 0

chest_press_stage = None
chest_press_counter = 0

close_counter = 0

second_image = None

## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()


        # gpu_frame = cv2.cuda_GpuMat()
        # gpu_frame.upload(frame)
        # frame = gpu_frame.download()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


        try:
            landmarks = results.pose_landmarks.landmark
            nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,landmarks[mp_pose.PoseLandmark.NOSE.value].y]
            left_eye_inner = [landmarks[mp_pose.PoseLandmark.LEFT_EYE_INNER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_EYE_INNER.value].y]
            left_eye = [landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].y]
            left_eye_outer = [landmarks[mp_pose.PoseLandmark.LEFT_EYE_OUTER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_EYE_OUTER.value].y]
            right_eye_inner = [landmarks[mp_pose.PoseLandmark.RIGHT_EYE_INNER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_EYE_INNER.value].y]
            right_eye = [landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].y]
            right_eye_outer = [landmarks[mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value].y]
            left_ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
            right_ear = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y]
            mouth_left = [landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].x,landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].y]
            mouth_right = [landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].x,landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].y]
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            left_pinky = [landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value].x,landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value].y]
            right_pinky = [landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value].y]
            left_index = [landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].y]
            right_index = [landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].y]
            left_thumb = [landmarks[mp_pose.PoseLandmark.LEFT_THUMB.value].x,landmarks[mp_pose.PoseLandmark.LEFT_THUMB.value].y]
            right_thumb = [landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            left_heel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]
            right_heel = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]
            left_foot_index = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
            right_foot_index = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]

            # biceps curl
            if selected == "biceps_curl":
                biceps_curl_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                biceps_curl_angle_2 = calculate_angle(right_shoulder, right_elbow, right_wrist)
                cv2.putText(image, str(biceps_curl_angle),
                                tuple(np.multiply(left_elbow, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )

                if biceps_curl_angle > 160 and biceps_curl_angle_2 > 160:
                    biceps_curl_stage = "down"
                if biceps_curl_angle < 50 and biceps_curl_angle_2 < 50 and biceps_curl_stage == "down":
                    close_counter += 1
                if biceps_curl_angle < 30 and biceps_curl_angle_2 < 30 and biceps_curl_stage =='down':
                    biceps_curl_stage="up"
                    close_counter = 0
                    biceps_curl_counter +=1

            # overhead_press
            if selected == "overhead_press":
                overhead_press_angle_left = calculate_angle(left_elbow, left_shoulder, left_hip)
                overhead_press_angle_right = calculate_angle(right_elbow, right_shoulder, right_hip)

                overhead_press_angle_hand_right = calculate_angle(right_shoulder, right_elbow, right_wrist)
                overhead_press_angle_hand_left = calculate_angle(left_shoulder, left_elbow, left_wrist)

                if overhead_press_angle_left <= 90 and overhead_press_angle_right <= 90 and overhead_press_angle_hand_left <= 90 and overhead_press_angle_hand_right <= 90:
                    overhead_press_stage = "down"
                if overhead_press_angle_left >= 150 and overhead_press_angle_right >= 150 and overhead_press_angle_hand_left >= 150 and overhead_press_angle_hand_right >= 150 and overhead_press_stage == 'down':
                    overhead_press_stage = "up"
                    overhead_press_counter +=1
                    close_counter = 0

                if overhead_press_angle_left > 130 and overhead_press_angle_right > 130 and overhead_press_angle_hand_left > 130 and overhead_press_angle_hand_right > 130 and overhead_press_stage == 'down':
                    close_counter += 1


            if selected == "tricep":
                tricep_angle_left = calculate_angle(left_shoulder, left_elbow, left_wrist)
                shoulder_angle_left = calculate_angle(right_shoulder, left_shoulder, left_elbow)

                tricep_angle_right = calculate_angle(right_shoulder, right_elbow, right_wrist)
                shoulder_angle_right = calculate_angle(left_shoulder, right_shoulder, right_elbow)

                if tricep_angle_left <= 90 and shoulder_angle_left <= 90:
                    tricep_stage = "down"
                if tricep_angle_left >= 150 and shoulder_angle_left >= 150 and tricep_stage == 'down':
                    tricep_stage = "up"
                    tricep_counter +=1
                    close_counter = 0

                if tricep_angle_left > 130 and shoulder_angle_left > 130 and tricep_stage == 'down':
                    close_counter += 1

                if tricep_angle_right <= 90 and shoulder_angle_right <= 90:
                    tricep_stage = "down"
                if tricep_angle_right >= 150 and shoulder_angle_right >= 150 and tricep_stage == 'down':
                    tricep_stage = "up"
                    tricep_counter +=1
                    close_counter = 0

                if tricep_angle_right > 130 and shoulder_angle_right > 130 and tricep_stage == 'down':
                    close_counter += 1


            if selected == "leg squat":
                leg_squat_angle_left = calculate_angle(left_hip, left_knee, left_ankle)
                leg_squat_angle_right = calculate_angle(right_hip, right_knee, right_ankle)

                left_ankle_knee_angle = calculate_angle(left_knee, left_foot_index, left_ankle)
                right_ankle_knee_angle = calculate_angle(right_knee, right_foot_index, right_ankle)

                if leg_squat_angle_left >= 85 and left_ankle_knee_angle >= 85:
                    if leg_squat_angle_left < 15 and leg_squat_angle_right > 15:
                        leg_squat_stage = "down"
                    if leg_squat_angle_left >= 80 and leg_squat_angle_right >= 80 and leg_squat_stage == 'down':
                        leg_squat_stage = "up"
                        leg_squat_counter +=1


                if leg_squat_angle_left > 65 and leg_squat_angle_right > 65 and leg_squat_stage == 'down':
                    close_counter += 1


            if selected == "chest press":
                chest_press_angle_left = calculate_angle(left_shoulder, left_elbow, left_wrist)
                chest_press_angle_right = calculate_angle(right_shoulder, right_elbow, right_wrist)

                if chest_press_angle_left <= 90 and chest_press_angle_right <= 90:
                    chest_press_stage = "down"
                if chest_press_angle_left >= 150 and chest_press_angle_right >= 150 and chest_press_stage == 'down':
                    chest_press_stage = "up"
                    chest_press_counter +=1
                    close_counter = 0

                if chest_press_angle_left > 130 and chest_press_angle_right > 130 and chest_press_stage == 'down':
                    close_counter += 1


            if close_counter > 3:
                    close_counter = 0
                    os.system("afplay voices/correct_pose.mp3")
                    playsound(voices + '/correct_pose.mp3')

        except:
            pass


        cv2.putText(
            image,
            'POSITION',
            (900, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)

        cv2.putText(
            image,
            'REPS',
            (1100, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)

        cv2.putText(
            image,
            'STAGE',
            (1200, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)

        cv2.putText(
            image,
            "BICEPS CURL **" if selected == "biceps_curl" else "BICEPS CURL",
            (900, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, ((0,128,0)if selected == "biceps_curl" else (255,0,0)), 1, cv2.LINE_AA)

        cv2.putText(
            image,
            str(biceps_curl_counter),
            (1100, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, ((0,128,0)if selected == "biceps_curl" else (255,0,0)), 1, cv2.LINE_AA)

        cv2.putText(
            image,
            str(biceps_curl_stage),
            (1200, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, ((0,128,0)if selected == "biceps_curl" else (255,0,0)), 1, cv2.LINE_AA)


        cv2.putText(
            image,
            "OVERHEAD PRESS **" if selected == "overhead_press" else "OVERHEAD PRESS",
            (900, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, ((0,128,0)if selected == "overhead_press" else (255,0,0)), 1, cv2.LINE_AA)

        cv2.putText(
            image,
            str(overhead_press_counter),
            (1100, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, ((0,128,0)if selected == "overhead_press" else (255,0,0)), 1, cv2.LINE_AA)

        cv2.putText(
            image,
            str(overhead_press_stage),
            (1200, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, ((0,128,0)if selected == "overhead_press" else (255,0,0)), 1, cv2.LINE_AA)


        cv2.putText(
            image,
            "TRICEP **" if selected == "tricep" else "TRICEP",
            (900, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, ((0,128,0)if selected == "tricep" else (255,0,0)), 1, cv2.LINE_AA)

        cv2.putText(
            image,
            str(tricep_counter),
            (1100, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, ((0,128,0)if selected == "tricep" else (255,0,0)), 1, cv2.LINE_AA)

        cv2.putText(
            image,
            str(tricep_stage),
            (1200, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, ((0,128,0)if selected == "tricep" else (255,0,0)), 1, cv2.LINE_AA)

        cv2.putText(
            image,
            "LEG SQUAT **" if selected == "leg_squat" else "LEG SQUAT",
            (900, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, ((0,128,0)if selected == "leg_squat" else (255,0,0)), 1, cv2.LINE_AA)

        cv2.putText(
            image,
            str(leg_squat_counter),
            (1100, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, ((0,128,0)if selected == "leg_squat" else (255,0,0)), 1, cv2.LINE_AA)

        cv2.putText(
            image,
            str(leg_squat_stage),
            (1200, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, ((0,128,0)if selected == "leg_squat" else (255,0,0)), 1, cv2.LINE_AA)

        cv2.putText(
            image,
            "CHEST PRESS **" if selected == "chest_press" else "CHEST PRESS",
            (900, 120),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, ((0,128,0)if selected == "chest_press" else (255,0,0)), 1, cv2.LINE_AA)

        cv2.putText(
            image,
            str(chest_press_counter),
            (1100, 120),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, ((0,128,0)if selected == "chest_press" else (255,0,0)), 1, cv2.LINE_AA)

        cv2.putText(
            image,
            str(chest_press_stage),
            (1200, 120),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, ((0,128,0)if selected == "chest_press" else (255,0,0)), 1, cv2.LINE_AA)


        cv2.putText(
            image,
            str(int(cap.get(cv2.CAP_PROP_FPS)))
            + " FPS",
            (400, 120),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)


        if editable:
            cv2.putText(
            image,
            "What you are willing to do ?",
            (900, 140),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)

            cv2.putText(
            image,
            "Press 1 for biceps curl",
            (900, 160),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)

            cv2.putText(
            image,
            "Press 2 for overhead press",
            (900, 180),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)

            cv2.putText(
            image,
            "Press 3 for tricep",
            (900, 200),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)

            cv2.putText(
            image,
            "Press 4 for leg squat",
            (900, 220),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)

            cv2.putText(
            image,
            "Press 5 for chest press",
            (900, 240),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)


            key = cv2.waitKey(1) & 0xFF
            if key == ord("1"):
                selected = "biceps_curl"
                os.system("afplay voices/biceps_curl.mp3")
                playsound(voices + '/biceps_curl.mp3')
                second_image = cv2.VideoCapture("videos/biceps_curl.gif")
                editable = False
            elif key == ord("2"):
                selected = "overhead_press"
                os.system("afplay voices/overhead_press.mp3")
                playsound(voices + '/overhead_press.mp3')
                second_image = cv2.VideoCapture("videos/overhead_press.gif")
                editable = False
            elif key == ord("3"):
                selected = "tricep"
                os.system("afplay voices/tricep.mp3")
                playsound(voices + '/tricep.mp3')
                second_image = cv2.VideoCapture("videos/tricep.gif")
                editable = False
            elif key == ord("4"):
                selected = "leg_squat"
                os.system("afplay voices/leg_squat.mp3")
                playsound(voices + '/leg_squat.mp3')
                second_image = cv2.VideoCapture("videos/leg_squat.gif")
                editable = False
            elif key == ord("5"):
                selected = "chest_press"
                os.system("afplay voices/chest_press.mp3")
                playsound(voices + '/chest_press.mp3')
                second_image = cv2.VideoCapture("videos/chest_press.gif")
                editable = False

        else:
            if cv2.waitKey(10) & 0xFF == ord('m'):
                os.system("afplay voices/select_exercise.mp3")
                playsound(voices + '/select_exercise.mp3')
                editable = True
                selected = ""
                second_image = None

            cv2.putText(
            image,
            "Press m twice to choose another exercise",
            (900, 140),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            cv2.putText(image, str(int(fps)), (15, 160), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                )

            ## show biceps_curl_image as watermark
            # if selected == "biceps_curl":
            #     cv2.imshow("biceps_curl_image", biceps_curl_image)
            # elif selected == "overhead_press":
            #     cv2.imshow("overhead_press_image", overhead_press_image)
            # elif selected == "tricep":
            #     cv2.imshow("tricep_image", tricep_image)
            # elif selected == "leg_squat":
            #     cv2.imshow("leg_squat_image", leg_squat_image)

        #second_image = cv2.imread("videos/eatezi.png")
        # if second_image is not None:
        #     second_image_height, second_image_width = second_image.shape[:2]
        #     image[0:second_image_height, 0:second_image_width] = second_image

        # run the frame through the inference engine
        # and use the return values to update the
        # model

        if second_image is not None:
            if second_image.isOpened():
                ret, frame = second_image.read()
                if ret:
                    frame_height, frame_width = frame.shape[:2]
                    image[0:frame_height, 0:frame_width] = frame


        cv2.imshow("Blaze Pose", image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


#