biceps_curl_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
biceps_curl_angle_2 = calculate_angle(right_shoulder, right_elbow, right_wrist)
cv2.putText(image, str(biceps_curl_angle), 
                tuple(np.multiply(left_elbow, [640, 480]).astype(int)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                    )

if biceps_curl_angle > 160 and biceps_curl_angle_2 > 160:
    biceps_curl_stage = "down"
if biceps_curl_angle < 30 and biceps_curl_angle_2 < 30 and biceps_curl_stage =='down':
    biceps_curl_stage="up"
    biceps_curl_counter +=1