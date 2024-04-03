overhead_press_angle_left = calculate_angle(left_elbow, left_shoulder, left_hip)
overhead_press_angle_right = calculate_angle(right_elbow, right_shoulder, right_hip)

overhead_press_angle_hand_right = calculate_angle(right_shoulder, right_elbow, right_wrist)
overhead_press_angle_hand_left = calculate_angle(left_shoulder, left_elbow, left_wrist)

if overhead_press_angle_left <= 90 and overhead_press_angle_right <= 90 and overhead_press_angle_hand_left <= 90 and overhead_press_angle_hand_right <= 90:
    overhead_press_stage = "down"
if overhead_press_angle_left >= 150 and overhead_press_angle_right >= 150 and overhead_press_angle_hand_left >= 150 and overhead_press_angle_hand_right >= 150 and overhead_press_stage == 'down':
    overhead_press_stage = "up"
    overhead_press_counter +=1