from gtts import gTTS
import os

language = 'en'

sound = gTTS(text="Welcome to smart pose detection", lang=language, slow=False)
sound.save("voices/welcome.mp3")

sound = gTTS(text="You selected biceps curl", lang=language, slow=False)
sound.save("voices/biceps_curl.mp3")

sound = gTTS(text="You selected overhead press", lang=language, slow=False)
sound.save("voices/overhead_press.mp3")


sound = gTTS(text="You have selected tricep", lang=language, slow=False)
sound.save("voices/tricep.mp3")

sound = gTTS(text="You have selected leg squat", lang=language, slow=False)
sound.save("voices/leg_squat.mp3")

sound = gTTS(text="Select your exercise", lang=language, slow=False)
sound.save("voices/select_exercise.mp3")

sound = gTTS(text="You were really close to complete the rep", lang=language, slow=False)
sound.save("voices/correct_pose.mp3")


sound = gTTS(text="You selected chest press", lang=language, slow=False)
sound.save("voices/chest_press.mp3")