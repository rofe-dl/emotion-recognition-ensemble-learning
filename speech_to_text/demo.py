import speech_recognition as sr

r1 = sr.Recognizer()

# From mic
with sr.Microphone() as source:
    print('speak now')
    r1.adjust_for_ambient_noise(source=source)
    audio = r1.listen(source)

# From file
file = sr.AudioFile("dogs.wav")
with file as source:
    audio = r1.record(source)

try:
    print (r1.recognize_google(audio))
except sr.UnknownValueError:
    print("Could not understand audio")
except sr.RequestError as e:
    print("Could not request results from google", e)

