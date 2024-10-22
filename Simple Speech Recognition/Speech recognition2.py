
import speech_recognition as sr
def recognize_speech(recognizer, microphone):
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        print("We are listening... Syntax should be eg. OPERAND + 1 NUMBER + BY + 2 NUMBER")
        audio = recognizer.listen(source)
    try:
        response = recognizer.recognize_google(audio)
        print("You said the following: " + response)
        return response
    except sr.RequestError:
        print("API unavailable")
    except sr.UnknownValueError:
        print("Unable to recognize your speech")

def calculate(expression):
    words = expression.split()
    if len(words) == 4:
        num1 = int(words[1])
        num2 = int(words[3])
        if words[0] == "add" or words[0] == "+":
            return num1 + num2
        elif words[0] == "subtract" or words[0] == "-":
            return num1 - num2
        elif words[0] == "multiply" or words[0] == "*":
            return num1 * num2
        elif words[0] == "divide" or words[0] == "/":
            return num1 / num2 if num2 != 0 else "Cannot divide by zero"
    return "Invalid input format"

if __name__ == "__main__":
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    while True:
        print("Please say a command to calculate:")
        command = recognize_speech(recognizer, microphone)
        if command:
            result = calculate(command)
            print("Result: " + str(result))
