# Install required libraries 
import librosa
import soundfile
import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pyaudio
import wave
import tkinter as tk
import sys
import pyttsx3


# Emotions in the RAVDESS dataset
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Emotions to observe
observed_emotions = ['calm', 'happy', 'fearful', 'disgust']

# Initialize the Multi Layer Perceptron Classifier
model = MLPClassifier(alpha=0.001, batch_size=256, epsilon=1e-08,
hidden_layer_sizes=(300), learning_rate='adaptive', max_iter=1000, activation='tanh', solver='adam')

# Recording user audio
def recordAudio():
    chunk = 1024  
    sample_format = pyaudio.paInt16  
    channels = 1
    fs = 48100  
    seconds = 5
    filename = "Recorded-Audio.wav"

    p = pyaudio.PyAudio()  # Create an interface to PortAudio

    print('Recording Started')

    stream = p.open(format=sample_format,channels=channels,rate=fs,frames_per_buffer=chunk,input=True)

    frames = []  # Initialize array to store frames

    # Store data in chunks for 10 seconds
    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
    p.terminate()

    print('Recording Ended')
    

# Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel))
    return result

# Load the data and extract features for each sound file
def load_data(test_size=0.2):
    x, y = [], []
    for file in glob.glob("E:\Major_Project\Dataset\\Actor_*\\*.wav"):
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[2]]
               
        if emotion not in observed_emotions:
            continue
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

# Train model on starting program
def trainModel():

    engine = pyttsx3.init()
    engine.say("Please wait Model is now getting Trained")
    engine.runAndWait()

    # Split the dataset
    x_train, x_test, y_train, y_test = load_data(test_size=0.25)

    # Get the shape of the training and testing datasets
    print((x_train.shape[0], x_test.shape[0]))

    # Get the number of features extracted
    print(f'Features extracted: {x_train.shape[1]}')

    # Train the model
    model.fit(x_train, y_train)

    # Predict for the test set
    y_pred = model.predict(x_test)

    # Calculate the accuracy of our model
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

    # Print the accuracy
    print("Accuracy: {:.2f}%".format(accuracy*100))
    acc = "{:.2f}".format(accuracy*100)

    engine = pyttsx3.init()
    engine.say("Model is Trained with accuracy")
    engine.say(acc)
    engine.say("You can start recording now.")
    engine.runAndWait()

# record your audio and predict emotion
def record_predictAudio():
    x_predictAudio = []
    recordAudio()  # Record audio to predict
    file = r'E:\Major_Project\Recorded-Audio.wav'  # Recorded audio filepath using raw string
    featurePredictAudio = extract_feature(file, mfcc=True, chroma=True, mel=True)  # Extract features of recorded audio
    x_predictAudio.append(featurePredictAudio)
    y_predictAudio = model.predict(np.array(x_predictAudio))
    return y_predictAudio




# Create the main application window
root = tk.Tk()
root.title("Speech Emotion Recognition")

root.attributes("-fullscreen", True)

def toggle_fullscreen(event=None):
    root.attributes("-fullscreen", not root.attributes("-fullscreen"))

root.geometry("1400x790")

# Toggle fullscreen mode
root.bind("<F11>", toggle_fullscreen)

engine = pyttsx3.init()
engine.say("Welcome to Speech Emotion Recognition System Developed By Group 14")
engine.runAndWait()

# Load the images
background_image = tk.PhotoImage(file="Assets/wall.png") 
icon_image1 = tk.PhotoImage(file="Assets/icon1.png")
icon_image2 = tk.PhotoImage(file="Assets/mic1.png")
icon_image3 = tk.PhotoImage(file="Assets/exit3.png")

# Create a label to hold the background image
background_label = tk.Label(root, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)  # Stretch the label to cover the entire window


# Create a label widget
label = tk.Label(root, text="[ Welcome to Speech Emotion Recognition System ]\n Record a clip of 3-5 seconds and we will try to determine the major emotion.", font=("Helvetica", 18, 'italic'), bg="lightblue", fg="navy")
# Pack the label widget into the window
label.pack(side=tk.TOP, padx=25, pady=15, fill='none', anchor=tk.W)



# Function 1 button
button1 = tk.Button(root, image=icon_image1, compound=tk.LEFT,
                    text="Start Model", 
                    font=('Times new roman',15, 'bold'), 
                    command=trainModel, 
                    background="lightblue", 
                    foreground="black", 
                    activebackground="#052b50", 
                    activeforeground="white", 
                    highlightthickness=8, 
                    highlightbackground="white", 
                    highlightcolor="white",
                    cursor="hand2",
                    
                    )
button1.pack(padx=350, anchor=tk.W)

def record_predict_and_display():
    # Call your function to record and predict audio
    y_predictAudio = record_predictAudio()
    
    # Redirect stdout temporarily to a StringIO object
    original_stdout = sys.stdout
    sys.stdout = sys.__stdout__

    # Capture the printed text
    printed_text = text_widget.get("1.0", tk.END)

    # Redirect stdout back to its original destination
    sys.stdout = original_stdout

    # Append the new print statement to the existing text in the text widget
    printed_text += "Emotion Predicted: {}\n".format(y_predictAudio)

    # Update the text widget with the combined text
    text_widget.config(state=tk.NORMAL)  # Allow modification
    text_widget.delete("1.0",tk.END)  # Clear existing text
    text_widget.insert(tk.END, printed_text)  # Insert new text
    text_widget.config(state=tk.DISABLED)  # Disable further modification

  


button2 = tk.Button(root, image=icon_image2, compound=tk.LEFT,
                    text="Record and Predict Audio", 
                    font=('Times new roman',15, 'bold'), 
                    command=record_predict_and_display, 
                    background="#052b50", 
                    foreground="white", 
                    activebackground="lightblue", 
                    activeforeground="black", 
                    highlightthickness=8, 
                    highlightbackground="white", 
                    highlightcolor="white",
                    cursor="hand2",
                    
                    )
button2.pack(padx=292, anchor=tk.W)

# Create a text widget to display the print statements
text_widget = tk.Text(root, bg= "#04082f", wrap="word", fg="White",font=('Courier',12, 'bold'),padx=5,borderwidth=0)
# text_widget.pack(fill="none", expand=True, padx=40, anchor=tk.W)
text_widget.place( relx=0.03, rely=0.4, anchor=tk.NW)

button3 = tk.Button(root, image=icon_image3,background="#020630",
                    borderwidth=0,activebackground="#020630",
                    command=lambda: root.quit(), 
                    cursor="hand2")
button3.place(anchor=tk.SE,relx=0.97,rely=0.95)

# Start the Tkinter event loop
root.mainloop()
