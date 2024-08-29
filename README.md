
# Speech Emotion Recognition

- The basic idea behind this project is to build and train/test a suited machine learning algorithm that could recognize and detects real-time human emotions from speech.
- There are various applications for SER such as Human-Computer interaction, Customer Service and Sentiment Analysis, Market Research and Security and Authentication etc.
- Check this [Video](https://drive.google.com/file/d/1SkAto6ggFZTL3WS8uU50Wy0Whhv-_H-O/view?usp=drivesdk) for project demonstration.
## Requirements
- **Python 3.10+**
### Python Packages
- **librosa**
- **soundfile**
- **os**
- **glob**
- **numpy**
- **scikit-learn**
- **pyaudio**
- **wave**
- **tkinter**
- **sys**
- **pyttsx3**

### Dataset

**RAVDESS**: The **R**yson **A**udio-**V**isual **D**atabase of **E**motional **S**peech and **S**ong that contains 24 actors (12 male, 12 female), vocalizing two lexically-matched statements in a neutral North American accent. This dataset, created by Ryerson University, includes recordings of 24 professional actors representing eight different emotional states: **neutral**, **calm**, **happy**, **sad**, **angry**, **fearful**, **disgust**, and **surprised**. The performers provide a multimodal dataset with audio and video recordings by expressing these feelings through song and speech. Because of the dataset's wide range of emotional expressions, availability of both male and female actors, and standard audio formats (usually WAV files), it is widely used. Though we are focusing on just four of them in the project:- **calm**, **happy**, **fearful**, and **disgust**.

## Feature Extraction
Feature extraction is the main part of the speech emotion recognition system. It is basically accomplished by changing the speech waveform to a form of parametric representation at a relatively lesser data rate.

In this project, we have used the most used features that are available in [librosa](https://github.com/librosa/librosa) library including:
- [MFCC](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)
- Chromagram 
- MEL Spectrogram Frequency (mel)
```python
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
```

The extract_feature() function takes an audio file name and three Boolean arguments (mfcc, chroma, and mel) to specify which features to extract. It reads the audio file using soundfile, computes the STFT (Short-Time Fourier Transform) using librosa.stft(), and computes the mean of the MFCCs, chroma features, and mel-spectrogram features using librosa.feature.mfcc(), librosa.feature.chroma_stft(), and librosa.feature.melspectrogram(), respectively. It then concatenates the features into a single array and returns it.

## Load Data Splitting it into Training set and Testing set

The load_data() function reads all the audio files in a directory (in this case, "E:\dataset\\Actor_*\\*.wav") and extracts the features and emotion labels using extract_feature() and emotions. It then splits the data into training set (75%) and testing set (25%) using train_test_split() from sklearn.model_selection and returns the resulting arrays. The x_train, x_test, y_train, and y_test arrays are created by calling load_data(). The array created is 576,192 and 180 features are extracted.
```python
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

# Split the dataset
x_train, x_test, y_train, y_test = load_data(test_size=0.25)

# Get the shape of the training and testing datasets
print((x_train.shape[0], x_test.shape[0]))

# Get the number of features extracted
print(f'Features extracted: {x_train.shape[1]}')
```

## Building and Evaluating Model

The [MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html) class from sklearn.neural_network is used to create an MLP classifier with one hidden layer of 300 neurons, using the alpha, batch_size, epsilon, learning_rate, and max_iter activation. solver hyperparameters. The fit() method is called on the training data to train the classifier. The predict() method is called on the testing data to obtain the predicted labels, which are then compared to the true labels using accuracy_score() from sklearn.metrics. The accuracy is printed to the console using print().

```python
# Initialize the Multi Layer Perceptron Classifier
model = MLPClassifier(alpha=0.001, batch_size=256, epsilon=1e-08,
hidden_layer_sizes=(300), learning_rate='adaptive', max_iter=1000, activation='tanh', solver='adam')
# Train the model
model.fit(x_train, y_train)

# Predict for the test set
y_pred = model.predict(x_test)

# Calculate the accuracy of our model
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

# Print the accuracy
print("Accuracy: {:.2f}%".format(accuracy*100))
acc = "{:.2f}".format(accuracy*100)
```

## GUI of the project is created with the help of [Tkinter](https://docs.python.org/3/library/tkinter.html).

#### To Run the Speech Emotion Recognition System use python serapp.py.
```python
python serapp.py
```

#### Screenshots of the Project;

![Screenshot (2192)](https://github.com/user-attachments/assets/12044b12-9398-4fcf-9da2-b6605efdccd1)

![Screenshot (2193)](https://github.com/user-attachments/assets/6b7ef32f-f2f3-4a99-ba50-bc4312748df0)



**Note:** 
- **Install all the necessary libraries and make sure all the dependencies are in same folder where serapp.py script is present.**
- **If the serapp.py is running for the first time then it is recommended to click on start model button to train model otherwise no predictions will be given.**

## Citation

```bibtex
@software{SER_KS_2024,
  author       = {Kunal Singla},
  title        = {Speech Emotion Recognition},
  version      = {1.0.0},
  year         = {2024},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  url          = {https://github.com/K-Singla/Speech-Emotion-Recognition/tree/K's_Projects}
}
