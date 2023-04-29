# Moodify

![1653490895007.png](image/README/1653490895007.png)

## What Inspired me

In today's world, where everyone is striving for success, many people are becoming depressed as a result of their efforts. As a result, **Moodify** was created to help people relax by suggesting movies and tunes based on their moods.

## What does this app do

**Moodify** is a movie and music recommendation website that uses facial expression detection to brighten up users and save time while searching for a movie or song that fits their mood.

1. It recognizes facial expression based on the 7 categories i.e., angry, sad, fear, happy, disgust, surprise and neutral.
2. It gives the user the option of proposing movies or songs based on the emotion.
3. If the user wants to watch movies or listen to music, a selection of movies or songs that match their mood is offered along with the movie/song poster.
4. When user clicks on movie which he/she wishes to watch, they will be redirected to IMDB website and for songs it redirects them to Spotify website.
5. As a feature, there is an Animated Gallery that displays certain photos.
6. A 3D Flipping Movies Card function has also been created to recommend movies and generate curiosity. When the user clicks on the card, it spins to show their iconic dialogue.

## How to run

**For creating virtual environment follow the steps:**

* Create an app.py (any name) file to indicate that you are using Python.
* Open Git bash in terminal & type python -m venv ProjectnameEnv and then tell it only for this project by command ProjectnameEnv/Scripts/Activate.bat.
* Open Command Palette.. and select Python:Select Interpreter and select .\ProjectnameEnv\Scripts\python.exe path.
* After cloning the project with the command - git clone [https://github.com/YOUR-USERNAME/YOUR-REPOSITORY](https://github.com/YOUR-USERNAME/YOUR-REPOSITORY) in Git bash.
* Use the command **pip install -r requirements.txt**, install all of the packages listed in requirements.txt
* To execute the project, open the app.py file in a terminal and type **python -u "directory of the app.py"** OR **python -u app.py.**

## How I built it and technology used

Python is the programming language used to create the emotion recognition model and deploy it on the web application using flask. CV2, TensorFlow, NumPy, matplotlib, and other libraries are also utilized.

The model is build using the transfer learning approach for which MobileNet model is used. The FER-2013 dataset, which comprises around 35000 photos, was utilized for model training and validation. This model is deployed on a website created with HTML and CSS using the flask framework. Based on the seven emotions, a new dataset of movies and music was constructed. The data from movies and songs was utilized to create the various templates that correlate to various emotions.

## Challenges I ran into

On the internet, it was difficult to discover Movies and Songs datasets that showed various emotions. Finding templates for emotional films and songs proved to be difficult. The task of getting the website to retrieve 7 movie templates from the movies button and 7 song templates from the songs button based on the emotion sensed was difficult. Dynamic links were used to access the templates for movies/songs that corresponded to the model's output.

## Accomplishments

Developing a project that is relevant to today's generation. Even though they were far apart, they worked in unison. Putting together this fantastic product in such a short amount of time! I learned a lot about Python and different frameworks like Flask and its integration, as well as many new Python modules. I planned and built the full solution in such a short period of time, overcoming all hurdles and finding a workaround for each issue.

## My Learning

I learnt about the transfer learning approach for the model, which assisted me in face identification, as well as how to develop creative web pages using HTML, CSS and how to deploy the model on the web using flask.

## Future Step for **Moodify**

The precision of the model must be improved. I'll also introduce an AI-powered chat bot that will allow users to communicate their difficulties and, based on that, the Bot will recommend some solutions as well as the best appropriate movie or song for the situation.
