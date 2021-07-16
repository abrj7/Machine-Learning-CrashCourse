# Machine-Learning-CrashCourse
A Machine Learning Course to teach you the basics behind Issues with Machine Learning.
# Machine Learning Course.
Welcome to the complete Machine Learning Guide. Here you will learn everything about machine learning. You will be taught how to deal with different type of data. This course is more than enough for you to master Machine Learning's theorotical concepts.
## Course Content:
- What is Machine Learning
- What _exactly_ is Machine Learning?(Part 2)
- Machine Learning Playground 1.
- Machine Learning Playground 2.
- Types of Machine Learning.
    - Supervised Learning.
        - Classification.
        - Regression.
    - Transfer Learning.
    - Reinforcement Learning.
- Machine Learning Framework(Creating our own)
- A deeper look into Modelling.
    - Part 1: Splitting Data.
    - Part 2: Choosing Data.
    - Part 3: Tuning a model
- What tools are used for what step.
- Anaconda.
    - MiniConda, Conda
    - MiniConda Setup
    - Jupyter
    - Testing
    - Finishing Up.




## What is Machine Learning?
#### Computers are machines, and machines are _really_ fast in doing some things, that humans are not so fast in. But the computer requires programming from the humans to _actually_ perform a task. Now humans can't work all the time, so we as machine learn, we program the computer in a way that it can find solutions to problems, and provide it to us, without ever breaking a sweat. 
#### For example, if we need to find a route, that takes the least amount of time to reach a place, we ourselves, can take a map, and use a ruler to measure the smalest distance. But this will take us hours, maube longer. Wheras if we program the computer to use `if`, `else` blocks to find the shortest distance, it will take barely a few seconds. This can be done by simple coding.
#### Here is the real question? What if we want to find out if a customer is angry, or happy with a product, using the reviews? We can't just use `if` and `else` to that because each customer has his own language. Here is when Machine Learning comes in. 
## But what __*exactly*__ is Machine Learning?
#### You may have heard of [AI](https://en.wikipedia.org/wiki/Artificial_intelligence), [Data Science](https://en.wikipedia.org/wiki/Data_science), [Neural Networks](https://en.wikipedia.org/wiki/Artificial_neural_network), Deep Learning, but what are those things, and how does Machine Learning apply to them? There is a diagram that can help us identify the concepts behind allof these things. 
#### It all starts with AI, a machine that acts like a human. Currentwe only have Narrow AI, AI that is only smart at one thing, like chess, or detecting heart diseases from pictures. Whereas General AI, is one that is smart in everything. We are far from reaching that stage.  Machine Learning is a subset of AI, an approach to try and achieve AI, thorugh systems that can find patterns in Data. Deep Learning, is one of the techniques of implementing Machine Learning. Data Science, on the other hand, overlaps Machine Learning, which is analyzing data. When we talk about Machine Learning, we generally also talk about Data Science. Here is the chart to keep in mind:

![image](https://miro.medium.com/max/650/1*-XKVI5SAEpffNR7BusdvNQ.png)
***
## Machine Learning Playground 1.
#### We can visualize Machine Learning using an amazing playground made by google, called [Teachable Machine](https://teachablemachine.withgoogle.com/). It helps you play with Machine Learning projects. Hit Get Started and play around with a few projects to see how the models are trained. The rained models can differentiate between things like, humans and dogs.

## Machine Learning Playground 2.
#### Another amazing playground is "[Machine Learning Playground](https://ml-playground.com/)", which can be trained to show recommended areas of different types. Here is the graph screen you will be welcomed with. You will be welcomed with a graph screen.

#### Lets say, the y-axis represents the length of the video and the x-axis represents the likes of the video. The yellow dots will represent the likes a specific user gives to videos, and the purple represesnts the dislikes. We can plot the user's data, and get a trained model:
![image](https://storage.googleapis.com/replit/images/1621357365989_a9fff8aec36a7433e00a60dcc0c0213f.png)

#### Using this, we can train the model and get a portion of the graph, using we which we can recommend user videos that they will like. 
![image](https://storage.googleapis.com/replit/images/1621357497905_0543b59bce382a868381cf6caacca84f.png)
***
## Types of Machine Learning.
#### No matter how many sub categories of machine learning, all of them are meant to predict results, based on incoming data. There are few main types of Machine Learnings; Supervised, Unsupervised, Transfer Learning and Reinforcement.

![mltype](https://coschedule.s3.amazonaws.com/106308/910af4fa-63fa-4346-a2f2-ef280e8a250f/1576687016462.png)
### Supervised Learning:
#### Firstly, we have Supervised Learning, a subset of Machine Learning. In Supervised Learning, the data already has categories, like [CSV](https://en.wikipedia.org/wiki/Comma-separated_values) files, or data that is labeled. A supervised learning model, uses classification, or regression to differentiate data, by simply drawing a line. It can be used to predict stock prices, or hiring employees, based on experience, age, etc. 
### Supervised Learning:
|Classification|Regression
:------|:-----|:------
|Determines if a sample is one thing or other|Trying to predict a number|
|Binary classification: Two options involved|Numbers can go up or down|
|Multi-class classification: more than two options|-------------------------------|
![cls/reg](https://static.javatpoint.com/tutorial/machine-learning/images/regression-vs-classification-in-machine-learning.png) 

### Unsupervised Learning:
#### Unsupervised learning, uses an algorithm that learns to form patterns in _untagged_ data. Using ML(Machine Learning), the computer can find patterns in a set of data, and form groups, more like what was shown in the ML Playground of video recommendations. We can either make the machine use classification or Association, in which it learns what is what.  

### Transfer Learning:
#### Transfer Learning leverages what one machine learning model has learned, into another model. You can create a model that uses lots of images and knows what the images are, for examples trees or cars. That way, the same model can be used to recognize pictures, but the pictures are of cats vs lions. The concept is same, but the images are different, much like the ML model of google.

### Reinforcement Learning:
#### Reinforcement Learning teaches the machine using trial and error; reward and punishment. So the program can simply learn to play a game, by playing it over a million ti  mes. More or less like how humans learn. If it does something wrong, the machine will be like "Oh, I will not do that next time, maybe try doing the opposite.".
![image](https://storage.googleapis.com/replit/images/1621359775990_a03a5bf654ea73b9e45d72c41ed89549.png)
***
## Machine Learning Framework(Creating our own)
#### Using Python, we will now build our own Machine Learning Framework.
#### Machine Learning usually comes in three parts, Data Collection, Data modelling(aka, writing an alogrithm to find patterns), and Deployment. Right now, we will work on Data modelling, by first creating a framework, and matching it to other data science projects. 

### Data modelling:
#### There are 6 steps to follow while modelling Data:
|Steps|Use|
:------|:-----|:------|
|Problem Definition|It is important to define the problem, i.e, Supervised, Unsupervised, Collection or Regression, etc.|
|Data|Knowing the type of data you have, i.e Structered(CSV, XML), Unstructered|
|Evaluation|Having a specific goal, like 95%, because ML models can be improved forever.|
|Features|What do we know about the data. Any type of data has features, and the model uses these for predictng results.|
|Modelling|What machine model to use, because most ML models are already made.|
|Experimentation|How could your model be improved.|


![image](https://storage.googleapis.com/replit/images/1621370700059_0bedef87c56669e342d1dc6631006a16.png)
***
## Modelling in Machine Learning.
#### Modelling in Machine Learning is on of the most important concepts. Lets dig deeper into it. There are a few _very_ important parts in Machine Learning:
### Part 1: Splitting Data:
#### In this part, we first ask the question, "Based on our problem and data, what model should we use?". Modelling can be broken down into three parts, **_Choosing and training a model_**, **_Tuning a model_**, **_Model comparison_**. This is the most important part in Machine Learning. To achieve this, we split the data into three diffferent sets, one for _training_, one for _validation_, and one for _testing_.
![image](https://storage.googleapis.com/replit/images/1621451802727_7c06ca363647b5cd97fa7940f41186ea.png)

Think of training, as practicing course material for your exams. As your exams come nearer, you take a practice paper to test yourself, this can be said as validation. Lastly, as you are giving the exam itself, and even though you are faced with different, never-seen-before questions, you know how to tackle them because of the experience you got from practice. This is the same way ML Models work. But in ML it is called Generalization.

**Generalization**:  _The ability for a mchine learning model to perform well on data it hasn't seen before._

But what if we as students have already seen the exam paper? Easy, we will get top marks right? But did we learna anything? No. And that is where the training, validation, test splits come in, while dealing with Machine Learning. We slit our data into sets and work with them, by first training them, test it on a sample, and taking it out in the real world and use it on other data.

### Part 2: Choosing Data:
Now, that we have talked about splitting data, lets dig even deeper and alk about the first step, "Choosing the data and training it".
#### For Structered Data:
There are many ML models that are already made, so rather than creating an algorithm, we can simply use other already-developed ML models and use them. So our first goal is to choose which model works best with our data. When working with Structered Data, it is best to use Decision Tree models, like "[Random Forests](https://en.wikipedia.org/wiki/Random_forest)", or "[Gradient Boosting Algorithms](https://en.wikipedia.org/wiki/Gradient_boosting)"; like [CatBoost](https://catboost.ai/), [XGBoost](https://xgboost.readthedocs.io/en/latest/build.html).

![image](https://storage.googleapis.com/replit/images/1621554022366_9d3d2fdb8e33eb53c9005d1360b9af93.png)

#### For Unstructered Data:
Unstructured data is best dealt with using Deep Learning, and Transfer Learning, because there may not be an already created model for the type of output you need based on your data.

After chosing the model, we need to train our model. The goal here is to line up the inputs and outputs. We make our model to look at the inputs/features of the data, to make predictions about it, which is the output. Training the model should always be done on the training set that is divided from the main data. 
Another main goal while training your fata, is minimizing the time taken between your experiments. This means, training samll amounts of data first, and then moing to larger data sets. For example if you have 100,000 examples in your data set, you might start training with only 10,000.

### Part 3: Tuning a model:
Tuning normally takes place in your validation data split, but depending on your circumstances, it could also take place on your training split. Tuning in a model can be explained using a real-life example. Lets say you have to cook a meal in the oven without ever trying it before. First, you t ry cooking it with 180°C, but it utrns out raw, next you try it with 200 °C and it is still a little uncooked. But lastly, you try with 250°C, and the meal is cooked perfectly. But in a model, you can use tuning to change numbers, layers etc. (Will be looked later on)

### Part 3: Model comparison:
Comparing models takes place on the test set. This is used to see how your model generalizes with the types of problems it faces. A good model will give out similar results in all three sets. In model comparison, there are two types of results, "Overfitting", and "Underfitting". Underfitting means that your model gives a higher success rate in the training model rather than the test model. And overfitting is the opposite; higher results in the test, than the training data.

![image](https://storage.googleapis.com/replit/images/1621622781364_03b2bed2afc72c69b0c567a00968a5dc.png)

The best type of result, is the one that fits just right, or in other words, result that is balanced. Overfitting is not good because it literally is the same as you giving an exam after seeing the paper, which means you will be able to amswer only the specific exam rigtly, no other types of problems. One of the main reasons of overfitting, is data leakage from the test data set to the training set.      
***
## What tools are used for what steps
#### For steps like Data, Evaluation, and Features, we might use frameworks like [Pandas](https://pandas.pydata.org/), [Matplotlib](https://matplotlib.org/), and [NumPy](https://numpy.org/).  
#### Wheras for Modelling, most common frameworks are [TensorFlow](https://www.tensorflow.org/), [PyTorch](https://pytorch.org/), [SciKit-Learn](https://scikit-learn.org/) and the previously discussed, XGBoost, and CatBoost. Wheras, for the entire project, we will use [Jupyter](https://jupyter.org/) and Anaconda.

## Anaconda.
#### [Anaconda](https://www.anaconda.com/) is a software distribution that provides us with literally all types of Machine Learning frameworks we can use to build our projects. To do that we will need to install Anaconda, which takes around 3GB memory space in your computer. But because we won't need all the frameworks, we can use a smaller version on Anaconda, which contains all the necessary tools we need to get started with ML. The smaller version is called MiniConda. Think of Anaconda as a Hardware Store, and MiniConda as a work bench. But to actually get that packages, we will use Conda itself. You can think of Conda as our assistant aka the Package Manager. In python, the package manager is pip, and in MiniConda it is Conda.
![image](https://storage.googleapis.com/replit/images/1621704472798_d2738ad49d2c29b00859d5173d70a49c.png)

## Conda.
#### A workflow for a ML project, might involve creating a project folder, containg our data and the tools we are using. We normally have to use a collection of tools, like pandas, jupyter, etc.. Thus, this collection is called an Environment, which we make using Conda. This is handy for large projects, in which many people are working, and Conda makes it easy to fork the tools.

## MiniConda Setup.
Head over to [MiniConda Docs](https://docs.conda.io/en/latest/miniconda.html) and install MiniConda on your OS.  
After downloading it, head over to you Anaconda Terminal by searching it up, and move to the Desktop directory on Windows by using the command: `chdir Desktop`. Over there, run this command to create a new project folder: `mkdir <name of your project folder. can be anything>`. After that, you need to let conda install a few packages. To do that, just run this command: `conda create --prefic ./env pandas matplotlib numpy scikit-learn` and hit enter. After that, hit y again and you will see a few packages getting installed. Following the package installation, you will recieve a command that you can run to activate your environment.  
`conda activate C:\Users\<user>\Desktop\<file_name>\env`

To install Jupyter Notebook, where we will write all our code, run the conds command: `conda install jupyter`, and launch jupyter by the command: `jupyter notebook`. This will take you to the default browser opened with the jupyter notebook. Create a new Python file and we will be ready to start writing some code.
######  ~~FINALLY~~.

## Testing
To test if all the frameworks are installed, import all of them and run the code:
```py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
```

If no errors pop up, you are good to go.

## Finishing up
To close Jupyter, go to command prompt and hit Ctrl+C, this will close Jupyter. After that if you want to deactivate your environment, run the `conda deactivate` command.

## Conclusion.
This course is just meant to give you a head start and a comprehensive guide to machine learning. If you actually want some coding with pandas, etc, comment down below.
