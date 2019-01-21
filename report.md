# Machine Learning Engineer Nanodegree
## Capstone Project
Thomas J. Webb  
January 19th, 2019

## I. Definition

### Project Overview
Machine learning techniques can be very useful in the domain of computer music and audio. Music theory is the study of the general rules music tends to follow and due to the vast knowledge provided by music theory, programmers are able to easily write software that analyzes or even generates music based on these known rules. However, music theory doesn't always encapsulate all that makes music sound pleasing to the ears. Successive generations of pop music will violate old rules and establish their own. By using machine learning, the computer can build a statistical model of what makes a given genre of music tick, perhaps even allowing the computer to discover things about the music the developers are unaware of.

This is not to say using music theory knowledge and using statistical techniques are mutually exclusive. Far from it. In machine learning, _domain knowledge_, information the developer has about the problem in question, can be used to improve the quality of the machine learning techniques. By striking a balance between being overly prejudiced with music theory while not completely avoiding the perks of music theory knowledge, we can make software that can make good predictions about the sorts of things one should expect in a modern composition.

Here, I look into a potential means of using machine learning to spot inaccuracies in inputs, which can be useful as part of a speech to virtual instrument pipeline, or to generate novel musical sequences. Other techniques have been used to this end. Hidden Markov Models (hmms) are a popular way of detecting an underlying state (chord progression) from a monophonic sequence (see [Harmonizing Pop Melodies using Hidden Markov Models](https://luckytoilet.wordpress.com/tag/hidden-markov-model/) and [Data-Driven Recomposition](https://www.cs.princeton.edu/sound/publications/hdphmm_icmc2008.pdf)). Google has also used RNNs to generate novel sequences (see [magenta](https://github.com/tensorflow/magenta/tree/master/magenta)). [This blog post](https://medium.com/cindicator/music-generation-with-neural-networks-gan-of-the-week-b66d01e28200) also details the use of C-RNN-GAN to create novel sequences from training data, along with some samples out output.

### Problem Statement
Predicting novel sequences from an existing sequence of notes serves two major purposes one may encounter when creating user-facing applications with music and audio capabilities, such as music making games:

1. The ability to detect and fix errors in a sequence, whether due to user error (note being off key) or error in pitch detection by the software itself
2. The ability to create novel sequences to accompany user input or to enhance the reusability of premade music sections

Since music is, like all art forms, intended to surprise and not simply be a repeat of rigid rules, some wiggle room should be allowed for accidentals and the most likely note shouldn't allways be the note that is selected in generation. An objective evaluation of whether the model does well enough on the test set can demonstrate that the model is learning _something_ that can be applies, while this analysis should ultimately be coupled with a subjective listening of computer-generated sequences.

Here we are treating this as a classification problem. By removing octave information from the equation, we are using environment data, including the recent history of notes to predict the next note. Additionally, we also treat the key and mode as a classification problem as well and use that input as part of the environment for the other classification problem, predicting the next note.

### Metrics
Ultimately the best metric for evaluating these methods would be subjective. Have a number of musically-inclined people judge the output of the algorithm, whether corrected sequences or sequences with new notes added programmatically, based on how musically pleasing it is. However, that is beyond the scope of this since it requires a statistically appropriately large number of humans making these judgments. Before we can get to the point where it's worth doing such tests, it should clear a more objective test of accuracy first.

To this end, I split the data into a training set and a test set, and score based on how many predictions are correct. This output by itself doesn't tell us enough, however. It's simply a number between 0 and 1. A good baseline would be the simpler to implement benchmark, which is a function that guesses the following note by simply selecting a random note based on the distribution of probabilities in the composition. In other words, our model needs to do _better than chance_, with an educated understanding of the chance. Given the additional overhead of training a model, it should ideally do _significantly_ better than this benchmark to be worth the trouble.

## II. Analysis

### Data Exploration
Midi files are very simple and have been around for a long time. They are functionally like sheet music for software synthesizers (or could even with a midi interface be fed into hardware synthesizers). As they take up much less space than mp3 files they were a popular means of sharing music and embedding music in websites earlier in the internet's history. Tragically, with the cracking down on copyright violations and the rise of easy access to music on youtube and spotify, midi files have become harder to track down. Regardless, midi file collections remain one of the most convenient ways of quickly accessing large amounts of musical information with which to train models.

For this I propose using [this collection](https://www.reddit.com/r/WeAreTheMusicMakers/comments/3ajwe4/the_largest_midi_collection_on_the_internet/), though the code makes no assumption about which selection of midi files are being used. Simply drop midi files in the data directory and the code will crawl through and ingest any valid midi file it finds.

Midi files contain a collection of _tracks_, each track contains a sequence of midi _messages_. The midi file can contain metadata that applies to all the tracks such as tempo and key signature and time signature. Each message is the same kinds of midi messages that are passed between midi devices - information about notes being pressed, pitch bends, control changes and so on. For our purposes, we only look at note on and note off messages, combining the corresponding on and offs into note events. Each such note event has the following features:

- Start time
- End time
- Velocity
- Pitch (which we convert into two features - octave and note)

We also engineer an additional feature called measure position, which is the start time in terms of how long after the start of the measure it finds itself in. For example, if the song is in 4/4 time and the start time corresponds to the 6th quarter note in the song, then the measure position will be 2. Note that start time and end time is actually in ticks, not seconds, as is customary with midi applications. How many ticks are in a beat is also specified in the midi file.

In order to shorten training time, I zeroed in on a subset of the collection that focuses on rock hits one might find on the radio, to be found in the `data/Metal_Rock_rock.freemidis.net_MIDIRip` subfolder. Even with that restriction, it is still a hefty amount of data and still presents a challenge when trying to see the outcome of different tweaks being made. In the optimization step, I'd sometimes temporarily work with 100 or 1000-sized chunks of the data.

### Exploratory Visualization

![silhouette scores](img/silhouette_scores.png "Silhouette Scores")
![note distribution](img/note_distribution.png "Note Distribution in Cluster Centers")

In this section, you will need to provide some form of visualization that summarizes or extracts a relevant characteristic or feature about the data. The visualization should adequately support the data being used. Discuss why this visualization was chosen and how it is relevant. Questions to ask yourself when writing this section:
- _Have you visualized a relevant characteristic or feature about the dataset or input data?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Algorithms and Techniques
Broadly speaking, I propose two models used in concert with each other in order to predict new notes. One, a clustering algorithm that clusters all melodies based on the relative frequencies of the 12 chromatic notes. Two, a supervised learning algorithm that predicts the next note based on its location, the cluster placement of the melody it's in, the beat (placement in time in the measure it's in) and the previous n notes (to be tuned) in the melody. The clusters likely would correspond to different modes and keys.

For clustering, we propose k means as a good way to determine key and mode. Such information can also be found in the key signature in the midi file, but it is not guaranteed to be present or accurate in a valid midi file. For classifying next note from previous notes, we propose using a linear support vector machine. It is a good algorithm for preventing overfitting and isn't too algorithmically complex for the huge dataset we're dealing with.

In this section, you will need to discuss the algorithms and techniques you intend to use for solving the problem. You should justify the use of each one based on the characteristics of the problem and the problem domain. Questions to ask yourself when writing this section:
- _Are the algorithms you will use, including any default variables/parameters in the project clearly defined?_
- _Are the techniques to be used thoroughly discussed and justified?_
- _Is it made clear how the input data or datasets will be handled by the algorithms and techniques chosen?_

### Benchmark
I will select a few monophonic tracks from compositions for which the key and mode are known. Then generate a few measures of quarter notes at random, based on probability, based on the probabilities of the existing notes. Then do the same for the model we built. Then compare the results. See if the crude probabilistic generator is any worse at guessing the next note or any worse at generating good sounding melodies. This simple random generation is often build into instruments and music software, such as in arpeggiators and does a good enough job for many musicians' needs. But hopefully we can do a lot better.


## III. Methodology

### Data Preprocessing
(Talk about how we turn the sequence of notes into arrays of floating point values)

In this section, all of your preprocessing steps will need to be clearly documented, if any were necessary. From the previous section, any of the abnormalities or characteristics that you identified about the dataset will be addressed and corrected here. Questions to ask yourself when writing this section:
- _If the algorithms chosen require preprocessing steps like feature selection or feature transformations, have they been properly documented?_
- _Based on the **Data Exploration** section, if there were abnormalities or characteristics that needed to be addressed, have they been properly corrected?_
- _If no preprocessing is needed, has it been made clear why?_

### Implementation
In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
- _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
- _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_

### Refinement
In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques were used?_
- _Are intermediate and final solutions clearly reported as the process is improved?_


## IV. Results

### Model Evaluation and Validation
In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_

### Justification
In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_


## V. Conclusion

### Free-Form Visualization
In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

### Improvement
In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_

-----------

**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?