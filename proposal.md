# Machine Learning Engineer Nanodegree
## Capstone Proposal
Thomas J. Webb  
December 29th, 2018

## Proposal
_(approx. 2-3 pages)_

### Domain Background

Machine learning has been used extensively in classifying music and even in generative music. Generative music are techniques of using software or algorithms to create novel compositions, which can be useful for a variety of purposes, from assisting musicians by creating accompanyments to creating background music that isn't repetitive. Hidden Markov Models (hmms) are a popular way of detecting an underlying state (chord progression) from a monophonic sequence (see [Harmonizing Pop Melodies using Hidden Markov Models](https://luckytoilet.wordpress.com/tag/hidden-markov-model/) and [Data-Driven Recomposition](https://www.cs.princeton.edu/sound/publications/hdphmm_icmc2008.pdf)). Google has also used RNNs to generate novel sequences (see [magenta](https://github.com/tensorflow/magenta/tree/master/magenta)).

Using machine learning techniques with music compositions can be difficult to generalize across genres and contexts. Since it's not difficult to encode domain knowledge about music theory into source code, it can be tempting to front-load these techniques with said domain knowledge. However, pop music often defies the rules and soundtrack music is even more prone to defying them.

### Problem Statement

Predicting novel sequences from an existing sequence of notes serves two major purposes one may encounter when creating user-facing applications with music and audio capabilities, such as music making games:

1. The ability to detect and fix errors in a sequence, whether due to user error (note being off key) or error in pitch detection by the software itself
2. The ability to create novel sequences to accompany user input or to enhance the reusability of premade music sections

Since music is, like all art forms, intended to surprise and not simply be a repeat of rigid rules, some wiggle room should be allowed for accidentals and the most likely note shouldn't allways be the note that is selected in generation. An objective evaluation of whether the model does well enough on the test set can demonstrate that the model is learning _something_ that can be applies, while this analysis should ultimately be coupled with a subjective listening of computer-generated sequences.

### Datasets and Inputs

Midi files are very simple and have been around for a long time. They are functionally like sheet music for software synthesizers (or could even with a midi interface be fed into hardware synthesizers). As they take up much less space than mp3 files they were a popular means of sharing music and embedding music in websites earlier in the internet's history. Tragically, with the cracking down on copyright violations and the rise of easy access to music on youtube and spotify, midi files have become harder to track down. Regardless, midi file collections remain one of the most convenient ways of quickly accessing large amounts of musical information with which to train models.

For this I propose using [this collection](https://www.reddit.com/r/WeAreTheMusicMakers/comments/3ajwe4/the_largest_midi_collection_on_the_internet/), though the code makes no assumption about which selection of midi files are being used. Simply drop midi files in the data directory and the code will crawl through and ingest any valid midi file it finds.

### Solution Statement

Broadly speaking, I propose two models used in concert with each other in order to predict new notes. One, a clustering algorithm that clusters all melodies based on the relative frequencies of the 12 chromatic notes. Two, a supervised learning algorithm that predicts the next note based on its location, the cluster placement of the melody it's in, the beat (placement in time in the measure it's in) and the previous n notes (to be tuned) in the melody. The clusters likely would correspond to different modes and keys.

### Benchmark Model

In the benchmark model, 

### Evaluation Metrics

The overall quality of the algorithm can be assessed a number of ways. One, score the model on a separate testing subset of the data which wasn't involved in the training. This must do substantially better than chance. However, due to the novel and artistic nature of music, it shouldn't be expected to be perfect. If it predicts correctly at least 60% of the time, I'd say that's pretty good.

The distinctiveness of the clusters in the first step can be evaluated by looking at the silhouette score or any measurement of cluster separation. A lower score would signify that the model isn't able to classify music into distinct modes or keys, which could affect the quality of the supervised learning algorithm, which uses the clustering output.

Some more subjective means of evaluating the models can also be done. Visualizing the content of the different clusters could be done to see if they look more or less like different keys or modes. Looking at note sequences generated and the preceding input notes, as well as listening to the sequence, is indispensible in evaluating the model.

### Project Design
_(approx. 1 page)_

In this final section, summarize a theoretical workflow for approaching a solution given the problem. Provide thorough discussion for what strategies you may consider employing, what analysis of the data might be required before being used, or which algorithms will be considered for your implementation. The workflow and discussion that you provide should align with the qualities of the previous sections. Additionally, you are encouraged to include small visualizations, pseudocode, or diagrams to aid in describing the project design, but it is not required. The discussion should clearly outline your intended workflow of the capstone project.