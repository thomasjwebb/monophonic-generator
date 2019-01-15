# Machine Learning Engineer Nanodegree
## Capstone Proposal
Thomas J. Webb  
December 29th, 2018

## Proposal

### Domain Background

Machine learning has been used extensively in classifying music and even in generative music. Generative music are techniques of using software or algorithms to create novel compositions, which can be useful for a variety of purposes, from assisting musicians by creating accompaniments to creating background music that isn't repetitive. Hidden Markov Models (hmms) are a popular way of detecting an underlying state (chord progression) from a monophonic sequence (see [Harmonizing Pop Melodies using Hidden Markov Models](https://luckytoilet.wordpress.com/tag/hidden-markov-model/) and [Data-Driven Recomposition](https://www.cs.princeton.edu/sound/publications/hdphmm_icmc2008.pdf)). Google has also used RNNs to generate novel sequences (see [magenta](https://github.com/tensorflow/magenta/tree/master/magenta)). [This blog post](https://medium.com/cindicator/music-generation-with-neural-networks-gan-of-the-week-b66d01e28200) also details the use of C-RNN-GAN to create novel sequences from training data, along with some samples out output.

Using machine learning techniques with music compositions can be difficult to generalize across genres and contexts. Since it's not difficult to encode domain knowledge about music theory into source code, it can be tempting to front-load these techniques with said domain knowledge. However, pop music often defies the rules and soundtrack music is even more prone to defying them.

### Problem Statement

Predicting novel sequences from an existing sequence of notes serves two major purposes one may encounter when creating user-facing applications with music and audio capabilities, such as music making games:

1. The ability to detect and fix errors in a sequence, whether due to user error (note being off key) or error in pitch detection by the software itself
2. The ability to create novel sequences to accompany user input or to enhance the reusability of premade music sections

Since music is, like all art forms, intended to surprise and not simply be a repeat of rigid rules, some wiggle room should be allowed for accidentals and the most likely note shouldn't allways be the note that is selected in generation. An objective evaluation of whether the model does well enough on the test set can demonstrate that the model is learning _something_ that can be applies, while this analysis should ultimately be coupled with a subjective listening of computer-generated sequences.

Here we are treating this as a classification problem. By removing octave information from the equation, we are using environment data, including the recent history of notes to predict the next note. Additionally, we also treat the key and mode as a classification problem as well and use that input as part of the environment for the other classification problem, predicting the next note.

### Datasets and Inputs

Midi files are very simple and have been around for a long time. They are functionally like sheet music for software synthesizers (or could even with a midi interface be fed into hardware synthesizers). As they take up much less space than mp3 files they were a popular means of sharing music and embedding music in websites earlier in the internet's history. Tragically, with the cracking down on copyright violations and the rise of easy access to music on youtube and spotify, midi files have become harder to track down. Regardless, midi file collections remain one of the most convenient ways of quickly accessing large amounts of musical information with which to train models.

For this I propose using [this collection](https://www.reddit.com/r/WeAreTheMusicMakers/comments/3ajwe4/the_largest_midi_collection_on_the_internet/), though the code makes no assumption about which selection of midi files are being used. Simply drop midi files in the data directory and the code will crawl through and ingest any valid midi file it finds.

Midi files contain a collection of _tracks_, each track contains a sequence of midi _messages_. The midi file can contain metadata that applies to all the tracks such as tempo and key signature and time signature. Each message is the same kinds of midi messages that are passed between midi devices - information about notes being pressed, pitch bends, control changes and so on. For our purposes, we only look at note on and note off messages, combining the corresponding on and offs into note events. Each such note event has the following features:

- Start time
- End time
- Velocity
- Pitch (which we convert into two features - octave and note)

We also engineer an additional feature called measure position, which is the start time in terms of how long after the start of the measure it finds itself in. For example, if the song is in 4/4 time and the start time corresponds to the 6th quarter note in the song, then the measure position will be 2. Note that start time and end time is actually in ticks, not seconds, as is customary with midi applications. How many ticks are in a beat is also specified in the midi file.

### Solution Statement

Broadly speaking, I propose two models used in concert with each other in order to predict new notes. One, a clustering algorithm that clusters all melodies based on the relative frequencies of the 12 chromatic notes. Two, a supervised learning algorithm that predicts the next note based on its location, the cluster placement of the melody it's in, the beat (placement in time in the measure it's in) and the previous n notes (to be tuned) in the melody. The clusters likely would correspond to different modes and keys.

For clustering, we propose k means as a good way to determine key and mode. Such information can also be found in the key signature in the midi file, but it is not guaranteed to be present or accurate in a valid midi file. For classifying next note from previous notes, we propose using a linear support vector machine. It is a good algorithm for preventing overfitting and isn't too algorithmically complex for the huge dataset we're dealing with.

### Benchmark Model

I will select a few monophonic tracks from compositions for which the key and mode are known. Then generate a few measures of quarter notes at random, based on probability, based on the probabilities of the existing notes. Then do the same for the model we built. Then compare the results. See if the crude probabilistic generator is any worse at guessing the next note or any worse at generating good sounding melodies. This simple random generation is often build into instruments and music software, such as in arpeggiators and does a good enough job for many musicians' needs. But hopefully we can do a lot better.

### Evaluation Metrics

The overall quality of the algorithm can be assessed a number of ways. One, score the model on a separate testing subset of the data which wasn't involved in the training. This must do substantially better than chance. However, due to the novel and artistic nature of music, it shouldn't be expected to be perfect. The classifier used to categorize tracks by key needs to show sufficiently distinct clusters and that will be demonstrated with a sufficiently high silhouette score (at least .5). If the linear SVC predicts correctly following note correctly at least 60% of the time, I'd say that's pretty good, given the creative and semi-random nature of music. This would be based on checking with the test set.

The distinctiveness of the clusters in the first step can be evaluated by looking at the silhouette score or any measurement of cluster separation. A lower score would signify that the model isn't able to classify music into distinct modes or keys, which could affect the quality of the supervised learning algorithm, which uses the clustering output.

Some more subjective means of evaluating the models can also be done. Visualizing the content of the different clusters could be done to see if they look more or less like different keys or modes. Looking at note sequences generated and the preceding input notes, as well as listening to the sequence, is indispensable in evaluating the model.

### Project Design

Data from out in the wild can be pretty wild. There are inconsistencies and as these aren't official but often made by music fans, they aren't necessarily accurate. The hope is that with a large enough quantity from a good collection, errors can be smoothed out. Polyphony is also a very complex thing and difficult to deal with without employing a lot of music theory like other approaches already tend to use. I wish to focus here just on monophonic sequences. All the valid musical sequences are used to train the clustering algorithm since only the note frequency data is used for that anyway. But the supervised learning will only be done on monophonic melodies. By only focusing on monophonic melodies, I can make some simplifying assumptions. The notes are sequences of values with features (note, octave, start and end positions).

Before checking if melodies are monophonic or not and to be included in that set of data, I will quantize to some arbitrary precision. This reduces the dimensionality of the problem, though it will end up disqualifying many sequences from consideration (two 16th notes in a row will be quantized to the same position, making the sequence polyphonic even if it started out as monophonic). Now each note has an attribute of start position within the measure that is one of only x possibilities, with x being the precision. I'll start with 8th notes as that seems to strike a good balance.

Then for each monophonic sequences, I'll create an array of labels and data. The data is n+2 dimension, with n being the number of prior notes to look at:

1. The cluster classification of the sequences as a whole
2. The start position of the note in the measure it starts in (so 0-7 if we decided to quantize to 8th notes)
3. The previous n notes (note only, no octave, so 0-11)

The labels then are the note that followed those notes, one of 12 possible values. This means that we don't predict octave or rhythm. The techniques I'm using can easily be extended to include other variables, but I think it can be potentially more useful without that complexity and let the timing be dictated by user input or the much simpler music theory that governs rhythm.

I then train an svm (or whichever supervised learning algorithm works well) on this data, then evaluate it and benchmark it. For this model, novel notes are generated by constructing the data for a new note by adding notes one quarter note at a time and copying the previous note's octave value. Then I'll test to see if this does any better than the random probabilistic model.

To allow for entropy, I could ask the model for the probabilities of the different notes instead of simply going by the most likely note, then select randomly, according to the probabilities of the different possibilities. This allows the [small] possibility of accidentals and other surprising note choices. Perhaps it could even result in modulation (the "Truck Driver's Gear Change"). I'll have to play with the model to see what it is and isn't capable of.

I can tweak the parameters by hand to get better results, but it's also possible to loop through a range of possibilities in order to optimize the output. For example, I can select the best cluster number for the first classifier as [described here](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html). If the end result isn't satisfactory (low accuracy or performs no better than benchmark model), then I will look into other models, such as a non-linear SVC or ensemble learning.