## Getting it Working

To run the training, put a bunch of midi files in the data directory. I used [this collection](https://www.reddit.com/r/WeAreTheMusicMakers/comments/3ajwe4/the_largest_midi_collection_on_the_internet/).

Also depends on these packages:
mido

## Performance Tips

If it takes too long to run the code blocks in the jupyter notebook for your tastes, there are several easy ways to get a smaller subset of the data. One is changing the folder in the line that parses the midi files:

```for root, dirs, files in os.walk('data/Metal_Rock_rock.freemidis.net_MIDIRip'):```

That can be changed to other subdirectories with smaller amounts of data. Or simply change it to just `'data'` and copy as many or as few midi files as you wish into the directory. Finally, any of the loops that go through the data can work on subsets pretty easily. For example:

```
for midi_file in midi_files:#[:1000]:
    try:
        process_midi_file(midi_file)
    except Exception as e:
        pass # print(e) # this just gives us a bunch of errors from the invalid files in the lot
```

The first line can be changed to

```for midi_file in midi_files[:1000]:```

To only look at the first 1k files. Also this part of the code:

```
for composition in compositions:
        composition_labels, composition_notes = composition.correlations(previous_notes, kmeans_model)
        monophonic_tracks = composition.monophonic_tracks()
        if len(monophonic_tracks) > 0:
            melodies.append((composition_labels, composition_notes, monophonic_tracks))
        labels += composition_labels
        notes += composition_notes

    labels = Normalizer().transform(labels)
    
    X_train, X_test, y_train, y_test = train_test_split(labels, notes, test_size=0.2, random_state=0)
```

Can by sped up by changing `compositions` to `compositions[:100]` and also add the same kind of subscript to labels and notes in the `train_test_split` function. Note that you will drastically alter the kinds of results you get by truncating the data set. It might also help to shuffle these containers before choosing a subset to avoid introducing a bias.

## LICENSE

This is all available per the MIT license, see `LICENSE` but also keep in mind academic honesty when completing your own assignments.
