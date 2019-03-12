'''
Purpose: load the pickle file containing the Bach chorale data.
This script should be restricted to this focus, so should
only be rarely (if ever) modified.
12 February 2019
Colorado State University
'''

import pickle
import numpy as np

# Assume the data lies in the same folder as this
# script. If it doesn't, modify this string to
# indicate the correct path.
prefix = './'

try:
    with open(prefix + 'JSB_Chorales.pickle','rb') as f:
        all_data = pickle.load(f)
except:
    raise Exception('Loading the chorales pickle file failed. Please ensure the file is in the proper folder.')

#
# Data is structured as a dictionary broken
# in to training, testing and validation data.
# Unpack into separate variables so that
# the data can be called from import by
# doing, e.g. chorales.train[0].
#

encoded_test = all_data['test']
encoded_train = all_data['train']
encoded_valid = all_data['valid']

test = []
for encoded_chorale in encoded_test:
    chorale = np.zeros( (88,len(encoded_chorale)), dtype=float)

    for i,beat in enumerate(encoded_chorale):
        # According to Patrick, notes start at 21, so shift down.
        beat_shft = np.array(beat, dtype=int) - 21
        chorale[beat_shft,i] = 1.
    #
    test.append( chorale )
#

train = []
for encoded_chorale in encoded_train:
    chorale = np.zeros( (88,len(encoded_chorale)), dtype=float)

    for i,beat in enumerate(encoded_chorale):
        # According to Patrick, notes start at 21, so shift down.
        beat_shft = np.array(beat, dtype=int) - 21
        chorale[beat_shft,i] = 1.
    #
    train.append( chorale )
#

valid = []
for encoded_chorale in encoded_valid:
    chorale = np.zeros( (88,len(encoded_chorale)), dtype=float)

    for i,beat in enumerate(encoded_chorale):
        # According to Patrick, notes start at 21, so shift down.
        beat_shft = np.array(beat, dtype=int) - 21
        chorale[beat_shft,i] = 1.
    #
    valid.append( chorale )
#

###############################################
# 
# Internal parameters
#

# Used for one-hot-encodings
_note_encoding = {'A':0, 'A#':1, 'B-':1, 'B':2, 'B#':3,
                            'C':3, 'C#':4, 'D-':4, 'D':5, 'D#':6, 
                            'E-':6, 'E':7, 'E#':8, 'F':8, 'F#':9,
                            'G-':9, 'G':10, 'G#':11, 'A-':11}

_chord_encoding1 = {'major':0, 'minor':1, 'other':2}


###############################################

def get_chord(encoded_sequence):
    '''
    Given a sequence of integers indicating positions of hit notes,
    return the string indicating the corresponding chord.
    Utilizes the music21 package. You need to install this.
    '''
    try:
        import music21
    except:
        raise ImportError('You need to install the music21 package to use this function.')
    #

    if len(encoded_sequence)==88:
        # need to encode!
        encoded_sequence = encode(encoded_sequence)
    #

    # This package is very finnicky - only native python integers are supported.
    cleaned_seq = [int(t) for t in encoded_sequence]

    m21chord = music21.chord.Chord(cleaned_seq)

    chordname = m21chord.pitchedCommonName

    return chordname
#

def chord_type(note):
    '''
    Purpose: identify a chord as being either "major", "minor", or "other".
        This is based on a string comparison of the chord's name 
        returned from get_chord(). These would result in a 3-class problem.

    Inputs:
        note - either an encoded (length 3 or 4 tuple) or decoded note (binary array length 88).
            (yes, this is a misnomer)
    Outputs:
        label - string; one of "major", "minor", or "other".
    '''
    chordname = get_chord(note)
    for option in ['major','minor']:
        if option in chordname.lower():
            return option
    #
    return 'other'  # only reached if none of the options are seen
#

def chord_root(note):
    '''
    Purpose: identify the root of a chord; these can be notes "A" through "G", possibly 
        with a suffix indicating flat "-" or sharp "#". These would result in a 12-class problem
        in theory -- I'm not 100% sure if there are issues with "synonyms" (e.g., A# and B-).

    Inputs:
        note - either an encoded (length 3 or 4 tuple) or decoded note (binary array length 88).
            (yes, this is a misnomer)
    Outputs:
        label - string; one of "major", "minor", or "other".
    '''
    try:
        import music21
    except:
        raise ImportError('You need to install the music21 package to use this function.')
    #

    if len(note)==88:
        enc_note = encode(note)
    else:
        enc_note = note
    #
    ch = music21.chord.Chord( enc_note )
    root = ch.root().name
    return root
#

def one_hot_encode(label_input):
    '''
    Purpose: one-hot-encode the input string or list based on its type. 
        If the input looks like a single note, then it is encoded 
        in a 12-dimensional vector, obeying
            A  -> [1,0,0,...,0]
            A# -> [0,1,0,...,0]
        and so on.

        If the input is one of "major", "minor", or "other", 
        then it is encoded in a 3-dimensional vector, specifically
            major -> [1,0,0]
            minor -> [0,1,0]
            other -> [0,0,1]

    Inputs:
        label_input : a string, one of the types above.
    Outputs:
        label : a numpy *column* vector, defined according to the mapping above.
    '''
    import numpy as np

    if any( [isinstance(label_input,list), isinstance(label_input,np.ndarray)] ):
        # multiple inputs; sequentially process them and return
        # the result as an array.
        output = np.hstack( [one_hot_encode(l) for l in label_input])
        return output
    #

    if len(label_input)<=2:
        # Assumed to be a single note.
        output = np.zeros((12,1))
        output[_note_encoding[label_input]] = 1.
    else:
        output = np.zeros((3,1))
        output[_chord_encoding1[label_input]] = 1.
    #
    return output
#

def encode(note):
    '''
    Purpose: encode a single 88-vector into corresponding locations
        suitable to be used by music21.
    Inputs:
        note - a list-like of length 88 with 0-1 entries.
    Outputs:
        enote - a list of python native integers indicating locations
            of key presses, shifted by 21
    '''
    import numpy as np
    locs = np.where(note)[0]
    enote = [int(21+l) for l in locs]
    return enote
#

def save_to_midi(notes,filename,verbosity=0):
    '''
    Purpose: save the input as a midi file.
    Input:
        notes : A collection of notes; can either be "encoded" or "decoded" forms:
            Encoded: tuples/lists of integers between 21 to 109(?).
            Decoded: an array of dimension 88 by T, where T is the length of the song,
                and the entries are binary 0 or 1 indicating playing the corresponding note.
        filename : string, the name of the file to be saved. By convention,
            end the file in ".mid" to indicate a MIDI file.
        verbosity : integer indicating the level of output. Default: 0
    Output: None.
    '''
    try:
        import music21
    except:
        raise ImportError('You need to install the music21 package to use this function.')
    #
    import numpy as np

    dims = np.shape(notes)
    if len(dims)==1:
        # inferred to be the encoded format.
        decoded = False
        T = dims[0]
    else:
        # inferred to be decoded format.
        width,T = dims
        decoded = True
    #

    if verbosity>0: print('Encoding notes... ', end="")
    if decoded:
        enc_notes = [encode(note) for note in notes.T]
    else:
        # re-cast just in case....
        enc_notes = [[int(n) for n in note] for note in notes]
    #
    if verbosity>0: print('done.')

    # Basic idea: set up a music21 stream, then append the
    # chords one at a time. Then use the .write() function.
    if verbosity>0: print('Populating music21 Stream()... ', end='')
    m21stream = music21.stream.Stream()
    for en in enc_notes:
        if len(en)>0:
            # Are there any notes played?
            m21stream.append( music21.chord.Chord(en) )
        else:
            # Otherwise, append a rest instead.
            m21stream.append( music21.note.Rest() )
        #
    #
    if verbosity>0: print('done.')

    # save
    if verbosity>0: print('Saving %s to disk...'%filename, end='')
    m21stream.write('midi', filename)
    if verbosity>0: print('done.\n')



    return
#
