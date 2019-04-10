'''
Demonstrating how you can load a midi file using
the new functionality in chorales. Right now the
assumption is that the midi files have been created
in a similar format to what we've seen with our data.

No guarantee what will happen if you happen to use
a random midi file.
'''

import chorales
import numpy as np

choice = 33
fname = './chorale_train%i.mid'%choice

chorale = chorales.encoded_train[choice]
print('Saving an example chorale to disk...\n')
chorales.save_to_midi( chorale, fname, verbosity=1 )
print('\ndone.')

print('Loading this file...')
reloaded = chorales.load_from_midi(fname)
print('done.')

print('\n')
print('First four attacks of original chorale:')
for note in chorale[:4]:
    print(note)

print('\nFirst four attacks of the loaded chorale:')
for note in reloaded[:4]:
    print(note)
