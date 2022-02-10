# Can a simple neural network model predict who's behind keystrokes?
This is a simple experiment, to verify whether a (basic) nn model 
can detect patterns in keystroke timestamps

Data is collected by accessing the keyboard device handler
and recording timestamps every time a key is pressed (w/o key releases)

Currently (first phase) would be to determine an upper limit for the accuracy
by training the model to differentiate between random sequences of timestamps (with frequency in given range)
and real keystroke timestamp sequences

(Linux only)
