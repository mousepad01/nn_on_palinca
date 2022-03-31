# Can a simple neural network model predict who's behind keystrokes?
This is a simple experiment, to verify whether a (basic) nn model 
can detect patterns in keystroke timestamps

Data is collected by accessing the keyboard device handler
and recording timestamps every time a key is pressed (w/o key releases)

Currently (first phase) would be to determine an upper limit for the accuracy
by training the model to differentiate between random sequences of timestamps (with frequency in given range)
and real keystroke timestamp sequences

Quick update: with a relatively basic model of up to 8 residual layers with 1d convolutions,
              and about as many dense layers, the accuracy goes up to ~95% 
              on the binary classification: human data vs random
              (short documentation will be posted soon)

Another update: ~75% validation accuracy on human vs human (2 classes),
                with supervised contrastive learning and custom inception blocks

(Linux only)
