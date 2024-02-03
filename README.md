Implements a A-B-C-D network. Trains network using backpropagation.

The default configuration file is input.txt. 
The configuration file can be overridden by passing another file path to the command line.
 
If no output file name is specified in the configuration file, the default will be used to save the last known weights.
The default output file is output.txt.

Format of Input File:
R           //R for running, T for training
P           //R for randomized initial weights, P for preloaded initial weights
4           //number of layers
2 100 20 3  //number of units in each layer 
4           //number of training sets
1 1 1 1 0   //training sets 1-4:
1 0 0 1 1
0 1 0 1 1 
0 0 0 0 0
500         //maximum number of iterations 
0.3         //minimum error
3           //learning rate (lambda)  
-1 1        //minimum weight value and maximum weight value
6           //number of iterations to pass before weights are saved
out.txt     //the output file that weights will be saved to
read.txt    //the file that the weights will be read from (if initial weights are preloaded)
