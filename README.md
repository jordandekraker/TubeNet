# TubeNet

The broad direction of TubeNet is to create continuously learning AI systems that mimic aspects of biological life (the name is inspired by the 'neural tube' seen in ontogeny and phylogeny). Rather than reinforcing foraging behaviour, which is most commonly done, this network will instead focus on building up representations of the world around it (which could subsequently be used to solve a foraging or other task later on). Thus the primary learning singal will come from predicting sensory signals at the next sampling time. A secondary learning system will be used to ensure the motor system of the network actively explores its environment by reinforcing novelty (or poor prediction performance). 

Part of the motive for this scheme comes from a branch of cognitive neuroscience known as 'predictive coding', the idea that high order cortical regions are constantly trying to supress lower order regions that they predict will selectively become activated, and its only when that prediction fails that there is a strong sensory-driven feed-forward signal.

Another motivation for this project is the idea that the complex representational systems that exist in brains comes about through evolutionary reinforcement, but can also be shaped on a much shorter timescale by the statistics of an environment. For example, one long-term goal is that TubeNet will naturally develop Place- and Grid-cell representations to represent its envoronment, with no built in structure of this sort.


**TubeNet1** chooses fixation location (M+0) over a single image, and then applies a fisheye filter centered at M+0 to produce a new image (S+0). A new fixation (M+1) and corresponding image (S+1) are generated. S+0 and M+1 are vectorized and fed into a small fully connected NN, and S+1 is used as the training signal.

Running this model, we see that TubeNet1 quickly maps an appropriate predicted image to any given input. However, this can be accomplished equally well given only inputs M, and its not clear whether TubeNet1 is using correlations between, for example, features in the periphery of S+0 and M+1 (it may instead simply memorize the appropriate image for each M+1).

Tubenet1.1 adds a reinforcement algorithm to choose new fixation locations from a set of NN outputs, and optimizes S+1 prediction error. This encourages new fixations in poorly learned areas of the image, and will help future versions explore an environment in a more systematic way. 


**TubeNet2** controls a robot car with a webcam stream, and is able to perform simple camera pan/tilt movements in addition to moving the fixation as above. This greatly expands the set of possible input images. Fixation locations are now denoted as M(fix) and M(head), referring to the location of the fisheye center and the head direction. M(head) is randomly changed every 10 iterations, and is not given as an input to the NN. 

This expanded input image set necessitates either a greater capacity (which has not been granted) or greater compression to produce appropriate predictions, which encourages TubeNet2 to make use of heuristics in its prediction. An example heuristic might be the correlation between features in the periphery of S+0 and M(fix)+1. This is tested by placeing TubeNet2 in a new environment, which ??produces reasonable predictions about a new image given small changes in fixation location?? (to be tested still). 


**TubeNet3** provides M(head) as an input to the NN and allows both M(fix) and M(head) to be chosen be the reinforcement algorthim. TubeNet3 also introduced forward/backward/left/right wheel movements (M(loc)). 

Note that M(fix) and M(head) are passed forward at each iteration, but M(loc) passes forward only the locomation taken at that iteration. Thus the state (H(loc)) of the location of TubeNet3 can only be inferred from a history of past locomotions. This is analogous to having proprioceptive feedback, which is normally the case for fixation and head direction in animals but is not the case for relative location in an environment. 

An LSTM is introduced to try and track M(loc) history and infer H(loc), but this inference could also be made by the cues available in S+1 and (M(fix)+0 and/or M(head)+0). Note that since S and M are not explicitly separated the LSTM may also store information about S or M(fix) (which contain temporal autocorrelation because of the reinforcement algorithm introduced in 1.1). ??This produces a more stepwise pattern of explore - learn - explore.?? (to be tested still) 



## From this point on is undeveloped and very hypothetical

**TubeNet4** should have the additional capability of maintaining multiple, distinct representations of environments. This could be framed as a challenge of pattern separation and completion to maintain and access distinct context representations in a memory bank. This conext could be generated in a memory-augmented neural network (e.g. for non-overlapping environments), or in the hidden layer of an LSTM (e.g. if the environments are close, or even result simply from a movement within the same environment). 

TubeNet4 should be able to make predictions for S+1 that go beyond what is accessible in the periphery of image S+0. Similar to the process of imagining. Some of these predictions should be context-dependent.

**TubeNet5** should begin to predict the movements of other bodies within its environment. This is a major challenge is current machine learning. Hopefully it will also follow other animate objects given its now deeply reinforced novelty seeking behaviour.

