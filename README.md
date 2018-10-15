# TubeNet

The broad direction of TubeNet is to create continuously learning AI systems that mimic aspects of biological life (the name is inspired by the 'neural tube' seen in ontogeny and phylogeny). Rather than reinforcing foraging behaviour, which is most commonly done, this network will instead focus on building up representations of the world around it (which could subsequently be used to solve a foraging or other task later on). Thus the primary learning singal will come from predicting sensory signals at the next sampling time. A secondary learning system will be used to ensure the motor system of the network actively explores its environment by reinforcing novelty (or poor prediction performance). 

Part of the motive for this scheme comes from a branch of cognitive neuroscience known as 'predictive coding', the idea that high order cortical regions are constantly trying to supress lower order regions that they predict will selectively become activated, and its only when that prediction fails that there is a strong sensory-driven feed-forward signal.

TubeNet1.0 controls a robot car with a webcam stream, and is able to predict some basic principles, like how a given saccade or a camera pan will change its sensory input. At this point its still not clear whether its memorizing every possible fixation location or generalizing between images, for example, by mapping a feature from the periphery of one image (n=0) to the centre of the next image following a saccade to that feature (n=1). 

**H1:** Training over a long time and in many different environments should teach the network that correlations between image n=0 and n=1 are more reliable than correlations between a given motor action and image n=1.

<pending qualitative demonstrations that this is true>


TubeNet2.0 should have the additional capability of maintaining multiple, distinct representations of environments that it can draw on to make predictions about image n=1 rather than drawing only on the features available from image n=0. This could be framed as a challenge of pattern separation and completion to maintain and access distinct context representations in an external memory bank. This conext could be generated in a memory-augmented neural network (e.g. for non-overlapping environments), or in the hidden layer of an LSTM (e.g. if the environments are close, or even result simply from a movement within the same environment). Another consideration is that the generalizability of a given representation might be dependent on how deep it is (e.g. high-level or compressed features may be shared across environments more than many non-compressed low-level conjunctions of features). 

TubeNet3.0 should be able to represent environments in a viewpoint-invariant manner, a prerequisite for navigating within them fluidly. This is currently a major hurdle in deep learning, but it might arise naturally from Tubenet2.0... will need to do more research. 

TubeNet4.0 should begin to predict the movements of other bodies within its environment.
