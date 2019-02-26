# TubeNet

The broad direction of TubeNet is to create continuously learning AI systems that mimic aspects of biological life (the name is inspired by the 'neural tube'). Rather than blindly reinforcing a behaviour, this network will instead focus on building up representations of the world around it (which could subsequently be used to solve a foraging or other task later on). Thus the primary learning singal will come from predicting sensory signals at the next sampling time. A secondary learning system will be used to ensure the motor system of the network actively explores its environment by reinforcing novelty (or poor prediction performance). 

Part of the motive for this scheme comes from a branch of cognitive neuroscience known as 'predictive coding', the idea that high order cortical regions are constantly trying to supress lower order regions that they predict will selectively become activated, and its only when that prediction fails that there is a strong sensory-driven feed-forward signal.

Another motivation for this project is the idea that the complex representational systems that exist in brains comes about through evolutionary reinforcement, but can also be shaped on a much shorter timescale by the statistics of an environment. For example, one long-term goal is that TubeNet will naturally develop Place- and Grid-cell representations of its envoronment, with no built in structure of this sort.

The basic architecture will consist of feeding sensory (S) and motor (M) signals into a neural network (TubeNet). A physics engine will simulate or carry out M, and then acquire a corresponding new S (Starget), which will be used as a training signal for the TubeNet. A secondary training signal will reinforce M based on some desired behaviour (Mtarget).

![Alt text](https://github.com/jordandekraker/TubeNet/tree/master/diagrams/BasicLayout.jpg "Optional title")

