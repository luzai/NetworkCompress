# NetworkCompress

Inspired by net2net, network distillation.

## Environment
- keras 1.2.2: rather than keras 2.0.1, it seems `resnet.py` depends on keras 1
- backend: theano for its speed on GPU. Also consider tf+cuda-8.0 for debug. 
- Memory: I think `[lib] cnmem=0.1` is enough (for cifar-10).


## TODO list:
- [finish] check the completeness of `net2net.py` `kd.py`  
- [finish] generate transfer data 
- generate soft-taget
- experiments on random generate model
  - [finish] generate random feasible command 
  - check the completeness 
  - run code parallel, find some rule.
- experiments on  comparing two type models:
  - discuss and specify the requirements of these two model
  - left --> right , up --> down
  - diag grow
- verify the utility of soft-target
  - [finish] hard label + transfer label
  - hard label + soft-target (tune hyper-parameter T)
- write `net2branch` function, imitating inception module
- net2deeper for pooling layer
- net2wider for conv layer on kernel size dimension, i.e., 3X3 to 5X5
