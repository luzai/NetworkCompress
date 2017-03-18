# NetworkCompress

Inspired by net2net, network distillation

## environment
- keras 1.2.2: rather than keras 2.0.1, it seems `resnet.py` depends on keras 1
- backend: theano temporarily. Also consider tf+cuda-8.0. Meanwhile, I think `[lib] cnmem=0.1` is enough (for cifar-10).


## TODO list:
- check the completeness of `net2net.py` `kd.py`  
- generate transfer data 
- experiments on random generate model
- experiments on  comparing two type models:
	- left --> right , up --> down
	- diag grow
