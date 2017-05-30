
# Towards Model Parallel

## Mutation Operation

- Bug in Deeper Operation Fixed

| Before | ![2017-05-30_18-55-30](_media/2017-05-30_18-55-30.png) |
| ------ | ---------------------------------------- |
| After  | ![1496141903180](_media/1496141903180.png) |

- More tests on different mutation operation combination and order, we can see accuracy degradation of skip and pool operation. I think we should not stick to operation design, we can use refuse sampling technique, which is easy to implement when using multi processing.

![Screenshot from 2017-05-23 00-35-04](_media/with_pool.png)

![Screenshot from 2017-05-23 00-09-56](_media/with_pool2.png)

## Multi Threading

- Implementation needs to override build-in `fit` function in keras, ref to https://github.com/luzai/NetworkCompress/blob/master/experiments/parallel.py
- The Running Pattern 

![1496142561177](_media/1496142561177.png)

- Test speed

  | Number of Model | 1    | 3    | 5     | 10    | 30   |
  | --------------- | ---- | ---- | ----- | ----- | ---- |
  | Multi Thread    | 52   | 132  | 221   | 447   | 1426 |
  | Sequential      | 54   | 165  | 276   | 556   | 1732 |
  | Speedup         | 1    | 1.25 | 1.248 | 1.243 | 1.21 |

  ![1496142616906](_media/1496142616906.png)

## Multi Process

### Toy Example: 

| Wall Time     | ![Screenshot from 2017-05-29 12-50-47](_media/wall_time.png) |
| ------------- | ---------------------------------------- |
| Relative Time | ![](_media/rela_time.png)                |

### GA Example(Only deeper operation):

- Verify True Speedup!

![Screenshot from 2017-05-29 21-40-08](_media/speedup.png)

- GPU Utilization Factor 

| Single Model | ![Screenshot from 2017-05-29 21-40-08](_media/single.png) |
| ------------ | ---------------------------------------- |
| Multi Model  | ![2017-05-30_19-25-43](_media/multi.png) |


## Other Tips:

- How to monitor GPU status to block training process:

![Screenshot from 2017-05-22 16-05-09](_media/monitor_gpu.png)

