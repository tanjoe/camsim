# camsim

To run the simulation script:

```
python -m src.camsim.sim
```

Press Enter, and the image along with the calculated truth keypoints will be saved in the output directory.

Then, you can run the verification script and input the number of the image:

```
python -m src.camsim.plot_truth
```

Ideally, all truth keypoints (represented by crosses) should perfectly lie at the centers of the circles in the image. However, due to some unknown bugs, there is currently some small deviation between them.