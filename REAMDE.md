# Overview
This is the official code base for DD-PPO (at least we hope it to be)

# Environmental Setup
## Set up the docker container
1. Clone this repo
2. Go to the root directory of this repo
3. Type ```source ./docker/build.docker.sh``` to build a docker image
4. Type ```source ./docker/run.docker.sh``` to run the docker container
5. Choose your preferred way to work inside the docker container
	1. The author likes to use the docker extension in VSCode **Thumbs up**

## Genesis World
1. Follow the instruction in [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) to install the simulation environment
2. If you are too lazy to open the link, theoretically you could install Genesis just following the two lines of codes:
```
pip install --upgrade pip
pip install git+https://github.com/Genesis-Embodied-AI/Genesis.git
```
3. If strange bugs occur in installation, feel free to debug them on yourself or write them down here

## Diffusion Policy
1. Claims: 
	1. My implementation of diffusion policy is not adapted from the official repo of [Diffusion policy](). Instead, I deployed the codes from the official repo onto the [pytorch-implementation of diffusion model](https://github.com/lucidrains/denoising-diffusion-pytorch), because the latter one is more efficient and straight-forward to read
	2. I adapt the codes a little bit on my own. Not 100% sure they are correct (but >95%). Please feel free to point out any bug, if you discover it
	3. Also, my implementation is not synchronized with the latest version of [pytorch-implementation of diffusion model](https://github.com/lucidrains/denoising-diffusion-pytorch). It should be totally fine, since it's working correctly on my machine so far. If you think there is a strong emergency to synchronize the code base to the latest version, let me know and we could solve out the conflicts together
2. To install diffusion policy, please type the following codes in the terminal:
```
git clone https://github.com/TuXun1999/denoising-diffusion-pytorch.git
```

3. Then, go inside the directory, and install the python package locally from scratch:
```
cd denoising-diffusion-pytorch
pip install . 
```
Or, 
```
cd denoising-diffusion-pytorch
python3 setup.py install
```

4. Install other necessary packages, such as open3d and pickle by 
```
pip install open3d/pickle
```

## Test the codes!
1. At first, type the following the codes to verify that Gensis is working correctly
```
python3 move_obj_env.py
```
You should be able to move the banana using WASD-style virtual joystick. 

2. After you kill program, a small package file called "banana.pkl" will be generated in the local directory. This is the raw dataset for the diffusion policy. 
	1. You don't need to do anything about it. I have already created one for your test in denoising-diffusion-pytorch
3. You can also load a diffusion policy & verify its performance by loading it into the pipeline! This part of codes are commented out. Feel free to do it after you complete the testing of diffusion policy

4. Go inside denoising_diffusion_pytorch, and modify line 70 of test_ddpm_move_cube.py into the correct path of banana.pkl, i.e
```
filename = f"../collected_demos/{obj_type}.pkl"
```

Then, type
```
python3 test_ddpm_move_cube.py
```

5. You would firstly see a window popped out, containing several frames. These are the SE(2) poses of the object in the collected demos

6. Close the window (by pressing q). The diffusion model should start to train

7. After it's complete, you would see another window popped up, containing several frames. These are the SE(2) poses of the object predicted by the trained diffusion policy. Theoretically, they are similar (at least, you should expect the object to move back to the origin!)

8. The whole process may take 5-15 mins

9. After the diffusion model is trained, to evaluate the diffusion policy in our previous simulation platform, copy the following file/folder back to the root directory:
	1. ./results
	2. normalization_stats.pth

10. Go back to the root directory, comment the parts to collect demos & uncomment the parts to test diffusion policy, then you would expect to see the gripper moving the banana back to the origin

11. Feel free to read the codes & Explore on yourself. You should be able to push your commits directly to the repo. Once things are validated, I will clone the repo into my own machine & Test it. If it passed, we could move forward to our next step. 