import numpy as np

import os
obj_list = os.listdir("../collected_demos")

for obj in obj_list:
    log_msg = "Success rates of " + obj + "-"
    for policy in ["random_g1", "rewarding_g1", "random_g1-to-g2", "rewarding_g1-to-g2", "rewarding_g2"]:
        data = np.load("./results/" + obj  + "_baseline-" + policy + ".npy", "r+")
        success_rates = np.sum(data[:, 2]) / data.shape[0]
        log_msg += policy + ": " + str(success_rates) + " "
    print(log_msg)
