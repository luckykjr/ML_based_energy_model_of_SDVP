# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea  # import geatpy
from MyProblem import MyProblem  # 导入自定义问题接口


if __name__ == '__main__':
    M = 1                      # Set the number of objects.
    problem = MyProblem(M)     # Instantiate MyProblem class
    # Instantiate a algorithm class.
    algorithm = ea.soea_SEGA_templet(problem,
                                      ea.Population(Encoding='RI', NIND=20),  # Set 100 individuals.
                                      MAXGEN=50,  # Set the max iteration number.
                                      logTras=1)  # Set the frequency of logging. If it is zero, it would not log.
    # Do the optimization
    res = ea.optimize(algorithm, verbose=False, drawing=1, outputMsg=True, drawLog=True, saveFlag=True)
