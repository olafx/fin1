# fin1

Advanced Monte Carlo sampling and model fitting of various exotic option data to various market models.
See `docs/main.pdf` for more info.

---

Some early results of nonlinear fitting via deep neural networks (340 params), for much faster pricing.
Monte-Carlo integrator estimated relative error target is set to 1% (red).
Actual relative error is highly controlled (green), exceeding the target with exponentially vanishing probability.
Actual relative error of deep neural network contains many more outliers, but mean is ~2%.
Main limitations:
- The MC integration is pure MC, so $n^{-1/2}$ scaling. Switch to randomized quasi-Monte Carlo integration. But this requires dimensionality reduction due to the path-dependent stochastic processes, so Brownian bridge sampling. Asymptotically better convergence will allow higher quality data, part of the ML bottleneck.
- Adjust statistical distribution of model and option parameters away from uniform, to produce more of the samples the DNN struggles with.
- Improve the somewhat simplistic model, and hyperparameter optimization.

![early_MC_ML_BSM_E_4](https://raw.githubusercontent.com/olafx/fin0/refs/heads/main/docs/img/early_MC_ML_BSM_E/4.png)

---

Pricing down-and-out European barrier options under the bilateral gamma model.
Pricing done via Monte Carlo sampling with adaptive importance sampling, changing the measure of the process by adjusting the rates.
Pricing is done in efficiently in parallel, written in C++.

![BG_1](https://raw.githubusercontent.com/olafx/fin0/refs/heads/main/docs/img/BG/1.png)
![BG_2](https://raw.githubusercontent.com/olafx/fin0/refs/heads/main/docs/img/BG/2.png)
![BG_3](https://raw.githubusercontent.com/olafx/fin0/refs/heads/main/docs/img/BG/3.png)
![BG_4](https://raw.githubusercontent.com/olafx/fin0/refs/heads/main/docs/img/BG/4.png)
![BG_5](https://raw.githubusercontent.com/olafx/fin0/refs/heads/main/docs/img/BG/5.png)
![BG_6](https://raw.githubusercontent.com/olafx/fin0/refs/heads/main/docs/img/BG/6.png)
![BG_7](https://raw.githubusercontent.com/olafx/fin0/refs/heads/main/docs/img/BG/7.png)
![BG_8](https://raw.githubusercontent.com/olafx/fin0/refs/heads/main/docs/img/BG/8.png)
![BG_9](https://raw.githubusercontent.com/olafx/fin0/refs/heads/main/docs/img/BG/9.png)
![BG_10](https://raw.githubusercontent.com/olafx/fin0/refs/heads/main/docs/img/BG/10.png)
