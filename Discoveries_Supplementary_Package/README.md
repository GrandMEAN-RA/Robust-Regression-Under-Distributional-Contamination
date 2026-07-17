**Supplementary Material**

**Title:**

Robust Regression under Distributional Contamination: A Monte Carlo Study of OLS, LTS, Theil and Bayesian Estimators

**Author:**

Opeyemi B. Sadiku

**Affiliation:**

Independent Researcher, Sagamu, Ogun State, Nigeria

**Overview**

This supplementary material provides additional details, extended simulation tables, and technical specifications supporting the findings presented in the main article. It is intended to offer transparency, reproducibility, and a deeper dive into the sensitivity analysis conducted across the various sample sizes and error-generating environments.

**Data and Code Availability**

As noted in the main manuscript, all simulation code (Python scripts) and GDP/Population data used for empirical analysis are publicly available via the project GitHub repository:

**GitHub Repository:**

<https://github.com/GrandMEAN-RA/Robust-Regression-Under-Distributional-Contamination>

**Repository Organization:**

**Appendix A: KDE Plots of Error Models across Sample Sizes** $$\mathbf{(} \mathbf{n = 1 0 , 3 0 , 5 0 , 1 0 0 , 5 0 0 , 1 0 0 0 )}$$

-   **Figure S1:**

    KDE Plot of Simulated Errors for the Standard Error Models

-   **Figure S2**:

    KDE Plot of Simulated Errors for the Contaminations Error Models

-   **Figure S3**:

KDE Plot of Simulated Errors for the Outliers Error Models

-   **Figure S4**:

KDE Plot of Simulated Errors for the Mixtures Error Models

Note: To ensure visible and comparable densities, the x-axis scale for each figure was truncated as shown below:

-   Figure S1 and S2: n=50 at [-50,50]; n=100 and 1000 at [-40,40]; n=500 at [-10,10]
-   Figure S3: all sample sizes (n) at [-40,40]
-   Figure S4: n50 at [-1,1]; n100 at [-40,40]

**Appendix B: Extended Simulation Results**

-   **Table S1**:

    Simulation Results for Regression Slope Parameter

-   **Table S2**:

    Simulation Results for Regression Slope Parameter

**Appendix C: Empirical Data Analysis**

**Table S3**:

Empirical Data on GDP (Current) and Population (Total) for the Year 2018 and 2024 (Uncontaminated)

**Table S4**:

Empirical Data on GDP (Current) and Population (Total) for the Year 2018 and 2024 (20% contaminated)

**Figure S5**:

Scatter Plot of uncontaminated GDP vs Population data for the year 2024

**Figure S6**:

Scatter Plot of 20% contaminated GDP vs Population data for the year 2024

**Figure S7**:

KDE Plot of residuals of uncontaminated GDP vs Population data for the year 2024

**Figure S8**:

KDE Plot of residuals of 20% contaminated GDP vs Population data for the year 2024

**Figure S9**:

P-P Plot of uncontaminated GDP vs Population data for the year 2024

**Figure S10**:

P-P Plot of 20% contaminated GDP vs Population data for the year 2024

**Appendix D: Exploratory Data Analysis of Simulated Slope Estimates**

**Figure S11**:

KDE Plot of Replicated Beta Estimates for the Standard Normal Cell

**Figure S12**:

KDE Plot of Replicated Beta Estimates for the Standard Lognormal Cell

**Figure S13**:

KDE Plot of Replicated Beta Estimates for the Standard Cauchy Cell

**Figure S14**:

KDE Plot of Replicated Beta Estimates for the Outliers Normal Cell

**Figure S15**:

KDE Plot of Replicated Beta Estimates for the Outliers Lognormal Cell

**Figure S16**:

KDE Plot of Replicated Beta Estimates for the Outliers Cauchy Cell

**Figure S17**:

KDE Plot of Replicated Beta Estimates for the Mixtures Normal Cell

**Figure S18**:

KDE Plot of Replicated Beta Estimates for the Mixtures Lognormal Cell

**Figure S19**:

KDE Plot of Replicated Beta Estimates for the Mixtures Cauchy Cell

**Figure S20**:

KDE Plot of Replicated Beta Estimates for the Contaminations Normal Cell

**Figure S21**:

KDE Plot of Replicated Beta Estimates for the Contaminations Lognormal Cell

**Figure S22**:

KDE Plot of Replicated Beta Estimates for the Contaminations Cauchy Cell

**Appendix E: Code Scripts**

-   main_sim.py – this is the main simulation script
-   simulator.py – a module for dynamically generating random data based on specified model and error distribution
-   estimators.py – a module that contains algorithms for OLS, LTS, Theil’s pairwise median and Bayesian regression estimators and the estimates aggregator function.
-   analyser.py – a module that receives and analyzes simulated estimates data and computes estimators’ performance metrics
-   plotter.py – a module for plotting and visualizing data; basically scatter plots and KDE plots.
-   setup_tools.py – a module that contains simulation iteration setup instructions and parameters.
-   terminal_manager.py – controls and manages the console during code execution
-   writer – writes to .csv file
-   monte-carlo-empirical.ipynb – a jupyter notebook for analysis of empirical data
-   eda.ipynb – a jupyter notebook for exploratory data analysis of replicated simulation data
