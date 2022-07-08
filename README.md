![FARM_LOGO_rgb_black](https://user-images.githubusercontent.com/63424217/178042559-7f4c1b33-cd0f-4bd4-9a31-22e6f0d80dd8.png)
# FARM
Feasible Actuator Range Modifier (FARM) Plugin for RAVEN

# Authors:
- Haoyu Wang and Roberto Ponciroli, Argonne National Laboratory
- Andrea Alfonsi, Idaho National Laboratory

# Introduction
Feasible Actuator Range Modifier (FARM) is a RAVEN (https://github.com/idaholab/raven) plugin designed to solve the 
supervisory control problem in Integrated Energy System (IES) project. FARM utilizes the linear state-space representation 
(A,B,C matrices) of a model to predict the system state and output in the future time steps, and adjust the actuation variable 
to avoid the violation of implicit thermal mechanical constraints.

# Organization
This plugin folder is organized as follows:
- src folder containing the two officially supported plugin source code:
    - RefGov_parameterized_SIMO.py
    - RefGov_unparameterized_SIMO.py
- tests folder containing the tests that are needed to test the plugin linked with RAVEN:
    - test_RefGov_para_xmlABC.xml
    - test_RefGov_unpara_xmlABC.xml
- doc folder containing the plugin manual in LatEX:
    - run "bibtex user_manual" and then "pdflatex user_manual" to get the latest version of user manual.

# License
FARM itself is licensed as follows:

Copyright 2021 UChicago Argonne, LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  https://www.apache.org/licenses/LICENSE-2.0.txt

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
