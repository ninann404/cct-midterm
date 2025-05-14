# Cultural Consensus Theory Analysis Report

## Model Structure
For this project, I used PyMC to implement a Cultural Consensus Theory (CCT) model in order to analyze a given dataset of local plant knowledge. The CCT model is used to estimate the competence of the individual participants and the general consensus in the population. In this given dataset, we analyzed 10 different participants who were given 20 questions to evaluate their local plant knowledge. 

## Prior Choices
The prior I chose for informant competence (D) was Beta(2,1) distribution. This prior was chosen since it would allow for increased weight on responses with values greater than 0.5, allowing for better convergence statistics by safely assuming that the participants have some knowledge above random guessing (0.5). For consensus answers (Z), a non-informative Bernoulli(0.5) prior was chosen to represent an equal probability of the participants having some knowledge on the subject, not just randomly guessing.

## Results 
In the Convergence diagnostics summary, we can see that the R-hate values are all at 1.0, which indicate that the model did successfully converge to the same posterior distribution. 

Posterior Mean Competence for Each Informant:
- Informant 0: 0.86
- Informant 1: 0.90
- Informant 2: 0.86
- Informant 3: 0.90
- Informant 4: 0.90
- Informant 5: 0.75
- Informant 6: 0.65
- Informant 7: 0.83
- Informant 8: 0.81
- Informant 9: 0.56

Most competent: Informants 1, 3, and 4 (tied at 0.90)
Least competent: Informant 9 (0.56)

The simple majority vote answer for each question showed that there were differences in 5/20 questions, in which the CCT model consistently estimated a consensus of "1" while the majority actually chose "0". This may be due to the method in which the CCT model weighed the estimated informant competence vs how the majority voting treated everything with equal weight. This means the CCT was able to provide better convergence statistics by accounting for participant competence due to the fact that it gave more weight to more knowledgeable participants. 