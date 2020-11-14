# Pacman Project 2: Multi-Agent Search
All five questions were answered in `multiAgents.py`
## Question 1: Reflex Agent
Using an *evaluationFunction* for ReflexAgent taking into account:
- Manhattan distance of food pellets
- Manhattan distance of each ghost - adversary
- number of food pellets remaining

The resulting function:

![](./equation.svg)
## Question 2: Minimax
Implementing **getAction** for MinimaxAgent using auxiliary function *minimax*.
## Question 3: Alpha - Beta Pruning
Implementing **getAction** for AlphaBetaAgent using auxiliary function *minimaxAB*. 
## Question 4: Expectimax
Implementing **getAction** for ExpectimaxAgent using auxiliary function *expectimax*. 
## Question 5: Evaluation Function
Just like Question 1, a coefficient-based modeling was used for this function.