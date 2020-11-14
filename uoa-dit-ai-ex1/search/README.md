# Pacman Project 1: Search
The first four questions were answered in `search.py` that contains search algorithms and the following
four were answered in `searchAgents.py`.
## Question 1: Finding a Fixed Food Dot using Depth First Search
Implementing **DFS** based on the general *GRAPH-SEARCH* in AIMA book using LIFO list (stack) and 
holding a dictionary for the parent node in order to backtrack the correct path.
## Question 2: Breadth First Search
Implementing **BFS** based on the general *GRAPH-SEARCH* in AIMA book using FIFO list (queue).
## Question 3: Varying the Cost Function
Implementing **UCS** based on the general *GRAPH-SEARCH* in AIMA book using PriorityQueue and holding 
a dictionary for the parent node and the cost to reach him in order to backtrack the correct path.
## Question 4: A* search
Implementing **A*** based on the general *GRAPH-SEARCH* in AIMA book using PriorityQueue with priority 
function *priorityfunction* combining transition cost and heuristic function value. Again a 
dictionary for the parent node and the cost to reach him in order to backtrack the correct path is 
being used.
## Question 5: Finding All the Corners
Modeling *CornersProblem* as a search problem by modifying search/goal states.
## Question 6: Corners Problem: Heuristic
Solving *CornersProblem* using Manhattan heuristic function.
## Question 7: Eating All The Dots
Solving *FoodSearchProblem* using two heuristic functions: *MazeDistance* and *ManhattanDistance*. Both
of these functions take between 4.5 - 6 seconds to run not causing a timeout. The first is graded by 
5/4 and the second one with 3/4 (but it is faster ofc).
## Question 8: Suboptimal Search
Solving *FoodSearchProblem* using a greedy approach i.e. eating always the closest dot.