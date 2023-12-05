# ex no.1
## Program 1: write a prolog in prolog calculate addition of two number. 

Code : sum(X,Y):- 

S is X+Y, 

write('Sum is: '),write(S). 

output: 

## Program 2: write a prolog in prolog to find maximum of two number. 

Code: max(X,Y):- 

 X=Y, 

 write('both are equal') 

 ; 

 X>Y, 

 Z is X, 

 write(Z) 

 ; 

 Z is Y, 

 write(Z). 

output:

## Program 3: write a prolog in prolog that take number N from the user and count from N to 

10. 

Code : count(11). 

count(N):- 

 write(N),nl, 

 X is N+1, 

 count(X). 

output: 

## Program 4: write a prolog in prolog that take number N from the user and count from N to 1. 

Code: loop(0). 

loop(N):-N>0, 

 write(N),nl, 

 X is N-1, 

 loop(X). 

output:

## Program 5: write a prolog in prolog that take number N from the user calculate factorial of 

number. 

Code: factorial(0, 1). 

factorial(N, X) :- 

N > 0, 

Y is N - 1, 

factorial(Y, Z), 

X is Z * N. 

Output: 

## Program 6: write a prolog in prolog that take number N from the user calculate square of 

number from N to 20 and display it. 

Code: squares(21). 

squares(N) :- 

 Y is N * N, 

 write(Y), nl, 

 M is N + 1, 

 squares(M). 

output:

# EX NO.2

## Program 1: write a program in prolog to solve N X N queen problem. 

Code: queen(N,Queens):- 

range(1, N, Rows), 

permutation(Rows, Queens), 

safe(Queens). 

range(Start,End, [Start|Rest]) :- 

Start =<End, 

NewStart is Start +1, 

range(NewStart, End, Rest). 

range(End, End, []). 

safe([]). 

safe([Queen|Queens]):- 

safe(Queens, Queen, 1), 

safe(Queens). 

safe([], _, _). 

safe([OtherQueen|Queens], Queen, Offset):- 

Queen=\= OtherQueen + Offset, 

Queen=\= OtherQueen - Offset, 

NewOffset is Offset + 1, 

safe(Queens, Queen, NewOffset). 

Output:

## Program 2 write a program in python to solve N X N queen problem. 

Code: result = [] 

def isSafe(board, row, col): 

 for i in range(col): 

 if (board[row][i]): 

 return False 

 i = row 

 j = col 

 while i >= 0 and j >= 0: 

 if(board[i][j]): 

 return False 

 i -= 1 

 j -= 1 

 i = row 

 j = col 

 while j >= 0 and i < n: 

 if(board[i][j]): 

 return False 

 i = i + 1 

 j = j - 1 

 return True 

def solveNQUtil(board, col): 

 if (col == n): 

 v = []
 
 for i in board: 

 for j in range(len(i)): 

 if i[j] == 1: 

 v.append(j+1) 

 result.append(v) 

 return True 

 res = False 

 for i in range(n): 

 if (isSafe(board, i, col)): 

 board[i][col] = 1 

 res = solveNQUtil(board, col + 1) or res 

 board[i][col] = 0 

 return res 

def solveNQ(n): 

 result.clear() 

 board = [[0 for j in range(n)] 

 for i in range(n)] 

 solveNQUtil(board, 0) 

 result.sort() 

 return result 

n = 8 

res = solveNQ(n) 

for i in range(len(res)): 

 print("Solution ",i+1," : ",res[i])

# EX NO.4

## 1. Write a Program in Prolog to solve any problem using Best First Search. Answer:

% Define your graph with weighted edges and heuristics. % Replace these with your actual graph and heuristics.
edge(a, b, 2). edge(b, c, 3). edge(b, d, 4). edge(c, e, 5). edge(d, e, 1). edge(e, f, 2). 

% Define heuristic values (straight-line distances to the goal). heuristic(a, 6). % Replace with your specific values. heuristic(b, 5). heuristic(c, 4). heuristic(d, 3). heuristic(e, 2). heuristic(f, 0). 

% Define a predicate to calculate the total estimated cost. total_cost(Node, Path, Cost) :- path_cost(Path, PCost), heuristic(Node, HCost), Cost is PCost + HCost. path_cost([_], 0). path_cost([A, B | Tail], Cost) :- 
edge(A, B, EdgeCost), path_cost([B | Tail], RestCost), Cost is EdgeCost + RestCost. % Define Best-First Search algorithm. best_first_search(Start, Goal, Path) :- best_first_search_internal([ [Start] ], Goal, RevPath), reverse(RevPath, Path). best_first_search_internal([ [Goal | Path] | _ ], Goal, [Goal | Path]). best_first_search_internal([ [Node | Path] | Rest ], Goal, Result) :-

findall([Next, Node | Path], (edge(Node, Next, _), not(member(Next, Path))), NewPaths), append(Rest, NewPaths, AllPaths), predsort(compare, AllPaths, SortedPaths), best_first_search_internal(SortedPaths, Goal, Result). % Comparison function for sorting paths based on total cost. compare(Result, [_, _, Path1], [_, _, Path2]) :-

total_cost(Path1, Path1, Cost1), total_cost(Path2, Path2, Cost2), compare_paths(Cost1, Cost2, Result).
compare_paths(Cost1, Cost2, Result) :-

(Cost1 < Cost2 -> Result = (<);

Cost1 > Cost2 -> Result = (>);

Result = (=)). % Example usage:

% To find the path, call best_first_search(StartNode, GoalNode, Path). % Replace StartNo

## 2. Write a Program in Python to solve any problem using Best First Search. Answer:

from queue import PriorityQueue

def best_first_search(graph, start, goal):

frontier = PriorityQueue()

frontier.put(start) # Use a priority queue with the initial node

came_from = {} # Dictionary to store the best path

came_from[start] = None

while not frontier.empty():

current = frontier.get()

if current == goal:

return reconstruct_path(came_from, current)

for neighbor, weight in graph.get(current, []):

if neighbor not in came_from:

came_from[neighbor] = current

frontier.put(neighbor)

return None

def reconstruct_path(came_from, current):

path = []

while current:

path.insert(0, current)

current = came_from[current]

return path

# Example usage:

graph = {

'a': [('b', 2)],

'b': [('c', 3), ('d', 4)],

'c': [('e', 5)],

'd': [('e', 1)],

'e': [('f', 2)], }

start_node = 'a' goal_node = 'f' path = best_first_search(graph, start_node, goal_node)

if path:

print(f"Shortest path from {start_node} to {goal_node}: {path}")

else:

print(f"No path found from {start_node} to {goal_node}")

Output:

#EX NO.5
