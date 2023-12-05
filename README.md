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

## 
