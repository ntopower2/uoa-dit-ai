animal(mouse).
animal(gruffalo).
inFront(mouse,gruffalo).
follows(X,Y):-animal(X),animal(Y),inFront(Y,X).
