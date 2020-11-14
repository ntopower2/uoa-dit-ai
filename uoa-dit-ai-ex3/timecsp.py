import csp


def constraint(A, a, B, b):
    if (A == 't1' and B == 't3'):
        return (a > b)
    elif (B == 't1' and A == 't3'):
        return (b > a)
    elif (A == 't3' and B == 't4'):
        return (b > a)
    elif (B == 't3' and A == 't4'):
        return (a > b)
    elif (A == 't3' and B == 't5'):
        return (a > b)
    elif (B == 't3' and A == 't5'):
        return (b > a)
    elif (A == 't1' and B == 't2') or (A == 't2' and B == 't1'):
        return (abs(a - b) > 60)
    elif (A == 't2' and B == 't4') or (A == 't4' and B == 't2'):
        return (abs(a - b) > 60)
    else:
        print(A, a, B, b)


variables = ['t1', 't2', 't3', 't4', 't5']
domains = {var: [540, 600, 660] for var in variables}
domains['t4'] = [540, 660]
neighs = """t1: t2 t3;t2: t4;t4: t3;t3: t5"""
neighs = csp.parse_neighbors(neighs)
timecsp = csp.CSP(variables, domains, neighs, constraint)
