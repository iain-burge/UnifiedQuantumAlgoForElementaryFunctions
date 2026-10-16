
import numpy as np
from functools import cache
import matplotlib.pyplot as plt
import tests as test

#=============================================================================#
#===Subroutines/Helper Functions==============================================#
#=============================================================================#
def bitshift(x: int, shift: int, n_bits: int) -> int:
    '''
    Does right bitshifting in two's complement.
    Note: The leftmost bit is copied leftwards, i.e., 10100>>1 == 11010.

    x      (int): Value to be bitshifted.
    shift  (int): How far the value is shifted right, shift is non-negative.
    n_bits (int): Number of bits representing x.

    return (int): Right bitshifted x.
    '''
    if shift > n_bits: return 0

    x = x%(2**n_bits)
    leftBit = (x&(1<<(n_bits-1)))>>(n_bits-1)
        
    x  = x>>shift
    x += (((1<<shift)-leftBit)<<(n_bits-shift))%(2**n_bits)

    return x

def add(
        x: int, y: int, n: int = 8, shift: int=0
    ) -> tuple[int, int]:
    '''
    Bitshifts y in two's complement and adds result to x.
    Should be unitary.
    Note: when bitshifting right, copies most significant bit left.
    
    x          (int): Number to be added to
    y          (int): Number adding into x
    shift      (int): Bitshift applied to y before addition
    n          (int): Number of bits
    '''
    N = 2**n
    x%=N; y%=N;

    x += bitshift(y, shift, n)

    return x%N, y%N

def fib(i:int) -> int: 
    '''
    Fibonacci numbers { 1,1,3,5,8,13, ... }

    i (int): Index of Fibonacci number

    return (int): i^th Fibonacci number
    '''
    return int(np.round(
        1/(5**.5) * (((1+5**.5)/2)**(i+1)-((1-5**.5)/2)**(i+1))
    ))


def mult(
        x: int, aux: int, n: int, ell: int, c: bool=False, _debug: bool = False
    ) -> tuple[int, int]:
    '''
    In place multiplication integer by 1+(-1)^c*2^(-ell) with some error depending on 
    how clean the auxiliary registers are.
    Note: the aux register can gain some error

    x     (int): Value being multiplied
    aux   (int): auxiliary register
    n     (int): Number of bits representing x
    ell   (int): x is multiplied by 1 + ((-1)^c * 2^(-ell))
    c     (bool): x is multiplied by 1 + ((-1)^c * 2^(-ell))

    return (tuple[int, int, int]):
        newX (int): the new value for x
        newAux (int): New value for aux, should equal to init value
    '''

    numIter = 2*int(.5*np.sqrt(5)*n/((1+np.sqrt(5))/2)**(ell))
    if ell>=n:
        return x, aux

    if _debug: print(f"in:\t[{x}, {aux}]")
    aux, x  = add(aux, x, n=n)
    if _debug: print(f"init_aux:\t[{x}, {aux}]")


    for i in range(numIter, -1, -1):
        if ell*fib(i) >= n: continue
        if i%2 == 0:
            if c: x = bitwiseNOT(x, n)
            elif fib(i)%2==0: x = bitwiseNOT(x, n)
            x, aux = add(x, aux, n, ell*fib(i))
            if c: x = bitwiseNOT(x, n)
            elif fib(i)%2==0: x = bitwiseNOT(x, n)
            if _debug: print(f"bounce(aux->x):\t[{x}, {aux}]")
        else:
            if c: aux = bitwiseNOT(aux,n)
            elif fib(i)%2==0: aux = bitwiseNOT(aux, n)
            aux, x = add(aux, x, n, ell*fib(i))
            if c: aux = bitwiseNOT(aux,n)
            elif fib(i)%2==0: aux = bitwiseNOT(aux, n)
            if _debug: print(f"bounce(x->aux):\t[{x}, {aux}]")

    aux = bitwiseNOT(aux, n)
    aux, x = add(aux, x, n)
    aux = bitwiseNOT(aux, n)
    if _debug: print(f"clear_aux:\t[{x}, {aux}]")

    return x, aux

def div(
        x: int, aux: int, n: int, ell: int, c: bool=False, _debug: bool = False
    ) -> tuple[int, int]:
    '''
    In place division of integer by 1+(-1)^c*2^(-ell) with some error depending on 
    how clean the auxiliary registers are.
    Note: the aux register can gain some error

    x     (int): Value being multiplied
    aux   (int): auxiliary register
    n     (int): Number of bits representing x
    ell   (int): x is multiplied by 1 + ((-1)^c * 2^(-ell))
    c     (bool): x is multiplied by 1 + ((-1)^c * 2^(-ell))

    return (tuple[int, int, int]):
        newX (int): the new value for x
        newAux (int): New value for aux, should equal to init value
    '''

    numIter = 2*int(.5*np.sqrt(5)*n/((1+np.sqrt(5))/2)**(ell))
    if ell>=n:
        return x, aux

    if _debug: print(f"in:\t[{x}, {aux}]")
    aux, x  = add(aux, x, n=n)
    if _debug: print(f"init_aux:\t[{x}, {aux}]")


    for i in range(numIter+1):
        if ell*fib(i) >= n: continue
        if i%2 == 0:
            x = bitwiseNOT(x,n)
            if c: x = bitwiseNOT(x, n)
            elif fib(i)%2==0: x = bitwiseNOT(x, n)
            x, aux = add(x, aux, n, ell*fib(i))
            x = bitwiseNOT(x,n)
            if c: x = bitwiseNOT(x, n)
            elif fib(i)%2==0: x = bitwiseNOT(x, n)
            if _debug: print(f"bounce(aux->x):\t[{x}, {aux}]")
        else:
            aux = bitwiseNOT(aux,n)
            if c: aux = bitwiseNOT(aux,n)
            elif fib(i)%2==0: aux = bitwiseNOT(aux, n)
            aux, x = add(aux, x, n, ell*fib(i))
            aux = bitwiseNOT(aux,n)
            if c: aux = bitwiseNOT(aux,n)
            elif fib(i)%2==0: aux = bitwiseNOT(aux, n)
            if _debug: print(f"bounce(x->aux):\t[{x}, {aux}]")

    aux = bitwiseNOT(aux, n)
    aux, x = add(aux, x, n)
    aux = bitwiseNOT(aux, n)
    if _debug: print(f"clear_aux:\t[{x}, {aux}]")

    return x, aux


def isneg(value: int, n_bits: int) -> bool:
    '''
    Returns true if the value is negative in two's complement

    value  (int): The value being assessed
    n_bits (int): Number of bits    

    return (bool): True if value is negative
    '''
    return (value & (1<<(n_bits-1))) != 0

def bitwiseNOT(x: int, n: int):
    return (~x) % (1<<n)

def bitwiseCNOT(c: bool, x: int, n: int):
    if c: return (~x) % (1<<n)
    return x % (1<<n)

#=============================================================================#
#===Printing Functions========================================================#
#=============================================================================#
def intToReal(
        xs: Union[int,list[int]], n: int, point: int
    ) -> Union[float,list[float]]:
    listmode = (type(xs)==type([]))
    if not listmode: xs = [xs]
    results = []
    for x in xs:
        signed = x if x<(1<<(n-1)) else -bitwiseNOT(x,n)-1
        results.append(signed/(1<<(n-point)))
    return results if listmode else results[0]

def fixedpointToStr(x: int, n: int, point: int) -> str:
    left  = f'{x>>(n-point):0{point}b}'
    right = f'{x&((1<<(n-point))-1):0{n-point}b}'
    # print(f'{:0{8}b}')

    if point==0: return '.'+right
    if point==n: return left
    return left+'.'+right

#=============================================================================#
#===Unified CORDIC Algorithm==================================================#
#=============================================================================#
def CORDIC(x:float,y:float,z:float,m:int,mode:int,
       depth:int, n: int = 10, point: int = 2, debug: bool = False
    ):
    ''' 
    Reversable CORDIC implementation roughly based on
    A unified algorithm for elementary functions, J. S. WALTHER, 
    Spring Joint Computer Conference, 1971

    Also, uses the double rotation trick from (and rough alg for arcsine),
        Computing Functions cos^-1 and sin^-1 Using Cordic
        IEEE Transactions on Computers, VOL. 42, NO. 1, January 1993          
        Christophe Mazenc, Xavier Merrheim, and Jean-Michel Muller


    d_func is traditionally equal to sign(z) or -sign(y)
    m is in {-1:'hyperbolic', 0:'linear', 1:'circular', 2:'identity'}
    mode in {0:'rot', 1:'vec', 2:'arc', 3: 'identity'}
    '''
    d    = [] #bool array
    aux  = 0

    norm = 1<<(n-point)
    vecs = [np.array([x,y])/norm] #list of vecs for debugging
    
    if m==-1: e_func = lambda k: np.atanh(2**(-k))
    if m== 0: e_func = lambda k: 2**(-k)
    if m== 1: e_func = lambda k: np.atan( 2**(-k))
    if m== 2: e_func = lambda k: 0

    if mode == 0: d_func=lambda a,b,c: isneg(c,n)
    if mode == 1: d_func=lambda a,b,c: not isneg(b,n)
    if mode == 2: 
        d_func=lambda a,b,c: (
            (isneg(a,n) and isneg(bitwiseNOT(bitwiseNOT(c,n)+b,n),n))
            ^ isneg(a,n) ^ isneg(bitwiseNOT(bitwiseNOT(c,n)+b,n),n)
            ^ (isneg(a,n) and isneg(b,n))
        )
        e_func = lambda k: 0
    if mode == 3: d_func=lambda a,b,c: False

    shiftSequence = []
    k = 1; repeat = 4
    while k < depth:
        if k > repeat: k-=1; repeat=3*repeat+1
        shiftSequence.append(k)
        k+=1

    for k in shiftSequence:
        #Choose rotation dir
        d.append(d_func(x,y,z))
        if debug: print(f'{k=:02d} | {(x,y,z,d[-1])=}')
        if debug: print(f'         | {intToReal([x,y,z],n,point)}')

        #Perform rotation
        #Addition 1 (Note bitwiseCNOT logic can be done on the d[-1] qubit)
        x   = bitwiseCNOT(not ((m==-1)^d[-1]), x, n) 
        x,y = add(x,y,n,k)
        x   = bitwiseCNOT(not ((m==-1)^d[-1]), x, n)
        if debug: print(f'\tadd 1:  [x,y,z]={
              [float(f'{r:.3f}') for r in intToReal([x,y,z],n,point)]}')

        #Mult
        y, aux = mult(y,aux,n,2*k,(m==-1))
        if debug: print(f'\tmult:   [x,y,z]={
              [float(f'{r:.3f}') for r in intToReal([x,y,z],n,point)]}')

        #Addition 2
        y   = bitwiseCNOT(d[-1], y, n)
        y,x = add(y,x,n,k)
        y   = bitwiseCNOT(d[-1], y, n)
        if debug: print(f'\tadd 2:  [x,y,z]={
              [float(f'{r:.3f}') for r in intToReal([x,y,z],n,point)]}')

        #Swap for identity to uncompute
        if m==2: x,y = y,x

        #Addition 3
        x   = bitwiseCNOT(not (m!=1)^d[-1]^(m==2), x, n)
        x,y = add(x,y,n,k)
        x   = bitwiseCNOT(not (m!=1)^d[-1]^(m==2), x, n)
        if debug: print(f'\tadd 3:  [x,y,z]={
              [float(f'{r:.3f}') for r in intToReal([x,y,z],n,point)]}')

        #Div
        x, aux = div(x,aux,n,2*k,(m==-1))
        if debug: print(f'\tdiv:    [x,y,z]={
              [float(f'{r:.3f}') for r in intToReal([x,y,z],n,point)]}')

        #Addition 4
        y   = bitwiseCNOT(d[-1], y, n)
        y,x = add(y,x,n,k)
        y   = bitwiseCNOT(d[-1], y, n)
        if debug: print(f'\tadd 4:  [x,y,z]={
              [float(f'{r:.3f}') for r in intToReal([x,y,z],n,point)]}')

        #Undo swap for identity
        if m==2: x,y = y,x

        #Update ang
        if mode!=2: 
            z   = bitwiseCNOT(not d[-1], z, n)
            z,_ = add(z,int(2*e_func(k)*norm),n)
            z   = bitwiseCNOT(not d[-1], z, n)
        if debug: print(f'\tup ang: [x,y,z]={
              [float(f'{r:.3f}') for r in intToReal([x,y,z],n,point)]}')

        #Update Debug Vector List
        vecs.append(np.array([x,y])/norm)

    return x,y,z,d,vecs

def main():
    n     = 12
    point = 2
    depth = n

    test.circularRotTest(depth, CORDIC, n, point)
    test.linearRotTest(depth, CORDIC, n, point)
    test.hyperbolicRotTest(depth, CORDIC, n, point)
    test.circularVecTest(depth, CORDIC, n, point)
    test.linearVecTest(depth, CORDIC, n, point)
    test.hyperbolicVecTest(depth, CORDIC, n, point)
    test.arcTest(depth, CORDIC, n, point)
    test.hypArcTest(depth, CORDIC, n, point)
    test.identityTest(depth, CORDIC, n, point)


if __name__=='__main__':
    main()

