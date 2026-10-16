
import numpy as np
import matplotlib.pyplot as plt

#=============================================================================#
#===Helper Functions==========================================================#
#=============================================================================#
def plot_vecs(vecs, title='') -> None:
    if type(vecs)!=type([]): vecs = [vecs]
    plt.figure()
    for i, vec in enumerate(vecs):
        t = i/len(vecs)
        plt.quiver(0,0,vec[0],vec[1],color=(.8-(t/2)**(.75), .1, (.1+(3/5)*t)**(1.25)),
            scale=1, angles='xy', scale_units='xy'
        )

    ticks = [(k-8)/3 for k in range(32)]
    plt.xticks(ticks); plt.yticks(ticks)
    plt.grid(True, ls='--', markevery=.1)
    plt.axis('scaled')
    plt.xlim(-1.1,1.1); plt.ylim(-1.1,1.1)
    plt.title(title)
    plt.show()

def bitwiseNOT(x: int, n: int):
    return (~x) % (1<<n)

def intToReal( xs, n: int, point: int):
    listmode = (type(xs)==type([]))
    if not listmode: xs = [xs]
    results = []
    for x in xs:
        signed = x if x<(1<<(n-1)) else -bitwiseNOT(x,n)-1
        results.append(signed/(1<<(n-point)))
    return results if listmode else results[0]

#=============================================================================#
#===Tests=====================================================================#
#=============================================================================#

#---Rotation Mode Tests-------------------------------------------------------#
def circularRotTest(depth:int,local_cordic: callable,
        n: int=10, point: int=2
    ) -> None:
    N  = 1<<(n-point)
    ts = np.arange(N<<point,dtype=int)/N

    results_x = []
    results_y = []
    for j in range(-2*N,2*N):
        x_init,y_init,z_init = 1<<(n-point),0,int(N*ts[j])
        x,y,z,d,vecs = local_cordic(x_init,y_init,z_init,
            m=1, mode=0, depth=depth, n=n, point=point
        )
        results_x.append(intToReal(x,n,point))
        results_y.append(intToReal(y,n,point))

    plt.plot(results_x, label='cos')
    plt.plot(results_y, label='sin')
    plt.legend()
    plt.title(f'CORDIC Cicular Rotation ({'X'*point}.{'X'*(n-point)})')
    plt.show()


def linearRotTest(depth,local_cordic: callable = None,
      n: int = 10, point: int = 2
    ) -> None:
    if local_cordic == None: local_cordic = oldCORDIC

    N = 1<<(n-point)
    samples = 1<<4 #results in samples^2 tests
    width = N//samples
    results_xz = np.zeros((samples,samples))
    truevals   = np.zeros((samples,samples))

    for x_0 in range(0,N,width):
        for z_0 in range(0,N,width):
            x,y,z,d,vecsPred = local_cordic(x_0,0,z_0,
                m=0, mode=0, depth=depth, n=n, point=point
            )

            # x,y,z,d,_ = revCORDIC(x_0/N,0,z_0/N,
            #     m=0,
            #     d_func=lambda a,b: b<0,
            #     depth=depth
            # )

            results_xz[x_0//width,z_0//width] = intToReal(y,n,point)
            truevals[x_0//width,z_0//width] = (x_0*z_0)/(N**2)

    fig, ax = plt.subplots(1,2)
    im = []
    fig.colorbar(ax[0].imshow(results_xz))#, vmax=vmax,vmin=vmin))
    ax[0].set_title('results')
    fig.colorbar(ax[1].imshow(truevals))#, vmax=vmax,vmin=vmin))
    ax[1].set_title('ground truth')

    fig.suptitle('Linear Rotation Test - x*z')
    plt.show()

def hyperbolicRotTest(depth,local_cordic: callable,
        n: int, point: int
    ) -> None:
    #Note getting some weird stuff for y when z near 0
    N = 1<<(n-point)
    samples = 1<<6 #results in samples^2 tests
    width = N//samples

    ts = 2.5*np.arange(samples)/samples
    xs = np.cosh(ts); ys = np.sinh(ts)

    results_x = []
    results_y = []
    for k in range(0,N,width):

        x,y,z,d,vecs = local_cordic(N,0,int(N*ts[k//width]),
            m=-1, mode=0, depth=depth, n=n, point=point
        )

        results_x.append(intToReal(x,n,point))
        results_y.append(intToReal(y,n,point))

    plt.plot(results_x, label='result x')
    plt.plot(xs, label='true x')
    plt.legend()
    plt.title('hyperbolicRotTest - cosh')
    plt.show()

    plt.plot(results_y, label='result y')
    plt.plot(ys, label='true y'), 
    plt.legend()
    plt.title('hyperbolicRotTest - sinh')
    plt.show()

#---Vector Mode Tests---------------------------------------------------------#
def circularVecTest(
        depth: int, local_cordic: callable, n: int, point: int
    ) -> None:
    N = 1<<(n-point)
    samples = 1<<5 #results in samples^2 tests
    width = N//samples

    results_x  = np.zeros((samples,samples))
    truevals_x = np.zeros((samples,samples))
    results_z  = np.zeros((samples,samples))
    truevals_z = np.zeros((samples,samples))

    for x_0 in range(0,N,width):
        for y_0 in range(0,N,width):
            x,y,z,d,vecs = local_cordic(x_0,y_0,0,
                m=1, mode=1, depth=depth, n=n, point=point
            )
            results_x[ x_0//width,y_0//width] = x/N
            truevals_x[x_0//width,y_0//width] = np.sqrt(x_0**2+y_0**2)/N
            if x_0!=0 and y_0!=0: 
                truevals_z[x_0//width,y_0//width] = np.atan(y_0/x_0)
                results_z[ x_0//width,y_0//width] = z/N

    fig, ax = plt.subplots(1,2)
    im = []
    fig.colorbar(ax[0].imshow(results_x)); ax[0].set_title('x results')
    fig.colorbar(ax[1].imshow(truevals_x));   ax[1].set_title('x ground truth')
    fig.suptitle('Circular Vec Test - sqrt(x^2+y^2)')
    plt.show()

    fig, ax = plt.subplots(1,2)
    im = []
    fig.colorbar(ax[0].imshow(results_z)); ax[0].set_title('z results')
    fig.colorbar(ax[1].imshow(truevals_z));   ax[1].set_title('z ground truth')
    fig.suptitle('Circular Vec Test - atan(y/x)')
    plt.show()

    results_atan  = []
    truevals_atan = []
    for y_0 in range(-2*N,2*N,width):
        y_init = int(y_0)
        x_init = N
        x,y,z,d,vecs = local_cordic(x_init,y_init,0,
            m=1, mode=1, depth=depth, n=n, point=point
        )
        results_atan.append(intToReal(z,n,point))
        truevals_atan.append(np.atan(y_init/N))

    plt.plot(results_atan, label='atan result')
    plt.plot(truevals_atan, label='atan ground truth')
    plt.legend()
    plt.title('Circular Vector Test - arctan')
    plt.show()

def linearVecTest(
        depth: int,local_cordic: callable, n: int, point: int
    ) -> None:
    N = 1<<(n-point)
    samples = 1<<5 #results in samples^2 tests
    width = N//samples

    results  = np.zeros((samples,samples))
    truevals = np.zeros((samples,samples))

    for x_0 in range(0,N,width):
        for y_0 in range(0,N,width):
            x,y,z,d,_ = local_cordic(x_0,y_0,0,
                m=0, mode=1, depth=depth, n=n, point=point
            )
            if x_0!=0: 
                results[x_0//width,y_0//width] = intToReal(z,n,point)
                truevals[x_0//width,y_0//width] = (y_0/x_0)

    fig, ax = plt.subplots(1,2)
    im = []
    vmax = 2#np.max((np.max(results),np.max(truevals)))
    vmin = 0#np.min((np.min(results),np.min(truevals)))
    fig.colorbar(ax[0].imshow(results[4:,:], vmax=vmax, vmin=vmin)) 
    ax[0].set_title('results')
    fig.colorbar(ax[1].imshow(truevals[4:,:], vmax=vmax, vmin=vmin))
    ax[1].set_title('ground truth')

    fig.suptitle('Linear Vector Test x/y')
    plt.show()

def hyperbolicVecTest(
        depth: int, local_cordic: callable, n:int, point: int
    ) -> None:
    N = 1<<(n-point)
    samples = 1<<5 #results in samples^2 tests
    width = N//samples
    results_x  = np.zeros((samples,samples))
    truevals_x = np.zeros((samples,samples))
    results_z  = np.zeros((samples,samples))
    truevals_z = np.zeros((samples,samples))

    for x_0 in range(0,N,width):
        for y_0 in range(0,N,width):
            x,y,z,d,_ = local_cordic(x_0,y_0,0,
                m=-1, mode=1, depth=depth, n=n, point=point
            )

            if x_0>y_0:
                results_x[ x_0//width,y_0//width] = intToReal(x,n,point)
                truevals_x[x_0//width,y_0//width] = np.sqrt(x_0**2-y_0**2)
                if x_0!=0 and y_0!=0: 
                    results_z[ x_0//width,y_0//width] = intToReal(z,n,point)
                    truevals_z[x_0//width,y_0//width] = np.atanh(y_0/x_0)

    fig, ax = plt.subplots(1,2)
    im = []
    fig.colorbar(ax[0].imshow(results_x,vmin=0));  ax[0].set_title('x results')
    fig.colorbar(ax[1].imshow(truevals_x,vmin=0)); ax[1].set_title('x ground truth')
    fig.suptitle('Hyperbolic Vector Test sqrt(x^2-y^2)')
    plt.show()

    fig, ax = plt.subplots(1,2)
    im = []
    fig.colorbar(ax[0].imshow(results_z,vmin=0));  ax[0].set_title('z results')
    fig.colorbar(ax[1].imshow(truevals_z,vmin=0)); ax[1].set_title('z ground truth')
    fig.suptitle('Hyperbolic Vecor Test arctanh(y/x)')
    plt.show()

    results_atanh  = []
    truevals_atanh = []
    for y_0 in range(-N+width,N,width):
        x,y,z,d,_ = local_cordic(N,y_0,0,
            m=-1, mode=1, depth=depth, n=n, point=point
        )
        results_atanh.append(intToReal(z,n,point))
        truevals_atanh.append(np.atanh(y_0/N))

    plt.plot(results_atanh, label='result')
    plt.plot(truevals_atanh, label='ground truth')
    plt.legend()
    plt.title('Hyperbolic Vector Test Arctanh')
    plt.show()

    results_sqrt  = []
    results_ln    = []
    truevals_sqrt = []
    truevals_ln   = []


    for w in range(N):
        y_init = bitwiseNOT(bitwiseNOT(w,n)+N,n); x_init = w+N; z_init=0
        x,y,z,d,_ = local_cordic(x_init,y_init,z_init,
            m=-1, mode=1, depth=depth, n=n, point=point
        )
        results_sqrt.append(intToReal(x,n,point)/2)
        truevals_sqrt.append(np.sqrt(intToReal(w,n,point)))
        if w!=0: 
            results_ln.append(intToReal(2*z,n,point))
            truevals_ln.append(np.log(intToReal(w,n,point)))


    plt.plot(results_sqrt, label='result')
    plt.plot(truevals_sqrt, label='ground truth')
    plt.legend()
    plt.title('hyperbolicVecTest - sqrt')
    plt.show()

    plt.plot(results_ln, label='result')
    plt.plot(truevals_ln, label='ground truth')
    plt.legend()
    plt.title('hyperbolicVecTest - ln')
    plt.show()

#---Arc Mode Tests------------------------------------------------------------#
def arcTest(
        depth:int,local_cordic: callable, n:int, point:int
    ) -> None:
    N = 1<<(n-point)
    samples = 1<<8 #results in samples^2 tests
    width = N//samples
    ts = (2*np.pi) * (np.arange(N)/N)

    xs = np.cos(ts); ys = np.sin(ts)

    shiftSequence = []
    k = 1; repeat = 4
    while k < depth:
        if k > repeat: k-=1; repeat=3*repeat+1
        shiftSequence.append(k)
        k+=1

    trueval = []
    results = []
    for k in range(-N,N+width,width):
        x,y,z,d,_ = local_cordic(N,0,k,
            m=1, mode=2, depth=depth, n=n, point=point
        )   

        asint = np.sum(np.array(
            [(-1 if d[j] else 1)*2*np.atan(2**(-shiftSequence[j])) 
             for j in range (depth)]
        ))
        trueval.append(np.asin(k/N))
        results.append(asint)

    plt.plot(results, label='result')
    plt.plot(trueval, label='ground truth')
    plt.legend()
    plt.title('asin')
    plt.show()

def hypArcTest(
        depth:int,local_cordic:callable,n:int,point:int
    ) -> None:
    N = 1<<(n-point)
    samples = 1<<5
    width = N//samples
    ts = (2*np.pi) * (np.arange(N)/N)

    xs = np.cosh(ts); ys = np.sinh(ts)

    shiftSequence = []
    k = 1; repeat = 4
    while k < depth:
        if k > repeat: k-=1; repeat=3*repeat+1
        shiftSequence.append(k)
        k+=1

    trueval = []
    results = []
    for k in range(0,2*N):

        x,y,z,d,_ = local_cordic(N,0,k,
            m=-1, mode=2, depth=depth, n=n, point=point#, debug=True
        )   

        asinh = np.sum(np.array(
            [(1 if d[j] else -1)*2*np.atanh(2**(-shiftSequence[j])) for j in range (depth)]
        ))

        if k!=0:
            trueval.append(np.asinh(k/N))
            results.append(-asinh)
        else:
            trueval.append(0)
            results.append(0)

    plt.title('Hyperbolic Arc Test - arcsinh')
    plt.plot(results, label='result')
    plt.plot(trueval, label='expected')
    plt.legend()
    plt.show()

#---Identity Mode Tests-------------------------------------------------------#
def identityTest(depth,local_cordic: callable = None,
      n: int = 10, point: int = 2
    ) -> None:
    N = 1<<(n-point)
    samples = 1<<4
    width = N//samples

    x_tru = np.random.randint(0,N,samples)
    y_tru = np.random.randint(0,N,samples)
    z_tru = np.random.randint(0,N,samples)
    x_res = []; y_res = []; z_res = []

    for k in range(samples):

        x,y,z,d,vecsPred = local_cordic(x_tru[k], y_tru[k], z_tru[k],
            m=2, mode=3, depth=depth, n=n, point=point
        )

        x_res.append(x); y_res.append(y); z_res.append(z)

    fig, ax = plt.subplots(1,3)

    fig.suptitle('Identity Test')
    ax[0].scatter(x_tru, x_res, label='x')
    ax[0].set_title('x')
    ax[1].scatter(y_tru, y_res, label='y')
    ax[1].set_title('y')
    ax[2].scatter(z_tru, z_res, label='z')
    ax[2].set_title('z')
    plt.show()

def main():
    pass

if __name__ == '__main__':
    main()

