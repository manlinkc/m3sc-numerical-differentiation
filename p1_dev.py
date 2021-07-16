"""M345SC Homework 3, part 1
Manlin Chawla, CID: 01205586
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import odeint
from scipy.signal import hann
from scipy.sparse import diags
import time

def nwave(alpha,beta,Nx=256,Nt=801,T=200,display=False):
    """
    Question 1.1
    Simulate nonlinear wave model

    Input:
    alpha, beta: complex model parameters
    Nx: Number of grid points in x
    Nt: Number of time steps
    T: Timespan for simulation is [0,T]
    Display: Function creates contour plot of |g| when true

    Output:
    g: Complex Nt x Nx array containing solution
    """

    # Set period
    L = 100
    # Generate grid
    x = np.linspace(0,L,Nx+1)
    # Discard endpoint of grid
    x = x[:-1]

    # Set up list of points, not including the last N/2 point
    # Rearrange n into fft order
    n = np.fft.fftshift(np.arange(-Nx/2,Nx/2))

    # Construct k, avoids having to recompute k within each call of RHS
    k = 2*np.pi*n/L
    # Construct (i*k)**2,a voids having to recompute k within each call of RHS
    ksquared=-np.square(k)

    # Pre-compute 1/N as multiplication is efficient compared to division
    Nreciprocal=float(1)/Nx

    def RHS(f,t,alpha,beta):
        """Computes dg/dt for model eqn.,
        f[:N] = Real(g), f[N:] = Imag(g)
        Called by odeint below
        """
        # Complex periodic function with period L=100
        g = f[:Nx]+1j*f[Nx:]

        # Fourier coefficients of c
        c = np.fft.fft(g)*Nreciprocal

        # Compute d2g
        d2g = Nx*np.fft.ifft(ksquared*c)
        #-----------
        # Rearrange non linear wave to make dgdt the subject
        dgdt = alpha*d2g + g -beta*g*g*g.conj()
        df = np.zeros(2*Nx)
        df[:Nx] = dgdt.real
        df[Nx:] = dgdt.imag
        return df

    #set initial condition
    g0 = np.random.rand(Nx)*0.1*hann(Nx)
    f0 = np.zeros(2*Nx)
    f0[:Nx] = g0
    t = np.linspace(0,T,Nt)

    #compute solution
    f = odeint(RHS,f0,t,args=(alpha,beta))
    g = f[:,:Nx] + 1j*f[:,Nx:]

    if display:
        plt.figure()
        plt.contour(x,t,g.real)
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('Contours of Real(g)')
        plt.show()

    return g

def modified_nwave(alpha,beta,g0,Nx=256,Nt=801,T=200):
    """
    Function which takes g0 as an input and uses g0 as the initial condition.
    """

    # Set period
    L = 100
    # Generate grid
    x = np.linspace(0,L,Nx+1)
    # Discard endpoint of grid
    x = x[:-1]

    # Set up list of points, not including the last N/2 point
    # Rearrange n into fft order
    n = np.fft.fftshift(np.arange(-Nx/2,Nx/2))

    # Construct k, avoids having to recompute k within each call of RHS
    k = 2*np.pi*n/L
    # Construct (i*k)**2,a voids having to recompute k within each call of RHS
    ksquared=-np.square(k)

    # Pre-compute 1/N as multiplication is efficient compared to division
    Nreciprocal=float(1)/Nx

    def RHS(f,t,alpha,beta):
        """Computes dg/dt for model eqn.,
        f[:N] = Real(g), f[N:] = Imag(g)
        Called by odeint below
        """
        # Complex periodic function with period L=100
        g = f[:Nx]+1j*f[Nx:]

        # Fourier coefficients of c
        c = np.fft.fft(g)*Nreciprocal

        # Compute d2g
        d2g = Nx*np.fft.ifft(ksquared*c)
        #-----------
        # Rearrange non linear wave to make dgdt the subject
        dgdt = alpha*d2g + g -beta*g*g*g.conj()
        df = np.zeros(2*Nx)
        df[:Nx] = dgdt.real
        df[Nx:] = dgdt.imag
        return df

    # Set initial condition
    f0 = np.zeros(2*Nx)
    f0[:Nx] = g0.real
    f0[Nx:] = g0.imag
    t = np.linspace(0,T,Nt)

    #compute solution
    f = odeint(RHS,f0,t,args=(alpha,beta))
    g = f[:,:Nx] + 1j*f[:,Nx:]

    return g

def analyze(figurenumber,casenumber,display=True):
    """
    Question 1.2
    Add input/output as needed

    Discussion:
    To compare the simulation results for the two different cases A: alpha = 1-2i,
    beta = 1+2i and case B: alpha =1-i, beta = 1+2i, I focused my investigation on
    the most energetic wavenumbers. From the notes in lecture 16 we know that the
    amplitude of a Fourier coefficient describes the ‘energy’ at a particular frequency.
    Figure 1 is a plot of the energy over different wavenumbers for case A in orange
    and case B in green. To produce this plot, I have created a new function called
    modified_nwave which has the same functionality as the nwave function with the
    only difference that it takes in g0 as an input. I have decided to create this
    function so I can directly compare simulation results for case A and case B based
    on the same initial conditions. Figure 1 shows that for both case A and case B, the
    amplitudes of the Fourier coefficients are centred around zero and there are more
    spikes in energy when the wavenumber gets closer to 0.

    From looking at figure 1, both case A and case B seem to peak with similar amplitudes.
    I have decided to further the investigation by focusing on the most energetic wavenumbers.
    To do so, I have found the maximum amplitude reached by the Fourier coefficients in case
    A and in case B. I have used these values to create a threshold which is
    0.75*min(max(cn_A),max(cn_B)). For any Fourier coefficient lower than this frequency I
    have set the coefficient to 0, this acts as a filter where only the contributions of the
    most energetic wavenumbers are left behind. Figure 2 is a plot of the amplitude of the
    Fourier coefficients for the most energetic wavenumbers for both Case A and Case B with
    the threshold as given in the title. This plot shows that for both case A and case B the
    highest frequencies are focused around 0 and the range of amplitudes reached is around
    60 – 80. The figure also shows that Case B has more energetic wavenumbers above the
    threshold than case A.

    Figure 3 is a semilog plot of the power spectral density against the frequency for case
    A for 6 different x values between 0 and 250. Figure 4 is a semilog plot of the power
    spectral density against the frequency for case B for 6 different x values between 0 and 250.
    We are given that the signal is periodic in the x-plane as a hann windowing signal is applied.
    However, in general there is no reason to expect that signal is periodic in the t-plane.
    I have used windowing to account for this, in particular I have used scipy.signal.welch to
    use Welch’s method and this also makes it clearer to analyse certain characteristics of the
    signal. I have used Welch’s method to produce a plot of the power spectral density.
    Both figure 3 and figure 4 show that for each x value the power spectral density increases
    to a maximum at a frequency just smaller than 0 and then gradually decreases to a minimum
    before spiking upwards against as frequency reaches around 2.0. Within my code I have
    outputted the frequency (fmin) that corresponds to the minimum power spectral density
    and the frequency (fmax) that corresponds to the maximum power spectral density. For
    case A the most energetic frequency/ fmax = -0.8251953125 and fmin = 1.25416992188 and
    for case B the most energetic frequency/ fmax = -0.2503125. and fmin =1.7834765625. From
    the notes in lecture 16 we need T >> 1/fmax to ensure the slow components are contained
    within the signal. In this case, we need T to be greater than approximately 5. Throughout
    my investigation, this inequality has been satisfied and I have set T=200 in both cases.
    Another inequality from the notes is that we need delta t < 2/fmin to resolve the highest
    frequency components. In this case we need delta t to be smaller than approximately 1.6 and
    in my code this inequality has been satisfied as delta t is 100/256=0.390625.

    From lecture 16 notes we know that abs(c_n)**2 is the energy spectral density. Intuitively,
    we can describe the sum of the energy spectral density across all n values as the ‘total energy’.
    Figure 5 is a plot of the total energy at each time point with t<50 discarded. Figure 5 shows
    that both case A and case B have stable total energy that stays between a range of values.
    However, case B has a higher total energy meaning that case B has more energy in the system.

    To measure if/to what degree each case is chaotic I have decided to use the Lyapunov exponents
    as a way of measuring the sensitivity of each case to initial conditions. To do this I have first
    run the simulation from arbitrary initial condition up to t=50 (the initial transience) and took
    the solution to be the initial condition. I then used this initial condition as an input into my
    modified_nwave function to obtain an output. I then repeated this step using a perturbed initial
    condition where I added 1**10-6 to on of the entries in the initial condition. I then computed the
    distance between the two solutions and plotted this on a semilog scale. Figure 6 is a plot of the
    distance squared between the two solutions against the time in seconds for case A. Figure 7 is a plot
    of the distance squared between the two solutions against the time in seconds for case B. In a
    chaotic system we expect there to be some sensitivity to initial condition. With the plots in figure 6
    and figure 7 if the system is chaotic this distance will increase exponentially. The rate of the
    exponential growth is the Lyaponov exponent and I have used these plots to estimate the Lyapunov
    exponent using least-squares fit. Figure 6 shows that the slope is 0.588813260269 so the Lyapunov
    exponent is 0.29 (2dp) for case A, figure 7 shows that the slope is 0.404224781031 so the Lyapunov
    exponent is 0.20 (2dp) for case B. This analysis indicates that although the Lyaponuv exponents are
    fairly close, case A is more chaotic than case B.
    """

    # Plot of Gaussian Fourier coefficient's for A: alpha=1-2j beta=1+2j and B: alpha=1-j beta=1+2j
    if figurenumber==1:
        # Set up parameters
        Nx = 256
        Nreciprocal=float(1)/Nx
        L = 100

        # Set up an initial condition to put in to modifiedg function
        g0 = np.random.rand(Nx)*0.1*hann(Nx)
        # Output for case A, compute g and extract the value of g at the last time
        gA = modified_nwave(1-2j,1+2j,g0,Nx=256,Nt=801,T=200)[-1,:]
        # Output for case B,compute g and extract the value of g at the last time
        gB = modified_nwave(1-1j,1+2j,g0,Nx=256,Nt=801,T=200)[-1,:]

        # Set up range of values for N
        n = np.arange(-Nx/2,Nx/2)
        #n = np.fft.fftshift(np.arange(-Nx/2,Nx/2))
        # Construct list of wavenumbers
        k = 2*np.pi*n/L

        # Get fourier coefficients for case A
        gfourier_A=np.fft.fft(gA)*Nreciprocal
        # Get absolute values of fourier coefficients and rearrange order, case A
        gfourier_A=Nx*np.fft.fftshift(np.abs(gfourier_A))

        # Get fourier coefficients for case B
        gfourier_B=np.fft.fft(gB)*Nreciprocal
        # Get absolute values of fourier coefficients and rearrange order, case A
        gfourier_B=Nx*np.fft.fftshift(np.abs(gfourier_B))

        # Plot amplitude of Fourier coefficients
        plt.figure()
        plt.hold(True)
        plt.plot(k,gfourier_A,'x--', label='Case A')
        plt.plot(k,gfourier_B,'x--', label='Case B')
        plt.hold(False)
        plt.xlabel('Wavenumbers')
        plt.ylabel('Amplitude of Fourier coefficients')
        plt.legend()
        plt.title('analyze('+str(figurenumber)+',1) \n Energy for all wavenumbers \n Nx=256,Nt=801,T=200')

        if display == True:
            plt.show()

    if figurenumber ==2:
        # Set up parameters
        Nx = 256
        Nreciprocal=float(1)/Nx
        L = 100

        # Set up an initial condition to put in to modifiedg function
        g0 = np.random.rand(Nx)*0.1*hann(Nx)
        # Output for case A, compute g and extract the value of g at the last time
        gA = modified_nwave(1-2j,1+2j,g0,Nx=256,Nt=801,T=200)[-1,:]
        # Output for case B,compute g and extract the value of g at the last time
        gB = modified_nwave(1-1j,1+2j,g0,Nx=256,Nt=801,T=200)[-1,:]

        # Set up range of values for N
        n = np.arange(-Nx/2,Nx/2)
        #n = np.fft.fftshift(np.arange(-Nx/2,Nx/2))
        # Construct list of wavenumbers
        k = 2*np.pi*n/L

        # Get fourier coefficients for case A
        gfourier_A=np.fft.fft(gA)*Nreciprocal
        # Get absolute values of fourier coefficients and rearrange order, case A
        gfourier_A=Nx*np.fft.fftshift(np.abs(gfourier_A))

        # Get fourier coefficients for case B
        gfourier_B=np.fft.fft(gB)*Nreciprocal
        # Get absolute values of fourier coefficients and rearrange order, case A
        gfourier_B=Nx*np.fft.fftshift(np.abs(gfourier_B))

        # Find threshold
        maxcn_A = np.max(np.abs(gfourier_A))
        thresholdA = 0.75*maxcn_A
        maxcn_B = np.max(np.abs(gfourier_B))
        thresholdB = 0.75*maxcn_B
        finalthreshold=min(thresholdA,thresholdB)

        for i in range(Nx):
            if gfourier_A[i] < finalthreshold:
                gfourier_A[i] = 0
            if gfourier_B[i] < finalthreshold:
                gfourier_B[i] = 0

        # Plot amplitude of Fourier coefficients
        plt.figure()
        plt.hold(True)
        plt.plot(k,gfourier_A,'x--', label='Case A')
        plt.plot(k,gfourier_B,'x--', label='Case B')
        plt.hold(False)
        plt.xlabel('Wavenumbers')
        plt.ylabel('Amplitude of Fourier coefficients')
        plt.legend()
        plt.title('Manlin Chawla: analyze('+str(figurenumber)+',1) \n Most energetic wavenumbers \n Nx=256,Nt=801,T=200 threshold='+str(finalthreshold))

        if display == True:
            plt.show()

    if figurenumber == 3:

        if casenumber==1:
            alpha = 1-2j
            case = 'A'
        elif casenumber==2:
            alpha = 1-1j
            case = 'B'

        Nx = 256
        Nt = 801
        T = 200
        g = nwave(alpha,1+2j,Nx=Nx,Nt=Nt,T=T,display=False)

        # Discard times t<50
        g = g[200:,:]

        xvalues=[0,50,100,150,200,250]
        fmax = np.zeros(6)
        fmin = np.zeros(6)

        plt.figure()
        plt.hold(True)

        for i,x in enumerate(xvalues):
            Y2 = g[:,x]
            w2,Pxx2 = scipy.signal.welch(Y2)
            w2 = w2*Nt/T
            w2 = np.fft.fftshift(w2)
            Pxx2 = np.fft.fftshift(Pxx2)
            fmax[i] = w2[np.argmax(Pxx2)]
            fmin[i] = w2[np.argmin(Pxx2)]
            plt.semilogy(w2,Pxx2,label=str(x))

        plt.hold(False)
        plt.legend()
        plt.xlabel('frequency')
        plt.ylabel('Pxx')
        plt.title('Manlin Chawla: analyze('+str(figurenumber)+','+str(casenumber)+') \n Power Spectral Density, alpha='+str(alpha))
        fmax_ave=np.mean(fmax)
        fmin_ave=np.mean(fmin)
        print('fmin=',fmin_ave)
        print('fmax=',fmax_ave)

    if figurenumber == 4:
        #Nx = 256
        Nx = 256
        Nt = 801
        T = 200
        t = np.linspace(0,T,Nt)
        t =t[200:]

        Nreciprocal = float(1)/Nx

        # Set up an initial condition to put in to modifiedg function
        g0 = np.random.rand(Nx)*0.1*hann(Nx)
        # Output for case A, compute g and discard output for t<50
        gA = modified_nwave(1-2j,1+2j,g0,Nx=256,Nt=801,T=200)
        gA = gA[200:,:]
        # Output for case B,compute g and extract the value of g at the last time
        gB = modified_nwave(1-1j,1+2j,g0,Nx=256,Nt=801,T=200)
        gB = gB[200:,:]

        gfourier_A = np.zeros_like(gA)
        gfourier_B = np.zeros_like(gB)

        for i in range(Nt-200):
            gfourier_A[i,:]=np.fft.fft(gA[i,:])*Nreciprocal
            gfourier_B[i,:]=np.fft.fft(gB[i,:])*Nreciprocal
            # Get absolute values of fourier coefficients and rearrange order
            #gfourier_A[i,:]=np.fft.fftshift(np.abs(gfourier_A[i,:]))
            #gfourier_B[i,:]=np.fft.fftshift(np.abs(gfourier_B[i,:]))

        gfourier_A = np.square(np.abs(gfourier_A))
        gfourier_B = np.square(np.abs(gfourier_B))
        cnsquared_A = Nx*np.sum(gfourier_A,axis=1)
        cnsquared_B = Nx*np.sum(gfourier_B,axis=1)

        plt.figure()
        plt.hold(True)
        plt.plot(t, cnsquared_A, label = 'Case A')
        plt.plot(t, cnsquared_B, label = 'Case B')
        plt.hold(False)
        plt.title('Manlin Chawla: analyze('+str(figurenumber)+',1) \n Total energy')
        plt.xlabel('T')
        plt.ylabel('Total Energy')
        plt.legend()

        if display == True:
            plt.show()

    if figurenumber == 5:
        if casenumber==1:
            alpha = 1-2j
            case = 'A'
        elif casenumber==2:
            alpha = 1-1j
            case = 'B'

        Nx = 256
        T = 200
        Nt = 801
        t = np.linspace(0,T,Nt)

        # Initial condition at T=50
        g0 = np.random.rand(Nx)*0.1*hann(Nx)
        g0 = modified_nwave(alpha,1+2j,g0,Nx=Nx,Nt=801,T=50)[-1]
        # Initial condition with small perturbation for T=200
        g1 = g0.copy()
        g1[0]+=1e-6

        # Initial condition for T=200
        original_g = modified_nwave(alpha,1+2j,g0,Nx=Nx,Nt=801,T=200)
        perturbed_g = modified_nwave(alpha,1+2j,g1,Nx=Nx,Nt=801,T=200)
        distance = np.abs(original_g - perturbed_g)**2
        distance = np.sum(distance,axis=1)
        print('shape of distance=',distance.shape)

        x=t[50:200]
        m,c=np.polyfit(x,np.log(distance[50:200]),deg=1)
        y= np.exp(m*t[:200]+c)

        plt.hold(True)
        plt.semilogy(t,distance,'-')
        plt.semilogy(t[:200],y,"r--",label='Least Squares Slope='+str(m))
        plt.hold(False)
        plt.legend()
        plt.ylabel('Distance squared')
        plt.xlabel('T')
        plt.title('Manlin Chawla: analyze('+str(figurenumber)+','+ str(casenumber)+') \n Sensitivity due to initial conditions')

        if display == True:
            plt.show()

    return None #modify as needed

def finitediffmethod(g, Nx, Nt, T):
        # x values go from 0,100
        L = 100
        # Compute h the spacing between the grid
        h = float(L)/Nx

        # Define coefficients for finite difference
        alpha_fd = 3/8
        beta_fd  = 0
        a = 25/16
        b = 1/5
        c = -1/80

        # Pre-compute a/2h, b/4h ,c/6h
        ah = a/(2*h)
        bh = b/(4*h)
        ch = c/(6*h)

        # Define coefficients for the one-sided scheme
        alpha_onefd = 3
        a_onefd = -17/6
        b_onefd = 3/2
        c_onefd = 3/2
        d_onefd = -1/6

        # LHS matrix
        # Construct vector for the diagonal of the LHS matrix
        l0 = np.ones(Nx)
        # Construct vector for the off-diagonal of the LHS matrix
        l1 = alpha_fd*l0[1:]
        # Use alpha=3 for the one-sided 4th order scheme at endpoints
        l1[0] = 3
        # Construct LHS matrix
        #A = scipy.sparse.diags([np.flip(l1,axis=0),l0,l1],[-1,0,1])
        # Collapse scipy matrix into an array

        # RHS matrix
        # Construct vectors for the diagonal
        # Pre-allocate a vector Nx -1 of ones
        r0 = np.ones(Nx-1)
        # Construct vector for the diagonal of the RHS matrix
        r1 = np.zeros(Nx)
        r1[0] = a_onefd/h
        r1[-1] = -r1[0]
        # Construct vector for the first off-diagonal of the RHS matrix
        r2 = ah*r0
        r2[0] = b_onefd/h
        # Construct vector for the second off-diagonal of the RHS matrix
        r3 = bh*r0[1:]
        r3[0] = c_onefd/h
        # Construct vector for the third off-diagonal of the RHS matrix
        r4 = ch*r0[2:]
        r4[0] = d_onefd/h
        # Construct vectors for the wrap-around diagonal of the RHS matrix
        r5 = ch*r0[:3]
        r5[-1] = 0
        r6 = bh*r0[:2]
        r6[-1] = 0

        # Construct RHS matrix
        RHS = scipy.sparse.diags([r6,r5,np.flip(-r4,axis=0),np.flip(-r3,axis=0),np.flip(-r2,axis=0),r1,r2,r3,r4,np.flip(-r5,axis=0),np.flip(-r6,axis=0)],[-Nx+2,-Nx+3,-3,-2,-1,0,1,2,3,Nx-3,Nx-2])

        # Construct b=RHS*g, where b is from linear system of equations Ax=b
        B = RHS*g

        # Constuct banded A matrix
        # Set up a Nx x 3 array
        A_banded = np.zeros((3,Nx))
        # Fill up rows of banded matrix with reversed off-diagonal, diagonal, off-diagonal
        A_banded[0,1:] = l1
        A_banded[1,:] = l0
        A_banded [2,:Nx-1] = np.flip(l1,axis=0)

        # Now we have a linear system of equation Ax=B
        # Solve Ax=b to get dg/dx from the finitedifference method
        #dgdx_finitedifference = scipy.sparse.linalg.spsolve(A,B)

        dgdx_finitedifference = scipy.linalg.solve_banded((1, 1), A_banded, B)


        return dgdx_finitedifference

def dicretefouriermethod(g, Nx):
        L = 100
        # Compute dgdx using discrete fourier transform method
        n = np.fft.fftshift(np.arange(-Nx/2,Nx/2))
        k = 2*np.pi*n/L
        c = np.fft.fft(g)/Nx
        dgdx_discretefourier = Nx*np.fft.ifft(1j*k*c)

        return dgdx_discretefourier

def wavediff(figurenumber, display=True):
    """
    Question 1.3
    Add input/output as needed

    Discussion:
    To investigate whether the compact finite difference method is superior to Fourier
    differentiation for non-linear waves I have considered the simulation results for
    case B at t=100. In my implementation of the finite difference method I have used
    the banded matrix to make the scheme more efficient so it requires less memory
    and less operations.

    I have first considered efficiency in terms of runtime. The variable Nx which is the
    number of x values in the grid is common to both the finite difference method and to
    the Fourier differentiation method. Figure 8 is a plot of the run time of each method
    as Nx is increased. The plot clearly shows that for the range of Nx values chosen the
    discrete fourier transform is a lot faster than the finite difference method. Interestingly
    as Nx increases the runtime for the discrete fourier transform increases but at a much
    slower rate compared to the finite difference. From this figure it seems as though the
    runtime for the finite difference method increases as Nx increase but at a much larger
    rate as Nx increases (perhaps exponentially). Using the notes form lecture 16 we c

    I have next considered the accuracy of both methods. Figure 9, Figure 10, Figure 11 and
    Figure 12 each show the derivative output from the Finite difference method and from the
    Discrete Fourier Transform in comparison to an estimate of the exact derivate for different
    values of Nx. These plots show how the number of grid points (Nx) affect the accuracy of
    the method. Figure 9 shows this plot for Nx =32, in this plot neither method is very accurate.
    As Nx increases to Nx = 64 both methods seem to do better at fitting the exact derivative.
    However it’s interesting to note that the finite difference method dgdx deviates at the end
    point whereas the discrete fourier transform dgdx follows the correct shape. Again as Nx
    increases to Nx =128 both methods better approximate the exact solution and there is much
    less deviation at the end points for the finite difference method. Finally as Nx =256 both
    methods perfectly match the exact solution. Overall these plots show that the finite difference
    method takes longer to converge to the exact solution at the end points.

    Overall although Figure 9 shows that the finite difference method takes longer to run than
    the discrete fourier transform method, it is still important to note that the finite difference
    method is still very competitive with respect to runtime. Another advantage of the finite
    difference scheme is that it uses less memory than the fourier method making it more memory
    efficient. However, a weakness of the finite difference method is that it is difficult to apply
    to complex geometries. Overall my conclusion is that the compact finite difference method is
    superior compared to the discrete fourier transform given it’s competitive efficiency and it’s
    advantages in memory efficiency and it’s comparative simplicity compared to the spectral method.
    """
    if figurenumber == 1:
        # Vary Nt
        numpoints = 7
        # Set parameters
        Nt = 801
        T = 200

        Nxvalues = np.array([1,2,4,8,16,32,64])
        time_finitedifference = np.zeros(numpoints)
        time_discretefourier = np.zeros(numpoints)

        g0 = nwave(1-1j,1+2j,Nx=512,Nt=Nt,T=T,display=False)
        Nxlist = 256/Nxvalues
        j= 0
        for i in Nxvalues:
            # Extract simulation results from t=100
            g = g0[400,::i]

            t1=time.time()
            dgdx_finitedifference= finitediffmethod(g,Nx=g.shape[0],Nt=Nt,T=T)
            t2=time.time()

            time_finitedifference[j] = t2-t1
            t3=time.time()
            dgdx_discretefourier= dicretefouriermethod(g, Nx=g.shape[0])
            t4=time.time()

            time_discretefourier[j] = t4-t3
            j+=1

        plt.figure()
        plt.hold(True)
        plt.plot(Nxlist,time_finitedifference,'x--',label='Finite Difference')
        plt.plot(Nxlist,time_discretefourier,'x--',label='Discrete Fourier Transform')
        plt.hold(False)
        plt.legend()
        plt.title('Manlin Chawla: wavediff(1) \n Plot of runtime as Nx is varied')
        plt.xlabel('Nx')
        plt.ylabel('Run Time')

        if display == True:
            plt.show()

    if figurenumber==2:
        Nx = 32
        Nt = 801
        T = 200
        x = np.linspace(0,100,Nx)

        g0 = nwave(1-1j,1+2j,Nx=256,Nt=Nt,T=T,display=False)
        g0_exact = g0[400,:]
        dgdx_exact= dicretefouriermethod(g0_exact, Nx=256)

        skippoints=int(256/Nx)
        g1 = g0_exact[::skippoints]

        dgdx_finitedifference = finitediffmethod(g1,Nx=Nx,Nt=Nt,T=T)
        dgdx_discretefourier = dicretefouriermethod(g1, Nx=Nx)

        plt.figure()
        plt.hold(True)
        # Plot of dgdx using finite difference
        plt.plot(x,dgdx_finitedifference,label='Finite Difference')
        # Plot of dgdx using discrete fourier transform
        plt.plot(x,dgdx_discretefourier,label='Discrete Fourier Transform')
        plt.plot(np.linspace(0,100,256),dgdx_exact,'--',label='Exact')
        plt.hold(False)
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('dgdx')
        plt.title('Manlin Chawla: wavediff(2) \n Plot of the derivative computed via different methods for Nx = 32')

        if display == True:
            plt.show()

    if figurenumber==3:
        Nx = 64
        Nt = 801
        T = 200
        x = np.linspace(0,100,Nx)

        g0 = nwave(1-1j,1+2j,Nx=256,Nt=Nt,T=T,display=False)
        g0_exact = g0[400,:]
        dgdx_exact= dicretefouriermethod(g0_exact, Nx=256)

        skippoints=int(256/Nx)
        g1 = g0_exact[::skippoints]

        dgdx_finitedifference = finitediffmethod(g1,Nx=Nx,Nt=Nt,T=T)
        dgdx_discretefourier = dicretefouriermethod(g1, Nx=Nx)

        plt.figure()
        plt.hold(True)
        # Plot of dgdx using finite difference
        plt.plot(x,dgdx_finitedifference,label='Finite Difference')
        # Plot of dgdx using discrete fourier transform
        plt.plot(x,dgdx_discretefourier,label='Discrete Fourier Transform')
        plt.plot(np.linspace(0,100,256),dgdx_exact,'--',label='Exact')
        plt.hold(False)
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('dgdx')
        plt.title('Manlin Chawla: wavediff(3) \n Plot of the derivative computed via different methods for Nx = 64')

        if display == True:
            plt.show()

    if figurenumber==4:
        Nx = 128
        Nt = 801
        T = 200
        x = np.linspace(0,100,Nx)

        g0 = nwave(1-1j,1+2j,Nx=256,Nt=Nt,T=T,display=False)
        g0_exact = g0[400,:]
        dgdx_exact= dicretefouriermethod(g0_exact, Nx=256)

        skippoints=int(256/Nx)
        g1 = g0_exact[::skippoints]

        dgdx_finitedifference = finitediffmethod(g1,Nx=Nx,Nt=Nt,T=T)
        dgdx_discretefourier = dicretefouriermethod(g1, Nx=Nx)

        plt.figure()
        plt.hold(True)
        # Plot of dgdx using finite difference
        plt.plot(x,dgdx_finitedifference,label='Finite Difference')
        # Plot of dgdx using discrete fourier transform
        plt.plot(x,dgdx_discretefourier,label='Discrete Fourier Transform')
        plt.plot(np.linspace(0,100,256),dgdx_exact,'--',label='Exact')
        plt.hold(False)
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('dgdx')
        plt.title('Manlin Chawla: wavediff(4) \n Plot of the derivative computed via different methods for Nx = 128')

        if display == True:
            plt.show()

    if figurenumber==5:
        Nx = 256
        Nt = 801
        T = 200
        x = np.linspace(0,100,Nx)

        g0 = nwave(1-1j,1+2j,Nx=256,Nt=Nt,T=T,display=False)
        g0_exact = g0[400,:]
        dgdx_exact= dicretefouriermethod(g0_exact, Nx=256)

        skippoints=int(256/Nx)
        g1 = g0_exact[::skippoints]

        dgdx_finitedifference = finitediffmethod(g1,Nx=Nx,Nt=Nt,T=T)
        dgdx_discretefourier = dicretefouriermethod(g1, Nx=Nx)

        plt.figure()
        plt.hold(True)
        # Plot of dgdx using finite difference
        plt.plot(x,dgdx_finitedifference,label='Finite Difference')
        # Plot of dgdx using discrete fourier transform
        plt.plot(x,dgdx_discretefourier,label='Discrete Fourier Transform')
        plt.plot(np.linspace(0,100,256),dgdx_exact,'--',label='Exact')
        plt.hold(False)
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('dgdx')
        plt.title('Manlin Chawla: wavediff(5) \n Plot of the derivative computed via different methods for Nx = 256')

        if display == True:
            plt.show()

    return None #modify as needed

if __name__=='__main__':
    x=None
    #x=nwave(1-2j,1+2j,Nx=256,Nt=801,T=200,display=True)
    #x=nwave(1-1j,1+2j,Nx=256,Nt=801,T=200,display=True)
    #generate figures you are submitting
    output_fig = analyze(1,1,display=False)
    plt.savefig('fig1.png', bbox_inches="tight")
    plt.clf()
    print('plot(1) figure saved!')

    output_fig = analyze(2,1,display=False)
    plt.savefig('fig2.png', bbox_inches="tight")
    plt.clf()
    print('plot(2) figure saved!')

    output_fig = analyze(3,1,display=False)
    plt.savefig('fig3.png', bbox_inches="tight")
    plt.clf()
    print('plot(3) figure saved!')

    output_fig = analyze(3,2,display=False)
    plt.savefig('fig4.png', bbox_inches="tight")
    plt.clf()
    print('plot(4) figure saved!')

    output_fig = analyze(4,1,display=False)
    plt.savefig('fig5.png', bbox_inches="tight")
    plt.clf()
    print('plot(5) figure saved!')

    output_fig = analyze(5,1,display=False)
    plt.savefig('fig6.png', bbox_inches="tight")
    plt.clf()
    print('plot(6) figure saved!')

    output_fig = analyze(5,2,display=False)
    plt.savefig('fig7.png', bbox_inches="tight")
    plt.clf()
    print('plot(7) figure saved!')

    output_fig = wavediff(1,display=False)
    plt.savefig('fig8.png', bbox_inches="tight")
    plt.clf()
    print('plot(8) figure saved!')

    output_fig = wavediff(2,display=False)
    plt.savefig('fig9.png', bbox_inches="tight")
    plt.clf()
    print('plot(9) figure saved!')

    output_fig = wavediff(3,display=False)
    plt.savefig('fig10.png', bbox_inches="tight")
    plt.clf()
    print('plot(10) figure saved!')

    output_fig = wavediff(4,display=False)
    plt.savefig('fig11.png', bbox_inches="tight")
    plt.clf()
    print('plot(11) figure saved!')

    output_fig = wavediff(5,display=False)
    plt.savefig('fig12.png', bbox_inches="tight")
    plt.clf()
    print('plot(12) figure saved!')
