    # Plot of Gaussian Fourier coefficient's for A: alpha=1-2j beta=1+2j and B: alpha=1-j beta=1+2j
    if figurenumber==1:
        # Set up parameters
        Nx=256
        Nreciprocal=1/Nx

        if casenumber==1:
            alpha = 1-2j
            case = 'A'
        elif casenumber==2:
            alpha = 1-1j
            case = 'B'

        print('calling now')
        # Run non linear wave function to get g
        g = nwave(alpha,1+2j,Nx=256,Nt=801,T=200,display=False)
        print('done calling now')
        # Discard rows where t<50
        g = g[200:,:]

        # n values
        n = np.arange(-Nx/2,Nx/2)

        # Get a mean value of g at each x point across all times
        # get fourier coefficients
        gfourier=np.fft.fft(np.mean(g,axis=0))*Nreciprocal
        # Get absolute values of fourier coefficients and rearrange order
        gfourier=np.fft.fftshift(np.abs(gfourier))

        # Plot amplitude of Fourier coefficients
        if display==True:
            plt.figure()
            plt.semilogy(n,gfourier)
            plt.xlabel('n')
            plt.ylabel('Amplitude of Fourier coefficients')
            plt.title('analyze('+str(figurenumber)+','+str(casenumber)+') \n Amplitude of the Fourier Coefficients \n Case '+str(case)+': alpha='+str(alpha)+',beta=(1+2j),Nx=256,Nt=801,T=200')



#------------------------------------------------------------------------------------------

        plt.figure()
        plt.hold(True)
        # Plot of dgdx using finite difference
        plt.plot(dgdx_finitedifference,label='Finite Difference')
        # Plot of dgdx using discrete fourier transform
        plt.plot(dgdx_discretefourier,label='Discrete Fourier Transform')
        plt.legend()


        # Highest wavenumbers of original_g
        wavenumber_originalg = wavenumber(original_g)




        wavenumber_perturbedg = wavenumber(perturbed_g)



        # Take a mean acorss all points at each time


        distance1 = np.abs(wavenumber_originalg -  wavenumber_perturbedg)
        distance1=np.mean(distance1,axis=1)


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
