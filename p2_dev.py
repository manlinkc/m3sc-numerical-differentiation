"""M345SC Homework 3, part 2
Manlin Chawla, CID: 01205586
"""
import numpy as np
import networkx as nx
import scipy

def growth1(G,params=(0.02,6,1,0.1,0.1),T=6):
    """
    Question 2.1
    Find maximum possible growth, G=e(t=T)/e(t=0) and corresponding initial
    condition.

    Input:
    G: Networkx graph
    params: contains model parameters, see code below.
    T: time for energy growth

    Output:
    G: Maximum growth
    y: 3N-element array containing computed initial condition where N is the
    number of nodes in G and y[:N], y[N:2*N], y[2*N:] correspond to S,I,V

    Discussion:
    The growth1 function computes the maximum growth e(t=T)/e(t=0) where e(t=T) = abs(y)**2 and
    returns the growth and the initial conditions which generates this growth. My approach towards
    completing the growth1 function was to first convert the differential equations given for
    dS_i/dt, dI_i/dt and dV_i/dt into a system of linear ODEs of the form dy/dt = My. To do this,
    I constructed the 3N x 3N matrix M. In this case y is a 3N-element array y[:N], y[N:2*N], y[2*N:]
    which corresponds to S,I,V and the 3N x 3N and when matrix M is multiplied with y returns the
    model equations dy/dt = (dS/dt,dI/dt,dV/dt)tranpose.

    The next step was to exponentiate the matrix M multiplied by the given time T to get the matrix
    C =exp(M*T). The solution of the system of linear of ODES, can be written as y(t) = C(t)*y_0. Now
    we have a system of linear equations of the form Ax = b, finding the maximum growth and the corresponding
    initial conditions can be thought of generally as given a matrix A, find x so |b| is maximized.
    To find the initial conditions which generate the maximum growth we can use the properties of C=exp(M*T)
    such as: Ctranspose*C is symmetric and can be orthogonally diagonalized, the eigenvalues of Ctransposed*C
    are real and non-negative. Using the theory given in lecture 12, Ctranspose*C*v_i=lambda_1*v_1 so
    the largest eigenvalue corresponds to the maximum growth and the corresponding eigenvector v_1
    is to the initial conditions which generates the maximum growth.

    To find the maximum growth and the corresponding initial conditions, I used singular value decomposition.
    The function np.linalg.svd returns a vector s which contains the square root of the eigenvalues in descending order.
    The maximum growth is given by the largest eigenvalue which is the first entry of s squared. The initial
    conditions that generate this maximum growth is given by the corresponding eigenvector which is the first
    row of vh. Another way of obtaining the eigenvalues and eigenvectors of Ctranspose*C is to use variations
    of the QR algorithm. The cost of the this method is estimated to be O(N cubed) but depends on the rate of
    convergence of iterations. To use the QR algorithm Ctranspose*C has to explicitly be computed whereas SVD
    avoids this. This makes SVD more efficient in practise and is why I opted to use np.linalg.svd over
    np.linalg.eigs which uses QR.
    """

    a,theta,g,k,tau=params
    N = G.number_of_nodes()

    #Construct flux matrix (use/modify as needed)
    Q = [t[1] for t in G.degree()]

    Pden = np.zeros(N)
    Pden_total = np.zeros(N)
    for j in G.nodes():
        for m in G.adj[j].keys():
            Pden[j] += Q[m]
    Pden = 1/Pden
    Q = tau*np.array(Q)
    F = nx.adjacency_matrix(G).toarray()
    F = F*np.outer(Q,Pden)

    #-------------------------------------
    G=0
    y = np.zeros(3*N)

    # Construct matrix M
    # Pre-allocate 3N x 3N matrix of zeros
    M = np.zeros((3*N,3*N))
    v = np.ones(N)
    # Top left N x N matrix in M : Add -(gamma+k+tau) to every diagonal element in F
    M[:N,:N] = F + np.diag(-(g+k+tau)*v)
    # Middle left N x N matrix in M: Matrix with theta's on the diagonals
    M[N:2*N,:N] = np.diag(theta*v)
    # Bottom left N x N matrix in M: Matrix with -theta on the diagonals
    M[2*N:,:N] = np.diag(-theta*v)
    # Middle top N x N matrix in M: Matrix with alpha's on the diagonals
    M[:N,N:2*N] = np.diag(a*v)
    # Middle middle N x N matrix in M: Add -(k+alpha+tau) to every diagonal element in F
    M[N:2*N,N:2*N] = F + np.diag(-(k+a+tau)*v)
    # Right bottom N x N matrix in M: Add -(k-tau) to every diagonal element in F
    M[2*N:,2*N:] = F + np.diag((k-tau)*v)

    # Exponentiate matrix
    C = scipy.linalg.expm(M*T)
     # Singular Value Decomposition to obtain the eigenvalues and eigenvectors
    u,s,vh = np.linalg.svd(C)
    # Extract first eigenvalue, this corresponds to the maximum growth
    G = s[0]**2
    # Extract corresponding eigen vector
    y = vh[0]

    return G,y

def growth2(G,params=(0.02,6,1,0.1,0.1),T=6):
    """
    Question 2.2
    Find maximum possible growth, G=sum(Ii^2)/e(t=0) and corresponding initial
    condition.

    Input:
    G: Networkx graph
    params: contains model parameters, see code below.
    T: time for energy growth

    Output:
    G: Maximum growth
    y: 3N-element array containing computed initial condition where N is the
    number of nodes in G and y[:N], y[N:2*N], y[2*N:] correspond to S,I,V

    Discussion:
    The growth2 function computes the maximum growth e(T=T)/e(T=0) where e(t=T)=abs(I)**2 and
    returns the growth and the initial conditions which generates this growth. Similar to growth1,
    my approach was to first construct a 3N x 3N matrix M which enables us to write the differential
    equations for this model in the form of dydt = My.  (Where dy/dt = (dS/dt,dI/dt,dV/dt)transpose
    and  y is a 3N-element array y[:N], y[N:2*N], y[2*N:] which corresponds to S,I,V).  Next, I have
    exponentiated the matrix M multiplied by the given time T to get the matrix C =  exp(M*T).  The
    aim of the growth2 function is to find the maximum growth of abs(I)**2 so we only need to focus
    on this portion of the matrix C.  So the next step I did was to slice the matrix C to get a smaller
    square matrix which only contains the rows corresponding to I1,I2,…,In.  Now, we have a system of
    linear equations of the form Ax = b, but in growth2 this matrix A corresponds to the matrix of I’s.

    To find the initial conditions which generate the maximum growth we can use the properties of the
    matrix I such as: Itranspose*I is symmetric and can be orthogonally diagonalized, the eigenvalues
    of Itransposed*I are real and non-negative. Using the theory given in lecture 12, Itranspose*I*v_i=lambda_1*v_1
    so the largest eigenvalue corresponds to the maximum growth and the corresponding eigenvector v_1 is
    to the initial conditions which generates the maximum growth. Again, similar to growth1 to find the
    maximum growth and the corresponding initial conditions, I used singular value decomposition. The function
    np.linalg.svd returns a vector s which contains the square root of the eigenvalues in descending order.
    The maximum growth is given by the largest eigenvalue which is the first entry of s squared. The initial
    conditions that generate this maximum growth is given by the corresponding eigenvector which is the first
    row of vh. See the docstring for growth1 for a discussion about SVD vs QR algorithm.
    """
    a,theta,g,k,tau=params
    N = G.number_of_nodes()

    #Construct flux matrix (use/modify as needed)
    Q = [t[1] for t in G.degree()]

    Pden = np.zeros(N)
    Pden_total = np.zeros(N)
    for j in G.nodes():
        for m in G.adj[j].keys():
            Pden[j] += Q[m]
    Pden = 1/Pden
    Q = tau*np.array(Q)
    F = nx.adjacency_matrix(G).toarray()
    F = F*np.outer(Q,Pden)
    #-------------------------------------
    G=0
    y = np.zeros(3*N)

    #Add code here
    # Construct matrix M
    # Pre-allocate 3N x 3N matrix of zeros
    M = np.zeros((3*N,3*N))
    v = np.ones(N)
    # Top left N x N matrix in M : Add -(gamma+k+tau) to every diagonal element in F
    M[:N,:N] = F + np.diag(-(g+k+tau)*v)
    # Middle left N x N matrix in M: Matrix with theta's on the diagonals
    M[N:2*N,:N] = np.diag(theta*v)
    # Bottom left N x N matrix in M: Matrix with -theta on the diagonals
    M[2*N:,:N] = np.diag(-theta*v)
    # Middle top N x N matrix in M: Matrix with alpha's on the diagonals
    M[:N,N:2*N] = np.diag(a*v)
    # Middle middle N x N matrix in M: Add -(k+alpha+tau) to every diagonal element in F
    M[N:2*N,N:2*N] = F + np.diag(-(k+a+tau)*v)
    # Right bottom N x N matrix in M: Add -(k-tau) to every diagonal element in F
    M[2*N:,2*N:] = F + np.diag((k-tau)*v)

    # Exponentiate matrix
    C = scipy.linalg.expm(M*T)
    # Extract all of the I cells
    I = C[N:2*N,:]
    # Singular Value Decomposition to obtain the eigenvalues and eigenvectors
    u,s,vh = np.linalg.svd(I)
    # Extract first eigenvalue, which correpsonds to the maximum growth e(t)/e(0)
    G = s[0]**2
    # Extract corresponding eigenvector
    y =vh[0]

    return G,y


def growth3(G,params=(2,2.8,1,1.0,0.5),T=6):
    """
    Question 2.3
    Find maximum possible growth, G=sum(Si Vi)/e(t=0)
    Input:
    G: Networkx graph
    params: contains model parameters, see code below.
    T: time for energy growth

    Output:
    G: Maximum growth

    Discussion:
    The growth3 function computes the maximum growth e(T=t)/e(T=0) where e(t=T) is as given
    in the question. Similar to growth1 and growth 2, my approach was to first construct a 3N x 3N
    matrix M which enables us to write the differential equations for this model in the form of
    dydt = My.  (Where dy/dt = (dS/dt,dI/dt,dV/dt)transpose and  y is a 3N-element array y[:N],
    y[N:2*N], y[2*N:] which corresponds to S,I,V).  Next, I have exponentiated the matrix M multiplied
    by the given time T to get the matrix C = exp(M*T).

    Next, I have expressed S_i*V_i as the sum of squares by using S_i*V_i = 0.5*((S_i+V_i)**2 - (S_i)**2 – (V_i)**2),
    to get rid of the minus signs we can use complex numbers to get the final expression
    S_i*V_i = 0.5*((S_i+V_i)**2 +(1j*S_i)**2 + (1j*V_i)**2). I have constructed a 3N x 3N matrix K which
    contains the coefficients for S, I, V in this expression.  K multiplied with C=exp(M*T) is used to
    construct the constituent terms of SV in a matrix called SVmat. Now, we have a system of linear equations
    of the form Ax = b, but in growth3 this matrix A corresponds to the matrix of SV terms. Using the theory
    given in lecture 12,  the largest eigenvalue corresponds to the maximum growth.

    Finally to obtain the maximum growth this requires finding the largest eigenvalues of SVmat transpose SVmat.
    In the previous functions growth1 and growth2 I have explained my reasoning behind why SVD has been better
    than the QR algorithm in those situations. In this function, the matrix containing SVmat contains complex terms
    which causes problem when using Singular Value Decomposition. Instead to obtain the eigenvalues and eigenvectors
    I have used the np.linalg.eigvals function. However this involves some extra steps such as explicitly computing
    SVmattransposeSVmat and finding the largest eigenvalue explicity from the returned list.
    The maximum growth of e(T) in growth3 is the largest eigenvalue.

    """
    a,theta,g,k,tau=params
    N = G.number_of_nodes()

    #Construct flux matrix (use/modify as needed)
    Q = [t[1] for t in G.degree()]

    Pden = np.zeros(N)
    Pden_total = np.zeros(N)
    for j in G.nodes():
        for m in G.adj[j].keys():
            Pden[j] += Q[m]
    Pden = 1/Pden
    Q = tau*np.array(Q)
    F = nx.adjacency_matrix(G).toarray()
    F = F*np.outer(Q,Pden)

    #-------------------------------------
    G=0

    # Construct matrix M
    # Pre-allocate 3N x 3N matrix of zeros
    M = np.zeros((3*N,3*N))
    v = np.ones(N)
    # Top left N x N matrix in M : Add -(gamma+k+tau) to every diagonal element in F
    M[:N,:N] = F + np.diag(-(g+k+tau)*v)
    # Middle left N x N matrix in M: Matrix with theta's on the diagonals
    M[N:2*N,:N] = np.diag(theta*v)
    # Bottom left N x N matrix in M: Matrix with -theta on the diagonals
    M[2*N:,:N] = np.diag(-theta*v)
    # Middle top N x N matrix in M: Matrix with alpha's on the diagonals
    M[:N,N:2*N] = np.diag(a*v)
    # Middle middle N x N matrix in M: Add -(k+alpha+tau) to every diagonal element in F
    M[N:2*N,N:2*N] = F + np.diag(-(k+a+tau)*v)
    # Right bottom N x N matrix in M: Add -(k-tau) to every diagonal element in F
    M[2*N:,2*N:] = F + np.diag((k-tau)*v)

    # Exponentiate matrix
    C = scipy.linalg.expm(M*T)

    # Make matrix, use this to get the norm
    # Matrix with coeffiecients of expression 0.5*((S_i+V_i)^2 +(i*S_i)^2+(i*V_i)^2)
    K = np.zeros((3*N,3*N),dtype=complex)
    K[:N,:N] = np.diag(v)
    K[:N,2*N:] = np.diag(v)
    K[N:2*N,:N] = np.diag(1j*v)
    K[2*N:,2*N:] = np.diag(1j*v)
    K = np.sqrt(0.5)*K

    # Matix containing terms for SV
    SVmat = np.matmul(K,C)

    # QR factorisation to obtain the eigenvalues
    l = np.linalg.eigvals(np.matmul(SVmat.T,SVmat))
    # Extract first eigenvalue
    G = np.max(l)
    G = G.real

    return G

def Inew(D):
    """
    Question 2.4

    Input:
    D: N x M array, each column contains I for an N-node network

    Output:
    I: N-element array, approximation to D containing "large-variance"
    behavior

    Discussion:
    The function Inew takes measurements based on one time of I_i for M different organisms
    in the early stages of infections. The aim of the Inew function is to construct an N-element
    array which is an approximation to D containing "large-variance" behaviour. The approach I took
    is to use Principal Component Analysis to identify the most important nodes.

    PCA assumes that the column vectors of our data have been scaled to zero mean. To ensure this is
    true, I have constructed a matrix which corresponds to the data D with the mean of each column
    removed.  Now, we can consider the variance-covariance matrix C = DtransposeD  using our given
    data. Now the diagonal elements of the variance-covariance matrix contain the variances of each
    organism and the trace(C) corresponds to the total variance.

    Using PCA we want to project our data onto new variables so the covariance matrix is diagonalized.
    The means we want to find a projection matrix ‘F’ which when applied to our data (in our case D.T
    to get the right dimensions) gives us our transformed data ‘G’. We want our projection matrix to
    satisfy the condition that F*D.T= G such that the new variance-covariance matrix of our transformed
    data G*Gtransposed=diagonal matrix. Using the theory in Lecture 12 we can use the properties that C=D.T*D
    is symmetric and orthogonally diagonalizable meaning singular value decomposition can be used to obtain our
    projection matrix F. F happens to be the transpose of the eigenvectors of the variance-covariance matrix C.
    As described earlier, now that we have obtained F we can use this to explicitly compute our transformed data
    G. As SVD orders the eigenvalues in descending order the largest eigenvalues is given by the first entry
    of S. Using the theory from lecture 12 we know that the eigenvectors of C are the principal components
    and the first component points in the direction of maximum variance.  So, I have identified Inew to be
    the first row of G as the n-element array which best approximates the total variance of the dataset as
    closely as possible.

    """
    # Get shape of data
    N,M = D.shape
    # Initialize a vector of zeros for the output
    I = np.zeros(N)
    # Mean of each column
    mean = np.mean(D,axis=0)
    # Scale each column so it has zero mean
    D = D - mean

    # Check if columns have zero mean
    # print(np.mean(scaled_D,axis=0))

    # Compute svd of D
    U, S, VT = np.linalg.svd(D.T)
    #F = U transpose
    F = U.T
    # Transform D using U
    G = np.matmul(F,D.T)
    # Inew, first column of G is the product of the first eigenvector and D
    I = G[0,:]

    return I

if __name__=='__main__':
    G=None
    #add/modify code here if/as desired
    N,M = 100,5
    #N,M = 2,1
    G = nx.barabasi_albert_graph(N,M,seed=1)
    D = np.loadtxt('q22test.txt')
