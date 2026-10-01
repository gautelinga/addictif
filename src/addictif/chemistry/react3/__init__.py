"""
This implements the first reaction from de Simoni et al. WRR 2007.

__author__ = Gaute Linga <gaute.linga@mn.uio.no>
with strong support from Tanguy Le Borgne

"""
import numpy as np
from numpy.polynomial import polynomial as npol

c_ref = 1.0 #mol/L 

nspec = 6

def compute_secondary_spec_initial(c_, K_):
    c_[2] = c_[0] * c_[1]**-1 *K_[0]
    c_[3] = c_[0] * c_[1]**-2 * K_[1]
    c_[4] = c_[0]**-1 * c_[1]**2 *K_[2]
    c_[5] = c_[1]**-1 *K_[3]

def compute_conserved_spec_initial(c):
    u_ = np.zeros(2)
    u_[0] = c[0] + c[2] + c[3] - c[4]
    u_[1] = c[1] - c[2] - 2*c[3] + 2*c[4] - c[5]
    return u_

def compute_primary_spec_initial(c_, u_, K_):
    K1, K2, K3, K4 = K_
    u1, u2 = u_

    a = np.zeros(7)
    a[6] = 4*K3 - 1 #4*K_[2] - 1
    a[5] = 4*K1*K3 - K1 + 2*u1 + 2*u2 #2*(u_[0] + u_[1]) - K_[0] + 4*K_[0]*K_[2]
    a[4] = K1**2*K3 + 3*K1*u1 + 2*K1*u2 - K2 + 2*K4 - 2*u1*u2 - u2**2 #-u_[1] * (2*u_[0] + u_[1]) + K_[0] * (3*u_[0] + 2*u_[1]) + 2*K_[3] - K_[1] + K_[0]**2*K_[2]
    a[3] = 2*K1*K4 - 2*K1*u1**2 - 3*K1*u1*u2 - K1*u2**2 + 4*K2*u1 + 2*K2*u2 - 2*K4*u1 - 2*K4*u2 #-K_[0] * (u_[0]+u_[1])*(2*u_[0]+u_[1]) - 2*(2*u_[0]+u_[1]) * K_[3] + 2*K_[3]*K_[0] + 2*K_[1]*(2*u_[0]+u_[1])
    a[2] = -3*K1*K4*u1 - 2*K1*K4*u2 + 2*K2*K4 - 4*K2*u1**2 - 4*K2*u1*u2 - K2*u2**2 - K4**2 #-K_[0] * K_[3] * (3*u_[0] + 2*u_[1]) - K_[3]**2 - K_[1] * (2*u_[0]+u_[1])**2 + 2*K_[1]*K_[3]
    a[1] = -K1*K4**2 - 4*K2*K4*u1 - 2*K2*K4*u2 #- 2*K_[1]*K_[3]*(2*u_[0]+u_[1]) - K_[0]*K_[3]**2
    a[0] = - K2*K4**2 #- K_[1]*K_[3]**2

    c1_ = npol.polyroots(a)
    
    sols = []
    for c1 in c1_:
        #print("c1=", c1)
        if abs(c1.imag) < 1e-8 and c1.real > 0:
            c1 = c1.real  # concentration of h+
            c0 = (- c1**2 + c1 * (2*u_[0]+u_[1]) + K_[3])/(2*c1 + K_[0])  # concentration of co2
            #print(c0)
            if c0 > 0:
                sols.append((c0, c1))

    if not (len(sols) == 1 or (len(sols) == 2 and np.linalg.norm(np.array(sols[0])-np.array(sols[1])) < 1e-7) ):
        #print(c1_)
        #print(sols)
        exit()
    
    c_[0] = sols[0][0]
    c_[1] = sols[0][1]

def compute_secondary_spec(c_, u_, K_):
    c_[2] = c_[0] * c_[1]**-1 *K_[0]
    c_[3] = c_[0] * c_[1]**-2 * K_[1]
    c_[5] = c_[1]**-1 *K_[3]
    c_[4] = u_[2] #ca2+

def compute_conserved_spec(c):
    u_ = np.zeros(3)
    u_[0] = c[0] + c[2] + c[3]
    u_[1] = c[1] - c[2] - 2*c[3] - c[5]
    u_[2] = c[4]
    return u_

def compute_primary_spec(c_, u_, K_):
    K1, K2, K3, K4 = K_
    u1, u2, u3 = u_

    a = np.zeros(5)
    a[4] = -1 #-u_[1] * (2*u_[0] + u_[1]) + K_[0] * (3*u_[0] + 2*u_[1]) + 2*K_[3] - K_[1] + K_[0]**2*K_[2]
    a[3] = u2 - K1 #-K_[0] * (u_[0]+u_[1])*(2*u_[0]+u_[1]) - 2*(2*u_[0]+u_[1]) * K_[3] + 2*K_[3]*K_[0] + 2*K_[1]*(2*u_[0]+u_[1])
    a[2] = K4 - K2 + K1*u1 + K1*u2 #-K_[0] * K_[3] * (3*u_[0] + 2*u_[1]) - K_[3]**2 - K_[1] * (2*u_[0]+u_[1])**2 + 2*K_[1]*K_[3]
    a[1] = K1*K4 + 2*K2*u1 + K2*u2 #- 2*K_[1]*K_[3]*(2*u_[0]+u_[1]) - K_[0]*K_[3]**2
    a[0] = K2*K4 #- K_[1]*K_[3]**2

    c1_ = npol.polyroots(a)
    
    sols = []
    for c1 in c1_:
        #print("c1=", c1)
        if abs(c1.imag) < 1e-8 and c1.real > 0:
            c1 = c1.real  # concentration of h+
            c0 = (c1**2*u1)/(c1**2+c1*K1+K2)  # concentration of co2
            #print(c0)
            if c0 > 0:
                sols.append((c0, c1))

    if not (len(sols) == 1 or (len(sols) == 2 and np.linalg.norm(np.array(sols[0])-np.array(sols[1])) < 1e-7) ):
        #print(c1_)
        #print(sols)
        exit()
    
    c_[0] = sols[0][0]
    c_[1] = sols[0][1]

def get_gamma(Is, z_, aom_):  # activity coefficient
    A = 0.5092
    B = 0.3282  
    bo = 0.041
    loggamma = bo * Is - A * z_**2 * np.sqrt(Is) / (1 + B * aom_ * np.sqrt(Is))
    #return np.exp(loggamma)
    return 10**loggamma

def get_gamma_coeff(Is, z_, aom_):
    gamma_ = get_gamma(Is, z_, aom_)
    #return np.array([gamma_[2]*gamma_[1]/gamma_[0], gamma_[1]**2*gamma_[3]/gamma_[0], gamma_[0]*gamma_[4]/gamma_[1]**2, gamma_[1]*gamma_[5]])
    return np.array([gamma_[0]/(gamma_[2]*gamma_[1]), gamma_[0]/(gamma_[1]**2*gamma_[3]),gamma_[1]**2/(gamma_[0]*gamma_[4]), 1.0/(gamma_[1]*gamma_[5])])

def equilibrium_constants(c_ref):
    # inverse of equilibrium constants
    K_1 = 10**-6.3447; #10**6.3447
    K_2 = 10**-16.6735 # 16.6735
    K_3 = 10**8.1934 #-8.1934
    K_4 = 10**-13.9951 #13.9951
    K_ = np.array([K_1 / c_ref, K_2 / c_ref**2, K_3, K_4 / c_ref**2])
    #K_ = np.array([K_1, K_2, K_3, K_4])
    return K_

def saturation_product_calcite():
    K_sp = 10**-8.48
    return K_sp

def rate_constants(): #mol/m2/s
    k1 = 0.89
    k2 = 5.01*10**-4
    k3 = 6.6*10**-7
    k = np.array([k1, k2, k3])
    return k

if __name__ == "__main__":
    ## Testing

    from scipy.special import erf

    # mixing ratio
    # alpha=1/2;

    # vector of chemical species 
    # 1: CO2, 2: H^+, 3: HCO3^-, 4: CO3^2-, 5: Ca^2+, 6: OH^-
    c_a = np.zeros(nspec)
    c_b = np.zeros(nspec)

    c_ref = 1 #mol/L
    #c_ref = 1
    
    K_ = equilibrium_constants(c_ref)
    #K_ = np.array([0.001, 2., 3., 5.])

    z_ = np.array([0., 1., -1., -2., 2., -1.])
    aom_ = np.array([3.0, 9.0, 4.0, 5.0, 6.0, 3.0])

    # end member fluids to be mixed 
    # primary species
    c_a[0] = 0.5 / c_ref
    c_a[1] = 10**-6 / c_ref
    #c_a[0] = 1.0
    #c_a[1] = 1.0
    
    Is_a = 0
    gamma_coeff_a_ = get_gamma_coeff(Is_a, z_, aom_)
    Ks_a_ = K_* gamma_coeff_a_
    #print(Ks_a_)

    c_b[0] = 0.01 / c_ref # 3.51*10**-4 / c_ref # 3.51*10**-4
    c_b[1] = 10**-8 / c_ref # c_ref 10**-7.16 / c_ref #10**-7.16
    
    Is_b = 0
    gamma_coeff_b_ = get_gamma_coeff(Is_b, z_, aom_)
    Ks_b_ = K_ * gamma_coeff_b_
    #print(Ks_b_)

    #print("gamma_", gamma_coeff_a_)

    compute_secondary_spec_initial(c_a, Ks_a_)
    compute_secondary_spec_initial(c_b, Ks_b_)
    u_a = compute_conserved_spec(c_a)
    #print(u_a)
    u_b = compute_conserved_spec(c_b)
    
    nx = 1000
    eta = np.linspace(-5, 5, nx)
    alpha = 0.5 * ( 1 - erf(eta/2))
    #alpha = np.linspace(0, 1, nx)

    u_ = np.outer(1-alpha, u_a) + np.outer(alpha, u_b)
    Is_ = np.outer(1-alpha, Is_a) + np.outer(alpha, Is_b)
    #print(u_.shape)

    c_ = np.zeros((nx, nspec))
    for i in range(nx):
        gamma_coeff_ = get_gamma_coeff(Is_[i], z_, aom_)
        K_new = K_ * gamma_coeff_
        compute_primary_spec(c_[i, :], u_[i, :], K_new)
        compute_secondary_spec(c_[i, :], u_[i, :], K_new)
    

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 6)

    # 1: CO2, 2: H^+, 3: HCO3^-, 4: CO3^2-, 5: Ca^2+, 6: OH^-

    ax[0].plot(alpha, c_ref * c_[:, 0])
    ax[0].plot(alpha, c_ref * c_a[0]*np.ones_like(alpha))
    ax[0].plot(alpha, c_ref * c_b[0]*np.ones_like(alpha))
    ax[0].set_title("CO2")
    ax[1].plot(alpha, -np.log10(c_ref * c_[:, 1]))
    ax[1].set_title("pH")
    ax[2].plot(alpha, c_ref * c_[:, 2])
    ax[2].set_title("HCO3^-")
    ax[3].plot(alpha, c_ref * c_[:, 3])
    ax[3].set_title("CO3^2-")
    ax[4].plot(alpha, c_ref * c_[:, 4])
    ax[4].set_title("Ca^2+")
    ax[5].plot(alpha, c_ref * c_[:, 5])
    ax[5].set_title("OH^-")
    #plt.semilogy()

    sat_rat = c_ref * c_[:, 4] * c_ref * c_[:, 3] / saturation_product_calcite()
    k = rate_constants()
    rea_rate = -(k[0]*c_[:,1] + k[1]*c_[:,0] +k[2])*(1-sat_rat)

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(alpha, sat_rat)
    ax[0].set_title("Saturation ratio")
    ax[1].plot(alpha, rea_rate)
    ax[1].set_title("Reaction rate (mol/m2/s)")

    plt.show() 
