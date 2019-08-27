from scipy import optimize
import numpy

# salt2 ###########

def model_salt2(m0, a, b, x):
    return(-m0 + (x[:,0])*a - x[:,1]*b)

def func_salt2(z, *p): 
    # this is the way brute wants its parameters, with tuppels. 
    # the z tupple is calculated in each brute loop from the rranges given in lin_fit_salt2
    m0, a, b = z # m0 is absolute M fitting parameter
    x, dx, y, dy = p
    chi2 = ((model_salt2(m0, a, b, x)) - y)**2
    chi2 = chi2/((dx[:,0]*a)**2 + (dx[:,1]*b)**2 + dy**2)
    return numpy.nansum(chi2)

# do the fit for m0,a,b (in this order) according to ranges of slice
def lin_fit_salt2(x, dx, y, dy):
    p = (x, dx, y, dy)
    rranges = (slice(-19.5, -19., 0.01), slice(0.08, 0.18, 0.01), slice(0.4, 4.0, 0.05)) # 0.01, 0.005, 0.05
    resbrute = optimize.brute(func_salt2, rranges, args=p, full_output=True, finish=optimize.fmin)

    return resbrute[0]


# salt ###########

def model_salt(m0, a, b, x):
    return(-m0 + (x[:,0])*a - x[:,1]*b)

def func_salt(z, *p):
    m0, a, b = z # m0 is absolute M
    x, dx, y, dy = p
    chi2 = ((model_salt(m0, a, b, x)) - y)**2
    chi2 = chi2/((dx[:,0]*a)**2 + (dx[:,1]*b)**2 + dy**2)
    return numpy.nansum(chi2)

# do the fit for m0,a,b (in this order) according to ranges of slice
def lin_fit_salt(x, dx, y, dy):
    p = (x, dx, y, dy)
    rranges = (slice(-20., -19., 0.05), slice(0.5, 2.5, 0.05), slice(1.5, 3.5, 0.05))
    resbrute = optimize.brute(func_salt, rranges, args=p, full_output=True, finish=optimize.fmin)

    return resbrute[0]


# mlcs ###########

def model_mlcs(m0, a, b, x):
    return(-m0 + (x[:,0])*a + x[:,1]*b)

def func_mlcs(z, *p):
    m0, a, b = z # m0 is absolute M
    x, dx, y, dy = p
    chi2 = ((model_mlcs(m0, a, b, x)) - y)**2
    chi2 = chi2/((dx[:,0]*a)**2 + (dx[:,1]*b)**2 + dy**2)
    return numpy.nansum(chi2)

# do the fit for m0,a,b (in this order) according to ranges of slice
def lin_fit_mlcs(x, dx, y, dy):
    p = (x, dx, y, dy)

    #rranges = (slice(-20., -18., 0.1), slice(-2.5, 0, 0.2), slice(-2.5, -0.5, 0.2))
    #resbrute = optimize.brute(func_mlcs, rranges, args=p, full_output=True, finish=optimize.fmin)
    minimizer_kwargs = {}
    minimizer_kwargs['args']=p
    resbrute = optimize.basinhopping(func_mlcs, (-19, -1, -1), minimizer_kwargs=minimizer_kwargs)
    #print(resbrute['x'])
    return resbrute['x']

