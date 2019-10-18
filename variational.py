import numpy as np
from ase.units import *
import numba
from scipy.optimize import minimize
from scipy.integrate import quad,nquad

@numba.njit()
def _potential(r,r0,epsilon):
    return Hartree*Bohr/r0*(np.log(r/(r+r0))-0.116*np.exp(-r/r0))/epsilon

@numba.njit()
def _wave_exciton(x,a):
    return np.sqrt(2./(np.pi*a**2))*np.exp(-x/a)

@numba.njit()
def _wave_trion(x,y,a,b):
    wave =_wave_exciton(x,a)*_wave_exciton(y,b)
    wave+=_wave_exciton(x,b)*_wave_exciton(y,a)
    wave/=np.sqrt(2)
    wave/=np.sqrt(1+16*a**2*b**2/(a+b)**4)
    return wave

@numba.njit()
def _integral_exciton(r,r0,epsilon,a):
    return 2*np.pi*r*_potential(r,r0,epsilon) * _wave_exciton(r,a)**2


@numba.njit()
def _integral_trion_1(x,y,r0,epsilon,a):
    result=x*y*_potential(x,r0,epsilon)*_wave_trion(x,y,a[0],a[1])**2
    return result
@numba.njit()
def _integral_trion_2(x,y,r0,epsilon,a):
    result=x*y*_potential(y,r0,epsilon)*_wave_trion(x,y,a[0],a[1])**2
    return result
@numba.njit()
def _integral_trion_3(phi,x,y,r0,epsilon,a):
    radial=_potential(np.sqrt(x**2+y**2-2*np.cos(phi)*x*y),r0,epsilon)
    result=x*y*radial*_wave_trion(x,y,a[0],a[1])**2
    return result

class Variational(object):
    
    def __init__(self,mc1=0.5,mc2=0.5,mv=0.5,epsilon=1,r0=41.47):
        self.mc1=mc1
        self.mc2=mc2
        self.mv=mv
        self.epsilon=epsilon
        self.r0=r0/epsilon
        self.me1=1/(1/self.mc1-1/self.mv)
        self.me2=1/(1/self.mc2-1/self.mv)
    
    def potential(self,r):
        return _potential(r,self.r0,self.epsilon)
    
    def wave_exciton(self,r,a):
        return _wave_exciton(r,a)
    
    def kinetic_exciton(self,a):
        return Hartree*Bohr**2/(2*self.me1*a**2)

    def interaction_exciton(self,a):
        return quad(_integral_exciton,0,np.inf,args=(self.r0,self.epsilon,a,))[0]
    
    def exciton_energy(self,a):
        return self.kinetic_exciton(a)+self.interaction_exciton(a)
    
    def exciton(self,x0=[10]):
        energy=lambda a:self.kinetic_exciton(a)+self.interaction_exciton(a)
        self.exciton_result=minimize(energy, x0=x0,method='L-BFGS-B',options={'disp': True,'eps': 1e-4})
        return self.exciton_result
    
    def wave_trion(self,r,a):
        return _wave_trion(r[0],r[1],a[0],a[1])
    
    def kinetic_trion(self,a):
        norm=1+16*a[0]**2*a[1]**2/(a[0]+a[1])**4
        m1=self.me1/(1/(2*a[0]**2)+1/(2*a[1]**2)+16*(a[0]*a[1])/(a[0]+a[1])**4)
        m2=self.me2/(1/(2*a[0]**2)+1/(2*a[1]**2)+16*(a[0]*a[1])/(a[0]+a[1])**4)
        return Hartree*Bohr**2*(1/(2*m1)+1/(2*m2))/norm
        
    def interaction_trion(self,a):
        E1=nquad(_integral_trion_1,[[0,np.inf],[0,np.inf]],args=(self.r0,self.epsilon,a,))[0]
        E2=nquad(_integral_trion_2,[[0,np.inf],[0,np.inf]],args=(self.r0,self.epsilon,a,))[0]
        E3=nquad(_integral_trion_3,[[0,2*np.pi],[0,np.inf],[0,np.inf]],args=(self.r0,self.epsilon,a,))[0]
        return 2*np.pi*(2*np.pi*(E1+E2)-E3)
    
    def trion(self,x0=[10,25]):
        energy=lambda a:self.kinetic_trion(a)+self.interaction_trion(a)
        self.trion_result=minimize(energy, x0=x0,method='L-BFGS-B',options={'disp': True,'eps': 1e-4})
        return self.trion_result
    