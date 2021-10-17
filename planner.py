########################################################################
# This files hold classes and functions that plans swarm of milirobot 
# system.
# Author: Farshid Asadi, farshidasadi47@yahoo.com
########## Libraries ###################################################
import casadi

########## classes and functions #######################################
nx = 1
nz = 1
z = casadi.SX.sym('x',nz)
x = casadi.SX.sym('x',nx)
g0 = casadi.sin(x+z)
g1 = casadi.cos(x-z)
g = casadi.Function('g',[z,x],[g0,g1])
G = casadi.rootfinder('G','newton',g)
print(G)
casadi.disp(G)