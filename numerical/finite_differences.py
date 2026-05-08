#Implemented methods:
#1. Forward, backward, central differences
#2. Richardson extrapolation for higher accuracy
#3. Gradient computation for arrays (to get chain of position -> velocity -> acceleration)


import numpy as np

def forward_difference(f, x, h):
    return (f(x+h) - f(x))/h

def backward_difference(f,x,h):
    return (f(x) - f(x-h))/h

def central_difference(f, x, h):
    return (f(x+h) - f(x-h))/(2*h)

def second_derivative(f, x, h):
    return (f(x+h) - 2*f(x) + f(x-h))/ h**2



def richardson_extrapolation(f, x, h, order=2):
    D_h = central_difference(f,x,h)
    D_half_h = central_difference(f,x, h/2)
    factor = 2 ** order

    return (factor * D_half_h - D_h) / (factor - 1)

# This section is about differentiation methods operated on arrays

def differentiate(y, dt):
    n = len(y)
    dy = np.zeros(n)

    #Central difference
    dy[1:-1] = (y[2:] - y[:-2]) / (2 * dt)


    dy[0]  = (y[1]  - y[0])/dt     #forward
    dy[-1] = (y[-1] - y[-2])/dt     #backward

    return dy

def differentiate_richardson(y, dt):
    n = len(y)
    dy = np.zeros(n)

    # 5-point stencil
    dy[2:-2] = (-y[4:] + 8 * y[3:-1] - 8 * y[1:-3] + y[:-4]) / (12 * dt)

    dy[1] = (y[2] - y[0]) / (2 * dt)
    dy[-2] = (y[-1] - y[-3]) / (2 * dt)

    dy[0] = (y[1] - y[0]) / dt
    dy[-1] = (y[-1] - y[-2]) / dt

    return dy

#This method computes vx and vy from position arrays x_pos and y_pos
def compute_velocity(x_pos, y_pos, dt, method="central"):
    diff = differentiate_richardson if method == "richardson" else differentiate
    vx = diff(x_pos, dt)
    vy = diff(y_pos, dt)
    return vx, vy

#This method computes ax and ay from velocity arrays vx and vy
def compute_acceleration(vx, vy, dt, method="central"):
    diff = differentiate_richardson if method == "richardson" else differentiate
    ax = diff(vx, dt)
    ay = diff(vy, dt)
    return ax, ay

#Returns speed at certain frame
def compute_speed(vx, vy):
    return np.sqrt(vx**2, vy**2)

#This method gives launch angle in degrees
def compute_launch_angle(vx, vy, frame=0):
    return np.degrees(np.arctan2(vy[frame], vx[frame]))
