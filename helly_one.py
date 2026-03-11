import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from matplotlib.widgets import RadioButtons
# =========================================================
# NACA AIRFOIL GENERATOR
# =========================================================

def naca4(m, p, t, n=80):

    x = np.linspace(0,1,n)

    yt = 5*t*(0.2969*np.sqrt(x)
             -0.1260*x
             -0.3516*x**2
             +0.2843*x**3
             -0.1015*x**4)

    yc = np.zeros_like(x)
    dyc = np.zeros_like(x)

    for i in range(len(x)):

        if p == 0:
            yc[i] = 0
            dyc[i] = 0
        elif x[i] < p:
            yc[i] = m/p**2*(2*p*x[i]-x[i]**2)
            dyc[i] = 2*m/p**2*(p-x[i])
        else:
            yc[i] = m/(1-p)**2*((1-2*p)+2*p*x[i]-x[i]**2)
            dyc[i] = 2*m/(1-p)**2*(p-x[i])

    theta = np.arctan(dyc)

    xu = x - yt*np.sin(theta)
    yu = yc + yt*np.cos(theta)

    xl = x + yt*np.sin(theta)
    yl = yc - yt*np.cos(theta)

    xcoords = np.concatenate([xu[::-1], xl[1:]])
    ycoords = np.concatenate([yu[::-1], yl[1:]])

    return xcoords, ycoords


# =========================================================
# GENERATE BLADE SURFACE
# =========================================================

def generate_blade(R, chord, sections=20):

    xf, yf = naca4(0,0,0.30,80)

    span = np.linspace(0.2*R, R, sections)

    X=[]
    Y=[]
    Z=[]

    for r in span:

        x = chord*(xf-0.25)
        y = np.full_like(x,r)
        z = chord*yf

        X.append(x)
        Y.append(y)
        Z.append(z)

    return np.array(X), np.array(Y), np.array(Z)


# =========================================================
# ROTATION TRANSFORM
# =========================================================

def rotate_blade(X,Y,Z,psi,beta):

    Xr = X*np.cos(psi) - Y*np.sin(psi)
    Yr = X*np.sin(psi) + Y*np.cos(psi)
    Zr = Z + Y*np.sin(beta)

    return Xr,Yr,Zr


# =========================================================
# HELICOPTER ROTOR CLASS
# =========================================================

class HelicopterRotor:

    def __init__(self,p):

        self.R = p["R"]
        self.Nb = p["Nb"]
        self.RPM = p["RPM"]

        self.rho = p["rho"]
        self.chord = p["chord"]

        self.CLalpha = p["CLalpha"]
        self.Cd0 = p["Cd0"]

        self.theta0 = np.deg2rad(p["collective"])
        self.theta1c = np.deg2rad(p["cyclicLat"])
        self.theta1s = np.deg2rad(p["cyclicLon"])
        
        if p["hover"]:
            self.Vf=0
        else:
            self.Vf = p["forwardSpeed"]

        self.dt = p["dt"]
        self.simTime = p["simTime"]

        self.omega = self.RPM*2*np.pi/60

        self.Nr = 25
        self.r = np.linspace(0.2*self.R,self.R,self.Nr)
        self.dr = self.r[1]-self.r[0]

        self.beta = np.zeros(self.Nb)
        self.beta_dot = np.zeros(self.Nb)

        self.gamma = 20

        self.time = np.arange(0,self.simTime,self.dt)

        self.Xb,self.Yb,self.Zb = generate_blade(self.R,self.chord,sections=10)

        self.init_plot()
        

    # -----------------------------------------------------
    


    def init_plot(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111,projection='3d')
        R_hub = 0.3       # raggio hub
        H_hub = 1.0       # altezza hub
        self.plot_hub(self.ax, R_hub, H_hub, color='gray', alpha=0.9)
        maxR = self.R*0.02#*1.5
        
        self.ax.set_xlim(-maxR*32,maxR*32)
        self.ax.set_ylim(-maxR*32,maxR*32)
        self.ax.set_zlim(-self.R,self.R)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_axis_off()
        self.blades=[]
        self.blade_colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan']
        for i in range(self.Nb):
            color = self.blade_colors[i % len(self.blade_colors)]
            surf = self.ax.plot_surface(
                self.Xb,self.Yb,self.Zb,
                color=color,
                edgecolor='none'
            )

            self.blades.append(surf)

        self.circle_line, = self.ax.plot([], [], [], color='black', linewidth=1)
        # spazio per slider
        plt.subplots_adjust(bottom=0.25)

        # slider axes
        ax_collective = plt.axes([0.2, 0.15, 0.6, 0.03])
        ax_cyclicLat = plt.axes([0.2, 0.10, 0.6, 0.03])
        ax_cyclicLon = plt.axes([0.2, 0.05, 0.6, 0.03])

        # sliders
        self.slider_collective = Slider(
            ax_collective,
            'Collective',
            -5,
            20,
            valinit=np.rad2deg(self.theta0)
        )

        self.slider_cyclicLat = Slider(
            ax_cyclicLat,
            'Cyclic Lat',
            -10,
            10,
            valinit=np.rad2deg(self.theta1c)
        )

        self.slider_cyclicLon = Slider(
            ax_cyclicLon,
            'Cyclic Lon',
            -10,
            10,
            valinit=np.rad2deg(self.theta1s)
        )
        # view dropdown
        ax_view = plt.axes([0.85,0.5,0.12,0.15], facecolor='lightgray')
        self.views = {"Top":{"elev":90,"azim":-90},"Side":{"elev":0,"azim":0},"View 1":{"elev":15,"azim":0}}
        self.view_menu = RadioButtons(ax_view,list(self.views.keys()))
        self.view_menu.on_clicked(self.change_view)
    
    
    def change_view(self,label):
        v = self.views[label]
        self.ax.view_init(elev=v['elev'], azim=v['azim'])
        plt.draw()

    # -----------------------------------------------------

    def aerodynamic_model(self,psi):

        pitch = self.theta0 + self.theta1c*np.cos(psi) + self.theta1s*np.sin(psi)

        Lift=0
        Torque=0

        for ri in self.r:

            Ut = self.omega*ri + self.Vf*np.sin(psi)
            Up = self.Vf*np.cos(psi)

            Vrel = np.sqrt(Ut**2 + Up**2)

            phi = np.arctan2(Up,Ut)

            alpha = pitch - phi

            CL = self.CLalpha*alpha
            CD = self.Cd0 + 0.01*CL**2

            q = 0.5*self.rho*Vrel**2

            dL = q*self.chord*CL*self.dr
            dD = q*self.chord*CD*self.dr

            dT = dL*np.cos(phi) - dD*np.sin(phi)
            dQ = ri*(dD*np.cos(phi) + dL*np.sin(phi))

            Lift += dT
            Torque += dQ

        return Lift,Torque,pitch


    # -----------------------------------------------------

    def update(self,frame):
        # read slider values
        self.theta0 = np.deg2rad(self.slider_collective.val)
        self.theta1c = np.deg2rad(self.slider_cyclicLat.val)
        self.theta1s = np.deg2rad(self.slider_cyclicLon.val)
        
        az = self.omega*self.time[frame]
        # Generates the base circle
        tt = np.linspace(0, 2*np.pi, 200)
        x = self.R * np.cos(tt + az)   # az = omega * t
        y = self.R * np.sin(tt + az)
        z = np.zeros_like(x)

        # se il cerchio è una linea 3D
        if hasattr(self, 'circle_line'):
            self.circle_line.remove()

        self.circle_line, = self.ax.plot(x, y, z, color='black', linewidth=1)
        TotalLift=0
        TotalTorque=0
        
        for b in range(self.Nb):

            psi = az + 2*np.pi*b/self.Nb

            Lift,Torque,pitch = self.aerodynamic_model(psi)
            
            beta_dd = self.gamma*(pitch-self.beta[b]) - 1.0*self.omega*self.beta_dot[b]

            self.beta_dot[b] += beta_dd*self.dt
            self.beta[b] += self.beta_dot[b]*self.dt

            Xr,Yr,Zr = rotate_blade(self.Xb,self.Yb,self.Zb,psi,self.beta[b])

            self.blades[b].remove()
            color = self.blade_colors[b % len(self.blade_colors)]
            self.blades[b] = self.ax.plot_surface(
                Xr,Yr,Zr,
                color=color,
                edgecolor='none'
            )
            TotalLift += Lift
            TotalTorque += Torque
            
        Power = TotalTorque*self.omega

        self.ax.set_title(
            f"Lift={TotalLift/1000:.1f} kN   "
            f"Torque={TotalTorque/1000:.1f} kNm   "
            f"Power={Power/1000:.1f} kW"
        )
        


    # -----------------------------------------------------

    def run(self):

        anim = FuncAnimation(
            self.fig,
            self.update,
            frames=len(self.time),
            interval=self.dt*1000
        )

        plt.show()

    #======================================================
    #HUB PLOT
    #=====================================================
    def plot_hub(self,ax, R_hub, H_hub, color='gray', alpha=0.8, resolution=50):
    

        # Cylindrical coordinates
        theta = np.linspace(0, 2*np.pi, resolution)
        z = np.linspace(0, H_hub, resolution)
        Theta, Z = np.meshgrid(theta, z)
        X = R_hub * np.cos(Theta)
        Y = R_hub * np.sin(Theta)

        # Cylindrical surface
        ax.plot_surface(X, Y, Z, color=color, alpha=alpha, edgecolor='k')
        
        # Lower base
        ax.plot_surface(R_hub*np.cos(Theta), R_hub*np.sin(Theta), np.zeros_like(Z), color=color, alpha=alpha)
        # Upper base
        ax.plot_surface(R_hub*np.cos(Theta), R_hub*np.sin(Theta), H_hub*np.ones_like(Z), color=color, alpha=alpha)
    # =========================================================
# RUN SIMULATION
# =========================================================

params = {

"R":6,
"Nb":4,   ###number of blades
"RPM":360,

"rho":1.225,
"chord":0.45,

"CLalpha":5.7,
"Cd0":0.012,

"collective":0,
"cyclicLat":0,
"cyclicLon":0,

"hover":True,
"forwardSpeed":35,

"dt":0.02,
"simTime":15

}

sim = HelicopterRotor(params)
sim.run()
