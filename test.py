import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import axes3d

t = np.linspace(0, 0.1)

x0 = -0.006
y0 = 0
v0 = 0.5
alpha = 75
A = 0
B = 0
C = 400
g=9.81

betta = [-30, -15, 0, 15, 30]
zz = []

vx0 = v0 * np.cos(np.radians(alpha))
vy0 = -v0 * np.sin(np.radians(alpha))
x = x0 + vx0 * t
y = y0 + vy0 * t + 0.5 * g * t ** 2

z = np.zeros(x.shape)


# x = np.linspace(-10, 10)
# y = np.linspace(-10, 10)
# xx, yy = np.meshgrid(x, y)
# zz = A * xx + B * yy + C

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
for b in betta:
    theta = np.radians(b)
    x_rot = x * np.cos(theta) + z * np.sin(theta)
    z_rot = -x * np.sin(theta) + z * np.cos(theta)
    ax.plot(x_rot, y, z_rot, label=f'betta={b}°')
    

# Plot a basic wireframe.
# for z in zz:
#     ax.scatter(x, y, z)

plt.xlabel('x')
plt.ylabel('y')
# plt.xlim()

plt.show()

print("")



