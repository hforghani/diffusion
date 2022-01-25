import numpy as np
from matplotlib import pyplot as plt

ax = plt.axes(projection='3d')
# zdata = 15 * np.random.random(100)
# xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
# ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
xdata = np.array([
    155,
    229,
    45,
    87,
    282,
    177,
    252,
    114,
    236,
    230,
    349,
    114,
    185,
    193,
    187,
    255,
    32,
    78,
    230,
    28,
    151,
    168,
])

ydata = np.array([
    49131,
    61180,
    7108,
    18345,
    53426,
    29942,
    56613,
    34830,
    63085,
    40215,
    56345,
    18574,
    36028,
    55487,
    54113,
    50335,
    3770,
    11484,
    53203,
    2081,
    34318,
    67828,
])

rtdmemm = np.array([
    1.4,
    0.624591982364654,
    0.227688940366109,
    0.475859419504801,
    1.5,
    0.890997604529063,
    1.5,
    1,
    1.6,
    1.2,
    1.6,
    0.50767518679301,
    0.93781631787618,
    1.3,
    1.4,
    1.3,
    3.5,
    12.8,
    53.4,
    2.7,
    36.2,
    78,
])

aslt = np.array([
    17.4,
    10.6,
    0.852744309107463,
    2.1,
    11.3,
    3.7,
    8.6,
    5.8,
    13.1,
    4.4,
    6.7,
    2.1,
    4.7,
    7.1,
    10.3,
    6.8,
    1.9,
    6.4,
    39,
    0.769859596093496,
    20.3,
    150,
])

ax.set_xlabel('cascades')
ax.set_ylabel('nodes')
ax.set_zlabel('time')
ax.scatter3D(xdata, ydata, rtdmemm)
ax.scatter3D(xdata, ydata, aslt)
plt.show()
