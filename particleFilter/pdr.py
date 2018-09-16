import numpy as np
from scipy.signal import butter, filtfilt, lfilter
import peakutils


def emwfilter(x, a, axis=0):
    """The exponential moving average filter y(n) = (1-a)*x(n)+a*y(n-1)
    INPUT: 
        x -- input time series
        a -- weight
    OUTPUT: 
        y -- filter data
    """
           
    y = lfilter([1-a], [1, -a], x, axis)
    
    # remove the artifacts at the beginning
    scale = 1 - np.power(a, np.linspace(1, x.shape[axis]+1, x.shape[axis])).reshape(x.shape[axis],1)
    
    return y/scale

def butter_lfilter(data, fc, fs, order=5, fb = False):
    """
    Butterworth digital low-pass filter
    Inputs:
        - data: input
        - fc: cutoff freq (Hz)
        - fs: sampling freq (Hz)
        - order: order of the filter (default to 5)
        - fb: forward and backward filter to compensate for the phase shift (default to False)
    Output:
        - filtered data
    """
    nyq = 0.5 * fs
    low = fc / nyq
    b, a = butter(order, low, btype='low')
    if fb:
        y = filtfilt(b, a, data)
    else: 
        y = lfilter(b, a, data)
    
    return y
 
def count_step(acc, fs, fc, thres=0.3, min_dist=25):
    """
    step counting from evenly sampled accelerometer data
    Inputs: 
        - acc: Nx3 matrix, where N is the number of samples. 
        Each sample contains x, y, z acceleration in device frame.
        - fs: sampling frequency
        - fc: cut off frequency of the low power filter
        - thres: threshold in peakfinding
        - min_dist: min distance two peaks need to be in peakfinding
    Output:
        - indices of step events
    """
    
    assert(acc.shape[1] == 3)
    
    fc_hpf = 0.5
    fc_lpf = fc
    
    a_emw = np.exp(-2*np.pi*fc_hpf/fs)
    acc_emw_filtered = acc - emwfilter(acc, a_emw)

    acc_magnitude = np.linalg.norm(acc_emw_filtered, axis=1)

    #use a butterworth forward+backward filter of order 5
    acc_lpf = butter_lfilter(acc_magnitude, fc_lpf, fs, 5, True)
    
    idx = peakutils.indexes(acc_lpf, thres, min_dist)

    return (idx, acc_lpf, acc_magnitude)

def pose_estimate(A, E):
    """
    Estimate a stationary device's pose using accelerometer and magnetometer readings in the device frame
    Input:
        A -- 1-D vector of acceleration along device x, y, z axis
        E -- 1-D vector of magnetic field measurement along device x, y, z axis
    
    Output: 
        R -- 3 x 3 rotation matrix
    """
    H = np.cross(E, A)
    A = A/np.linalg.norm(A)
    H = H/np.linalg.norm(H)
    M = np.cross(A,H)
    R = np.array([H, M, A])
    
    return R


def vrrotvec2mat(ax_ang):
    """
    Inputs:
        ax_ang - [x, y, z, angle], angle is counter-clock-wise
    
    Output: 
        mtx - rotation matrix
    """

    assert(np.ndim(ax_ang) == 1)
    
    if np.size(ax_ang) == 4:
        ax_ang = np.reshape(ax_ang, (4, 1))
    
    direction = ax_ang[0:3]
    angle = ax_ang[3]

    d = np.array(direction, dtype=np.float64)
    d /= np.linalg.norm(d, axis=0)
    x = d[0, :]
    y = d[1, :]
    z = d[2, :]
    c = np.cos(angle)
    s = np.sin(angle)
    tc = 1 - c

    mt11 = tc*x*x + c
    mt12 = tc*x*y - s*z
    mt13 = tc*x*z + s*y

    mt21 = tc*x*y + s*z
    mt22 = tc*y*y + c
    mt23 = tc*y*z - s*x

    mt31 = tc*x*z - s*y
    mt32 = tc*y*z + s*x
    mt33 = tc*z*z + c

    mtx = np.column_stack((mt11, mt12, mt13, mt21, mt22, mt23, mt31, mt32, mt33))

    mtx = mtx.reshape(3, 3)

    return mtx    

def EstimatePoseChangeFromGyro(gyro, sampleInterval):
    """
    Inputs: 
        - gyro: 1-D gyro reading
        - sampleinterval: time to next sample
    Output:
        - Rotation matrix for the change
    """
    gyroMag = np.linalg.norm(gyro)*sampleInterval
    rotMatrix = vrrotvec2mat([gyro[0], gyro[1], gyro[2], gyroMag])
    
    return rotMatrix

def EstimateHeadingAccMag(acc, mag):
    """
    Estimate heading direction represented as the device y in the global frame
    
    Inputs:
        - acc: 1-D acc reading
        - mag: 1-D mag reading
    Ouptputs:
        - 1-D vector in global frame
    """
    
    rotMax = pose_estimate(acc,mag)
    
    heading = np.dot(rotMax,np.array([0, 1, 0]).reshape(3,1))

    return heading

def fuseTwoHeadingsXY(v1, v2, w):
    """
    Interpolate two vectors by projecting onto the x-y plane; 
    Special care has to be taken when angles diff by pi
    
    INPUT: 
        v1 -- the first heading in a 2x1 array
        -- the second heading in a 2x1 array
        - weight in [0, 1]
    
    OUTPUT: weighted heading	
    """


    theta1 = np.arctan2(v1[1], v1[0])
    theta2 = np.arctan2(v2[1], v2[0])
    
    if (np.abs(theta1 - theta2) > np.pi):
        if (theta1 < 0):
            theta = (2*np.pi + theta1)*w + theta2*(1-w)
        else:
            theta = theta1*w + (2*np.pi + theta2)*(1-w)
    else:
        theta = theta1*w + theta2*(1-w)
    
    v = [np.cos(theta), np.sin(theta), 0]
    
    return v
