#!/usr/bin/env python

def rebin(array, dimensions=None, scale=None):
    """
    Return the array 'array' to the new 'dimensions'
    conserving the flux in the bins. The sum of the
    array will remain the same as the original array.
    Congrid from the scipy recipies does not generally
    conserve surface brightness with reasonable accuracy.
    As such, even accounting for the ratio of old and new
    image areas, it does not conserve flux. This function
    nicely solves the problem and more accurately
    redistributes the flux in the output. This function
    conserves FLUX so input arrays in surface brightness
    will need to use the ratio of the input and output
    image areas to go back to surface brightness units.
    
    EXAMPLE
    -------
    
    In [0]:
    
    # input (1,4) array (sum of 6)
    y = np.array([0,2,1,3]).reshape(1,4).astype(float)
    # rebin to (1,3) array
    yy = rebin(y,dimensions=(1,3))
    print yy
    print np.sum(yy)
    
    Out [0]:
    
    Rebinning to Dimensions: 1, 3
    [[0.66666667 2.         3.33333333]]
    6.0
    RAISES
    ------
    AssertionError
        If the totals of the input and result array don't
        agree, raise an error because computation may have
        gone wrong.
        
    Copyright: Martyn Bristow (2015) and licensed under GPL v3:
    i.e. free to use/edit but no warranty.
    """
    import numpy as np
    if dimensions is not None:
        if isinstance(dimensions, float):
            dimensions = [int(dimensions)] * len(array.shape)
        elif isinstance(dimensions, int):
            dimensions = [dimensions] * len(array.shape)
        elif len(dimensions) != len(array.shape):
            raise RuntimeError('')
    elif scale is not None:
        if isinstance(scale, float) or isinstance(scale, int):
            dimensions = map(int, map(round, map(lambda x: x*scale, array.shape)))
        elif len(scale) != len(array.shape):
            raise RuntimeError('')
    else:
        raise RuntimeError('Incorrect parameters to rebin.\n\trebin(array, dimensions=(x,y))\n\trebin(array, scale=a')
    #print "Rebinning to Dimensions: %s, %s" % tuple(dimensions)
    import itertools
    dY, dX = map(divmod, map(float, array.shape), dimensions)
 
    result = np.zeros(dimensions)
    for j, i in itertools.product(*map(range, array.shape)):
        (J, dj), (I, di) = divmod(j*dimensions[0], array.shape[0]), divmod(i*dimensions[1], array.shape[1])
        (J1, dj1), (I1, di1) = divmod(j+1, array.shape[0]/float(dimensions[0])), divmod(i+1, array.shape[1]/float(dimensions[1]))
        
        # Moving to new bin
        # Is this a discrete bin?
        dx,dy=0,0
        if (I1-I == 0) | ((I1-I == 1) & (di1==0)):
            dx = 1
        else:
            dx=1-di1
        if (J1-J == 0) | ((J1-J == 1) & (dj1==0)):
            dy=1
        else:
            dy=1-dj1
        # Prevent it from allocating outide the array
        I_=min(dimensions[1]-1,I+1)
        J_=min(dimensions[0]-1,J+1)
        result[J, I] += array[j,i]*dx*dy
        result[J_, I] += array[j,i]*(1-dy)*dx
        result[J, I_] += array[j,i]*dy*(1-dx)
        result[J_, I_] += array[j,i]*(1-dx)*(1-dy)
    allowError = 0.1
    print(array.sum(),result.sum())
    assert (abs(array.sum()) < abs(result.sum()) * (1+allowError)) & (abs(array.sum()) > abs(result.sum()) * (1-allowError))
    return result
