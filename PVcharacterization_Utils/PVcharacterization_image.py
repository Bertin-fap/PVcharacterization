__all__ = ['apply_savgol_filter',
           'crop_image',
           'image_padding',
           'ines_crop',
           'laplacian_kern',
           'py2gwyddion',
           'read_electolum_file',
           'sgolay2d',
           'sgolay2d_kernel',]


def read_electolum_file(file, pack=True):

    """
    Reads raw files .data generated by the greateyes camera
    
    Args:
        file (Path): absolute path of the binary file
        pack (boolean): if true the F frame are stacked in one image
        
    Returns:
        electrolum (namedtuple):
           electrolum.imgWidth (integer): number N of rows
           electrolum.imgHeight (integer): number M of columns
           electrolum.numPatterns (integer): number F of frame
           electrolum.image (list of F NxM nparray of floats): list of F images
           
    todo: the info1, info2, info3 fields are not correctly decoded

    """

    # Standard library import
    import struct
    from collections import namedtuple

    # 3rd party imports
    import numpy as np

    data_struct = namedtuple(
        "PV_electrolum",
        [
            "imgWidth",
            "imgHeight",
            "numPatterns",
            "exptime",
            "info1",
            "info2",
            "info3",
            "image",
        ],
    )
    data = open(file, "rb").read()

    # Header parsing
    fmt = "2i"
    imgWidth, imgHeight = struct.unpack(fmt, data[: struct.calcsize(fmt)])
    pos = struct.calcsize(fmt) + 4
    fmt = "i"
    numPatterns = struct.unpack(fmt, data[pos : pos + struct.calcsize(fmt)])[0]

    pos = 18
    lastPaternIsFractional = struct.unpack(fmt, data[pos : pos + struct.calcsize(fmt)])[
        0
    ]
    if lastPaternIsFractional == 1:
        print("WARNING: the last image will contain overlapping information")

    pos = 50
    exptime = struct.unpack(fmt, data[pos : pos + struct.calcsize(fmt)])[0]

    fmt = "21s"
    pos = 100
    info1 = struct.unpack(fmt, data[pos : pos + struct.calcsize(fmt)])[
        0
    ]  # .decode('utf-8')

    fmt = "51s"
    pos = 130
    info2 = struct.unpack(fmt, data[pos : pos + struct.calcsize(fmt)])[
        0
    ]  # .decode('utf-8')

    fmt = "501s"
    pos = 200
    info3 = struct.unpack(fmt, data[pos : pos + struct.calcsize(fmt)])[
        0
    ]  # .decode('utf-8')

    # Images parsing
    list_images = []
    for numPattern in range(numPatterns):
        fmt = str(imgWidth * imgHeight) + "H"
        pos = 1024 * 4 + numPattern * struct.calcsize(fmt)

        y = struct.unpack(fmt, data[pos : pos + struct.calcsize(fmt)])
        list_images.append(np.array(y).reshape((imgHeight, imgWidth)))

    if pack:
        list_images = [np.concatenate(tuple(list_images), axis=0)]

    return data_struct(
        imgWidth, imgHeight, numPatterns, exptime, info1, info2, info3, list_images
    )


def py2gwyddion(image, file):

    """The function py2gwyddion stores an array as a simple field files Gwyddion
        format(.gsf). For more information see the Gwyddionuser guide §5.13 
        http://gwyddion.net/documentation/user-guide-en/gsf.html
    """
    # Standard library import
    import struct

    # 3rd party imports
    import numpy as np

    imgHeight, imgWidth = np.shape(image)
    a = b"Gwyddion Simple Field 1.0\n"  # magic line
    a += f"XRes = {str(imgWidth)}\n".encode("utf8")
    a += f"YRes = {str(imgHeight)}\n".encode("utf8")
    a += (chr(0) * (4 - len(a) % 4)).encode("utf8")  # Adds the NULL padding
                                                     # accordind to the Gwyddion .gsf format

    z = image.flatten().astype(
        dtype=np.float32
    )  # Gwyddion reads IEEE 32bit single-precision floating point numbers
    a += struct.pack(str(len(z)) + "f", *z)

    with open(file, "wb") as binary_file:
        binary_file.write(a)

def crop_image(file):

    """
    The function crop_image reads, crops and stitches a set of electroluminesence images.
    
    Args:
       file (Path) : absolute path of the electroluminescence image.
       
    Returns:
       
    """
    
    # 3rd party import
    import numpy as np
    from skimage import filters
 
    #Internal import 
    import PVcharacterization_Utils as pv

    SAFETY_WIDTH = 10 # The width of the top image is set to the coputed with - SAFETY_WIDTH
    BORNE_SUP = np.Inf
    N_SIGMA = 5       # Number of RMS used to discriminate outlier
    

    def crop_segment_image(img, mode="top", default_width=0):

        # Standard library import
        from collections import Counter

        get_modale = lambda a: (Counter(a[a != 0])).most_common(1)[0][0]

        shift_left = []  # list of the row image left border indice
        width = []       # list of the row image width
        height = np.shape(img)[0]
        for jj in range(height):  # Sweep the image by rows
            ii = np.nonzero(
                img[jj]
            )[0]  # Finds the left border and the image width
            if len(ii):
                shift_left.append(ii[0])
                width.append(ii[-1] - ii[0] + 1)

            else:  # The row contains only zero value
                shift_left.append(0)
                width.append(0)

        modale_shift_left = get_modale(
            np.array(shift_left)
        )  # Finds the modale value of the left boudary
        if mode == "top":
            modale_width = (
                get_modale(np.array(width)) - SAFETY_WIDTH
            )  # Reduces the width to prevent for
            # further overestimation
        else:  # Fixes the image width to the one of the top layer
            modale_width = default_width

        if (
            mode == "top"
        ):  # Slice the image throwing away the upper row with width < modale_width
            img_crop = img[
                np.where(width >= modale_width)[0][0] : height,
                modale_shift_left : modale_width + modale_shift_left,
            ]

        else:  # Slice the image throwing away the lower row with width < modale_width
            img_crop = img[
                0 : np.where(width >= modale_width)[0][-1],
                modale_shift_left : modale_width + modale_shift_left,
            ]

        return img_crop, modale_width

    electrolum = pv.read_electolum_file(file, pack=False)

    images_crop = []
    list_borne_inf =[]
    nbr_images = len(electrolum.image)
    for index, image in enumerate(electrolum.image):  # [:-1] to Get rid of the last image
        BORNE_INF = filters.threshold_otsu(image) # Otsu threshold is used to discriminate the noise from electolum signal
        list_borne_inf.append(BORNE_INF)
        if index == nbr_images - 1: # get rid of the last image if the image contains only noise
            if(np.abs(np.mean(list_borne_inf) - BORNE_INF) > N_SIGMA * np.sqrt(np.std(list_borne_inf))):break
        image = np.where((image < BORNE_INF) | (image > BORNE_SUP), 0, image)
        if index == 0:  # We process the top image
            image_crop, modale_width_0 = crop_segment_image(image, mode="top")
            images_crop.append(image_crop)
        else:
            image_crop, _ = crop_segment_image(
                image, mode="bottom", default_width=modale_width_0
            ) # We fix the image width to the one of the top image
            images_crop.append(image_crop)

    croped_image = np.concatenate(tuple(images_crop), axis=0)

    return croped_image
    
def laplacian_kern(size,sigma):
    """
    laplacian_kern computes 2D laplacian kernel. 
    See Digital Image Processing R. Gonzales, R. Woods p. 582; https://theailearner.com/2019/05/25/laplacian-of-gaussian-log/ 
    
    Args:
        size (int): the pixel size of the kernel
        sigma (float): the standard deviation
    
    Returns:
        (array): kernel matrix
  
    """
    import numpy as np
    

    mexican_hat = lambda x,y:-1/(np.pi*sigma**4)*(1-(x**2+y**2)/(2*sigma**2))*np.exp(-(x**2+y**2)/(2*sigma**2))

    size = size+1 if size%2==0 else size # Force size to be odd
    
    x = np.linspace(-(c:=size//2), c, size)
    x_1, y_1 = np.meshgrid(x, x)
    kern = mexican_hat(x_1, y_1)
    
    kern = kern - kern.sum()/(size**2) # The kernel coefficients must sum to zero so that the response of
                                       # the mask is zero in areas of constant grey level
    
    return kern
    
def sgolay2d_kernel ( window_size, order):
    
    """
    sgolay2d_kernel computes the kernel of solvay-Golay filter.
    see https://www.uni-muenster.de/imperia/md/content/physik_ct/pdf_intern/07_savitzky_golay_krumm.pdf
    
    Args:
        window_size (int): size of the  squared image patche
        order (int): order of the smoothing polynomial
        
    Return:
       (array): kernel of size window_size*window_size
    """
    import itertools
    
    import numpy as np
    
    set_jacobian_row = lambda x,y: [ x**(k-n) * y**n for k in range(order+1) for n in range(k+1) ]
    
    

    if  window_size % 2 == 0:
        raise ValueError('window_size must be odd')
        
    n_terms = ( order + 1 ) * ( order + 2)  / 2.0 #number of terms in the polynomial expressio
    if window_size**2 < n_terms:
        raise ValueError('order is too high for the window size')

    half_size = window_size // 2
 
    ind = np.arange(-half_size, half_size+1)
    
    jacobian_mat = [set_jacobian_row(x[0], x[1]) for x in itertools.product(ind, repeat=2)]
    
    jacobian_pseudo_inverse = np.linalg.pinv(jacobian_mat)
    jacobian_pseudo_inverse = [jacobian_pseudo_inverse[i].reshape(window_size, -1) 
                               for i in range(jacobian_pseudo_inverse.shape[0])]
    return jacobian_pseudo_inverse

 
def image_padding(z,window_size): 
    
    import numpy as np
    
    if  window_size % 2 == 0:
        raise ValueError('window_size must be odd')

    half_size = window_size // 2
    
    # pad input array with appropriate values at the four borders
    new_shape = z.shape[0] + 2*half_size, z.shape[1] + 2*half_size
    Z = np.zeros( (new_shape) )
    # top band
    band = z[0, :]
    Z[:half_size, half_size:-half_size] =  band -  np.abs( np.flipud( z[1:half_size+1, :] ) - band )
    # bottom band
    band = z[-1, :]
    Z[-half_size:, half_size:-half_size] = band  + np.abs( np.flipud( z[-half_size-1:-1, :] )  -band )
    # left band
    band = np.tile( z[:,0].reshape(-1,1), [1,half_size])
    Z[half_size:-half_size, :half_size] = band - np.abs( np.fliplr( z[:, 1:half_size+1] ) - band )
    # right band
    band = np.tile( z[:,-1].reshape(-1,1), [1,half_size] )
    Z[half_size:-half_size, -half_size:] =  band + np.abs( np.fliplr( z[:, -half_size-1:-1] ) - band )
    # central band
    Z[half_size:-half_size, half_size:-half_size] = z

    # top left corner
    band = z[0,0]
    Z[:half_size,:half_size] = band - np.abs( np.flipud(np.fliplr(z[1:half_size+1,1:half_size+1]) ) - band )
    # bottom right corner
    band = z[-1,-1]
    Z[-half_size:,-half_size:] = band + np.abs( np.flipud(np.fliplr(z[-half_size-1:-1,-half_size-1:-1]) ) - band )

    # top right corner
    band = Z[half_size,-half_size:]
    Z[:half_size,-half_size:] = band - np.abs( np.flipud(Z[half_size+1:2*half_size+1,-half_size:]) - band )
    # bottom left corner
    band = Z[-half_size:,half_size].reshape(-1,1)
    Z[-half_size:,:half_size] = band - np.abs( np.fliplr(Z[-half_size:, half_size+1:2*half_size+1]) - band )
    
    return Z 
    
def apply_savgol_filter(Z,jacobian_pseudo_inverse,derivative=None):

    import scipy.signal

    if derivative == None:
        m = jacobian_pseudo_inverse[0]
        return scipy.signal.fftconvolve(Z, m, mode='valid')
    elif derivative == 'col':
        c = jacobian_pseudo_inverse[1]
        return scipy.signal.fftconvolve(Z, -c, mode='valid')
    elif derivative == 'row':
        r = jacobian_pseudo_inverse[2]
        return scipy.signal.fftconvolve(Z, -r, mode='valid')
    elif derivative == 'both':
        c = jacobian_pseudo_inverse[1]
        r = jacobian_pseudo_inverse[2]
        return (scipy.signal.fftconvolve(Z, -r, mode='valid'),
                scipy.signal.fftconvolve(Z, -c, mode='valid'))
                
def sgolay2d (z, window_size=5, order=3, derivative=None):

    jacobian_pseudo_inverse = sgolay2d_kernel( window_size, order)
    z_padded = image_padding(z,window_size)
    z_filtered = apply_savgol_filter(z_padded,jacobian_pseudo_inverse,derivative=derivative)
    
    return z_filtered
    
def ines_crop(image,autocrop_para):

    import cv2
    import numpy as np
    
    im_sg = sgolay2d(np.float32(image),
                     autocrop_para['2D SG window_size'],
                     autocrop_para['2D SG order'],
                     derivative=None)

    array_im_lap = cv2.filter2D(im_sg,
                                -1,
                                laplacian_kern(autocrop_para['laplacian kernel size'],
                                                   autocrop_para['laplacian kernel sigma']))

    ind_v = np.where(np.abs(array_im_lap.sum(axis=1)) > 
                     np.std(array_im_lap.sum(axis=1))/autocrop_para['fraction of the std laplacian'])[0]

    ind_h = np.where(np.abs(array_im_lap.sum(axis=0)) > 
                     np.std(array_im_lap.sum(axis=0))/autocrop_para['fraction of the std laplacian'])[0]
    ind_h = ind_h[np.where((ind_h>autocrop_para['ind_h_min'])&(ind_h<autocrop_para['ind_h_max']))[0]]

    array_im_red = image[ind_v.min():ind_v.max(),ind_h.min():ind_h.max()]
    
    return array_im_red

    
