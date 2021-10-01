__all__ = ['read_electolum_file','py2gwyddion','crop_image']


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

    SAFETY_WIDTH = 10
    BORNE_SUP = np.Inf
    

    def crop_segment_image(img, mode="top", default_width=0):

        # Standard library import
        from collections import Counter

        get_modale = lambda a: (Counter(a)).most_common(1)[0][0]

        shift_left = []  # list of the row image left border indice
        width = []  # list of the row image width
        height = np.shape(img)[0]
        for jj in range(height):  # Sweep the image by rows
            for ii in np.nonzero(
                img[jj, :]
            ):  # Finds the left border and the image width
                try:
                    shift_left.append(ii[0])
                    width.append(ii[-1] - ii[0] + 1)

                except:  # The row contains only zero values
                    shift_left.append(0)
                    width.append(0)

        modale_shift_left = get_modale(
            shift_left
        )  # Finds the modale value of the left boudary
        if mode == "top":
            modale_width = (
                get_modale(width) - SAFETY_WIDTH
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
    for index, image in enumerate(electrolum.image):  # [:-1] to Get rid of the last image
        BORNE_INF = filters.threshold_otsu(image) # Otsu threshol is used to discriminate the noise from electolum signal
        image = np.where((image < BORNE_INF) | (image > BORNE_SUP), 0, image)
        if index == 0:  # We process the image as a top one
            image_crop, modale_width_0 = crop_segment_image(image, mode="top")
            images_crop.append(image_crop)
        else:
            image_crop, _ = crop_segment_image(
                image, mode="bottom", default_width=modale_width_0
            ) # We fix the image width to the one of the top image
            images_crop.append(image_crop)

    crop_image = np.concatenate(tuple(images_crop), axis=0)

    return crop_image