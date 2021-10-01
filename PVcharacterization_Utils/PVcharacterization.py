""" Creation: 2021.09.07
    Last update: 2021.09.20
    
    Useful functions for correctly parsing the aging data files
"""
__all__ = [
    "crop_image",
    "data_parsing",
    "df2sqlite",
    "parse_filename",
    "plot_diff_param",
    "py2gwyddion",
    "read_electolum_file",
    "sieve_files",
]

#Internal imports 
from .PVcharacterization_global import (DEFAULT_DIR,
                                        DATA_BASE_NAME,
		                                DATA_BASE_TABLE,
		                                USED_COLS,
		                                PARAM_UNIT_DIC)
                                       
def data_parsing(filepath, parse_all=True):

    """
    The function `data_parsing` reads a csv file organized as follow:
    
                ==========  =================================
                Title:       HET JNHM72 6x12 M2 0200W
                Comment:     
                Op:          Util
                ID:          JINERGY3272023326035_0200W_T0
                Mod Type:    ModuleType1
                Date:        2/15/2021
                ...          ...
                Voltage:     Current:
                -0.740710    1.8377770
                -0.740387    1.8374640
                -0.734611    1.8376460
                ...          ....
                Ref Cell:   Lamp I:
                199.9875    200.0105
                199.9824    200.1674
                ...         ...
                Voltage1:   Current1:
                -0.740710   1.8377770
                -0.740387   1.8374640
                -0.734611   1.8376460
                ...         ....
                Ref Cell1:  Lamp I1:
                ...         ....
                Voltage2:   Current2:
                -0.740710   1.8377770
                -0.740387   1.8374640
                -0.734611   1.8376460
                ...         ....
                Ref Cell2:  Lamp I2:
                0.008593    1.823402
                0.043122    1.823085
                ...         ....
                DarkRsh:    0
                DarkV:       ark I:
                ==========  =================================
    
    The `.csv` file is parsed in a namedtuple `data` where:
       
       - data.IV0, data.IV1, data.IV2 are dataframes containing the `IV` curves as :
       
                ======== ==========
                Voltage	 Current
                ======== ==========
                0.008593  1.823402
                0.043122  1.823085
                0.070891  1.823253
                xxxx      xxxx
                50.0      1.823253
                ======== ==========
       - data.Ref_Cell0, data.Ref_Cell1, data.Ref_Cell2 are dataframes containing
       the irradiance curves as:
       
                ======== ==========
                Ref_Cell  Lamp_I
                ======== ==========
                199.9875  200.0105
                199.9824  200.1674
                xxxxx     xxxxx
                199.9824  200.0074
                ======== ==========
       - data.meta_data is a dict containing the header :
    .. code-block:: python 
    
      data.meta_data = {
      "Title":"HET JNHM72 6x12 M2 0200W",
      "Comment":"",
      "Op":"Util",
      .... :.....,
      }
      
    Args:
        filename (Path): name of the .csv file
    
    Returns:
        data (namedtuple): results of the file parsing (see summary)
    
    """

    # Standard library imports
    from collections import namedtuple

    # 3rd party imports
    import numpy as np
    import pandas as pd

    data_struct = namedtuple(
        "PV_module_test",
        ["meta_data", "IV0", "IV1", "IV2", "Ref_Cell0", "Ref_Cell1", "Ref_Cell2"],
    )

    df_data = pd.read_csv(filepath, sep=",", skiprows=0, header=None)

    index_data_header = np.where(
        df_data.iloc[:, 0].str.contains(
            "^ Volt|Ref Cell",  # Find the indice of the
            case=True,  # headers of th IV and
            regex=True,
        )
    )[
        0
    ]  # Ref Cell data

    index_data_header = np.insert(
        index_data_header,  # Insersion of index 0 and the index of th
        [0, len(index_data_header)],  # last numerical value
        [0, len(df_data) - 3],
    )

    meta_data = df_data.iloc[np.r_[index_data_header[0] : index_data_header[1]]]
    meta_data = dict(zip(meta_data[0], meta_data[1]))
    meta_data = {key.split(":")[0]: val for key, val in meta_data.items()}

    if not parse_all:
        data = data_struct(
            meta_data=meta_data,
            IV0=None,
            IV1=None,
            IV2=None,
            Ref_Cell0=None,
            Ref_Cell1=None,
            Ref_Cell2=None,
        )
        return data

    list_df = []
    for i in range(1, len(index_data_header) - 1):
        dg = df_data.iloc[
            np.r_[index_data_header[i] + 1 : index_data_header[i + 1]]
        ].astype(float)

        dg = dg.loc[dg[0] > 0]
        dg.index = list(range(len(dg)))

        if "Voltage" in df_data.iloc[index_data_header[i]][0]:
            dg.columns = ["Voltage", "Current"]
        else:
            dg.columns = ["Ref_Cell", "Lamp_I"]

        list_df.append(dg)

    data = data_struct(
        meta_data=meta_data,
        IV0=list_df[0],
        IV1=list_df[2],
        IV2=list_df[4],
        Ref_Cell0=list_df[1],
        Ref_Cell1=list_df[3],
        Ref_Cell2=list_df[0],
    )
    return data


def parse_filename(file):

    """
    Let the string "file" structured as follow:
      '~/XXXXXXX<ddddddddddddd>_<dddd>W_T<d>.csv'
    where <> is a placeholder, d a digit, X a capital letter and ~ the relative or absolute path of the file
    
    parse_filename parses "file" in three chunks: JINERGY<ddddddddddddd>, <dddd>, T<d> and stores them in
    the nametuple FileInfo.
    
    Args:
       file (str): filename to parse
    
    Returns:
        data (namedtuple): results of the file parsing (see summary)
        
    Examples:
    let file = 'C:/Users/franc/PVcharacterization_files/JINERGY3272023326035_0200W_T2.csv'
    we obtain:
        FileInfo.power = 200
        FileInfo.time = "T2"
        FileInfo.time = "JINERGY3272023326035"
    
    """
    # Standard library imports
    from collections import namedtuple
    import re

    FileNameInfo = namedtuple("FileNameInfo", "power time name file")
    re_power = re.compile(r"(?<=\_)\d{4}(?=W\_)")
    re_time = re.compile(r"(?<=\_)T\d{1}(?=\.)")
    re_name = re.compile(r"[A-Z]*\d{1,15}(?=\_)")

    FileInfo = FileNameInfo(
        power=int(re.findall(re_power, file)[0]),
        time=re.findall(re_time, file)[0],
        name=re.findall(re_name, file)[0],
        file=file,
    )
    return FileInfo


def df2sqlite(dataframe, file=None, tbl_name="import"):

    """The function df2sqlite converts a dataframe into a squlite database.
    
    Args:
       dataframe (panda.DataFrame): the dataframe to convert in a data base
       file (Path): full pathname of the database
       tbl_name (str): name of the table
    """

    import sqlite3

    if file is None:
        conn = sqlite3.connect(":memory:")
    else:
        conn = sqlite3.connect(file)

    cur = conn.cursor()
    wildcards = ",".join(["?"] * len(dataframe.columns))
    data = [tuple(x) for x in dataframe.values]
    cur.execute(f"DROP TABLE IF EXISTS {tbl_name}")
    col_str = '"' + '","'.join(dataframe.columns) + '"'
    cur.execute(f"CREATE TABLE {tbl_name} ({col_str})")
    cur.executemany(f"insert into {tbl_name} values ({wildcards})", data)
    conn.commit()
    cur.close()
    conn.close()


def sieve_files(pow_select, time_select, name_select, database_path):

    """The sieve_files select 
    """
    # Standard library imports
    import sqlite3
    from string import Template
    

    conv2str = lambda list_: str(tuple(list_)).replace(",)", ")")

    conn = sqlite3.connect(database_path)
    cur = conn.cursor()

    querry_d = Template(
        """SELECT file
                        FROM $table_name 
                        WHERE name  IN $name_select
                        AND power IN $pow_select
                        AND time IN $time_select
                        ORDER BY name ASC
                        LIMIT 50"""
    )

    cur.execute(
        querry_d.substitute(
            {
                "table_name": DATA_BASE_TABLE,
                "name_select": conv2str(name_select),
                "pow_select": conv2str(pow_select),
                "time_select": conv2str(time_select),
            }
        )
    )

    querry = [x[0] for x in cur.fetchall()]
    cur.close()
    conn.close()
    return querry


def plot_diff_param(params, df_meta):
    
    '''Plots for different experiments and for different parameters the relative 
    evolution (in %) of the parameters vs power
    
      ID                          Voc         Isc   Rseries   power time                                                         
   JINERGY3272023326035_0200W_T0  50.5082    1.827  1.95841     200   T0  
   JINERGY3272023326035_0200W_T1  50.6780  1.82484  1.87985     200   T1   
   JINERGY3272023326035_0200W_T2  50.3452  1.79790  2.09313     200   T2  
   JINERGY3272023326035_0400W_T0  51.8321  3.61464  1.05142     400   T0 

    '''

    # Standard library imports
    from itertools import combinations

    # 3rd party import
    import matplotlib.pyplot as plt
    import numpy as np
    
    color = ["#8F3E3A", "#8F5B3A", "#8F7B3A", "#7B8F3A", "#458F3A",] # "#3A8F72", "#433A8F" ]
    marker = ["o", "v", ">", "<", "s", "p"]
    dic_ylim = {"Rseries":[-30, 30],
                "Rshunt":[-30, 30],
                "Voc":[-3.2, 1.6],
                "Isc":[-3.2, 1.6],
                "Pmax":[-3.2, 1.6],
                "Fill Factor":[-3.2, 1.6]}

    pow_list = sorted(list(set(df_meta["power"].tolist())))
    pow_add_nbr = 2
    pow_median = (min(pow_list) + max(pow_list)) / 2
    pow_add = (max(pow_list) - min(pow_list)) / 2
    pow_min, pow_max = (
        min(pow_list) - pow_add_nbr * pow_add,
        max(pow_list) + pow_add_nbr * pow_add,
    )

    nbr_time = len(set(df_meta["time"].tolist()))
    assert nbr_time > 1, "not enough time measurements. Should be greeter than 1"
    fig = plt.figure(figsize=(15, 15))
    gs = fig.add_gridspec(
        len(params), int(nbr_time * (nbr_time - 1) / 2), hspace=0, wspace=0
    )
    ax = gs.subplots(sharex="col", sharey="row")


    list_exp = list(set(df_meta["name"]))

    for idx_exp, exp in enumerate(list_exp): # Loop on the experiments
        df_exp = df_meta.query("name == @exp")

        # split df_exp into a dic keyed by time (T0,T1,...). The values are dataframe df_exp
        # with column time=T0,T1,...
        dic_time = {}
        set_times = set(df_exp["time"].tolist())
        for time in set_times:
            dic_time[time] = df_exp.loc[df_exp["time"] == time, :]

        list_t = sorted(list(set(df_exp["time"].tolist())))

        for idx_param, param in enumerate(params): # Loop on the parameter
            dic_time_cp = {}

            for time in combinations(list_t, 2): # Loop on time difference
                val = np.array(dic_time[time[1]][param].astype(float).tolist())
                ref = np.array(dic_time[time[0]][param].astype(float).tolist())
                delta = 100 * (val - ref) / ref
                dic_time_cp[time[1] + "-" + time[0]] = dic_time[time[1]].copy()
                dic_time_cp[time[1] + "-" + time[0]]["Delta_" + param] = delta

            list_items = sorted(dic_time_cp.keys())
            if len(list_items) == 1:
                key = list(dic_time_cp.keys())[0]
                ax[idx_param].scatter(
                    dic_time_cp[key]["power"],
                    dic_time_cp[key]["Delta_" + param],
                    c=color,
                )
                ax[idx_param].axhline(y=0, color="red", linestyle="--")
                if idx_param == 0:
                    ax[idx_param].set_title(key)
                ax[idx_param].set_xlabel("Power ($W/{m^2}$)")
                ax[idx_param].set_ylabel("$\Delta$ " + param + " (%)")
                ax[idx_param].tick_params(axis="x", rotation=90)
                ax[idx_param].set_xlim([pow_min, pow_max])
                ax[idx_param].set_ylim(dic_ylim.get(param,[-3.2, 1.6]))
                for axis in ["top", "bottom", "left", "right"]:
                    ax[idx_param].spines[axis].set_linewidth(2)
            else:
                for idx_time, key in enumerate(list_items):
                    ax[idx_param, idx_time].scatter(
                        dic_time_cp[key]["power"],
                        dic_time_cp[key]["Delta_" + param],
                        c=color,
                        marker=marker[idx_exp]
                    )
                    ax[idx_param, idx_time].axhline(y=0, color="red", linestyle="--")
                    if idx_param == 0:
                        ax[idx_param, idx_time].set_title(key)
                    ax[idx_param, idx_time].set_xlabel("Power ($W/{m^2}$)")
                    if idx_time == 0:
                        ax[idx_param, idx_time].set_ylabel("$\Delta$ " + param + " (%)")
                    ax[idx_param, idx_time].tick_params(axis="x", rotation=90)
                    ax[idx_param, idx_time].set_xticks(pow_list, minor=False)
                    ax[idx_param, idx_time].set_xticklabels(pow_list, fontsize=12)
                    ax[idx_param, idx_time].set_xlim([pow_min, pow_max])
                    ax[idx_param, idx_time].set_ylim(dic_ylim.get(param,[-3.2, 1.6]))
                    for axis in ["top", "bottom", "left", "right"]:
                        ax[idx_param, idx_time].spines[axis].set_linewidth(2)

    fig.suptitle(chr(9679) + " " + list_exp[0], fontsize=15)
    fig.subplots_adjust(top=0.95)


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
    import numpy as np

    SAFETY_WIDTH = 10
    BORNE_SUP = np.Inf
    BORNE_INF = 800

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

    electrolum = read_electolum_file(file, pack=False)

    images_crop = []
    for index, image in enumerate(electrolum.image[:-1]):  # Get rid of the last image

        image = np.where((image < BORNE_INF) | (image > BORNE_SUP), 0, image)
        if index == 0:  # We process the image as a top one
            image_crop, modale_width_0 = crop_segment_image(image, mode="top")
            images_crop.append(image_crop)
        else:
            image_crop, _ = crop_segment_image(
                image, mode="bottom", default_width=modale_width_0
            )
            images_crop.append(image_crop)

    crop_image = np.concatenate(tuple(images_crop), axis=0)

    return crop_image
