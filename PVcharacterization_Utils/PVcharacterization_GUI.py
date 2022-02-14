__all__ = ['input_treatment_labels',
           'get_date',
           'select_data_dir',
           'select_files',
           'select_items',]

# Global variables used by Select_multi_items function
from .config import GLOBAL

def select_items(list_item,title,mode = 'multiple'): 

    """interactive selection of items among the list list-item
    
    Args:
        list_item (list): list of items used for the selection
        mode (string): 'single' or 'multiple' (default = 'multiple'
        title (string): title of the window
        
    Returns:
        val (list): list of selected items without duplicate
        
    """
    import os
    import tkinter as tk
    
    GEOMETRY_ITEMS_SELECTION = GLOBAL['GEOMETRY_ITEMS_SELECTION']    # Size of the tkinter window

    global val

    window = tk.Tk()
    window.geometry(GEOMETRY_ITEMS_SELECTION)
    window.attributes("-topmost", True)
    window.title(title)

    yscrollbar = tk.Scrollbar(window)
    yscrollbar.pack(side = tk.RIGHT, fill = tk.Y)
    selectmode = tk.MULTIPLE
    if mode == 'single':selectmode = tk.SINGLE
    listbox = tk.Listbox(window, width=40, height=10, selectmode=selectmode,
                     yscrollcommand = yscrollbar.set)

    x = list_item
    for idx,item in enumerate(x):
        listbox.insert(idx, item)
        listbox.itemconfig(idx,
                           bg = "white" if idx % 2 == 0 else "white")
    
    def selected_item():
        global val
        val = [listbox.get(i) for i in listbox.curselection()]
        if os.name == 'nt':
            window.destroy()

    btn = tk.Button(window, text='OK', command=selected_item)
    btn.pack(side='bottom')

    listbox.pack(padx = 10, pady = 10,expand = tk.YES, fill = "both")
    yscrollbar.config(command = listbox.yview)
    window.mainloop()
    return val

def select_files():
    
    '''The function `select_files` interactively selects *.txt or *.txt files from
    a directory.
    
    Args:
       DEFAULT_DIR (Path, global): root directory used for the file selection.
       
    Returns:
       filenames (list of str): list of selected files
    '''
    
    # Standard library imports
    import os
    import tkinter as tk
    from tkinter import ttk
    from tkinter import filedialog as fd

    DEFAULT_DIR = GLOBAL['WORKING_DIR']

    root = tk.Tk()
    root.title('File Dialog')
    root.resizable(False, False)
    root.geometry('300x150')
    global filenames, filetypes
    filetypes = (
            ('csv files', '*.csv'),
            ('text files', '*.txt'), 
            ('excel files', '*.xlsx'),)

    def select_files_():
        global filenames,filetypes
        
        filenames = fd.askopenfilenames(
            title='Select files',
            initialdir=DEFAULT_DIR,
            filetypes=filetypes)

    open_button = ttk.Button(
        root,
        text='Select Files',
        command=select_files_)
    open_button.pack(expand=True)
    
    if os.name == 'nt':
        tk.Button(root,
                  text="EXIT",
                  command=root.destroy).pack(expand=True)

    root.mainloop()
    
    return filenames

    
def select_data_dir_old(root,title) :
 
    '''
    Selection of a folder
   
    Args:
        root (Path): initial folder.
        title (str): title specifying which kind of folder to be selected.
    Returns:
       (str): selected folder.
 
    '''
   
    # Standard library imports
    import os
    import tkinter as tk
    from tkinter import ttk
    from tkinter import filedialog
    
    GEOMETRY_SELECT_DIR = GLOBAL['GEOMETRY_SELECT_DIR']
   
    global in_dir, button
   
    win= tk.Tk()
    win.geometry(GEOMETRY_SELECT_DIR )
    win.title("Folder selection")
    
    def select_file():
        global in_dir, button
        button["state"] = "disabled"
        in_dir= filedialog.askdirectory(initialdir=str(root), title=title)
        tk.Label(win, text=in_dir, font=13).pack()
        
   
    tk.Label(win, text=title+'\nthen close the window', font=('Aerial 18 bold')).pack(pady=20)
    button= ttk.Button(win, text="Select", command= select_file)
    button.pack(ipadx=5, pady=15)
    if os.name == 'nt':
        tk.Button(win,
                  text="EXIT",
                  command=win.destroy).pack(pady=3)
        
    win.mainloop()
    return in_dir

def input_treatment_labels(list_diff): 
    
    '''Interactive choice of the treatment name.
    
    Args:
       list_diff (list of tuples): [(T1,T0),(T2,T0),...] where T<i> are the label of the <ieme> treatment
       
    Returns:
       A dict={T0:name of treatment T0, T1:name of the treatment T1,...}
    '''
    
    import tkinter as tk
    global dict_label, list_trt
    
    list_trt = []
    for trt in list_diff:
       list_trt.extend([trt[0],trt[1]])
    list_trt = list(set(list_trt))
    list_trt.sort()
    n_items = len(list_trt)
    
    root = tk.Tk()
    root.title("Python - Basic Register Form")
    
    FONT = ('arial', 12)
    FONT1 = ('arial', 15)

    width = 640
    height = 600
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width/2) - (width/2)
    y = (screen_height/2) - (height/2)
    root.geometry("%dx%d+%d+%d" % (width, height, x, y))
    root.resizable(1, 1)

    def Register():
        global dict_label,list_trt
        dict_label= {}
        for idx in range(len(list_t)):
            value = str(list_t[idx].get())
            dict_label[list_trt[idx]] = value if value else list_trt[idx]

    TitleFrame = tk.Frame(root, height=100, width=640, bd=1, relief=tk.SOLID)
    TitleFrame.pack(side=tk.TOP)
    RegisterFrame = tk.Frame(root)
    RegisterFrame.pack(side=tk.TOP, pady=20)

    lbl_title = tk.Label(TitleFrame, text="PVcharacterization treatment labels", font=FONT, bd=1, width=640)
    lbl_title.pack()
    list_t = ['']*n_items
    for idx, trt in enumerate(list_trt):
        list_t[idx] =  tk.StringVar()
        tk.Label(RegisterFrame, text= trt, font=FONT, bd=18).grid(row=1+idx)
        tk.Entry(RegisterFrame, font=FONT1, textvariable=list_t[idx], width=15).grid(row=1+idx, column=1)
    lbl_result = tk.Label(RegisterFrame, text="", font=FONT).grid(row=n_items+1, columnspan=2)
    

    btn_register = tk.Button(RegisterFrame, font=FONT1, text="Register", command=Register)
    btn_register.grid(row=n_items+2, columnspan=2)
    btn_exit = tk.Button(RegisterFrame, font=FONT1, text="EXIT", command=root.destroy)
    btn_exit.grid(row=n_items+3, columnspan=2)


    root.mainloop()
    return dict_label


################### New GUI ###################


def _str_size_mm(text, font, ppi):
    '''The function `_str_size_mm` computes the sizes in mm of a string.

    Args:
        text (str): the text of which we compute the size in mm.
        font (tk.font): the font of the text.
        ppi (int): pixels per inch of the display.

    Returns:
        `(tuple)`: width in mm `(string)`, height in mm `(string)`.

    Note:
        The use of this function requires a tkinter window availability 
        since it is based on a tkinter font definition.

    '''

    # Local imports
    from .config import GLOBAL
    
    in_to_mm = GLOBAL['IN_TO_MM']
       
    (w_px,h_px) = (font.measure(text),font.metrics("linespace"))
    w_mm = w_px * in_to_mm / ppi
    h_mm = h_px * in_to_mm / ppi

    return (w_mm,h_mm )


def _str_max_len_mm(list_strs,font,ppi): 
    '''The `_str_max_len_mm`function gets the maximum length in mm of a list of strings 
       given the font and the ppi of the used display and using the `_str_size_mm` function .
       
    Args:
        list_strs (list): list of strings to be sized to get their maximum length.
        font (tk.font): the font used for the strings size evaluation in mm.
        ppi (int): pixels per inch of the display.
        
    Returns:
        `(float)`: maximum length in mm of the strings in `list_strs`.
    '''                   
    
    max_length_mm = max([_str_size_mm(value, font, ppi)[0] 
                         for value in list_strs])
    return max_length_mm


def _mm_to_px(size_mm,ppi, fact=1):
    '''The `_mm_to_px' function converts a value in mm to a value in pixels
    using the ppi of the used display and a factor fact.
    
    Args:
        size_mm (float): value in mm to be converted.
        ppi ( float): pixels per inch of the display.
        fact (float): factor (default= 1).
        
    Returns:
        `(int)`: upper integer value of the conversion to pixels
        
    '''
    
    # Standard library imports 
    import math
    
    # Local imports
    from .config import GLOBAL
    
    in_to_mm = GLOBAL['IN_TO_MM']

    size_px = math.ceil((size_mm * fact / in_to_mm) * ppi)
    
    return size_px


def _split_path2str(in_str,sep,max_px,font,ppi):
    '''The `_split_path2str` function splits the `in_str` string 
    in substrings of pixels sizes lower than `max_px` using the separator `sep` .

    Args:
        in_str (str): the full path of a folder.
        sep (str): the character to be find in `in_str`.
        max_px (int): the maximum size in pixels for the substrings 
                      that should result from the split of `in-dir`.
        font (tk.font): the font used for the substrings size evaluation in mm.
        ppi (float): pixels per inch of the display.

    Returns:
        `(tuple)`: tuple of the substrings resulting from the split of `in-dir`.

    Note:
        The use of this function requires a tkinter window availability 
        since it is based on a tkinter font definition.

    '''        

    # Standard library imports 
    import numpy as np
    import re

    len_in_str,_ = _str_size_mm(in_str, font, ppi)
    if _mm_to_px(len_in_str,ppi)>int(max_px):
        pos_list = np.array([m.start() for m in re.finditer(r'[\\' + sep + ']', in_str)])
        list_len = [_mm_to_px(_str_size_mm(in_str[0:pos_slash], font, ppi)[0],ppi)
                    for pos_slash in pos_list ]
        try:
            pos_mid = pos_list[np.min(np.where(np.array(list_len) >= int(max_px))) - 1]
        except:
            pos_mid = pos_list[-1]
        out_str1 = str(in_str)[0:pos_mid]
        out_str2 = str(in_str)[pos_mid:]

    else:
        out_str1 = str(in_str)
        out_str2 = ''

    return (out_str1,out_str2)


def select_data_dir(in_dir, titles, buttons_labels, 
                    prime_disp=0, widget_ratio=1.2, button_ratio=2.5, max_lines_nb=3):
    
    '''The function `select_folder_gui_grid` allows the interactive selection of a folder.
    
    Args: 
        in_dir (str): name of the initial folder.
        titles (dict): title of the tk window.
        buttons_labels(list): list of button labels as strings.
        prime_disp (int): number identifying the used display (default: 0).
        widget_ratio (float): base ratio for defining the different widget ratii (default: 1.2).
        button_ratio (float): buttons-height to the label-height ratio 
                              to vertically center the label in the button (default: 2.5).
        max_lines_nb (int): maximum lines number for editing the selected folder name (default: 3).
    
    Returns:
        `(str)`: name of the selected folder.
        
    Note:
        Uses the globals: `DISPLAYS`, `GLOBAL('FOLDER_SELECTION_HELP_TEXT')`.
        Based on two frames in the main window and two buttons in the top frame.

    '''
    # Standard library imports 
    import math
    import re
    import tkinter as tk
    from tkinter import messagebox
    from tkinter import filedialog
    import tkinter.font as TkFont
    
    # Local imports
    from .PVcharacterization_sys import DISPLAYS
    from .config import GLOBAL
    
    global out_dir
 
    ############# Definition of local functions #############    

    def outdir_folder_choice():
        '''The function `outdir_folder_choice' allows the interactive choice of a folder, 
        puts it in the `out_dir` global variable and prints the result in the `folder_choice` frame.

        '''
        # Standard library imports
        import numpy as np

        global out_dir

        out_dir = filedialog.askdirectory(initialdir=in_dir,title=titles['main'])
        
        out_dir_split = [out_dir]
        while out_dir_split[len(out_dir_split)-1]!='':
            out_dir1,out_dir2 = _split_path2str(out_dir_split[len(out_dir_split)-1], 
                                               '/', frame_widthpx, text_font, ppi)
            out_dir_split[len(out_dir_split)-1] = out_dir1
            out_dir_split.append(out_dir2)

         # Creating the folder-result frame and set its geometry. 
        folder_result = tk.LabelFrame(master=win,              
                        text=titles['result'],
                        font=frame_font)
        folder_result.place(x=frame_xpx,
                            y=frame_ypx,
                            width=frame_widthpx,
                            height=frame_heightpx)

         # Editing the selected folder.       
        text_max_widthmm = _str_max_len_mm(out_dir_split, text_font, ppi)
        text_xmm = (frame_widthmm - text_max_widthmm) / 2
        text_xpx = _mm_to_px(text_xmm - mm_size_corr,ppi)
        text_ypx = _mm_to_px(frame_unit_heightmm,ppi)       
        text = '\n'.join(out_dir_split)
        folder_label = tk.Label(folder_result, text=text, font=text_font)
        folder_label.place(x=text_xpx,
                           y=text_ypx)
    
    def help():
        messagebox.showinfo('Folder selection info', folder_selection_help_text)
    
    ############# Local parameters setting #############  
    
     # Getting the ppi of the selected prime display.
    ppi = DISPLAYS[prime_disp]['ppi']
    
     # Getting the help text of the GUI
    folder_selection_help_text = GLOBAL['FOLDER_SELECTION_HELP_TEXT'] 
    
     # Checking the number of frames and buttons.
    frames_nb = len(titles) -1 
    buttons_nb = len(buttons_labels)
    if frames_nb!=1 or buttons_nb!=2:
        print('Number of titles:', len(titles) )
        print('Number of buttons:', len(button_labels) )
        print('The number of titles should be 2 \
               and the number of buttons should be 2.\
               Please define ad hoc number of widgets.')

     # Setting the ratio of frames-width to the titles-max-width.
    frame_ratio = widget_ratio
    
     # Setting the ratio of window-width to the frame-width.
    win_ratio = widget_ratio 
    
     # Setting a potential ratio for correcting the conversion from mm to px for the buttons sizes.
     # Todo: adapt these ratios to correct the discrepancy between the set mm sizes 
     # and the effective mm sizes on the screen for MacOs 
     # (correction still to be understood).
    buttonsize_mmtopx_ratios = (1,1,1)
    
     # Setting the value in mm for the correction of the sizes in milimeters 
     # before use for computing the widgets horizontal positions in pixels 
     # (correction still to be understood).
    mm_size_corr = 1
    
    ############# Tkinter window management #############
    
     # Creating the tk window.
    win = tk.Tk()
    win.attributes("-topmost", True)
    win.title(titles['main']) 
    
     # Setting the fonts to be used.
    frame_font = TkFont.Font(family='arial', size=16, weight='bold')
    text_font = TkFont.Font(family='arial', size=12, weight='normal')
    button_font = TkFont.Font(family='arial', size=12, weight='normal')
    
     # Computing the maximum size in mm of the list of titles.
    titles_mm_max = _str_max_len_mm(titles.values(), frame_font, ppi)
    
     # Computing button sizes in mm and pixels using button label sizes and button_ratio.
     # Buttons width is the button heigth added to the labels width 
     # to horizontally center the label in the button. 
    labels_widthmm = [_str_size_mm(buttons_labels[i],button_font, ppi)[0] for i in range(buttons_nb)]
    label_heightmm = _str_size_mm(buttons_labels[0],button_font, ppi)[1]
    button_heightmm =  label_heightmm * button_ratio
    buttons_widthmm = (labels_widthmm[0] + button_heightmm, labels_widthmm[1] + button_heightmm)
    buttons_widthpx = (_mm_to_px(buttons_widthmm[0],ppi,buttonsize_mmtopx_ratios[0]), 
                       _mm_to_px(buttons_widthmm[1],ppi,buttonsize_mmtopx_ratios[1]))
    button_heigthpx = _mm_to_px(button_heightmm,ppi,buttonsize_mmtopx_ratios[2])

     # Computing the frame width in pixels from titles maximum size in mm using frame_ratio.
    frame_widthmm = titles_mm_max * frame_ratio    
    frame_widthpx = str(_mm_to_px(frame_widthmm,ppi))

     # Computing the window width in pixels from the frame width and buttons width using win_ratio.
    win_widthmm = max(frame_widthmm,sum(buttons_widthmm)) * win_ratio 
    win_widthpx = str(_mm_to_px(win_widthmm,ppi))

     # Computing the buttons horizontal positions in pixels 
     # assuming 2 buttons and with correction of size in mm by mm_size_corr value.
    padx_ratio = buttons_nb * 2  
    pad_xmm = (win_widthmm - min(frame_widthmm,sum(buttons_widthmm))) / padx_ratio
    buttons_xmm = (pad_xmm, buttons_widthmm[0] + 3 * pad_xmm)
    buttons_xpx = (_mm_to_px(buttons_xmm[0] - mm_size_corr,ppi), _mm_to_px(buttons_xmm[1] - mm_size_corr,ppi))

     # Computing the frames heigth unit.
    _, text_heigthmm = _str_size_mm('Users/',text_font, ppi)
    frame_unit_heightmm = min(button_heightmm,text_heigthmm) 

     # Computing the buttons vertical position in pixels.
    button_ymm = frame_unit_heightmm 
    button_ypx = _mm_to_px(button_ymm,ppi)

     # Computing the frame heigth in mm and in pixels.
    pads_nb = 4  # 2 frame units above and 2 frame units under the edited text. 
    max_frame_unit_nb = pads_nb + max_lines_nb  
    frame_heightmm = frame_unit_heightmm * max_frame_unit_nb
    frame_heightpx = str(_mm_to_px(frame_heightmm,ppi))

     # Computing the frame positions in pixels.
    frame_xmm = (win_widthmm - frame_widthmm) / 2
    frame_ymm = button_ymm + button_heightmm + 2 * frame_unit_heightmm
    frame_xpx, frame_ypx = _mm_to_px(frame_xmm,ppi), _mm_to_px(frame_ymm,ppi)

    # Computing the window heigth in mm and in pixels .
    # with frame_unit_heightmm separating vertically the widgets.
    win_heightmm = button_ymm + button_heightmm + frame_heightmm + 3 * frame_unit_heightmm
    win_heightpx = str(_mm_to_px(win_heightmm,ppi))

     # Setting the window geometry.
    win_xpx = str(int(DISPLAYS[prime_disp]['x']) + 50)
    win_ypx = str(int(DISPLAYS[prime_disp]['y']) + 50)
    win.geometry(f'{win_widthpx}x{win_heightpx}+{win_xpx}+{win_ypx}')

     # Creates the folder result frame and set its geometry. 
    folder_result = tk.LabelFrame(master=win,              
                text=titles['result'],
                font=frame_font)
    folder_result.place(x=frame_xpx,
                    y=frame_ypx,
                    width=frame_widthpx,
                    height=frame_heightpx)

     # Creating the button for folder selection.
    select_button = tk.Button(win,
                          text=buttons_labels[0],
                          font=button_font,
                          command=outdir_folder_choice)
    select_button.place(x=buttons_xpx[0], 
                    y=button_ypx, 
                    width=buttons_widthpx[0], 
                    height=button_heigthpx)

     # Creating the help button.
    help_button = tk.Button(win,
                        text=buttons_labels[1],
                        font=button_font,
                        command=help)
    help_button.place(x=buttons_xpx[1], 
                  y=button_ypx, 
                  width=buttons_widthpx[1], 
                  height=button_heigthpx)

    win.mainloop()
    
    return out_dir
    
def get_date():
    
    # Standard library imports
    import datetime 
    import tkinter as tk
    
    # 3rd party import
    from tkcalendar import Calendar
    
    global selected_date
    
    def grad_date():
        global selected_date
        selected_date = cal.get_date()
        
        date.config(text = "Selected Date is: " + cal.get_date())

    root = tk.Tk()
    root.geometry("400x400")

    current_time = datetime.datetime.now() 
    cal = Calendar(root, selectmode = 'day',
                   year = current_time.year,
                   month = current_time.month,
                   day = current_time.day)
    cal.pack(pady = 20)

    tk.Button(root, text = "Get Date",
       command = grad_date).pack(pady = 20)
 
    date = tk.Label(root, text = "")
    date.pack(pady = 20)

    root.mainloop()
    return datetime.datetime.strptime(selected_date, '%m/%d/%y')