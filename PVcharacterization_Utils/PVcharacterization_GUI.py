__all__ = ['input_treatment_labels',
           'select_data_dir',
           'select_files',
           'select_items',]

# Global variables used by Select_multi_items function
from .PVcharacterization_global import GLOBAL

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

    DEFAULT_DIR = GLOBAL['DEFAULT_DIR']

    root = tk.Tk()
    root.title('File Dialog')
    root.resizable(False, False)
    root.geometry('300x150')
    global filenames, filetypes
    filetypes = (
            ('csv files', '*.csv'),
            ('text files', '*.txt'), 
            )

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

    
def select_data_dir(root,title) :
 
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
        dict_label = {list_trt[idx] : str(list_t[idx].get()) for idx in range(len(list_t))}

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

