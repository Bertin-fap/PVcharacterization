__all__ = ['Select_items','select_files','select_data_dir']

# Global variables used by Select_multi_items function
from .PVcharacterization_global import (DEFAULT_DIR,
                                       DATA_BASE_NAME,
		                               DATA_BASE_TABLE,
		                               USED_COLS,
		                               PARAM_UNIT_DIC)
GEOMETRY_ITEMS_SELECTION = '500x580+50+50'    # Size of the tkinter window
GEOMETRY_SELECT_DIR = "750x250"

def Select_items(list_item,title,mode = 'multiple'): 

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

