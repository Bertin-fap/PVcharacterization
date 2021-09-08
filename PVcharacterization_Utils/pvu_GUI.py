__all__ = ['Select_items']

# Global variables used by Select_multi_items function
GEOMETRY_ITEMS_SELECTION = '500x580+50+50'    # Size of the tkinter window

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
