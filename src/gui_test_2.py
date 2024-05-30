import PySimpleGUI as sg

lst1 = list("ABCDEFGHIJ")

layout = [
    [sg.Button('Select All')],
    [sg.Listbox(lst1, size=(5, 10), select_mode='multiple', key='-IN7-')],
]
window = sg.Window("test", layout, finalize=True)
while True:

    event, values = window.read()
    print(event)

    if event in (sg.WINDOW_CLOSED, 'Exit'):
        print('Exiting program')
        break
    # Get value of elements should be here to prevent value of `values` is None when sg.WINDOW_CLOSED
    pos_cols = values["-IN7-"]
    if event == 'Select All':
        window['-IN7-'].set_value(lst1)

window.close()