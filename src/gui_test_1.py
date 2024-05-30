import PySimpleGUI as sg


# Define the layout
layout = [[sg.Text("Hello, World!")],
          [sg.Button("OK")]]

# Create the window
window = sg.Window("Hello World", layout)



# Event loop
while True:
    event, values = window.read()
    print(event, values)
    if event == sg.WINDOW_CLOSED:
        break
    if event == 'OK':
        break    

# Close the window
window.close()
