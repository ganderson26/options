# Exercise 1: Personalized Greeting**

import PySimpleGUI as sg

layout = [[sg.Text("Enter your name:"), sg.InputText()],
          [sg.Button("Submit")]]

window = sg.Window("Greeting App", layout)

while True:
    event, values = window.read()
    if event == sg.WINDOW_CLOSED:
        break
    sg.popup(f"Hello, {values[0]}!")

window.close()
