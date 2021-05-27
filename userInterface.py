import io
import os
import PySimpleGUI as sg
import cv2
from PIL import Image

from functions_for_model import predict_one_image

sg.theme('DarkAmber')   # Add a touch of color

file_types = [("JPEG (*.jpg)", "*.jpg"),
              ("All files (*.*)", "*.*")]

# All the stuff inside your window.
layout = [  [sg.Text('This is the user interface for this project')],
            [sg.Text('Browse to an image to test the program:')],
            [sg.T("")], [sg.Text("Choose a file: "), sg.Input(key="FILE"), sg.FileBrowse(file_types=file_types)],
            [sg.Image(key="-IMAGE-")],
            [sg.T("")], [sg.Button("Check chosen image")],
            [sg.T("")], [sg.Button("Try the program with the chosen file")],
            [sg.T("")], [sg.Text('Here will appear the text prediction of the image:')],
            [sg.T("")], [sg.Text('....................................', key="RESULT")]]



# Create the Window
window = sg.Window('Window Title', layout, size=(750, 500))
# Event Loop to process "events" and get the "values" of the inputs


while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED:  # if user closes window
        break

    if event == "Check chosen image":
        filename = values["FILE"]
        if os.path.exists(filename):
            image = Image.open(values["FILE"])
            image.thumbnail((400, 400))
            bio = io.BytesIO()
            image.save(bio, format="PNG")
            window["-IMAGE-"].update(data=bio.getvalue())

    if event == "Try the program with the chosen file":
        filename2 = values["FILE"]
        if os.path.exists(filename2):
            result = predict_one_image(filename2)
            window["RESULT"].update(result)



window.close()