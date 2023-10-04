# Import the required libraries
from tkinter import *
import webview

# Create an instance of tkinter frame or window
win = Tk()

# Set the size of the window
win.geometry("700x350")

# Create a GUI window to view the HTML content
webview.create_window('tutorialspoint', 'https://www.tutorialspoint.com')
webview.start()