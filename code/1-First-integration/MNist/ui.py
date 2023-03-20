import tkinter as tk
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageOps
from main import predict_image

root = tk.Tk()
root.wm_title("MNist classifier test application")


class ImageDrawer:
    def __init__(self, parent, x, y, **kwargs):
        self.parent = parent
        self.x = x
        self.y = y
        self.width = 280
        self.height = 280
        self.mouse1 = "up"
        self.lastx = None
        self.lasty = None
        self.coordinates = []
        self.canvas = tk.Canvas(parent, width=self.width, height=self.height, bg="white")
        self.canvas.place(x=x, y=y)
        self.canvas.bind("<Motion>", self.motion)
        self.canvas.bind("<ButtonPress-1>", self.button_down)
        self.canvas.bind("<ButtonRelease-1>", self.button_release)
        # self.canvas.grid(row=self.y, column=self.x, sticky="nsew")

        self.button_clear = tk.Button(parent, text="Clear", command=self.clear)
        self.button_clear.place(x=10, y=self.height + 20)

        self.button_save = tk.Button(parent, text="Save", command=self.save)
        self.button_save.place(x=self.width - 20, y=self.height + 20)

        self.image = Image.new("RGB", (self.width, self.height), "white")
        # self.image_tk = ImageTk.PhotoImage(self.image)
        # self.canvas.create_image(0, 0, image=self.image_tk, anchor="nw")
        self.draw = ImageDraw.Draw(self.image)

    def save(self):
        filename = "temp.png"
        # im = self.image.resize((28, 28), 1)
        im = ImageOps.invert(self.image)
        im.save(filename)
        # self.image.save(filename)

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("RGB", (self.width, self.height), "white")
        # self.image_tk = ImageTk.PhotoImage(self.image)
        # self.canvas.create_image(0, 0, image=self.image_tk, anchor="nw")
        self.draw = ImageDraw.Draw(self.image)

    def motion(self, event):
        if self.mouse1 == "down" and self.lastx is not None and self.lasty is not None:
            event.widget.create_rectangle(self.lastx, self.lasty, event.x, event.y, width=10, fill="black")
            self.draw.rectangle((self.lastx - 5, self.lasty - 5, event.x + 5, event.y + 5), fill="black")

            self.coordinates.append((event.x, event.y))

        self.lastx = event.x
        self.lasty = event.y

    def button_down(self, event):
        self.mouse1 = "down"

    def button_release(self, event):
        self.mouse1 = "up"
        self.lastx = None
        self.lasty = None


def predict():
    predict_image("./temp.png")


def main():
    button = tk.Button(root, text="Quit", command=root.quit)
    button.pack(side=tk.BOTTOM)

    button = tk.Button(root, text="Predict", command=predict)
    button.pack(side=tk.BOTTOM)

    # Canvas to draw 28x28 image on
    root.wm_geometry("%dx%d+%d+%d" % (800, 400, 10, 10))
    root.config(bg='white')
    ImageDrawer(root, 10, 10)

    root.mainloop()


def on_key_press(event):
    print("pressed", repr(event.char))


if __name__ == '__main__':
    main()
