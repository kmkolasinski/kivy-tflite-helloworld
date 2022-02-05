import kivy

from logging_ops import measuretime

kivy.require('1.0.6') # replace with your current kivy version !

from kivy.app import App
from kivy.uix.label import Label


class MyApp(App):

    def build(self):

        from tflite_models import TensorFlowModel
        import os
        import numpy as np
        model = TensorFlowModel()
        path = os.path.join(os.getcwd(), 'model.tflite')
        np.random.seed(42)

        model.load(path, 1)

        with measuretime("tflite1"):
            for i in range(1000):
                x = np.array(np.random.random_sample((1, 28, 28)), np.float32)
                y = model.pred(x)
        print(f" >> Predictions: {y}")

        model.load(path, 2)

        with measuretime("tflite2"):
            for i in range(1000):
                x = np.array(np.random.random_sample((1, 28, 28)), np.float32)
                y = model.pred(x)
        print(f" >> Predictions: {y}")

        model.load(path, 2, True)

        with measuretime("tflite3"):
            for i in range(1000):
                x = np.array(np.random.random_sample((1, 28, 28)), np.float32)
                y = model.pred(x)
        print(f" >> Predictions: {y}")

        model.load(path, 4, True)

        with measuretime("tflite3"):
            for i in range(1000):
                x = np.array(np.random.random_sample((1, 28, 28)), np.float32)
                y = model.pred(x)
        print(f" >> Predictions: {y}")

        model.load(path, 4, False)

        with measuretime("tflite3"):
            for i in range(1000):
                x = np.array(np.random.random_sample((1, 28, 28)), np.float32)
                y = model.pred(x)
        print(f" >> Predictions: {y}")

        return Label(text='Hello world')


if __name__ == '__main__':
    MyApp().run()