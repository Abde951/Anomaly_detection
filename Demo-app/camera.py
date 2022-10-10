from pypylon import pylon
import imageio

def Basler():
    # create InstantCamera and grab image
    camera = pylon.InstantCamera(
        pylon.TlFactory.GetInstance().CreateFirstDevice())
    grab_result = camera.GrabOne(1000)

    # create and configure ImageFormatConverter
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_RGB8packed

    # convert image to RGB, create NumPy array
    # and save RGB image as color BMP
    converted = converter.Convert(grab_result)
    image_rgb = converted.GetArray()
    return image_rgb