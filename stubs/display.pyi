"""
Display driver.

MicroPython module: https://docs.micropython.org/en/preview/library/display.html

The ``display`` module is used for driving SPI LCDs, 24-bit parallel LCDs, MIPI DSI LCDs, HDMI output, and Display Port output.
"""

# source version: preview
# origin module:: repos/micropython/docs/library/display.rst
# + module: omv.display.spidisplay.rst
# + module: omv.display.rgbdisplay.rst
# + module: omv.display.dsidisplay.rst
# + module: omv.display.displaydata.rst
# + module: omv.display.ST7701.rst
# + module: omv.display.DACBacklight.rst
# + module: omv.display.PWMBacklight.rst
from __future__ import annotations
from typing import IO, Any, Callable, Coroutine, Dict, Generator, Iterator, List, NoReturn, Optional, Tuple, Union, NamedTuple, TypeVar
from _typeshed import Incomplete
class SPIDisplay():
    """
        ``width`` SPI LCD width. By default this is 128 to match the OpenMV 128x160 LCD shield.
    
        ``height`` SPI LCD height. By default this is 160 to match the OpenMV 128x160 LCD shield.
    
        ``refresh`` Sets the LCD refresh rate in hertz. This controls the SPI LCD shield clock.
    
        ``bgr`` set to True to swap the red and blue channels.
        This argument allows you to use our driver with more types of displays.
    
        ``byte_swap`` set to True to swap RGB565 pixel bytes sent to the LCD.
        This argument allows you to use our driver with more types of displays.
    
        ``triple_buffer`` If True then makes updates to the screen non-blocking at the cost of 3X the
        display size in RAM. This is on by default for OpenMV Cam boards with SDRAM.
    
        ``controller`` Pass the controller chip class here to initialize it along with the display.
    
        ``backlight`` specify a backlight controller module to use. By default the backlight will be
        controlled via a GPIO pin.
    
        .. note::
    
            Uses pins P0, P2, P3, P6, P7, and P8.
    """
    def __init__(self, width=128, height=160, refresh=60, bgr=False, byte_swap=False, triple_buffer, controller, backlight) -> None:
        ...
    def deinit(self) -> Incomplete:
        """
           Releases the I/O pins and RAM used by the class. This is called automatically on destruction.
        """
        ...
    def width(self) -> Incomplete:
        """
           Returns the width of the screen.
        """
        ...
    def height(self) -> Incomplete:
        """
           Returns the height of the screen.
        """
        ...
    def refresh(self) -> Incomplete:
        """
           Returns the refresh rate.
        """
        ...
    def bgr(self) -> Incomplete:
        """
           Returns if the red and blue channels are swapped.
        """
        ...
    def byte_swap(self) -> Incomplete:
        """
           Returns if the RGB565 pixels are displayed byte reversed.
        """
        ...
    def triple_buffer(self) -> Incomplete:
        """
           Returns if triple buffering is enabled.
        """
        ...
    def write(self, image, x=0, y=0, x_scale=1.0, y_scale=1.0, roi=None, rgb_channel=-1, alpha=256, color_palette=None, alpha_palette=None, hint=0) -> Incomplete:
        """
           Displays an ``image`` whose top-left corner starts at location x, y.
        
           You may also pass a path instead of an image object for this method to automatically load the image
           from disk and draw it in one step. E.g. ``write("test.jpg")``.
        
           ``x_scale`` controls how much the displayed image is scaled by in the x direction (float). If this
           value is negative the image will be flipped horizontally. Note that if ``y_scale`` is not specified
           then it will match ``x_scale`` to maintain the aspect ratio.
        
           ``y_scale`` controls how much the displayed image is scaled by in the y direction (float). If this
           value is negative the image will be flipped vertically. Note that if ``x_scale`` is not specified
           then it will match ``x_scale`` to maintain the aspect ratio.
        
           ``roi`` is the region-of-interest rectangle tuple (x, y, w, h) of the image to display. This
           allows you to extract just the pixels in the ROI to scale.
        
           ``rgb_channel`` is the RGB channel (0=R, G=1, B=2) to extract from an RGB565 image (if passed)
           and to render on the display. For example, if you pass ``rgb_channel=1`` this will
           extract the green channel of the RGB565 image and display that in grayscale.
        
           ``alpha`` controls how opaque the image is. A value of 256 displays an opaque image while a
           value lower than 256 produces a black transparent image. 0 results in a perfectly black image.
        
           ``color_palette`` if not ``-1`` can be `image.PALETTE_RAINBOW`, `image.PALETTE_IRONBOW`, or
           a 256 pixel in total RGB565 image to use as a color lookup table on the grayscale value of
           whatever the input image is. This is applied after ``rgb_channel`` extraction if used.
        
           ``alpha_palette`` if not ``-1`` can be a 256 pixel in total GRAYSCALE image to use as a alpha
           palette which modulates the ``alpha`` value of the input image being displayed at a pixel pixel
           level allowing you to precisely control the alpha value of pixels based on their grayscale value.
           A pixel value of 255 in the alpha lookup table is opaque which anything less than 255 becomes
           more transparent until 0. This is applied after ``rgb_channel`` extraction if used.
        
           ``hint`` can be a logical OR of the flags:
        
              * `image.AREA`: Use area scaling when downscaling versus the default of nearest neighbor.
              * `image.BILINEAR`: Use bilinear scaling versus the default of nearest neighbor scaling.
              * `image.BICUBIC`: Use bicubic scaling versus the default of nearest neighbor scaling.
              * `image.CENTER`: Center the image being drawn on the display. This is applied after scaling.
              * `image.HMIRROR`: Horizontally mirror the image.
              * `image.VFLIP`: Vertically flip the image.
              * `image.TRANSPOSE`: Transpose the image (swap x/y).
              * `image.EXTRACT_RGB_CHANNEL_FIRST`: Do rgb_channel extraction before scaling.
              * `image.APPLY_COLOR_PALETTE_FIRST`: Apply color palette before scaling.
              * `image.SCALE_ASPECT_KEEP`: Scale the image being drawn to fit inside the display.
              * `image.SCALE_ASPECT_EXPAND`: Scale the image being drawn to fill the display (results in cropping)
              * `image.SCALE_ASPECT_IGNORE`: Scale the image being drawn to fill the display (results in stretching).
              * `image.ROTATE_90`: Rotate the image by 90 degrees (this is just VFLIP | TRANSPOSE).
              * `image.ROTATE_180`: Rotate the image by 180 degrees (this is just HMIRROR | VFLIP).
              * `image.ROTATE_270`: Rotate the image by 270 degrees (this is just HMIRROR | TRANSPOSE).
        """
        ...
    def clear(self, display_off=False: Optional[Any]=None) -> Incomplete:
        """
           Clears the lcd screen to black.
        
           ``display_off`` if True instead turns off the display logic versus clearing the frame LCD
           frame buffer to black. You should also turn off the backlight too after this to ensure the
           screen goes to black as many displays are white when only the backlight is on.
        """
        ...
    def backlight(self, value: Optional[Any]=None) -> Incomplete:
        """
           Sets the lcd backlight dimming value. 0 (off) to 100 (on).
        
           Note that unless you pass `DACBacklight` or `PWMBacklight` the backlight will be controlled
           as a GPIO pin and will only go from 0 (off) to !0 (on).
        
           Pass no arguments to get the state of the backlight value.
        """
        ...
    def bus_write(self, cmd, args=None: Optional[Any]=None) -> None:
        """
           Send the SPI Display ``cmd`` with ``args``.
        """
        ...
class RGBDisplay():
    """
        ``framesize`` One of the standard supported resolutions.
    
        ``refresh`` Sets the screen refresh rate in hertz. This controls the RGB LCD clock.
    
        ``display_on`` Enables the display. Pass `False` here when the 24-bit parallel LCD output is
        shared by multiple devices like the TFP410 chip for driving HDMI displays to keep the display
        off while driving the databus still.
    
        ``portrait`` Swap the framesize width and height.
    
        ``controller`` Pass the controller chip class here to initialize it along with the display.
    
        ``backlight`` specify a backlight controller module to use. By default the backlight will be
        controlled via a GPIO pin.
    """
    def __init__(self, framesize=FWVGA, refresh=60, display_on=True, portrait=False, controller, backlight) -> None:
        ...
    def deinit(self) -> Incomplete:
        """
           Releases the I/O pins and RAM used by the class. This is called automatically on destruction.
        """
        ...
    def width(self) -> Incomplete:
        """
           Returns the width of the screen.
        """
        ...
    def height(self) -> Incomplete:
        """
           Returns the height of the screen.
        """
        ...
    def refresh(self) -> Incomplete:
        """
           Returns the refresh rate.
        """
        ...
    def write(self, image, x=0, y=0, x_scale=1.0, y_scale=1.0, roi=None, rgb_channel=-1, alpha=256, color_palette=None, alpha_palette=None, hint=0) -> Incomplete:
        """
           Displays an ``image`` whose top-left corner starts at location x, y.
        
           You may also pass a path instead of an image object for this method to automatically load the image
           from disk and draw it in one step. E.g. ``write("test.jpg")``.
        
           ``x_scale`` controls how much the displayed image is scaled by in the x direction (float). If this
           value is negative the image will be flipped horizontally. Note that if ``y_scale`` is not specified
           then it will match ``x_scale`` to maintain the aspect ratio.
        
           ``y_scale`` controls how much the displayed image is scaled by in the y direction (float). If this
           value is negative the image will be flipped vertically. Note that if ``x_scale`` is not specified
           then it will match ``x_scale`` to maintain the aspect ratio.
        
           ``roi`` is the region-of-interest rectangle tuple (x, y, w, h) of the image to display. This
           allows you to extract just the pixels in the ROI to scale.
        
           ``rgb_channel`` is the RGB channel (0=R, G=1, B=2) to extract from an RGB565 image (if passed)
           and to render on the display. For example, if you pass ``rgb_channel=1`` this will
           extract the green channel of the RGB565 image and display that in grayscale.
        
           ``alpha`` controls how opaque the image is. A value of 256 displays an opaque image while a
           value lower than 256 produces a black transparent image. 0 results in a perfectly black image.
        
           ``color_palette`` if not ``-1`` can be `image.PALETTE_RAINBOW`, `image.PALETTE_IRONBOW`, or
           a 256 pixel in total RGB565 image to use as a color lookup table on the grayscale value of
           whatever the input image is. This is applied after ``rgb_channel`` extraction if used.
        
           ``alpha_palette`` if not ``-1`` can be a 256 pixel in total GRAYSCALE image to use as a alpha
           palette which modulates the ``alpha`` value of the input image being displayed at a pixel pixel
           level allowing you to precisely control the alpha value of pixels based on their grayscale value.
           A pixel value of 255 in the alpha lookup table is opaque which anything less than 255 becomes
           more transparent until 0. This is applied after ``rgb_channel`` extraction if used.
        
           ``hint`` can be a logical OR of the flags:
        
              * `image.AREA`: Use area scaling when downscaling versus the default of nearest neighbor.
              * `image.BILINEAR`: Use bilinear scaling versus the default of nearest neighbor scaling.
              * `image.BICUBIC`: Use bicubic scaling versus the default of nearest neighbor scaling.
              * `image.CENTER`: Center the image being drawn on the display. This is applied after scaling.
              * `image.HMIRROR`: Horizontally mirror the image.
              * `image.VFLIP`: Vertically flip the image.
              * `image.TRANSPOSE`: Transpose the image (swap x/y).
              * `image.EXTRACT_RGB_CHANNEL_FIRST`: Do rgb_channel extraction before scaling.
              * `image.APPLY_COLOR_PALETTE_FIRST`: Apply color palette before scaling.
              * `image.SCALE_ASPECT_KEEP`: Scale the image being drawn to fit inside the display.
              * `image.SCALE_ASPECT_EXPAND`: Scale the image being drawn to fill the display (results in cropping)
              * `image.SCALE_ASPECT_IGNORE`: Scale the image being drawn to fill the display (results in stretching).
              * `image.ROTATE_90`: Rotate the image by 90 degrees (this is just VFLIP | TRANSPOSE).
              * `image.ROTATE_180`: Rotate the image by 180 degrees (this is just HMIRROR | VFLIP).
              * `image.ROTATE_270`: Rotate the image by 270 degrees (this is just HMIRROR | TRANSPOSE).
        """
        ...
    def clear(self, display_off=False: Optional[Any]=None) -> Incomplete:
        """
           Clears the lcd screen to black.
        
           ``display_off`` if True instead turns off the display logic versus clearing the frame LCD
           frame buffer to black. You should also turn off the backlight too after this to ensure the
           screen goes to black as many displays are white when only the backlight is on.
        """
        ...
    def backlight(self, value: Optional[Any]=None) -> Incomplete:
        """
           Sets the lcd backlight dimming value. 0 (off) to 100 (on).
        
           Note that unless you pass `DACBacklight` or `PWMBacklight` the backlight will be controlled
           as a GPIO pin and will only go from 0 (off) to !0 (on).
        
           Pass no arguments to get the state of the backlight value.
        """
        ...
class DSIDisplay():
    """
        ``framesize`` One of the standard supported resolutions.
    
        ``refresh`` Sets the screen refresh rate in hertz. This controls the DSI LCD clock.
    
        ``portrait`` Swap the framesize width and height.
    
        ``channel`` The virtual MIPI DSI channel to use to talk to the display.
    
        ``controller`` Pass the controller chip class here to initialize it along with the display. E.g.
        `display.ST7701()` which is a standard display controller for MIPI DSI displays.
    
        ``backlight`` specify a backlight controller module to use. By default the backlight will be
        controlled via a GPIO pin.
    """
    def __init__(self, framesize=FWVGA, refresh=60, portrait=False, channel=0, controller, backlight) -> None:
        ...
    def deinit(self) -> Incomplete:
        """
           Releases the I/O pins and RAM used by the class. This is called automatically on destruction.
        """
        ...
    def width(self) -> Incomplete:
        """
           Returns the width of the screen.
        """
        ...
    def height(self) -> Incomplete:
        """
           Returns the height of the screen.
        """
        ...
    def refresh(self) -> Incomplete:
        """
           Returns the refresh rate.
        """
        ...
    def write(self, image, x=0, y=0, x_scale=1.0, y_scale=1.0, roi=None, rgb_channel=-1, alpha=256, color_palette=None, alpha_palette=None) -> Incomplete:
        """
           Displays an ``image`` whose top-left corner starts at location x, y.
        
           You may also pass a path instead of an image object for this method to automatically load the image
           from disk and draw it in one step. E.g. ``write("test.jpg")``.
        
           ``x_scale`` controls how much the displayed image is scaled by in the x direction (float). If this
           value is negative the image will be flipped horizontally. Note that if ``y_scale`` is not specified
           then it will match ``x_scale`` to maintain the aspect ratio.
        
           ``y_scale`` controls how much the displayed image is scaled by in the y direction (float). If this
           value is negative the image will be flipped vertically. Note that if ``x_scale`` is not specified
           then it will match ``x_scale`` to maintain the aspect ratio.
        
           ``roi`` is the region-of-interest rectangle tuple (x, y, w, h) of the image to display. This
           allows you to extract just the pixels in the ROI to scale.
        
           ``rgb_channel`` is the RGB channel (0=R, G=1, B=2) to extract from an RGB565 image (if passed)
           and to render on the display. For example, if you pass ``rgb_channel=1`` this will
           extract the green channel of the RGB565 image and display that in grayscale.
        
           ``alpha`` controls how opaque the image is. A value of 256 displays an opaque image while a
           value lower than 256 produces a black transparent image. 0 results in a perfectly black image.
        
           ``color_palette`` if not ``-1`` can be `image.PALETTE_RAINBOW`, `image.PALETTE_IRONBOW`, or
           a 256 pixel in total RGB565 image to use as a color lookup table on the grayscale value of
           whatever the input image is. This is applied after ``rgb_channel`` extraction if used.
        
           ``alpha_palette`` if not ``-1`` can be a 256 pixel in total GRAYSCALE image to use as a alpha
           palette which modulates the ``alpha`` value of the input image being displayed at a pixel pixel
           level allowing you to precisely control the alpha value of pixels based on their grayscale value.
           A pixel value of 255 in the alpha lookup table is opaque which anything less than 255 becomes
           more transparent until 0. This is applied after ``rgb_channel`` extraction if used.
        
           ``hint`` can be a logical OR of the flags:
        
              * `image.AREA`: Use area scaling when downscaling versus the default of nearest neighbor.
              * `image.BILINEAR`: Use bilinear scaling versus the default of nearest neighbor scaling.
              * `image.BICUBIC`: Use bicubic scaling versus the default of nearest neighbor scaling.
              * `image.CENTER`: Center the image being drawn on the display. This is applied after scaling.
              * `image.HMIRROR`: Horizontally mirror the image.
              * `image.VFLIP`: Vertically flip the image.
              * `image.TRANSPOSE`: Transpose the image (swap x/y).
              * `image.EXTRACT_RGB_CHANNEL_FIRST`: Do rgb_channel extraction before scaling.
              * `image.APPLY_COLOR_PALETTE_FIRST`: Apply color palette before scaling.
              * `image.SCALE_ASPECT_KEEP`: Scale the image being drawn to fit inside the display.
              * `image.SCALE_ASPECT_EXPAND`: Scale the image being drawn to fill the display (results in cropping)
              * `image.SCALE_ASPECT_IGNORE`: Scale the image being drawn to fill the display (results in stretching).
              * `image.ROTATE_90`: Rotate the image by 90 degrees (this is just VFLIP | TRANSPOSE).
              * `image.ROTATE_180`: Rotate the image by 180 degrees (this is just HMIRROR | VFLIP).
              * `image.ROTATE_270`: Rotate the image by 270 degrees (this is just HMIRROR | TRANSPOSE).
        """
        ...
    def clear(self, display_off=False: Optional[Any]=None) -> Incomplete:
        """
           Clears the lcd screen to black.
        
           ``display_off`` if True instead turns off the display logic versus clearing the frame LCD
           frame buffer to black. You should also turn off the backlight too after this to ensure the
           screen goes to black as many displays are white when only the backlight is on.
        """
        ...
    def backlight(self, value: Optional[Any]=None) -> Incomplete:
        """
           Sets the lcd backlight dimming value. 0 (off) to 100 (on).
        
           Note that unless you pass `DACBacklight` or `PWMBacklight` the backlight will be controlled
           as a GPIO pin and will only go from 0 (off) to !0 (on).
        
           Pass no arguments to get the state of the backlight value.
        """
        ...
    def bus_write(self, cmd, args=None, dcs=False) -> None:
        """
           Send the DSI Display ``cmd`` with ``args``.
        """
        ...
    def bus_read(self, cmd, len, args=None, dcs=False: Optional[Any]=None) -> Incomplete:
        """
           Read ``len`` using ``cmd`` with ``args`` from the DSI Display.
        """
        ...
class DisplayData():
    """
        ``cec`` Pass `True` to enable CEC communication to an external display (if possible).
    
        ``ddc`` Pass `True` to enable DDC communication to an external display (if possible).
    
        ``ddc_addr`` The I2C address to use to talk to the external display EEPROM.
    """
    def __init__(self, cec=False, ddc=False, ddc_addr=0x50: Optional[Any]=None) -> None:
        ...
    def send_frame(self, dst_addr, src_addr, bytes) -> None:
        """
           Sends a packet on the HDMI-CEC bus to ``dst_addr`` with source ``src_addr`` and data ``bytes``.
        """
        ...
    def receive_frame(self, dst_addr, timeout=1000) -> bool:
        """
           Waits ``timeout`` milliseconds to receive an HDMI-CEC
           frame for address ``dst_addr``. Returns True if the received frame was for ``dst_addr`` and False
           if not. On timeout throws an `OSError` Exception.
        """
        ...
    def frame_callback(self, callback, dst_addr) -> Incomplete:
        """
           Registers a ``callback`` which will be called on reception of an
           HDMI-CEC frame. The callback will receive two arguments of the frame src_addr as an int and
           payload as a `bytes()` object.
        
           ``dst_addr`` sets the filter address to listen to on the CEC bus.
        
           If you use this method do not call `DisplayData.receive_frame()` anymore until the callback is
           disabled by passing ``None`` as the callback for this method.
        """
        ...
class display():
    """ """
    def display_id(self) -> Incomplete:
        """
           Returns the external display EDID data as a bytes()
           object. Verifying the EDID headers, checksums, and concatenating all sections into one bytes()
           object is done for you. You may then parse this information by `following this guide <https://en.wikipedia.org/wiki/Extended_Display_Identification_Data>`__.
        """
        ...
class ST7701():
    """
       Creates a controller object to initialize the ST7701 display controller which typically powers
       MIPI DSI displays. This class should be passed as the ``cotnroller`` argument to the `DSIDisplay()`
       class constructor which will take care of calling the `ST7701.init()` method for you.
    """
    def __init__(self) -> None:
        ...
    def init(self, display_controller) -> Incomplete:
        """
           Initializes the display using the display controller which must provide `display.DSIDisplay.bus_write()` and
           `display.DSIDisplay.bus_read()` methods.
        """
        ...
    def read_id(self) -> Incomplete:
        """
           Returns the display id.
        """
        ...
class DACBacklight():
    """
       Creates a backlight object to initialize the display backlight. This class should be passed as
       the ``backlight`` argument to any display object constructor which can use a backlight controller.
    
       ``channel`` specifies the DAC channel to use. This can be the GPIO pin also. For STM32 based
       OpenMV Cams this is ``P5``.
    
       ``bits`` specifies the resolution of the DAC. The default value of 8-bits should be good enough.
    """
    def __init__(self, channel, bits=8: Optional[Any]=None) -> None:
        ...
    def deinit(self) -> Incomplete:
        """
           Deinitializes the backlight controller.
        """
        ...
    def backlight(self, value: Optional[Any]=None) -> None:
        """
           Sets the backlight strength from 0-100. Note that a linear voltage on the backlight output
           will not necessary result in a linear brightness change on the screen. Typically there's
           a small region where the screen brightness will change drastically.
        """
        ...
class PWMBacklight():
    """
       Creates a backlight object to initialize the display backlight. This class should be passed as
       the ``backlight`` argument to any display object constructor which can use a backlight controller.
    
       ``pin`` specifies the Pin to use.
    
       ``timer`` specifies the Timer number to use.
    
       ``channel`` specifies the Timer channel to use.
    
       ``frequency`` specifies the PWM frequency.
    """
    def __init__(self, pin, timer=3, channel=3, frequency=200: Optional[Any]=None) -> None:
        ...
    def deinit(self) -> Incomplete:
        """
           Deinitializes the backlight controller.
        """
        ...
    def backlight(self, value: Optional[Any]=None) -> None:
        """
           Sets the backlight strength from 0-100. Note that a linear pwm duty cycle on the backlight output
           will not necessary result in a linear brightness change on the screen. Typically there's
           a small region where the screen brightness will change drastically.
        """
        ...
