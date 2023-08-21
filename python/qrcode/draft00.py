import qrcode

# qrcode.image.pil.PilImage
img = qrcode.make('https://www.google.com')
# img.save("tbd00.png")

from qrcode.image.styledpil import StyledPilImage
from qrcode.image.styles.moduledrawers.pil import RoundedModuleDrawer
from qrcode.image.styles.colormasks import RadialGradiantColorMask

qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_L)
qr.add_data('https://www.google.io')
img_1 = qr.make_image(image_factory=StyledPilImage, module_drawer=RoundedModuleDrawer())
# img_2 = qr.make_image(image_factory=StyledPilImage, color_mask=RadialGradiantColorMask())
# img_3 = qr.make_image(image_factory=StyledPilImage, embeded_image_path="/path/to/image.png")
