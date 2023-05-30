from PIL import Image
import matplotlib.pyplot as plt
from super_resolve import *

input_image = r"C:\Users\hlia981\Downloads\Linnaeus 5 256X256\Linnaeus 5 256X256\bird\test\1083_256.jpg"
raw_ = Image.open(input_image)
width, height = raw_.size
new_ = raw_.resize((int(width/3),int(height/3)))
input = new_.resize((width,height),Image.BICUBIC)

fig, ax = plt.subplots(1, 3)
ax[0].imshow(raw_)
ax[0].axis('off')  # Hide axes
ax[1].imshow(input)
ax[1].axis('off')  # Hide axes

output = super_resolve(new_)
ax[2].imshow(output)
ax[2].axis('off')  # Hide axes
raw_.save("raw.png")
input.save("input.png")
output.save("output.png")
# Show the figure
plt.show()

# raw_.show()
# input.show()