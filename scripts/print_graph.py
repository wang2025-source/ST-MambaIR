import torch
from torchviz import make_dot
from basicsr.archs.NAFNet_arch import NAFNet  # replace with your model

# init model and input
model = NAFNet(img_channel=3, width=32, middle_blk_num=1, enc_blk_nums=[1,1,1,28], dec_blk_nums=[1,1,1,1])
x = torch.randn(1, 3, 256, 256)

# forward
y = model(x)

# plot
make_dot(y, params=dict(model.named_parameters())).render("nafnet_graph", format="png")
