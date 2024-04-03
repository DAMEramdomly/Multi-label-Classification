import torch
from thop import profile
import time

# -------------------- import networks  --------------------

# --------------------------- CNN ---------------------------
from resnet import resnet18
from resnet import resnet34
from resnet import resnet50
from model.DenseNet import DenseNet
from torchvision.models import MobileNetV2
from torchvision.models import efficientnet_b0
from torchvision.models import efficientnet_b1
from torchvision.models import efficientnet_b2
from torchvision.models import convnext_tiny
from torchvision.models import convnext_small

# ----------------------- Transformer -----------------------
from model.VIT import ViT
from model.SwinT import SwinTransformer
from model.restv1 import rest_lite, rest_small
from model.restv2 import restv2_tiny, restv2_small
from model.ShuntedT import shunted_t, shunted_s
# -------------------------- Hybrid --------------------------
from model.fitnet import FITNet
from model.medvit import MedViT_small
# -------------------------- Django --------------------------
from model.MINE.hybrid4 import hybrid4


def choice(model_name):
    # --------------------------- CNN ---------------------------
    if model_name == "resnet18":
        model = resnet18()
    elif model_name == "resnet34":
        model = resnet34()
    elif model_name == "resnet50":
        model = resnet50()
    elif model_name == "densenet":
        model = DenseNet()
    elif model_name == "mobilenet":
        model = MobileNetV2()
    elif model_name == "efficientnet0":
        model = efficientnet_b0()
    elif model_name == "efficientnet1":
        model = efficientnet_b1()
    elif model_name == "efficientnet2":
        model = efficientnet_b2()
    elif model_name == "convnext_tiny":
        model = convnext_tiny()
    elif model_name == "convnext_small":
        model = convnext_small()

    # ----------------------- Transformer -----------------------
    elif model_name == "swint":
        model = SwinTransformer()
    elif model_name == "vit":
        model = ViT()
    elif model_name == "restv1_lite":
        model = rest_lite()
    elif model_name == "restv1_small":
        model = rest_small()
    elif model_name == "restv2_tiny":
        model = restv2_tiny()
    elif model_name == "restv2_small":
        model = restv2_small()
    elif model_name == "shunted_tiny":
        model = shunted_t()
    elif model_name == "shunted_small":
        model = shunted_s()

    # -------------------------- Hybrid --------------------------
    elif model_name == "fitnet":
        model = FITNet()
    elif model_name == "medvit_s":
        model = MedViT_small()

    # ------------------------- GAFromer -------------------------
    elif model_name == "mine":
        model = hybrid4()

    return model

if __name__ == "__main__":

    model_name = "restv1_lite"
    model = choice(model_name).cuda()

    model.eval()
    dummy_input = torch.randn(4, 3, 256, 512).cuda()
    start_time = time.time()
    for i in range(152):
        output = model(dummy_input)
    end_time = time.time()
    print("test_time: {:.5f}s".format(end_time - start_time))
    print("test_time: {:.5f}ms".format((end_time - start_time) / 608 * 1000))

    flops, params = profile(model, inputs=(dummy_input,))
    print(f"Total Params: {params}")
    print(f"Total FLOPs: {flops / 4.0 / 1e9} GFLOPs")