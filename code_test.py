import timm
from timm.models import create_model
import segmentation_models_pytorch as smp


def load_model():
    # model = create_model(
    #     model_name='convnext',
    #     pretrained='imagenet',
    # )
    # print(model)
    model_list = timm.list_models()
    print(model_list)
    model = smp.UnetPlusPlus(
        encoder_name='timm-convnext'
    )


if __name__ == '__main__':
    load_model()