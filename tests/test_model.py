import torch

from src.model import get_model


def test_model_output_shape():
    model = get_model(num_classes=10)
    model.eval()

    x = torch.randn(4, 3, 32, 32)

    with torch.no_grad():
        output = model(x)

    assert output.shape == (4, 10)


def test_model_is_resnet18_for_cifar10():
    model = get_model(num_classes=10)

    assert model.conv1.kernel_size == (3, 3)
    assert model.conv1.stride == (1, 1)
    assert model.fc.out_features == 10