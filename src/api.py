import model
import utils
import torch
from PIL import Image
import io


def obtain_model(path):
    """
    Obtains a pretrained model from the given path.

    Args:
        path (str): The path to the pretrained model file.

    Returns:
        torch.nn.Module: The loaded pretrained model.

    Raises:
        FileNotFoundError: If the pretrained model file is not found.
    """
    obtained = model.get_model(pretrained=True, path=path)
    return obtained


def get_prediction(network, image_data):
    """
    Get a prediction from the given network using the provided image data.

    Parameters:
        network (torch.nn.Module): The network used to make predictions.
        image_data (bytes): The image data to be used for prediction.

    Returns:
        dict: A dictionary containing the prediction results. The keys are 'safe' and 'inappropriate',
              and the values are the corresponding probabilities.

    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image = Image.open(io.BytesIO(image_data))
    # image = Image.open('image.png')
    # image.show()
    img = utils.preprocessors(image)
    img = img.unsqueeze(0)
    network.to(device)

    img = img.to(device)
    print(img.shape)
    with torch.no_grad():
        results = network(img)
    res = {'safe': results[0][0].item(), 'inappropriate': results[0][1].item()}
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res


print(get_prediction(obtain_model('frozen_MSE_lr0.1_best.pt'), 'image.png'))
