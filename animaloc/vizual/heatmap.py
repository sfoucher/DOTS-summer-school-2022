import torch
import matplotlib
import PIL

import torchvision.transforms.functional as TF

from PIL import Image

from typing import Optional

from .objects import draw_points

__all__ = ['overlay_heatmap']

def overlay_heatmap(
    image: torch.Tensor,
    map: torch.Tensor,
    alpha: float = 0.5,
    points: Optional[list] = None
    ) -> PIL.Image.Image :
    ''' Overlay a probability map (values in [0,1]) onto a RGB image

    Args:
        image (torch.Tensor): RGB image of shape [3,H,W], values in [0,255]
        map (torch.Tensor): 2D tensor of shape [H,W], values in [0,1]
        alpha (float, optional): transparency of the map from transparent (0;0)
            to opaque (1.0). Defaults to 0.3.
        points (list, optional): list of points coordinates [y,x] to plot over
            the image. Defaults to None.

    Returns:
        PIL.Image.Image:
            the image with the probability map overlaid
    '''

    assert image.dim() == 3, \
        f'image must be a 3D torch.Tensor (RGB), got {image.dim()} dim'
    
    assert map.dim() == 2, \
        f'map must be a 2D torch.Tensor, got {map.dim()} dim'
    
    color_map = torch.Tensor([matplotlib.cm.viridis(x)[:3] for x in range(256)])
    # color_map[:,-1] = torch.linspace(0.0, 1.0, 256)
    gray_image = (map * 255).long()
    probmap = color_map[gray_image]

    image = TF.to_pil_image(image)
    probmap_image = TF.to_pil_image(probmap.permute(2,0,1))

    map_image = Image.blend(image, probmap_image, alpha=alpha)
    # map_image = Image.alpha_composite(image.convert('RGBA'), probmap_image)

    if points is not None:
        map_image = draw_points(map_image, points, color='yellow')

    return map_image