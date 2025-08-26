import torch
from PIL import Image
import torchvision.transforms.functional as F
    
def count_bp(tensor: torch.Tensor):
    # Sum along the channel dimension to get a 2D tensor [height, width]
    channel_sum = torch.sum(tensor, dim=0)

    # Check if any pixel has a sum of 0, indicating black
    black_pixels = (channel_sum.view(-1) == 0)
    return torch.sum(black_pixels == True).item()

def condition_black_pixels(path: str, threshold: int):
    '''
    Return True if the image has more than threshold of black pixels. A black pixel is a 0 value on three channels. 
    '''
    tensor = F.pil_to_tensor(Image.open(path))

    count_black_pixels = count_bp(tensor)
    return (count_black_pixels > threshold) 

def condition_border_bp_tensor(tensor:torch.Tensor, threshold: int, border_width: int):
    '''
    Return True if the image has more than threshold of black pixels in the borders of the image. 
    ''' 
    height, width = tensor.shape[1:3]

    top_mask = tensor[:, height-border_width:height, :]
    bottom_mask = tensor[:, :border_width, :]
    left_mask = tensor[:, :, :border_width]
    right_mask = tensor[:, :, width-border_width:width,]

    top_count_bp = count_bp(top_mask)
    bottom_count_bp = count_bp(bottom_mask)
    left_count_bp = count_bp(left_mask)
    right_count_bp = count_bp(right_mask)
    
    return (top_count_bp+bottom_count_bp+left_count_bp+right_count_bp) > threshold

def condition_border_bp_path(path:str, threshold: int, border_width: int):
    '''
    Return True if the image has more than threshold of black pixels in the borders of the image. 
    ''' 
    tensor = F.pil_to_tensor(Image.open(path))
    return condition_border_bp_tensor(tensor, threshold=threshold, border_width=border_width)

