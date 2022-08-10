def sensitivity(output, target):
    smooth = 1e-5
    intersection = (output * target).sum()
    return (intersection + smooth) / (target.sum() + smooth)
