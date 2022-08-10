def ppv(output, target):
    smooth = 1e-5
    intersection = (output * target).sum()
    return (intersection + smooth) / (output.sum() + smooth)

