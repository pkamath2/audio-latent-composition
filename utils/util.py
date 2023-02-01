import torch
import urllib

def set_requires_grad(requires_grad, *models):
    for model in models:
        if isinstance(model, torch.nn.Module):
            for param in model.parameters():
                param.requires_grad = requires_grad
        elif isinstance(model, (torch.nn.Parameter, torch.Tensor)):
            model.requires_grad = requires_grad
        else:
            assert False, 'unknown type %r' % type(model)

def is_url(url):
    scheme = urllib.parse.urlparse(url).scheme
    return scheme in ('http', 'https')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def renormalize(n, range1, range2):
    delta1 = range1[1] - range1[0]
    delta2 = range2[1] - range2[0]
    return (delta2 * (n - range1[0]) / delta1) + range2[0]

def threshold_spectrogram(im0, threshold=17):
    im0_ = (im0  * 127.5+ 128).clamp(0, 255)/255.0
    im0_ = -50+im0_*50
    
    im0_ = -im0_
    im0_ = (im0_ * ((im0_>threshold).int()*1000)).clamp(0, 50)
    im0_ = -im0_ 

    im0_ = (im0_ + 50.0)/50.0
    im0_ = ((im0_ * 255.0) - 128.0)/127.5
    return im0_