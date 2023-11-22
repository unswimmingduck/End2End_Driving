import math


def lr_stepping(optimizer, current_epoch, setp_epoch, all_epoch,lr):
    
    if current_epoch % setp_epoch == 0 and current_epoch!= all_epoch:
        rate = float(current_epoch/all_epoch)
        lr *= float(math.exp(rate*(-1)))
        optimizer.param_groups[0]['lr'] = lr