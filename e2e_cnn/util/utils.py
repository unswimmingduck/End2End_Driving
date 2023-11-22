import os
import torch



class AverageMeter(object):
    def __init__(self) -> None:
        self.val = 0
        self.sum = 0
        self.avg = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def get_avg(self):
        return self.avg


def checkpoint_save(epoch, model, optimizer, work_dir):
    if hasattr(model, "module"):
        model = model.module.to("cpu")
    
    save_path = os.path.join(work_dir, f'epoch_{epoch}.pth')
    save_checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
    torch.save(save_checkpoint, save_path)

    # update the latest.pth
    if os.path.exists(f'{work_dir}/latest.pth'):
        os.remove(f'{work_dir}/latest.pth')
    os.system(f'cd {work_dir}; ln -s {os.path.basename(save_path)} latest.pth')