from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import os
import time
import sys
import torch
# my import
import matplotlib.pyplot as plt
import numpy as np
#

USE_TENSORBOARD = True
try:
  import tensorboardX
  print('Using tensorboardX')
except:
  USE_TENSORBOARD = False

class Logger(object):
  def __init__(self, opt):
    """Create a summary writer logging to log_dir."""
    if not os.path.exists(opt.save_dir):
      os.makedirs(opt.save_dir)
    if not os.path.exists(opt.debug_dir):
      os.makedirs(opt.debug_dir)
   
    time_str = time.strftime('%Y-%m-%d-%H-%M')

    args = dict((name, getattr(opt, name)) for name in dir(opt)
                if not name.startswith('_'))
    file_name = os.path.join(opt.save_dir, 'opt.txt')
    with open(file_name, 'wt') as opt_file:
      opt_file.write('==> torch version: {}\n'.format(torch.__version__))
      opt_file.write('==> cudnn version: {}\n'.format(
        torch.backends.cudnn.version()))
      opt_file.write('==> Cmd:\n')
      opt_file.write(str(sys.argv))
      opt_file.write('\n==> Opt:\n')
      for k, v in sorted(args.items()):
        opt_file.write('  %s: %s\n' % (str(k), str(v)))
          
    log_dir = opt.save_dir + '/logs_{}'.format(time_str)
    if USE_TENSORBOARD:
      self.writer = tensorboardX.SummaryWriter(log_dir=log_dir)
    else:
      if not os.path.exists(os.path.dirname(log_dir)):
        os.mkdir(os.path.dirname(log_dir))
      if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    self.log = open(log_dir + '/log.txt', 'w')
    try:
      os.system('cp {}/opt.txt {}/'.format(opt.save_dir, log_dir))
    except:
      pass
    self.start_line = True
    #mycode================
    self.log_dir = log_dir
    #======================
  def draw_loss_graph(self, train_loss, val_loss):
    # Epochs for the validation loss (e.g., [2, 5, 8] if validation occurs at every 3 epochs)
    val_epochs = np.arange(2, len(train_loss), 3)

    # Plotting
    plt.plot(train_loss, label="Training Loss")
    plt.plot(val_epochs, val_loss, label="Validation Loss", marker='o', linestyle='--')

    # Adding labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(self.log_dir, 'training_validation_loss.png'))
    # Show plot
    plt.show()
    return


  def write(self, txt):
    if self.start_line:
      time_str = time.strftime('%Y-%m-%d-%H-%M')
      self.log.write('{}: {}'.format(time_str, txt))
    else:
      self.log.write(txt)  
    self.start_line = False
    if '\n' in txt:
      self.start_line = True
      self.log.flush()
  
  def close(self):
    self.log.close()
  
  def scalar_summary(self, tag, value, step):
    """Log a scalar variable."""
    if USE_TENSORBOARD:
      self.writer.add_scalar(tag, value, step)
