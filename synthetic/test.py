import os
from synthetic.options.test_options import TestOptions
from synthetic.data import create_dataset
from synthetic.models import create_model
from synthetic.util.visualizer import save_images
from itertools import islice
from synthetic.util import html
import torch

# options
opt = TestOptions().parse()
#opt = TrainOptions().parse()
opt.num_threads = 1   # test code only supports num_threads=1
opt.batch_size = 1   # test code only supports batch_size=1
opt.serial_batches = True  # no shuffle

# create dataset
dataset = create_dataset(opt)
model = create_model(opt)
model.setup(opt)
#model.eval()
#model.train()
#model.optimize_parameters()
print('Loading model %s' % opt.model)

# create website
#web_dir = os.path.join(opt.results_dir, opt.phase + '_sync' if opt.sync else opt.phase)
#webpage = html.HTML(web_dir, 'Training = %s, Phase = %s, Class =%s' % (opt.name, opt.phase, opt.name))

# sample random z
if opt.sync:
    z_samples = model.get_z_random(opt.n_samples + 1, opt.nz)

def synthesize(src, tgt):
    batch_size=src.shape[0]
    #print(batch_size)
    if src.shape[1]==1:
        src = torch.cat([src, src, src],dim=1)

    tgt_cat = torch.cat([tgt, tgt, tgt], dim=1)
    paired_gt = torch.cat([tgt_cat, src], dim=3)
    model.set_input(paired_gt)
    if not opt.sync:
        z_samples = model.get_z_random(opt.n_samples + 1, opt.nz)
    batches=[]
    for nn in range(opt.n_samples + 1):
        encode = nn == 0 and not opt.no_encode
        real_A, fake_B, real_B = model.test(z_samples[[nn]].repeat(batch_size,1), encode=encode)
        if nn == 0:
            pair_gt = torch.cat([real_B, real_A], dim=3)
            images = [pair_gt]
        else:
            pair_fake = torch.cat([fake_B, real_A], dim=3)
            images.append(pair_fake)
      
    new_batch = torch.cat(images, dim=0)
    src_new = new_batch[:,:,:,:256]
    tgt_new = new_batch[:,:1,:,256:]
    return src_new, tgt_new
    #print(new_batch.shape) # (4,3,256,256)
'''
src = torch.randn((4,3,256,256))
tgt = torch.randn((4,1,256,256))
sc, tc = synthesize(src, tgt)
print(sc.shape, tc.shape)
'''

