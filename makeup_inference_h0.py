import argparse, os, sys, glob
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
# from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
import torchvision
from ldm.util import instantiate_from_config,log_txt_as_img
from ldm.models.diffusion.ddim_test import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.data.makeup_dataset import MakeupDatasetTest
import torch.nn.functional as F

# from ldm.util import log_txt_as_img, exists, instantiate_from_config


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model



def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                            (0.5, 0.5, 0.5))]
    return torchvision.transforms.Compose(transform_list)



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/makeup_results"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=500,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=256,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=256,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--n_imgs",
        type=int,
        default=100,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=3,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=4,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given reference image. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--source_image_path",
        type=str,
        help="evaluate at this precision",
        default=""
    )
    parser.add_argument(
        "--source_seg_path",
        type=str,
        help="evaluate at this precision",
        default=""
    )
    parser.add_argument(
        "--source_depth_path",
        type=str,
        help="evaluate at this precision",
        default=""
    )
    parser.add_argument(
        "--ref_image_path",
        type=str,
        help="evaluate at this precision",
        default=""
    )
    parser.add_argument(
        "--ref_seg_path",
        type=str,
        help="evaluate at this precision",
        default=""
    )

    opt = parser.parse_args()

    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size

    
    grid_path = os.path.join(outpath, "grid_"+f'scale{opt.scale}_step{opt.ddim_steps}')
    
    os.makedirs(grid_path, exist_ok=True)

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    dataset = MakeupDatasetTest(mode='test_pair', source_image_path=opt.source_image_path, source_seg_path=opt.source_seg_path,
                                source_depth_path=opt.source_depth_path,
                                ref_image_path=opt.ref_image_path,ref_seg_path=opt.ref_seg_path,)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    count=0

    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for iter, batch in enumerate(test_loader):
                    count+=1
                    source_image = batch['source_image'].to(device).float()
                    ref_image = batch['ref_image'].to(device).float()

                    source_depth = batch['source_depth'].to(device).float()
                    source_bg= batch['source_bg'].to(device).float()
                    source_face_gray = batch['source_face_gray'].to(device).float()
                    ref_face = batch['ref_face'].to(device).float()

                    source_seg_onehot = batch['source_seg_onehot'].to(device).float()
                    ref_seg_onehot = batch['ref_seg_onehot'].to(device).float()

                    source_face_seg = batch['source_face_seg'].to(device).float()

                    # get h0
                    source_HF_0 = model.lap_pyr_c1.pyramid_decom(source_face_gray)[0] # h0
                    source_HF_0_down4 = F.pixel_unshuffle(source_HF_0, downscale_factor=4)

                    source_depth_down4 = F.pixel_unshuffle(source_depth, downscale_factor=4)

                    source_HF = torch.cat([source_HF_0_down4,source_depth_down4],dim=1)

                    ref_LF_64 = F.pixel_unshuffle(ref_face, downscale_factor=4)


                    source_face_seg_64 = F.interpolate(source_face_seg, size=(64, 64), mode='bilinear')


                    encoder_posterior_bg = model.encode_first_stage(source_bg)
                    z_bg = model.get_first_stage_encoding(encoder_posterior_bg).detach()

                    encoder_posterior_ref_LF = model.encode_first_stage(ref_face)
                    z_ref_LF = model.get_first_stage_encoding(encoder_posterior_ref_LF).detach()

                    test_model_kwargs = {}
                    test_model_kwargs['z_bg'] = z_bg.to(device)
                    test_model_kwargs['source_HF'] = source_HF.to(device)
                    test_model_kwargs['z_ref_LF'] = z_ref_LF.to(device)
                    test_model_kwargs['ref_LF_64'] = ref_LF_64.to(device)
                    test_model_kwargs['source_face_seg_64'] = source_face_seg_64.to(device)

                    test_model_kwargs['source_seg_onehot'] = source_seg_onehot.to(device)
                    test_model_kwargs['ref_seg_onehot'] = ref_seg_onehot.to(device)

                    uc = None
                    if opt.scale != 1.0:
                        uc = model.learnable_vector
                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                    samples_ddim, intermediates = sampler.sample(S=opt.ddim_steps,
                                                     batch_size=opt.n_samples,
                                                     shape=shape,
                                                     verbose=False,
                                                     unconditional_guidance_scale=opt.scale,
                                                     unconditional_conditioning=uc,
                                                     eta=opt.ddim_eta,
                                                     x_T=start_code,
                                                    log_every_t=1,
                                                     test_model_kwargs=test_model_kwargs)

                    
                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    source_image = torch.clamp((source_image + 1.0) / 2.0, min=0.0, max=1.0)
                    ref_image = torch.clamp((ref_image + 1.0) / 2.0, min=0.0, max=1.0)
                    
                    if not opt.skip_save:
                        row_0=torch.cat([source_image[0,::],ref_image[0, ::],x_samples_ddim[0, ::]],dim=2)
                        grid = make_grid(row_0)
                        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                        img = Image.fromarray(grid.astype(np.uint8))
                        img.save(os.path.join(grid_path, 'grid_' + '%05d'%count + '_seed_'+str(opt.seed)+'.jpg'))

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()
