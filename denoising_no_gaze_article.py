import torch
from tqdm import tqdm
import torchvision.utils as tvu
import torchvision
import os
from PIL import Image
import torchvision.transforms as T
from torch.cuda.amp import autocast, GradScaler

from .clip.base_clip import CLIPEncoder
from .face_parsing.model import FaceParseTool
from .anime2sketch.model import FaceSketchTool
from .landmark.model import FaceLandMarkTool
from .arcface.model import IDLoss
from .arcface.model import ExpressionArcFaceTool

from .perceptual.model import PerceptualFaceTool


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def clip_ddim_diffusion(x, seq, model, b, cls_fn=None, rho_scale=1.0, prompt=None, stop=100, domain="face"):
    clip_encoder = CLIPEncoder().cuda()

    # setup iteration variables
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]

    # iterate over the timesteps
    for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        at = compute_alpha(b, t.long())
        at_next = compute_alpha(b, next_t.long())
        xt = xs[-1].to('cuda')

        if domain == "face":
            repeat = 1
        elif domain == "imagenet":
            if 800 >= i >= 500:
                repeat = 10
            else:
                repeat = 1
        
        for idx in range(repeat):
        
            xt.requires_grad = True
            
            et = model(xt, t)

            if et.size(1) == 6:
                et = et[:, :3]

            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            
            # get guided gradient
            residual = clip_encoder.get_residual(x0_t, prompt)
            norm = torch.linalg.norm(residual)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=xt)[0]

            c1 = at_next.sqrt() * (1 - at / at_next) / (1 - at)
            c2 = (at / at_next).sqrt() * (1 - at_next) / (1 - at)
            c3 = (1 - at_next) * (1 - at / at_next) / (1 - at)
            c3 = (c3.log() * 0.5).exp()
            xt_next = c1 * x0_t + c2 * xt + c3 * torch.randn_like(x0_t)
            
            l1 = ((et * et).mean().sqrt() * (1 - at).sqrt() / at.sqrt() * c1).item()
            l2 = l1 * 0.02
            rho = l2 / (norm_grad * norm_grad).mean().sqrt().item()
            
            xt_next -= rho * norm_grad
            
            x0_t = x0_t.detach()
            xt_next = xt_next.detach()
            
            x0_preds.append(x0_t.to('cpu'))
            xs.append(xt_next.to('cpu'))

            if idx + 1 < repeat:
                bt = at / at_next
                xt = bt.sqrt() * xt_next + (1 - bt).sqrt() * torch.randn_like(xt_next)

    # return x0_preds, xs
    return [xs[-1]], [x0_preds[-1]]


def parse_ddim_diffusion(x, seq, model, b, cls_fn=None, rho_scale=1.0, stop=100, ref_path=None):
    parser = FaceParseTool(ref_path=ref_path).cuda()

    # setup iteration variables
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]

    # iterate over the timesteps
    for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        at = compute_alpha(b, t.long())
        at_next = compute_alpha(b, next_t.long())
        xt = xs[-1].to('cuda')
        
        xt.requires_grad = True
        
        if cls_fn == None:
            et = model(xt, t)
        else:
            print("use class_num")
            class_num = 281
            classes = torch.ones(xt.size(0), dtype=torch.long, device=torch.device("cuda"))*class_num
            et = model(xt, t, classes)
            et = et[:, :3]
            et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)

        if et.size(1) == 6:
            et = et[:, :3]

        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
        
        residual = parser.get_residual(x0_t)
        norm = torch.linalg.norm(residual)
        norm_grad = torch.autograd.grad(outputs=norm, inputs=xt)[0]

        
        eta = 0.5
        c1 = (1 - at_next).sqrt() * eta
        c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
        xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x0_t) + c2 * et

        # use guided gradient
        rho = at.sqrt() * rho_scale
        if not i <= stop:
            xt_next -= rho * norm_grad
        
        x0_t = x0_t.detach()
        xt_next = xt_next.detach()
        
        x0_preds.append(x0_t.to('cpu'))
        xs.append(xt_next.to('cpu'))

    # return x0_preds, xs
    return [xs[-1]], [x0_preds[-1]]


def sketch_ddim_diffusion(x, seq, model, b, cls_fn=None, rho_scale=1.0, stop=100, ref_path=None):
    img2sketch = FaceSketchTool(ref_path=ref_path).cuda()

    # setup iteration variables
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]

    # iterate over the timesteps
    for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        at = compute_alpha(b, t.long())
        at_next = compute_alpha(b, next_t.long())
        xt = xs[-1].to('cuda')
        
        xt.requires_grad = True
        
        if cls_fn == None:
            et = model(xt, t)
        else:
            # print("use class_num")
            class_num = 7
            classes = torch.ones(xt.size(0), dtype=torch.long, device=torch.device("cuda"))*class_num
            et = model(xt, t, classes)
            et = et[:, :3]
            et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)

        if et.size(1) == 6:
            et = et[:, :3]

        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
        
        residual = img2sketch.get_residual(x0_t)
        norm = torch.linalg.norm(residual)
        norm_grad = torch.autograd.grad(outputs=norm, inputs=xt)[0]
        
        eta = 0.5
        c1 = (1 - at_next).sqrt() * eta
        c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
        xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x0_t) + c2 * et
        
        # use guided gradient
        rho = at.sqrt() * rho_scale
        if not i <= stop:
            xt_next -= rho * norm_grad
        
        x0_t = x0_t.detach()
        xt_next = xt_next.detach()
        
        x0_preds.append(x0_t.to('cpu'))
        xs.append(xt_next.to('cpu'))

    return [xs[-1]], [x0_preds[-1]]

"""
def landmark_ddim_diffusion(x, seq, model, b, cls_fn=None, rho_scale=1.0, stop=100, ref_path=None):
    img2landmark = FaceLandMarkTool(ref_path=ref_path).cuda()
    parser = FaceParseTool(ref_path=ref_path).to(x.device) #.cuda() is implicitly handled by FaceParseTool

    # Define face part IDs for segmentation mask
    # Typical IDs: 1:skin, 2:l_brow, 3:r_brow, 4:l_eye, 5:r_eye, 6:eye_g, 7:l_ear, 8:r_ear, 10:nose, 11:mouth, 12:u_lip, 13:l_lip, 17:hair
    face_part_ids = [1, 2, 3, 4, 5, 6,  10, 11, 12, 13]

    # Load original reference image for compositing and mask generation
    pil_ref_image = Image.open(ref_path).convert("RGB")
    
    # Transform for compositing: 256x256, range [-1, 1]
    transform_to_diffusion_range = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(), # HWC uint8 -> CHW float [0,1]
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # [0,1] -> [-1,1]
    ])
    original_ref_img_for_composite = transform_to_diffusion_range(pil_ref_image).unsqueeze(0).to(x.device)

    # Get face mask from the original reference image.
    # FaceParseTool.get_mask expects input tensor in [-1, 1] range, and will resize to 512 for parsing.
    # The output mask is 256x256.
    # Ensure the input tensor for get_mask is on the same device as the parser's model (CUDA).
    face_mask = parser.get_mask(original_ref_img_for_composite.cuda(), id_num=face_part_ids)
    face_mask = face_mask.to(x.device) # Ensure mask is on the same device as generated image for compositing

    # setup iteration variables
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]

    # iterate over the timesteps
    for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
        print("step ",i)
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        at = compute_alpha(b, t.long())
        at_next = compute_alpha(b, next_t.long())
        xt = xs[-1].to('cuda')
        
        xt.requires_grad = True
        
        if cls_fn == None:
            et = model(xt, t)
        else:
            print("use class_num")
            class_num = 281
            classes = torch.ones(xt.size(0), dtype=torch.long, device=torch.device("cuda"))*class_num
            et = model(xt, t, classes)
            et = et[:, :3]
            et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)

        if et.size(1) == 6:
            et = et[:, :3]
        
        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
        
        residual = img2landmark.get_residual(x0_t)
        norm = torch.linalg.norm(residual)
        norm_grad = torch.autograd.grad(outputs=norm, inputs=xt)[0]

        
        eta = 0.5
        c1 = (1 - at_next).sqrt() * eta
        c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
        xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x0_t) + c2 * et
        
        # use guided gradient
        rho = at.sqrt() * rho_scale
        if not i <= stop:
            xt_next -= rho * norm_grad
        
        x0_t = x0_t.detach()
        xt_next = xt_next.detach()
        
        x0_preds.append(x0_t.to('cpu'))
        xs.append(xt_next.to('cpu'))
    
        # Composite the generated face onto the original reference image
        # Use x0_preds[-1] as it's the explicit prediction of the clean image
        generated_img = x0_preds[-1].to(x.device) # This should be the clean generated image, range [-1, 1]
        
        # Ensure all tensors for compositing are on the same device
        original_ref_img_for_composite = original_ref_img_for_composite.to(x.device)
        face_mask = face_mask.to(x.device)
        generated_img = generated_img.to(x.device)
    
        final_output_img = face_mask * generated_img + (1 - face_mask) * original_ref_img_for_composite
        
    return [final_output_img.to('cpu')], [x0_preds[-1].to('cpu')]
    
 """
def landmark_ddim_diffusion(x, seq, model, b, cls_fn=None, rho_scale=1.0, stop=100, ref_path=None):
    img2landmark = FaceLandMarkTool(ref_path=ref_path).cuda()
    parser = FaceParseTool(ref_path=ref_path).to(x.device) #.cuda() is implicitly handled by FaceParseTool

    # Define face part IDs for segmentation mask
    # Typical IDs: 1:skin, 2:l_brow, 3:r_brow, 4:l_eye, 5:r_eye, 6:eye_g, 7:l_ear, 8:r_ear, 10:nose, 11:mouth, 12:u_lip, 13:l_lip, 17:neck
    face_part_ids = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13,14]

    # Load original reference image for compositing and mask generation
    pil_ref_image = Image.open(ref_path).convert("RGB")
    
    # Transform for compositing: 256x256, range [-1, 1]
    transform_to_diffusion_range = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(), # HWC uint8 -> CHW float [0,1]
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # [0,1] -> [-1,1]
    ])
    original_ref_img_for_composite = transform_to_diffusion_range(pil_ref_image).unsqueeze(0).to(x.device)

    # Get face mask from the original reference image.
    # FaceParseTool.get_mask expects input tensor in [-1, 1] range, and will resize to 512 for parsing.
    # The output mask is 256x256.
    # Ensure the input tensor for get_mask is on the same device as the parser's model (CUDA).
    face_mask = parser.get_mask(original_ref_img_for_composite.cuda(), id_num=face_part_ids)
    face_mask = face_mask.to(x.device) # Ensure mask is on the same device as generated image for compositing

    # setup iteration variables
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]

    # iterate over the timesteps
    for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        at = compute_alpha(b, t.long())
        at_next = compute_alpha(b, next_t.long())
        xt = xs[-1].to('cuda')
        
        xt.requires_grad = True
        
        if cls_fn == None:
            et = model(xt, t)
        else:
            print("use class_num")
            class_num = 281
            classes = torch.ones(xt.size(0), dtype=torch.long, device=torch.device("cuda"))*class_num
            et = model(xt, t, classes)
            et = et[:, :3]
            et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)

        if et.size(1) == 6:
            et = et[:, :3]
        
        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
        
        residual = img2landmark.get_residual(x0_t)
        norm = torch.linalg.norm(residual)
        norm_grad = torch.autograd.grad(outputs=norm, inputs=xt)[0]

        
        eta = 0.5
        c1 = (1 - at_next).sqrt() * eta
        c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
        xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x0_t) + c2 * et
        
        # use guided gradient
        rho = at.sqrt() * rho_scale
        if not i <= stop:
            xt_next -= rho * norm_grad
        
        x0_t = x0_t.detach()
        xt_next = xt_next.detach()
        
        x0_preds.append(x0_t.to('cpu'))
        xs.append(xt_next.to('cpu'))
    
        # Composite the generated face onto the original reference image
        generated_img = xs[-1].to(x.device) # This is the raw output from diffusion, range [-1, 1]
        
        # Ensure all tensors for compositing are on the same device
        original_ref_img_for_composite = original_ref_img_for_composite.to(x.device)
        face_mask = face_mask.to(x.device)
        generated_img = generated_img.to(x.device)
    
        final_output_img = face_mask * generated_img + (1 - face_mask) * original_ref_img_for_composite
        
    # --- Code pour afficher/sauvegarder le masque binaire ---
    # Convertir le tenseur du masque en format image PIL
    # face_mask est de forme [1, 1, H, W] float [0, 1]
    mask_display = face_mask.squeeze(0).squeeze(0).cpu() # Supprimer le batch et le canal, déplacer vers le CPU
    mask_display = mask_display * 255 # Mettre à l'échelle entre [0, 255]
    mask_display_np = mask_display.byte().numpy() # Convertir en byte (uint8) puis en numpy
    mask_display_img = Image.fromarray(mask_display_np, 'L') # Créer une image PIL à partir du tableau numpy (Niveaux de gris)

    # Sauvegarder l'image du masque dans un fichier
    mask_save_path = "./face_mask_debug.png" # Définir un chemin de sauvegarde
    mask_display_img.save(mask_save_path)
    print(f"Masque facial sauvegardé pour le débogage à {mask_save_path}") # Afficher une confirmation
    # --- Fin du code pour afficher/sauvegarder le masque binaire ---

    return [final_output_img.to('cpu')], [x0_preds[-1].to('cpu')]    

import torch
import torch.nn.functional as F
from torchvision import transforms as T
from PIL import Image
from skimage.exposure import match_histograms
from tqdm import tqdm
import cv2
import numpy as np
import torchvision.transforms.functional as TF



from .perceptual.model import PerceptualFaceTool


#vgg +id+mask+color woooooooooooooooooork
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from PIL import Image
from skimage.exposure import match_histograms
from tqdm import tqdm

import torchvision.transforms.functional as TF
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from PIL import Image
from skimage.exposure import match_histograms
from tqdm import tqdm
import cv2
import numpy as np





def reinhard_color_transfer(source, target, mask=None):
    # source, target: [C, H, W] torch tensors, valeurs [-1, 1]
    # mask: [H, W] ou [1, H, W], valeurs 0/1 (optionnel)
    def to_uint8(img):
        img = (img * 0.5 + 0.5).clamp(0,1) * 255
        return img.permute(1,2,0).cpu().numpy().astype(np.uint8)
    src_np = to_uint8(source)
    tgt_np = to_uint8(target)
    src_lab = cv2.cvtColor(src_np, cv2.COLOR_RGB2LAB)
    tgt_lab = cv2.cvtColor(tgt_np, cv2.COLOR_RGB2LAB)
    if mask is not None:
        mask_np = mask.squeeze().cpu().numpy().astype(bool)
        for i in range(3):
            src_mean, src_std = src_lab[...,i][mask_np].mean(), src_lab[...,i][mask_np].std()
            tgt_mean, tgt_std = tgt_lab[...,i][mask_np].mean(), tgt_lab[...,i][mask_np].std()
            src_lab[...,i][mask_np] = ((src_lab[...,i][mask_np] - src_mean) / (src_std+1e-8)) * tgt_std + tgt_mean
    else:
        for i in range(3):
            src_mean, src_std = src_lab[...,i].mean(), src_lab[...,i].std()
            tgt_mean, tgt_std = tgt_lab[...,i].mean(), tgt_lab[...,i].std()
            src_lab[...,i] = ((src_lab[...,i] - src_mean) / (src_std+1e-8)) * tgt_std + tgt_mean
    result_rgb = cv2.cvtColor(src_lab, cv2.COLOR_LAB2RGB)
    result_tensor = torch.from_numpy(result_rgb).permute(2,0,1).float() / 255
    result_tensor = (result_tensor - 0.5) / 0.5  # [0,1] -> [-1,1]
    return result_tensor



def normalize_grad(grad, eps=1e-8):
    norm = grad.norm()
    return grad / (norm + eps)

def project_out_id_component_soft(g, id_g, attenuation=0.5):
    """
    Soft orthogonalization: attenuate the component of `g` aligned with `id_g`.
    Returns g - attenuation * proj_{id_g}(g).
    attenuation: 0.0 -> no removal, 1.0 -> full removal (hard orthogonalization).
    """
    if not isinstance(g, torch.Tensor) or not isinstance(id_g, torch.Tensor):
        return g
    id_flat = id_g.view(-1)
    g_flat = g.view(-1)
    denom = torch.dot(id_flat, id_flat) + 1e-8
    alpha = torch.dot(g_flat, id_flat) / denom
    return g - attenuation * alpha * id_g


def _gram_schmidt_orthonormalize(vecs):
    """Simple Gram-Schmidt orthonormalization for a list of 1D flattened vectors.
    Expects `vecs` as list of 1D tensors of same shape. Returns list of orthonormal 1D tensors.
    """
    orthonorm = []
    for v in vecs:
        w = v.clone()
        for u in orthonorm:
            proj = (torch.dot(w, u) / (torch.dot(u, u) + 1e-12)) * u
            w = w - proj
        nrm = w.norm()
        if nrm > 1e-12:
            orthonorm.append(w / nrm)
    return orthonorm


def project_out_id_subspace(g, id_list, attenuation=0.5):
    """Project `g` out of the subspace spanned by `id_list` using Gram-Schmidt.
    - `g`: tensor shape [C,H,W]
    - `id_list`: list of tensors shape [C,H,W]
    Returns g - attenuation * P_sub g, where P_sub = U U^T.
    """
    if not isinstance(g, torch.Tensor) or len(id_list) == 0:
        return g

    device = g.device
    g_flat = g.view(-1)

    # Build flattened vectors and orthonormalize
    flat_list = [v.view(-1).to(device).clone() for v in id_list]
    if len(flat_list) == 0:
        return g

    U = _gram_schmidt_orthonormalize(flat_list)
    if len(U) == 0:
        return g

    # Project g onto span(U)
    proj = torch.zeros_like(g_flat)
    for u in U:
        coeff = torch.dot(g_flat, u)
        proj = proj + coeff * u

    proj = proj.view_as(g)
    return g - attenuation * proj


def grad_conflict_cosine(g, id_g, eps=1e-8):
    """|kappa|: cosine alignment between attribute and identity gradients in [0, 1]."""
    if not isinstance(g, torch.Tensor) or not isinstance(id_g, torch.Tensor):
        return 0.0
    g_flat = g.view(-1).float()
    id_flat = id_g.view(-1).float()
    denom = g_flat.norm() * id_flat.norm() + eps
    return (torch.dot(g_flat, id_flat).abs() / denom).clamp(0.0, 1.0).item()


def conflict_aware_attenuation(kappa, alpha_min, alpha_max):
    """Map gradient conflict kappa to soft-projection strength alpha_t."""
    kappa = float(max(0.0, min(1.0, kappa)))
    return alpha_min + (alpha_max - alpha_min) * kappa


def bidirectional_subspace_decouple(
    g,
    src_buffer,
    tgt_buffer,
    id_g,
    tgt_id_g,
    alpha_src,
    alpha_tgt,
    use_bidirectional=True,
):
    """Project attribute gradient g out of source and (optionally) target identity subspaces."""
    if not isinstance(g, torch.Tensor):
        return g
    g = project_out_id_subspace(g, src_buffer, attenuation=alpha_src)
    if use_bidirectional and len(tgt_buffer) > 0:
        g = project_out_id_subspace(g, tgt_buffer, attenuation=alpha_tgt)
    elif use_bidirectional and isinstance(tgt_id_g, torch.Tensor):
        g = project_out_id_component_soft(g, tgt_id_g, attenuation=alpha_tgt)
    return g


def compute_pair_compatibility(idloss_src, ref_tensor, expr_tool=None, src_tensor=None):
    """
    Estimate source-target compatibility for weight scheduling.
    Returns cos_sim in [-1, 1] and pose_gap in [0, 1] (higher = harder pair).
    """
    with torch.no_grad():
        src_norm = torch.clamp((idloss_src.ref + 1.0) / 2.0, 0.0, 1.0)
        tgt_norm = torch.clamp((ref_tensor + 1.0) / 2.0, 0.0, 1.0)
        src_emb = idloss_src.extract_feats(src_norm)
        tgt_emb = idloss_src.extract_feats(tgt_norm)
        cos_sim = F.cosine_similarity(src_emb, tgt_emb, dim=1).mean().item()

        pose_gap = max(0.0, 1.0 - cos_sim)
        if expr_tool is not None and src_tensor is not None:
            src_feats = expr_tool._extract_midlevel_features(src_tensor)
            tgt_feats = expr_tool.ref_feats
            feat_dist = 0.0
            for fa, fb in zip(src_feats, tgt_feats):
                fa_n = fa / (fa.flatten(2).norm(dim=2, keepdim=True) + 1e-6).unsqueeze(2)
                fb_n = fb / (fb.flatten(2).norm(dim=2, keepdim=True) + 1e-6).unsqueeze(2)
                feat_dist += (fa_n - fb_n).pow(2).mean().item()
            feat_dist /= max(1, len(src_feats))
            pose_gap = max(pose_gap, min(1.0, feat_dist * 0.25))

    return cos_sim, pose_gap


def apply_compatibility_weight_schedule(
    id_weight,
    expression_weight,
    color_weight,
    id_attenuation,
    id_attenuation_min,
    id_attenuation_max,
    cos_sim,
    pose_gap,
):
    """
    Adapt guidance weights and ISGD strength to pair difficulty.
    Hard pairs (low cos_sim / large pose gap): stronger identity + stronger decoupling.
    """
    difficulty = max(0.0, min(1.0, 0.55 * (1.0 - cos_sim) + 0.45 * pose_gap))

    sched_id = id_weight * (1.0 + 0.40 * difficulty)
    sched_expr = expression_weight * (1.0 - 0.35 * difficulty)
    sched_color = color_weight * (1.0 + 0.20 * difficulty)

    alpha_boost = 1.0 + 0.50 * difficulty
    sched_alpha = min(1.0, id_attenuation * alpha_boost)
    sched_alpha_min = min(id_attenuation_max, id_attenuation_min * alpha_boost)
    sched_alpha_max = min(1.0, id_attenuation_max * alpha_boost)

    return {
        "id_weight": sched_id,
        "expression_weight": max(1.0, sched_expr),
        "color_weight": sched_color,
        "id_attenuation": sched_alpha,
        "id_attenuation_min": sched_alpha_min,
        "id_attenuation_max": sched_alpha_max,
        "difficulty": difficulty,
    }


def sparse_guidance_active_streams(step_t, seq_max, use_sparse=True,
                                   expr_progress_min=0.20, color_progress_min=0.10,
                                   tgt_id_progress_min=0.15, tgt_id_progress_max=0.90):
    """
    Stage-Adaptive Sparse Guidance (SASG).
    step_t is the current diffusion index (large = early/noisy, small = late/clean).
    Returns which expensive guidance branches to evaluate at this step.
    """
    if not use_sparse:
        return {"id": True, "expr": True, "color": True, "tgt_id": True}

    progress = float(step_t) / float(max(seq_max, 1))
    return {
        "id": True,
        "expr": progress >= expr_progress_min,
        "color": progress >= color_progress_min,
        "tgt_id": tgt_id_progress_min <= progress <= tgt_id_progress_max,
    }


def match_face_color(generated_face, ref_face):
    # generated_face, ref_face: [C, H, W] tensors, valeurs [-1, 1]
    import torch
    gen_np = (generated_face.permute(1,2,0).cpu().numpy() * 0.5 + 0.5).clip(0,1)
    ref_np = (ref_face.permute(1,2,0).cpu().numpy() * 0.5 + 0.5).clip(0,1)
    matched = match_histograms(gen_np, ref_np, channel_axis=-1)
    matched_tensor = torch.from_numpy(matched).permute(2,0,1)
    matched_tensor = (matched_tensor - 0.5) / 0.5
    return matched_tensor



def erode_mask(mask, ksize=5):
    # mask: [1, 1, H, W] torch tensor, valeurs [0,1]
    mask_np = (mask[0,0].cpu().numpy() * 255).astype(np.uint8)
    kernel = np.ones((ksize, ksize), np.uint8)
    eroded = cv2.erode(mask_np, kernel, iterations=1)
    eroded = torch.from_numpy(eroded / 255.0).float().to(mask.device).unsqueeze(0).unsqueeze(0)
    return eroded

def q_sample(x0, t, betas):
    """
    Forward diffusion: q(x_t | x_0)
    x0: [B, C, H, W] image in [-1, 1]
    t: [B] or int timestep
    betas: [T] schedule
    """
    alphas = 1. - betas
    alphas_cumprod = alphas.cumprod(0)
    if isinstance(t, int):
        t = torch.full((x0.size(0),), t, dtype=torch.long, device=x0.device)
    sqrt_alphas_cumprod = alphas_cumprod[t].sqrt().view(-1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod = (1 - alphas_cumprod[t]).sqrt().view(-1, 1, 1, 1)
    noise = torch.randn_like(x0)
    return sqrt_alphas_cumprod * x0 + sqrt_one_minus_alphas_cumprod * noise

def arcface_ddim_diffusion(
    x, seq, model, b,
    cls_fn=None, rho_scale=1.0,
    stop=100, ref_path=None, src_path="/kaggle/input/datasets/matheuseduardo/flickr-faces-dataset-resized/256x256/faces/04963.png", id_attenuation=0.9,
    id_buffer_k=5,
    id_weight=300, perceptual_weight=0, landmark_weight=0, parse_weight=0, expression_weight=150, color_weight=150,
    perceptual_feature_weights=None, perceptual_loss_type='l1',
    apply_hist_match=False,
    use_fp16=True,
    step_skip=1,
    preserve_accessories=True,
    use_inpainting=False,
    inpainting_timesteps=30,
    # --- CA-ISGD + bidirectional decoupling + compatibility scheduling ---
    use_ca_isgd=True,
    use_bidirectional_isgd=True,
    use_compatibility_schedule=True,
    id_attenuation_min=0.25,
    id_attenuation_max=0.90,
    tgt_id_attenuation_min=0.30,
    tgt_id_attenuation_max=0.95,
    # --- Stage-Adaptive Sparse Guidance (SASG): faster inference, same quality ---
    use_sparse_guidance=False,
    fast_mode=True,
    guidance_every_n=1,
    keep_latents_on_gpu=False,
    expr_progress_min=0.20,
    color_progress_min=0.10,
    tgt_id_progress_min=0.15,
    tgt_id_progress_max=0.90,
):
    """
    DDIM face swap with three-stream guidance (identity, expression, color).
    - ISGD / CA-ISGD / B-ISGD: bidirectional conflict-aware gradient decoupling
    - Compatibility-aware scheduling: adapt weights to source-target pair difficulty
    - SASG (optional): activate expression/color/tgt-id only on relevant timestep ranges;
      reuse cached fused gradient every guidance_every_n steps
    - fast_mode=True enables SASG + guidance_every_n=2 (does NOT store all latents on GPU)
    """
    if fast_mode:
        use_sparse_guidance = True
        guidance_every_n = max(2, guidance_every_n)
    # Utiliser directement src_path sans alignement
    src_path_for_guidance = src_path if src_path is not None else ref_path

    # --- Initialisation des outils de guidance ---
    # 1. IDLoss utilise la source alignée (src_path_for_guidance) pour l'identité
    idloss = IDLoss(ref_path=src_path_for_guidance).cuda().eval()
    tgt_idloss = None
    if use_bidirectional_isgd:
        tgt_idloss = IDLoss(ref_path=ref_path).cuda().eval()
        # Share one ArcFace backbone (saves ~500MB VRAM vs two full copies)
        old_tgt_facenet = tgt_idloss.facenet
        tgt_idloss.facenet = idloss.facenet
        del old_tgt_facenet
        torch.cuda.empty_cache()
    perceptual_tool = None
    if perceptual_weight > 0:
        perceptual_tool = PerceptualFaceTool(
            ref_path=ref_path,
            feature_weights=perceptual_feature_weights,
            loss_type=perceptual_loss_type
        ).cuda().eval()
    # 2. Expression guidance (ArcFace mid-level) si demandé
    expr_tool = None
    if expression_weight > 0:
        expr_tool = ExpressionArcFaceTool(ref_path=ref_path).cuda().eval()
    # 3. Landmarks et parsing doivent utiliser la référence (ref_path)
    parser = FaceParseTool(ref_path=ref_path).cuda()  # guidance parsing = destination
    landmark_tool = None
    if landmark_weight > 0.0:
        landmark_tool = FaceLandMarkTool(ref_path=ref_path).cuda()  # guidance landmarks = destination

    # Chargement de l'image de référence (toujours ref_path pour le masquage/fond)
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    ref_img = Image.open(ref_path).convert('RGB')
    ref_tensor = transform(ref_img).unsqueeze(0).cuda()
    src_img = Image.open(src_path_for_guidance).convert('RGB')
    src_tensor = transform(src_img).unsqueeze(0).cuda()

    # --- Compatibility-aware weight scheduling (once per pair) ---
    base_id_weight = id_weight
    base_expression_weight = expression_weight
    base_color_weight = color_weight
    base_id_attenuation = id_attenuation
    sched_alpha_min = id_attenuation_min
    sched_alpha_max = id_attenuation_max
    pair_difficulty = 0.0

    if use_compatibility_schedule:
        cos_sim, pose_gap = compute_pair_compatibility(
            idloss, ref_tensor, expr_tool=expr_tool, src_tensor=src_tensor
        )
        sched = apply_compatibility_weight_schedule(
            id_weight,
            expression_weight,
            color_weight,
            id_attenuation,
            id_attenuation_min,
            id_attenuation_max,
            cos_sim,
            pose_gap,
        )
        id_weight = sched["id_weight"]
        expression_weight = sched["expression_weight"]
        color_weight = sched["color_weight"]
        id_attenuation = sched["id_attenuation"]
        sched_alpha_min = sched["id_attenuation_min"]
        sched_alpha_max = sched["id_attenuation_max"]
        pair_difficulty = sched["difficulty"]
        print(
            f"[Compatibility] cos_sim={cos_sim:.3f} pose_gap={pose_gap:.3f} "
            f"difficulty={pair_difficulty:.3f} -> "
            f"id_w={id_weight:.1f} expr_w={expression_weight:.1f} color_w={color_weight:.1f} "
            f"alpha={id_attenuation:.3f} alpha_range=[{sched_alpha_min:.3f},{sched_alpha_max:.3f}]"
        )

    # Masques visage et fond
    face_mask = parser.create_face_mask(ref_tensor)
    mask_display = face_mask.squeeze(0).squeeze(0).cpu()
    mask_display = mask_display * 255
    mask_display_np = mask_display.byte().numpy()
    mask_display_img = Image.fromarray(mask_display_np, 'L')
    mask_save_path = "./face_mask_debug.png"
    mask_display_img.save(mask_save_path)
    print(f"Masque facial sauvegardé pour le débogage à {mask_save_path}")

    # Sauvegarde du masque facial pour le débogage
    mask_display = face_mask.squeeze(0).squeeze(0).cpu()
    mask_display = mask_display * 255
    mask_display_np = mask_display.byte().numpy()
    mask_display_img = Image.fromarray(mask_display_np, 'L')
    mask_save_path = "./enhanced_face_mask_debug.png"
    mask_display_img.save(mask_save_path)
    print(f"Masque facial amélioré sauvegardé à {mask_save_path}")

    # Sauvegarde du masque peau+lips+lunettes pour le débogage (si color_weight > 0)
    print(f"🔍 DEBUG: color_weight = {color_weight}")
    if color_weight > 0:
        print("🔍 DEBUG: color_weight > 0, création du masque peau...")
        try:
            skin_mask_debug = parser.create_skin_lips_glasses_mask(ref_tensor)
            print(f"🔍 DEBUG: Masque peau+lips+lunettes créé, forme: {skin_mask_debug.shape}")
            skin_mask_display = skin_mask_debug.squeeze(0).squeeze(0).cpu()
            skin_mask_display = skin_mask_display * 255
            skin_mask_display_np = skin_mask_display.byte().numpy()
            skin_mask_display_img = Image.fromarray(skin_mask_display_np, 'L')
            skin_mask_save_path = "./skin_lips_glasses_mask_debug.png"
            skin_mask_display_img.save(skin_mask_save_path)
            print(f"✅ Masque peau+lips+lunettes sauvegardé à {skin_mask_save_path}")
        except Exception as e:
            print(f"❌ ERREUR lors de la création du masque peau: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("🔍 DEBUG: color_weight = 0, pas de masque peau créé")

    background_mask = 1 - face_mask
    ref_background = ref_tensor * background_mask

    # --- VGG color guidance setup (low layers, masked + blurred) ---
    vgg_color = None
    ref_color_feat_cached = None
    to_skin_rgb_only = None
    if color_weight > 0:
        import torchvision.models as models
        vgg16 = models.vgg16(pretrained=True).features.eval().cuda()
        slice1 = torch.nn.Sequential(*[vgg16[i] for i in range(2)]).eval().cuda()
        vgg_color = (slice1,)

        def to_skin_rgb_only(img):
            img01 = (img + 1.0) * 0.5
            m = face_mask
            if m.shape[1] == 1:
                m = m.repeat(1, 3, 1, 1)
            skin_mask = parser.create_skin_lips_glasses_mask(img)
            combined_mask = skin_mask * m
            skin_rgb = img01 * combined_mask
            skin_rgb = TF.gaussian_blur(skin_rgb, kernel_size=[15, 15], sigma=[4.0, 4.0])
            normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])
            return normalize(skin_rgb)

        with torch.no_grad():
            ref_color_feat_cached = slice1(to_skin_rgb_only(ref_tensor))

    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    # Rolling latent: keep only the current step (NOT a list of 100 GPU tensors)
    xt_current = x.detach()
    if not keep_latents_on_gpu:
        xt_current = xt_current.cpu()
    last_x0 = None

    # Per-sample buffers for source and target identity subspaces (ISGD / B-ISGD)
    id_grad_buffers = [[] for _ in range(n)]
    tgt_id_grad_buffers = [[] for _ in range(n)] if use_bidirectional_isgd else None

    seq_max = max(seq) if len(seq) > 0 else 1
    cached_total_grad = None
    step_counter = 0
    if fast_mode or use_sparse_guidance:
        print(
            f"[SASG] sparse={use_sparse_guidance} guidance_every_n={guidance_every_n} "
            f"gpu_latents={keep_latents_on_gpu}"
        )

    for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
        step_counter += 1
        stream_active = sparse_guidance_active_streams(
            i, seq_max, use_sparse=use_sparse_guidance,
            expr_progress_min=expr_progress_min,
            color_progress_min=color_progress_min,
            tgt_id_progress_min=tgt_id_progress_min,
            tgt_id_progress_max=tgt_id_progress_max,
        )
        recompute_guidance = (
            guidance_every_n <= 1
            or cached_total_grad is None
            or (step_counter % guidance_every_n == 1)
        )

        t = torch.full((n,), i, dtype=torch.long, device=x.device)
        next_t = torch.full((n,), j, dtype=torch.long, device=x.device)
        at = compute_alpha(b, t)
        at_n = compute_alpha(b, next_t)

        xt = xt_current.cuda() if not keep_latents_on_gpu else xt_current
        xt = xt.detach().requires_grad_(True)

        # --- Chemin guidé (face swap) ---
        with autocast(enabled=use_fp16):
            if cls_fn is None:
                et = model(xt, t)
            else:
                class_num = 281
                classes = torch.full((n,), class_num, dtype=torch.long, device=x.device)
                et = model(xt, t, classes)[:, :3]
                et -= (1 - at).sqrt()[0] * cls_fn(x, t, classes)

            if et.size(1) == 6:
                et = et[:, :3]

            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

        # --- Guidance par gradient (skipped on cached steps for SASG) ---
        if recompute_guidance:
            with autocast(enabled=use_fp16):
                id_res = idloss.get_residual(x0_t)
                id_norm = torch.linalg.norm(id_res)

                if perceptual_weight > 0:
                    perc_res = perceptual_tool.get_residual(x0_t)
                else:
                    perc_res = None

                if (stream_active["color"] and color_weight > 0
                        and vgg_color is not None and ref_color_feat_cached is not None
                        and to_skin_rgb_only is not None):
                    (slice1,) = vgg_color
                    x_vgg_in = to_skin_rgb_only(x0_t)
                    f1 = slice1(x_vgg_in)
                    color_loss = F.mse_loss(f1, ref_color_feat_cached)
                else:
                    color_loss = None

            if stream_active["expr"] and expr_tool is not None:
                with autocast(enabled=use_fp16):
                    expr_residual = expr_tool.get_residual(x0_t)
                expr_loss = expr_residual
            else:
                expr_loss = None

            need_tgt_id = (stream_active["tgt_id"] and use_bidirectional_isgd
                    and tgt_idloss is not None)
            if need_tgt_id:
                tgt_id_res = tgt_idloss.get_residual(x0_t)
                tgt_id_norm = torch.linalg.norm(tgt_id_res)
            else:
                tgt_id_norm = None

            if landmark_tool is not None:
                landmark_residual = landmark_tool.get_residual(x0_t)
                landmark_norm = torch.linalg.norm(landmark_residual)
            else:
                landmark_norm = None

            if parse_weight > 0:
                parse_residual = parser.get_residual(x0_t)
                parse_norm = torch.linalg.norm(parse_residual)
            else:
                parse_norm = None

            grad_names = ["id"]
            if perceptual_weight > 0:
                grad_names.append("perceptual")
            if color_loss is not None:
                grad_names.append("color")
            if expr_loss is not None:
                grad_names.append("expr")
            if tgt_id_norm is not None:
                grad_names.append("tgt_id")
            if landmark_norm is not None:
                grad_names.append("landmark")
            if parse_norm is not None:
                grad_names.append("parse")

            last_grad = grad_names[-1]

            def compute_grad(loss, retain_graph):
                return torch.autograd.grad(loss, xt, retain_graph=retain_graph)[0]

            id_grad = compute_grad(id_norm, retain_graph=(last_grad != "id"))
            if perceptual_weight > 0:
                perceptual_grad = compute_grad(perc_res, retain_graph=(last_grad != "perceptual"))
            else:
                perceptual_grad = 0.0
            if color_loss is not None:
                color_grad = compute_grad(color_loss, retain_graph=(last_grad != "color"))
            else:
                color_grad = 0.0
            if expr_loss is not None:
                expr_grad = compute_grad(expr_loss, retain_graph=(last_grad != "expr"))
            else:
                expr_grad = 0.0
            if tgt_id_norm is not None:
                tgt_id_grad = compute_grad(tgt_id_norm, retain_graph=(last_grad != "tgt_id"))
            else:
                tgt_id_grad = 0.0
            if landmark_norm is not None:
                landmark_grad = compute_grad(landmark_norm, retain_graph=(last_grad != "landmark"))
            else:
                landmark_grad = 0.0
            if parse_norm is not None:
                parse_grad = compute_grad(parse_norm, retain_graph=False)
            else:
                parse_grad = 0.0

            if isinstance(id_grad, torch.Tensor):
                id_grad = id_grad.detach()
            if isinstance(perceptual_grad, torch.Tensor):
                perceptual_grad = perceptual_grad.detach()
            if isinstance(color_grad, torch.Tensor):
                color_grad = color_grad.detach()
            if isinstance(expr_grad, torch.Tensor):
                expr_grad = expr_grad.detach()
            if isinstance(tgt_id_grad, torch.Tensor):
                tgt_id_grad = tgt_id_grad.detach()
            if isinstance(landmark_grad, torch.Tensor):
                landmark_grad = landmark_grad.detach()
            if isinstance(parse_grad, torch.Tensor):
                parse_grad = parse_grad.detach()

            id_grad_norm = normalize_grad(id_grad)
            perceptual_grad_norm = normalize_grad(perceptual_grad) if perceptual_weight > 0 else 0.0
            color_grad_norm = normalize_grad(color_grad) if color_weight > 0 and stream_active["color"] else 0.0
            expr_grad_norm = normalize_grad(expr_grad) if expression_weight > 0 and stream_active["expr"] else 0.0
            if isinstance(tgt_id_grad, torch.Tensor):
                tgt_id_grad_norm = normalize_grad(tgt_id_grad)
            else:
                tgt_id_grad_norm = None
            landmark_grad_norm = normalize_grad(landmark_grad) if landmark_weight > 0 else 0.0
            parse_grad_norm = (
                normalize_grad(parse_grad)
                if parse_weight > 0 and isinstance(parse_grad, torch.Tensor) else 0.0
            )

            # --- CA-ISGD + Bidirectional subspace decoupling ---
            isgd_enabled = (
                id_attenuation is not None
                and id_buffer_k is not None
                and id_buffer_k > 0
            )
            if isgd_enabled:
                for bidx in range(n):
                    try:
                        cur_src = id_grad_norm[bidx].detach().clone()
                    except Exception:
                        cur_src = id_grad_norm.detach().clone()
                    id_grad_buffers[bidx].append(cur_src)
                    if len(id_grad_buffers[bidx]) > id_buffer_k:
                        id_grad_buffers[bidx].pop(0)

                    if use_bidirectional_isgd and tgt_id_grad_norm is not None:
                        try:
                            cur_tgt = tgt_id_grad_norm[bidx].detach().clone()
                        except Exception:
                            cur_tgt = tgt_id_grad_norm.detach().clone()
                        tgt_id_grad_buffers[bidx].append(cur_tgt)
                        if len(tgt_id_grad_buffers[bidx]) > id_buffer_k:
                            tgt_id_grad_buffers[bidx].pop(0)

                def _decouple_batch_tensor(batch_tensor, stream_attenuation_scale=1.0):
                    if not isinstance(batch_tensor, torch.Tensor):
                        return batch_tensor
                    out = torch.zeros_like(batch_tensor)
                    for bidx in range(batch_tensor.shape[0]):
                        g = batch_tensor[bidx]
                        try:
                            id_g = id_grad_norm[bidx]
                        except Exception:
                            id_g = id_grad_norm
                        tgt_g = None
                        if tgt_id_grad_norm is not None:
                            try:
                                tgt_g = tgt_id_grad_norm[bidx]
                            except Exception:
                                tgt_g = tgt_id_grad_norm

                        if use_ca_isgd:
                            kappa_src = grad_conflict_cosine(g, id_g)
                            alpha_src = conflict_aware_attenuation(
                                kappa_src, sched_alpha_min, sched_alpha_max
                            )
                            if use_bidirectional_isgd and tgt_g is not None:
                                kappa_tgt = grad_conflict_cosine(g, tgt_g)
                                alpha_tgt = conflict_aware_attenuation(
                                    kappa_tgt, tgt_id_attenuation_min, tgt_id_attenuation_max
                                )
                            else:
                                alpha_tgt = id_attenuation
                        else:
                            alpha_src = id_attenuation * stream_attenuation_scale
                            alpha_tgt = id_attenuation * stream_attenuation_scale

                        out[bidx] = bidirectional_subspace_decouple(
                            g,
                            id_grad_buffers[bidx],
                            tgt_id_grad_buffers[bidx] if use_bidirectional_isgd else [],
                            id_g,
                            tgt_g,
                            alpha_src=alpha_src,
                            alpha_tgt=alpha_tgt,
                            use_bidirectional=use_bidirectional_isgd,
                        )
                    return out

                if isinstance(perceptual_grad_norm, torch.Tensor):
                    perceptual_grad_norm = _decouple_batch_tensor(perceptual_grad_norm)
                if isinstance(color_grad_norm, torch.Tensor):
                    color_grad_norm = _decouple_batch_tensor(color_grad_norm, stream_attenuation_scale=1.05)
                if isinstance(expr_grad_norm, torch.Tensor):
                    expr_grad_norm = _decouple_batch_tensor(expr_grad_norm, stream_attenuation_scale=0.95)
                if isinstance(landmark_grad_norm, torch.Tensor):
                    landmark_grad_norm = _decouple_batch_tensor(landmark_grad_norm)
                if isinstance(parse_grad_norm, torch.Tensor) and parse_weight > 0:
                    parse_grad_norm = _decouple_batch_tensor(parse_grad_norm)

                if i == seq[-1]:
                    mode = []
                    if use_ca_isgd:
                        mode.append("CA-ISGD")
                    if use_bidirectional_isgd:
                        mode.append("B-ISGD")
                    print(
                        f"[{'+'.join(mode) if mode else 'ISGD'}] "
                        f"alpha={id_attenuation:.3f} range=[{sched_alpha_min:.3f},{sched_alpha_max:.3f}] "
                        f"k={id_buffer_k} difficulty={pair_difficulty:.3f}"
                    )

            progress = float(i) / float(seq_max) if seq_max > 0 else 0.0
            id_w_step = id_weight * (0.65 + 0.35 * (1.0 - progress))
            expr_w_step = expression_weight * (0.65 + 0.35 * progress) if stream_active["expr"] else 0.0
            color_w_step = color_weight if stream_active["color"] else 0.0

            total_weight = (
                id_w_step + perceptual_weight + color_w_step + expr_w_step
                + landmark_weight + parse_weight + 1e-8
            )
            id_w = id_w_step / total_weight
            perc_w = perceptual_weight / total_weight
            color_w = color_w_step / total_weight
            expr_w = expr_w_step / total_weight
            land_w = landmark_weight / total_weight
            parse_w = parse_weight / total_weight

            total_grad = (
                - id_w * id_grad_norm
                - perc_w * perceptual_grad_norm
                - color_w * color_grad_norm
                - expr_w * expr_grad_norm
                - land_w * landmark_grad_norm
                - parse_w * parse_grad_norm
            )
            cached_total_grad = total_grad.detach().clone()
        else:
            total_grad = cached_total_grad

        # DDIM step (chemin guidé)
        eta = 0.5
        c1 = (1 - at_n).sqrt() * eta
        c2 = (1 - at_n).sqrt() * (1 - eta**2)**0.5
        xt_next_fg = at_n.sqrt() * x0_t + c1 * torch.randn_like(x0_t) + c2 * et

        # --- Application du gradient blendé (chemin guidé) ---
        rho = at.sqrt() * rho_scale
        if i > stop:
            xt_next_fg = xt_next_fg + rho * total_grad

        # --- Chemin background (pas de guidance) ---
        with autocast(enabled=use_fp16):
            xt_bg = q_sample(ref_tensor, i, b)
            et_bg = model(xt_bg, t)
            if et_bg.size(1) == 6:
                et_bg = et_bg[:, :3]
            x0_t_bg = (xt_bg - et_bg * (1 - at).sqrt()) / at.sqrt()
            xt_next_bg = at_n.sqrt() * x0_t_bg + c1 * torch.randn_like(x0_t_bg) + c2 * et_bg

        # --- Blending spatial à chaque étape ---
        mask3 = face_mask
        if mask3.shape[1] == 1 and xt_next_fg.shape[1] == 3:
            mask3 = mask3.repeat(1, 3, 1, 1)
        xt_next_blend = xt_next_fg * mask3 + xt_next_bg * (1 - mask3)

        last_x0 = x0_t.detach()
        xt_current = xt_next_blend.detach()
        if not keep_latents_on_gpu:
            xt_current = xt_current.cpu()
        del xt, x0_t, xt_next_fg, xt_next_bg, xt_next_blend
        torch.cuda.empty_cache()

    final_generated_image = xt_current.cuda()

    if apply_hist_match:
        try:
            skin_m = parser.create_skin_lips_glasses_mask(final_generated_image)
            final_generated_image = reinhard_color_transfer(
                final_generated_image[0], ref_tensor[0], mask=skin_m[0, 0]
            ).unsqueeze(0).to(final_generated_image.device)
        except Exception as e:
            print(f"[hist_match] skipped: {e}")

    # Ajuster la luminosité si color_weight=0 pour éviter les images trop sombres
    if color_weight == 0:
        # Augmenter légèrement la luminosité sans affecter l'identité
        brightness_factor = 1.15  # Augmenter de 15%
        final_generated_image = torch.clamp(final_generated_image * brightness_factor, -1, 1)
        print("Luminosité ajustée (+15%) car color_weight=0")

    # Sauvegarde de l'image générée (résultat final)
    tvu.save_image((final_generated_image * 0.5 + 0.5).clamp(0,1), 'debug_generated_face.png')

    return [final_generated_image.cpu()], [last_x0.cpu() if last_x0 is not None else final_generated_image.cpu()]
