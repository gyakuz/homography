from argparse import Namespace
import torch
import numpy as np
import cv2

from model.loftr_src.loftr.utils.cvpr_ds_config import default_cfg
from model.full_model import GeoFormer as GeoFormer_
from model.geo_config import lower_config
from eval_tool.immatch.utils.data_io import load_gray_scale_tensor_cv
from eval_tool.immatch.utils.misc import load_im_padding
from model.geo_config import default_cfg as geoformer_cfg

class GeoFormer():
    def __init__(self, imsize, match_threshold, no_match_upscale=False, ckpt=None, device='cuda'):

        self.device = device
        self.imsize = imsize
        self.match_threshold = match_threshold
        self.no_match_upscale = no_match_upscale

        # Load model
        conf = dict(default_cfg)
        conf['match_coarse']['thr'] = self.match_threshold
        geoformer_cfg['coarse_thr'] = self.match_threshold
        self.model = GeoFormer_(conf)
        ckpt_dict = torch.load(ckpt, map_location=torch.device('cpu'))
        if 'state_dict' in ckpt_dict:
            ckpt_dict = ckpt_dict['state_dict']
        self.model.load_state_dict(ckpt_dict, strict=False)
        self.model = self.model.eval().to(self.device)

        # Name the method
        self.ckpt_name = ckpt.split('/')[-1].split('.')[0]
        self.name = f'GeoFormer_{self.ckpt_name}'
        if self.no_match_upscale:
            self.name += '_noms'
        print(f'Initialize {self.name}')

    def change_deivce(self, device):
        self.device = device
        self.model.to(device)

    def load_im(self, im1_path, enhanced=False):
        return load_gray_scale_tensor_cv(
            im1_path, self.device, imsize=self.imsize, dfactor=8, enhanced=enhanced, value_to_scale=min
        )


    def match_inputs_(self, gray1, gray2, is_draw=False):

        batch = {'image0': gray1, 'image1': gray2}
        with torch.no_grad():
            batch = self.model(batch)
        kpts1 = batch['mkpts0_f'].cpu().numpy()
        kpts2 = batch['mkpts1_f'].cpu().numpy()
        def draw(save_path):
            import matplotlib.pyplot as plt
            import cv2
            import numpy as np
            plt.figure(dpi=200)
            kp0 = kpts1
            kp1 = kpts2
            # if len(kp0) > 0:
            kp0 = [cv2.KeyPoint(int(k[0]), int(k[1]), 30) for k in kp0]
            kp1 = [cv2.KeyPoint(int(k[0]), int(k[1]), 30) for k in kp1]
            matches = [cv2.DMatch(_trainIdx=i, _queryIdx=i, _distance=1, _imgIdx=-1) for i in
                       range(len(kp0))]

            show = cv2.drawMatches((gray1.cpu()[0][0].numpy() * 255).astype(np.uint8), kp0,
                                   (gray2.cpu()[0][0].numpy() * 255).astype(np.uint8), kp1, matches,
                                   None)
            plt.imshow(show)
            plt.show()
            plt.savefig(save_path)
        if is_draw:
            draw('output_image_nms5.png')
        scores = batch['mconf'].cpu().numpy()
        matches = np.concatenate([kpts1, kpts2], axis=1)
        return matches, kpts1, kpts2, scores

    def match_pairs(self, im1_path, im2_path, cpu=False, is_draw=False):
        torch.cuda.empty_cache()
        tmp_device = self.device
        if cpu:
            self.change_deivce('cpu')

        gray1, sc1 = self.load_im(im1_path)
        gray2, sc2 = self.load_im(im2_path)
        # ori_rgb1, ori_rgb2, rgb1, rgb2, mask1, mask2, sc1, sc2 = load_im_padding(im1_path, im2_path)

        # print(gray1.shape)
        # print(gray2.shape)
        # print(rgb1.shape)
        # print(rgb2.shape)

        upscale = np.array([sc1 + sc2])
        matches, kpts1, kpts2, scores = self.match_inputs_(gray1, gray2, is_draw)

        if self.no_match_upscale:
            return matches, kpts1, kpts2, scores, upscale.squeeze(0)

        # Upscale matches &  kpts
        matches = upscale * matches
        kpts1 = sc1 * kpts1
        kpts2 = sc2 * kpts2
        print(len(matches))

        if cpu:
            self.change_deivce(tmp_device)

        # def _np_to_cv2_kpts(np_kpts):
        #     cv2_kpts = []
        #     for np_kpt in np_kpts:
        #         cur_cv2_kpt = cv2.KeyPoint()
        #         cur_cv2_kpt.pt = tuple(np_kpt)
        #         cv2_kpts.append(cur_cv2_kpt)
        #     return cv2_kpts
        # def draw(save_path):
        #     import cv2
        #     query_kpts, ref_kpts = _np_to_cv2_kpts(kpts1), _np_to_cv2_kpts(kpts2)
        #     matched_image = cv2.drawMatches(
        #         ori_rgb1,
        #         query_kpts,
        #         ori_rgb2,
        #         ref_kpts,
        #         [
        #             cv2.DMatch(_queryIdx=idx, _trainIdx=idx, _distance=0)
        #             for idx in range(len(query_kpts))
        #         ],
        #         None,
        #         flags=2,
        #     )
        #     import matplotlib.pyplot as plt
        #
        #     import numpy as np
        #     plt.imshow(matched_image)
        #     plt.show()
        #     plt.savefig(save_path)
        # if is_draw:
        #     draw('output_image_nms5.png')
        return matches, kpts1, kpts2, scores
def get_args():
    import argparse
    parser = argparse.ArgumentParser("test feature matching")
    parser.add_argument("--NMS", default=False, action='store_true')
    return parser.parse_args()

args = get_args()
conf = dict(default_cfg)
conf['match_coarse']['method'] = 'maxpool_nms' if args.NMS else None
conf['match_coarse']['window_size'] = 1 if args.NMS else None
g = GeoFormer(640, 0.2, no_match_upscale=False, ckpt='saved_ckpt/geoformer.ckpt', device='cuda')
g.match_pairs('/data/lyh/code/CasMTR/assets/demo_imgs/output1.jpg', '/data/lyh/code/CasMTR/assets/demo_imgs/output2.jpg', is_draw=True)
