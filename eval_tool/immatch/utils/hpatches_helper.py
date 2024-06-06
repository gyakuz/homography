import os
import numpy as np
import glob
import time
import pydegensac
import cv2
import torch
from tqdm import tqdm

from utils.homography import warp_points_batch


def cal_error_auc(errors, thresholds):
    if len(errors) == 0:
        return np.zeros(len(thresholds))
    N = len(errors)
    errors = np.append([0.], np.sort(errors))
    recalls = np.arange(N + 1) / N
    aucs = []
    for thres in thresholds:
        last_index = np.searchsorted(errors, thres)
        rcs_ = np.append(recalls[:last_index], recalls[last_index-1])
        errs_ = np.append(errors[:last_index], thres)
        aucs.append(np.trapz(rcs_, x=errs_) / thres)
    return np.array(aucs, dtype=float)

def cal_reproj_dists(p1s, p2s, homography):
    '''Compute the reprojection errors using the GT homography'''

    p1s_h = np.concatenate([p1s, np.ones([p1s.shape[0], 1])], axis=1)  # Homogenous
    p2s_proj_h = np.transpose(np.dot(homography, np.transpose(p1s_h)))
    p2s_proj = p2s_proj_h[:, :2] / p2s_proj_h[:, 2:]
    dist = np.sqrt(np.sum((p2s - p2s_proj) ** 2, axis=1))
    return dist

def eval_summary_homography(dists_sa, dists_si, dists_sv, thres, lprint=print):
    correct_sa = np.mean(
        [[float(dist <= t) for t in thres] for dist in dists_sa], axis=0
    )
    correct_si = np.mean(
        [[float(dist <= t) for t in thres] for dist in dists_si], axis=0
    )
    correct_sv = np.mean(
        [[float(dist <= t) for t in thres] for dist in dists_sv], axis=0
    )

    # Compute aucs
    auc_sa = cal_error_auc(dists_sa, thresholds=thres)
    auc_si = cal_error_auc(dists_si, thresholds=thres)
    auc_sv = cal_error_auc(dists_sv, thresholds=thres)    

    # Generate summary
    summary = f'Hest Correct: a={correct_sa}\ni={correct_si}\nv={correct_sv}\n'
    summary += f'Hest AUC: a={auc_sa}\ni={auc_si}\nv={auc_sv}\n'
    lprint(summary)
    return auc_sa[-1], dict(auc_sa=auc_sa, auc_si=auc_si, auc_sv=auc_sv, summary=summary)

def eval_summary_matching(results, thres=[1, 3, 5, 10], save_npy=None):
    np.set_printoptions(precision=4)
    summary = ''
    n_i = 52
    n_v = 56      
    i_err, v_err, stats = results
    seq_type, n_feats, n_matches = stats

    if save_npy:
        print(f'Save results to {save_npy}')
        np.save(save_npy, np.array(results, dtype=object))
        
    summary += '#Features: mean={:.0f} min={:d} max={:d}\n'.format(np.mean(n_feats), np.min(n_feats), np.max(n_feats))
    summary += '#(Old)Matches: a={:.0f}, i={:.0f}, v={:.0f}\n'.format(
                    np.sum(n_matches) / ((n_i + n_v) * 5), 
                    np.sum(n_matches[seq_type == 'i']) / (n_i * 5), 
                    np.sum(n_matches[seq_type == 'v']) / (n_v * 5)
                )
    summary += '#Matches: a={:.0f}, i={:.0f}, v={:.0f}\n'.format(
                    np.mean(n_matches),
                    np.mean(n_matches[seq_type == 'i']),
                    np.mean(n_matches[seq_type == 'v'])
                )

    thres = np.array(thres)
    ierr = np.array([i_err[th] / (n_i * 5) for th in thres ])
    verr = np.array([v_err[th] / (n_v * 5) for th in thres])
    aerr = np.array([(i_err[th] + v_err[th]) / ((n_i + n_v) * 5) for th in thres])
    summary += 'MMA@{} px:\na={}\ni={}\nv={}\n'.format(thres, aerr, ierr, verr)
    return summary

def scale_homography(sw, sh):
    return np.array([[sw,  0, 0],
                     [ 0, sh, 0],
                     [ 0,  0, 1]])

def eval_hpatches(
    matcher,
    data_root,
    method='',
    task='both',
    scale_H=False,
    h_solver='degensac',
    ransac_thres=2,
    thres=[1, 3, 5, 10],
    lprint_=print,
    print_out=False,
    save_npy=None,
    debug=False,
):
    """Evaluate a matcher on HPatches sequences for image matching and homogray estimation.
    The matching metric is adopted from D2Net paper, i.e., the precentage of correctly matched
    keypoints at the given re-projection error thresholds. 
    For homography estimation, the average distances between the corners transformed using 
    the estimated and GT homographies are computed. Both percentage of the corner distance at
    the given thresholds and the area under the cumulative error curve (AUC) at those thresholds
    are reported.
    
    Args:
        - matcher: the matching function that inputs an image pair paths and 
                   outputs the matches and keypoints. 
        - data_root: the folder directory of HPatches dataset.
        - method: the description of the evaluated method.
        - task: the target task, options = [matching|homography|both]
        - ransac_thres: the set of ransac thresholds used by the solver to estimate homographies.
                        Results under each ransac threshold are printed per line.
        - thres: error thresholds in pixels to compute the metrics.
        - lprint: the printing function. If needed it can be implemented to outstream to a log file.
        - print_out: when set to True, per-pair results are printed during the evaluation.
    """

    np.set_printoptions(precision=4)
    from PIL import Image

    if task == 'both':
        task = 'matching+homography'
    seq_dirs = sorted(glob.glob('{}/*'.format(data_root)))
    lprint_(f'\n>>>>Eval hpatches: task={task} method={method} scale_H={scale_H} rthres={ransac_thres} thres={thres} ')
    
    # Matching
    if 'matching' in task:
        thres_range = np.arange(1, 16)
        i_err = {thr: 0 for thr in thres_range}
        v_err = {thr: 0 for thr in thres_range}
        n_feats = []
        seq_type = []

    # Homography
    if 'homography' in task:
        inlier_ratio = []
        h_failed = 0
        dists_sa = []
        dists_si = []
        dists_sv = []

    match_failed = 0
    first_ransac_num = 0
    first_match_num = 0
    first_match = 0
    n_matches = []
    match_time = []
    start_time = time.time()
    for seq_idx, seq_dir in tqdm(enumerate(seq_dirs[::-1]), total=len(seq_dirs), smoothing=.5):
        if debug and seq_idx > 10:
            break
        sname = seq_dir.split('/')[-1]
        im1_path = os.path.join(seq_dir, '1.ppm')

        # Eval on composed pairs within seq
        for im_idx in range(2, 7):
            im2_path = os.path.join(seq_dir, '{}.ppm'.format(im_idx))
            H_gt = np.loadtxt(os.path.join(seq_dir, 'H_1_{}'.format(im_idx)))
            scale = np.ones(4)

            # Predict matches
            try:
                t0 = time.time()
                # if im2_path != './data/datasets/hpatches-sequences-release/v_adam/2.ppm':
                #     continue
                match_res = matcher(im1_path, im2_path)
                if len(match_res) > 5:
                    first_match_num += match_res[-2]
                    first_ransac_num += match_res[-1]
                    first_match += 1
                match_time.append(time.time() - t0)
                matches, p1s, p2s = match_res[0:3]
                H_gt_raw = H_gt
                if scale_H:
                    # scale = (wo / wt, ho / ht) for im1 & im2
                    scale = match_res[4]

                    # Scale gt homoragphies
                    H_scale_im1 = scale_homography(scale[0], scale[1])
                    H_scale_im2 = scale_homography(scale[2], scale[3])
                    H_gt = np.linalg.inv(H_scale_im2) @ H_gt @ H_scale_im1
            except Exception as e:
                print(str(e))
                p1s = p2s = matches = []
                match_failed += 1
            n_matches.append(len(matches))
            
            if 'matching' in task:
                n_feats.append(len(p1s))
                n_feats.append(len(p2s))
                seq_type.append(sname[0])
                if len(matches) == 0:
                    dist = np.array([float("inf")])
                else:
                    dist = cal_reproj_dists(matches[:, :2], matches[:, 2:], H_gt)
                for thr in thres_range:
                    if sname[0] == 'i':
                        i_err[thr] += np.mean(dist <= thr)
                    else:
                        v_err[thr] += np.mean(dist <= thr)

            if 'homography' in task:
                try:
                    if 'cv' in h_solver:
                        # H_pred, inliers = cv2.findHomography(matches[:, :2], matches[:, 2:4], cv2.RANSAC, ransac_thres)
                        H_pred, inliers = cv2.findHomography(matches[:, :2], matches[:, 2:4], cv2.RANSAC, ransac_thres, maxIters=8000, confidence=0.99995,)
                    else:
                        H_pred, inliers = pydegensac.findHomography(matches[:, :2], matches[:, 2:4], ransac_thres)
                except:
                    H_pred = None

                if H_pred is None:
                    corner_dist = np.nan
                    irat = 0
                    h_failed += 1
                    inliers = []
                else:
                    im = Image.open(im1_path)
                    w, h = im.size
                    w, h = w / scale[0], h / scale[1]
                    corners = np.array([[0, 0, 1],
                                        [0, h - 1, 1],
                                        [w - 1, 0, 1],
                                        [w - 1, h - 1, 1]])
                    real_warped_corners = np.dot(corners, np.transpose(H_gt))
                    real_warped_corners = real_warped_corners[:, :2] / real_warped_corners[:, 2:]
                    warped_corners = np.dot(corners, np.transpose(H_pred))
                    warped_corners = warped_corners[:, :2] / warped_corners[:, 2:]
                    corner_dist = np.mean(np.linalg.norm(real_warped_corners - warped_corners, axis=1))
                    irat = np.mean(inliers)

                if corner_dist > 10000000:
                    kp0 = p1s
                    kp1 = p2s
                    if scale_H:
                        # scale = (wo / wt, ho / ht) for im1 & im2
                        scale = match_res[4]
                        sc1 = scale[:2]
                        sc2 = scale[2:]
                        kp0 = sc1 * p1s
                        kp1 = sc2 * p2s
                    kp0, kp1 = torch.from_numpy(kp0), torch.from_numpy(kp1)
                    H_tmp = torch.from_numpy(H_gt_raw).to(kp0)

                    k0_warp = warp_points_batch(kp0[None], H_tmp.unsqueeze(0))[0]
                    errors = torch.sqrt(((kp1 - k0_warp) ** 2).sum(-1))
                    error = errors.mean()
                    # kp0, kp1 = kp0[errors>8].detach(), kp1[errors>8].detach()
                    kp0_raw = kp0
                    print(im1_path, im2_path, error.item())
                    if corner_dist > 10:
                        try:
                    # if im2_path == './data/datasets/hpatches-sequences-release/i_bridger/6.ppm':
                            import matplotlib.pyplot as plt
                            im1 = cv2.imread(im1_path, 0)
                            im2 = cv2.imread(im2_path, 0)

                            plt.figure(dpi=200)
                            kp0 = [cv2.KeyPoint(int(k[0]), int(k[1]), 30) for k in kp0]
                            kp1 = [cv2.KeyPoint(int(k[0]), int(k[1]), 30) for k in kp1]
                            matches = [cv2.DMatch(_trainIdx=i, _queryIdx=i, _distance=1, _imgIdx=-1) for i in range(len(kp0))]
                            show = cv2.drawMatches(im1, kp0,
                                                   im2, kp1, matches, None)
                            plt.imshow(show)
                            plt.title(f'cor:{corner_dist}, err:{error}')
                            plt.show()
                        except Exception:
                            pass

                    # kp1 = k0_warp[errors > 8]
                    # kp0 = kp0_raw
                    # plt.figure(dpi=200)
                    # kp0 = [cv2.KeyPoint(int(k[0]), int(k[1]), 30) for k in kp0]
                    # kp1 = [cv2.KeyPoint(int(k[0]), int(k[1]), 30) for k in kp1]
                    # matches = [cv2.DMatch(_trainIdx=i, _queryIdx=i, _distance=1, _imgIdx=-1) for i in range(len(kp0))]
                    # show = cv2.drawMatches(im1, kp0,
                    #                        im2, kp1, matches, None)
                    # plt.imshow(show)
                    # plt.title(f'cor:{corner_dist}, err:{error}')
                    # plt.show()
                inlier_ratio.append(irat)
                dists_sa.append(corner_dist)
                if sname[0] == 'i':
                    dists_si.append(corner_dist)
                if sname[0] == 'v':
                    dists_sv.append(corner_dist)
                    
            if print_out:
                print(f'Scene {sname}, pair:1-{im_idx} matches:{len(matches)}')
                if 'matching' in task:
                    print(f'Median matching dist:{np.median(dist):.2f} <1px:{np.mean(dist <= 1):.3f}')
                if 'homography' in task:
                    print(f'Corner dist:{corner_dist:.2f} inliers:{np.sum(inliers)}')

    lprint_(f'>>Finished, pairs={len(match_time)} match_failed={match_failed} matches={np.mean(n_matches):.1f} match_time={np.mean(match_time):.2f}s')
    if first_match != 0:
        print('----------------------------------------------------------------------------------------')
        print(f'first_matches_num: {first_match_num/first_match}, first_ransac_num: {first_ransac_num/first_match}')
        print('----------------------------------------------------------------------------------------')
    if 'matching' in task:
        results = i_err, v_err, [np.array(seq_type), np.array(n_feats), np.array(n_matches)]
        lprint_('==== Image Matching ====')
        lprint_(eval_summary_matching(results, thres, save_npy=save_npy))
    if 'homography' in task:
        lprint_('==== Homography Estimation ====')        
        lprint_(f'Hest solver={h_solver} est_failed={h_failed} ransac_thres={ransac_thres} inlier_rate={np.mean(inlier_ratio):.2f}')
        eval_summary_homography(dists_sa, dists_si, dists_sv, thres, lprint=lprint_)

