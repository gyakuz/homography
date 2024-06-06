# batch.keys()
import torch
torch.set_float32_matmul_precision('high')

def clear_min_to_zero(mat, thr):
    mat[mat < thr] = 0
    return mat


SELECTED_KEYS = {'name', 'index', 'dataset_name', 'image0', 'image1', 'sc1', 'sc2', 'H_gt', 'pair_id', 'pair_names', 'H_0to1', 'H_1to0', 'conf_matrix_gt', 'bs', 'hw0_i', 'hw1_i', 'hw0_c', 'hw1_c', 'hw0_f', 'hw1_f', 'b_ids', 'i_ids', 'j_ids', 'm_bids', 'mkpts0_c', 'mkpts1_c', 'mconf', 'conf_matrix', 'W', 'fine_matrix', 'mkpts0_f', 'mkpts1_f', 'conf_matrix_fine_gt', 'loss', 'loss_scalars', 'ret_dict', 
                 'image_path1', 'image_path2', 'scale0', 'scale1',
                'conf_matrixx_sim', 'dect_conf_matrixx_sim',
                 'vis_dict',
                }

def to_sparse(data_dict, thr=1e-6):
    for k, v in data_dict.items():
        if isinstance(v, dict):
            to_sparse(v)
        elif isinstance(v, torch.Tensor):
            if not v.is_sparse and v.numel() > 10 ** 6:
                data_dict[k] = clear_min_to_zero(v, thr).to_sparse()
                print(k)
    return data_dict


def prepare_data_to_save(data: dict):
    # pop dect conf
    # for key in ['dect_conf_matrix', ]:
    # # for key in ['dect_conf_matrix', 'fine_matrix']:
    #     if key in data:
    #         data.pop(key)

    # pop dino data
    # data = {k:v for k, v in data.items() if 'dino_' not in k}

    data = {k:v for k, v in data.items() if k in SELECTED_KEYS}

    # convert to sparse matirx, fp16
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            if value.dtype == torch.float32:
                value = value.to(torch.float16)
                data[key] = value
        
            if '_matrix' in key:
                value = value.detach()
                # value[value < 1e-4] = 0
                if key not in {'fine_matrix'}:
                    thr = 1e-4
                else:
                    thr = 1e-2
                # thr = 1e-3
                value = clear_min_to_zero(value, thr)
                data[key] = value.to_sparse()
            data[key] = data[key].cpu()
    # data = to_sparse(data)
    # data['fine_matrix'] = clear_min_to_zero(data['fine_matrix'], 1e-2)
    return data

def cache_load(batch, cache_path):
    return None
    try:
        batch = torch.load(cache_path)
        for k, v in batch.items():
            if torch.sparse.is_sparse(v):
                batch[k] = v.to_dense()
            
        print('load from cache: {}'.format(cache_path))
        return batch
    except:
        return None

def cache_create(batch, cache_path):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    ## reduce the data storage size
    data = batch

    # data preprocess
    data = prepare_data_to_save(data)

    # save to pt file
    torch.save(data, cache_path)
    filesz = os.path.getsize(cache_path)
    print('save cache at: {}, ({:.2}M)'.format(cache_path, filesz/2**20))

def cache_to_disk0(func):
    CACHE_FOLDER = './outputs/hpatches_cache'
    def wrapper(batch=None, model=None, version='default', cache_folder=None, **kwargs):
        nonlocal CACHE_FOLDER
        name = batch['name'].replace('/', '_')
        if cache_folder is not None:
            CACHE_FOLDER = cache_folder
        cache_path = os.path.join(CACHE_FOLDER, version, name + '.pt')

        ret = cache_load(batch, cache_path, )
        if ret is not None:
            batch = ret
        else:
            batch = func(batch, model, version=version, **kwargs)
            cache_create(batch, cache_path)

    return wrapper

def cache_to_disk(version='default', cache_folder=None):
    CACHE_FOLDER = './outputs/hpatches_cache'
    if cache_folder is not None:
        CACHE_FOLDER = cache_folder
    def cache_to_disk0(func):
        CACHE_FOLDER = './outputs/hpatches_cache'
        def wrapper(batch=None, model=None, version='default', **kwargs):
            name = batch['name'].replace('/', '_')
            cache_path = os.path.join(CACHE_FOLDER, version, name + '.pt')
    
            ret = cache_load(batch, cache_path)
            if ret is not None:
                batch = ret
            else:
                batch = func(batch=batch, model=model)
                cache_create(batch, cache_path)
    
        return wrapper
    return cache_to_disk0



def cache_on_local(func):
    CACHE_FOLDER = './outputs/hpatches_cache'
    def wrapper(batch, model, version='default'):
        name = batch['name'].replace('/', '_')
        cache_path = os.path.join(CACHE_FOLDER, version, name + '.pt')
        try:
            assert 0
            batch = torch.load(cache_path)
            for k, v in batch.items():
                if torch.sparse.is_sparse(v):
                    batch[k] = v.to_dense()
                
            print('load from cache: {}'.format(cache_path))
            return batch
        except:
            batch = func(batch, model)
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            ## reduce the data storage size
            data = batch
            for key in ['dect_conf_matrix', ]:
            # for key in ['dect_conf_matrix', 'fine_matrix']:
                data.pop(key)

            data = {k:v for k, v in data.items() if 'dino_' not in k}
            
            for key, value in data.items():
                if not isinstance(value, torch.Tensor): continue
                if value.dtype == torch.float32:
                    value = value.to(torch.float16)
                    data[key] = value
            
                if '_matrix' in key:
                    value = value.detach()
                    # value[value < 1e-4] = 0
                    if key not in {'fine_matrix'}:
                        thr = 1e-4
                    else:
                        thr = 1e-2
                    value = clear_min_to_zero(value, thr)
                    data[key] = value.to_sparse()
            # data['fine_matrix'] = clear_min_to_zero(data['fine_matrix'], 1e-2)

            torch.save(data, cache_path)
            return data
    return wrapper

@cache_on_local
def eval_one(batch, model, version=None):
    model = model.to('cuda')
    batch['dataset_name'] = (batch['dataset_name'], )
    for key in ['image0', 'image1', 'mask1', 'mask2', 'H_0to1', 'H_1to0', ]:
        if key not in batch: continue
        batch[key] = batch[key].unsqueeze(0).to('cuda')
    model.validation_step(batch, 0, dataloader_idx=0)
    return batch



import os
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'

from homodataset.hpatches import load_model, HpatchesDataset, Hpatches_Eval
from tqdm import tqdm

# args = {
#     "ckpt_path": 'logs/tb_logs/baseline_speed/version_3/checkpoints/epoch=7-auc@3px=0.69.ckpt',
#     'version': 'geo2',
# }
args = {
    "ckpt_path": None,
    'version': 'ours2',
}


import glob
def cal_hpatch_score(model=None, path='./outputs/hpatches_cache/loftr/*.pt'):
    
    hpatches = Hpatches_Eval(None, None, None)

    files = glob.glob(path)

    # model = load_model(**args)
    print('find {} pt files'.format(len(files)))

    # out_dict = [torch.load(_)['ret_dict'] for _ in files]
    out_dict = []
    for file in tqdm(files):
        d = torch.load(file)
        if d is None:
            print('d=None')
        val = d.get('ret_dict', None)
        if val is not None:
            out_dict.append(val)
        else:
            print('out_dict not found')
            # assert 0, d
    res = hpatches.cal_scores(out_dict)
    print(res)
    print(res['summary'])



def main():
        
    model = load_model(**args)
    model = model.to('cuda')
    
    
    obj = HpatchesDataset()
    
    for idx, batch in enumerate(tqdm(obj)):
        # if idx >= 4: break
        batch = eval_one(batch, model, args['version'])


if __name__ == '__main__':
    # cal_hpatch_score()
    # main()
    cal_hpatch_score()



