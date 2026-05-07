from datasets.multiple_datasets import datasets_dict     

ds_names = ['cisyn']
# ds_names = ['coco', 'mpii', 'insta', 'aic', 'agora', 'bedlam', 'cisyn']
ds_splits =  ['train']
# ds_splits = ['train_opt', 'train_opt', 'train_opt', 'train_opt', 'train', 'train_6fps', 'train']

# use it to visualize GTs
if __name__ == '__main__':
    kwargs = {'input_size': 1288, 'aug': False, 'mode': 'eval', 'human_type':'smpl', 'use_kid':True}
    # kwargs = {'input_size': 1288, 'aug': False, 'mode': 'eval', 'human_type':'smpl', 'use_kid':True, 'force_aspect_ratio': [720, 1280]}
    for name, split in zip(ds_names, ds_splits):
        kwargs['sat_cfg'] = {'use_sat': True, 'num_lvls':3}
        print(f'Loading {name}_{split}...')
        ds = datasets_dict[name](split = split, **kwargs)
        print(f'Length of {name}_{split}: {len(ds)}')
        ds.visualize(vis_num = 10)
    