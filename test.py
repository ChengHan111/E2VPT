import torch
results_dict = torch.load('/home/ch7858/vpt/output_finalfinal/vtab-caltech101_P5_VK5_SHARED_0/sup_vitb16_224/lr2.5_wd0.0001/run1/eval_results.pth', "cpu")
val_result = results_dict[f"epoch_{100}"]["classification"][t_name]["top1"]
val_result = float(val_result)
print(val_result)
exit()