import glob
import pandas as pd

# from src.utils.vis_utils import get_df, average_df

from src.utils.vis_utils_pvk import get_df, average_df

LOG_NAME = "logs.txt"
MODEL_NAME = "sup_vitb16_224"
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# 如果路径中有引号已经括号 需要手动删除！example: （task="closest_object_distance"） --> None 
# 两个文件夹下:output_before_pruning + output_rewind
root = "/home/ch7858/vpt/output_rewind/vtab-smallnorb_P100_VK20_SHARED_1_INIT_2_ACC_0_ONVK_0"
dataset_type = 'vtab_rewind' # currently support vtab and fgvc

df_list=[]
for idx, seed in enumerate(["42", "44", "82", "100", "800"]):
    run = idx + 1

    files = glob.glob(f"{root}/{MODEL_NAME}/*/run{run}/{LOG_NAME}")
    print(files)
    for f in files:
        df = get_df(
            files, f"run{run}", root, MODEL_NAME, is_best=False, is_last=True, dataset_type=dataset_type)
        if df is None:
            continue
        df["seed"] = seed
    df_list.append(df)

df= pd.concat(df_list)
df["type"] = "P_VK"
print(df)

f_df = average_df(df, metric_names=["l-test_top1"], take_average=True)
print(f_df)

best_top_1 = pd.to_numeric(f_df["l-test_top1"]).tolist()[0]
Prompt_length = f_df["Prompt_length"].tolist()[0][1:]
VK_length = f_df["VK_length"].tolist()[0][2:]
lr = f_df["lr"].tolist()[0]
wd = f_df["wd"].tolist()[0]
tuned_percentage = f_df["tuned / total (%)"].tolist()[0]
batch_size = f_df["batch_size"].tolist()[0]
runs = df["l-test_top1"].tolist()
print(f"{best_top_1}--{runs}({Prompt_length}+{VK_length}+lr{lr}_wd{wd} {tuned_percentage} {batch_size})")

