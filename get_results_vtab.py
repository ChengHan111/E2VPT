import glob
import pandas as pd

# from src.utils.vis_utils import get_df, average_df

from src.utils.vis_utils_pvk import get_df, average_df

LOG_NAME = "logs.txt"
MODEL_NAME = "sup_vitb16_224"
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# 如果路径中有引号 需要手动删除！example: task="closest_object_distance" --> task=closest_object_distance
root = "/home/ch7858/vpt/output_finalfinal/vtab-clevr(task=closest_object_distance)_P20_VK5_SHARED_1"
df_list=[]
# for seed in ["42", "44", "82", "100", "800"]:
for idx, seed in enumerate(["42", "44", "82", "100", "800"]):
    run = idx + 1
#     model_type = f"adapter_{r}"
    files = glob.glob(f"{root}/{MODEL_NAME}/*/run{run}/{LOG_NAME}")
    for f in files:
        df = get_df(files, f"run{run}", root, MODEL_NAME, is_best=False, is_last=True)
        if df is None:
            continue
        df["seed"] = seed
    df_list.append(df)

df= pd.concat(df_list)
df["type"] = "P_VK"
print(df)

f_df = average_df(df, metric_names=["l-test_top1"], take_average=True)
print(f_df)
