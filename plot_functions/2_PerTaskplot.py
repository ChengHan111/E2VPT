import csv
import os
import glob
import matplotlib.pyplot as plt


def read_loss_from_txt(filename, log_scale):
    train_loss, val_loss, test_loss = [], [], []
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            train_loss.append(float(row[0]))
            val_loss.append(float(row[1]))
            test_loss.append(float(row[2]))
    # print('train_loss', train_loss)
    # print('val_loss', val_loss)
    # print('test_loss', test_loss)
    plt.plot(range(len(train_loss)), train_loss, color='red')
    plt.plot(range(len(val_loss)), val_loss, color='green')
    plt.plot(range(len(test_loss)), test_loss, color='blue')
    if log_scale:
        plt.yscale('log')

    plt.legend(['train_loss', 'val_loss', 'test_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # SavedPath = filename.split('output_folder/')[1].split('/')[0]
    SavedPath = filename.split('output_folder/')[1].split('/run')[0]
    plt.savefig(f'output_folder/{SavedPath}/Per_Task.png')
    return train_loss, val_loss, test_loss

# filename = '/home/ch7858/vpt/plot_functions/output_folder/vtab-svhn/P20_VK5_TwoSteps/run1.txt'
FolderName = '/home/ch7858/vptSelf/plot_functions/output_folder/vtab-cifar(num_classes=100)'
subfolders = [f.path for f in os.scandir(FolderName) if f.is_dir()]
log_scale = True # True to enable log scale
print(subfolders)

for folder in subfolders:
    plt.figure(dpi=300)
    txt_files = glob.glob(os.path.join(folder, "*.txt")) # 查找以.txt结尾的文件
    print(f"Files in {folder}:")
    for file_path in txt_files:
        print(os.path.basename(file_path)) # 打印文件名
        train_loss, val_loss, test_loss = read_loss_from_txt(file_path, log_scale)
        