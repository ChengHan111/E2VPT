import os
import csv
import glob
import matplotlib.pyplot as plt


def read_loss_from_txt(filename, loss_type):
    loss = []
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            loss.append(float(row[loss_type]))
    # plt.plot(range(len(loss)), loss)

    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    
    # # SavedPath = filename.split('output_folder/')[1].split('/')[0]
    # SavedPath = filename.split('output_folder/')[1].split('/run')[0]
    # plt.savefig(f'output_folder/{SavedPath}/Per_Task.png')
    
    # SavedPath = filename.split('output_folder/')[1].split('/')[0]
    # plt.savefig(f'output_folder/{SavedPath}/Per_Task_{loss_type}.png')
    
    return loss

FolderName = '/home/ch7858/vpt/plot_functions/output_folder/vtab-sun397_finish'
log_scale = False # True to enable log scale
subfolders = [f.path for f in os.scandir(FolderName) if f.is_dir()]
# print(subfolders)

# color_map = ['red', 'green', 'blue']

SavedPath = FolderName.split('output_folder/')[1]
plt.figure(dpi=300)
for folder in subfolders:
    txt_files = glob.glob(os.path.join(folder, "*.txt"))
    task = folder.split('/')[-1]
    # print(task)
    print(f"Files in {folder}:")
    counter_train = 0
    for file_path in txt_files:
        print(os.path.basename(file_path))
        
        train_loss = read_loss_from_txt(file_path, 0)
        if task.endswith('TwoSteps'):
            if counter_train % 5 == 0:
                counter_train += 1
                plt.plot(range(len(train_loss)), train_loss, color='blue', label=task)
                if log_scale:
                    plt.yscale('log')
            else:
                plt.plot(range(len(train_loss)), train_loss, color='blue')
                if log_scale:
                    plt.yscale('log')
            
        elif task.endswith('PromptTuning'):
            if counter_train % 5 == 0:
                counter_train += 1
                plt.plot(range(len(train_loss)), train_loss, color='red', label=task)
                if log_scale:
                    plt.yscale('log')
            else:
                plt.plot(range(len(train_loss)), train_loss, color='red')
                if log_scale:
                    plt.yscale('log')


        else:
            if counter_train % 5 == 0:
                counter_train += 1
                plt.plot(range(len(train_loss)), train_loss, color='green', label=task)
                if log_scale:
                    plt.yscale('log')
            else:
                plt.plot(range(len(train_loss)), train_loss, color='green')
                if log_scale:
                    plt.yscale('log')

        # plt.legend([task])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
            
        plt.legend()
        plt.title(os.path.basename('Train_Loss'))
        plt.savefig(f'output_folder/{SavedPath}/Train_Loss.png')

plt.clf()
# plt.legend()

plt.figure(dpi=300)
for folder in subfolders:
    txt_files = glob.glob(os.path.join(folder, "*.txt"))
    task = folder.split('/')[-1]
    # print(task)
    print(f"Files in {folder}:")
    counter_train = 0
    for file_path in txt_files:
        print(os.path.basename(file_path))
        
        val_loss = read_loss_from_txt(file_path, 1)
        if task.endswith('TwoSteps'):
            if counter_train % 5 == 0:
                counter_train += 1
                plt.plot(range(len(val_loss)), val_loss, color='blue', label=task)
                if log_scale:
                    plt.yscale('log')
            else:
                plt.plot(range(len(val_loss)), val_loss, color='blue')
                if log_scale:
                    plt.yscale('log')
            
        elif task.endswith('PromptTuning'):
            if counter_train % 5 == 0:
                counter_train += 1
                plt.plot(range(len(val_loss)), val_loss, color='red', label=task)
                if log_scale:
                    plt.yscale('log')
            else:
                plt.plot(range(len(val_loss)), val_loss, color='red')
                if log_scale:
                    plt.yscale('log')


        else:
            if counter_train % 5 == 0:
                counter_train += 1
                plt.plot(range(len(val_loss)), val_loss, color='green', label=task)
                if log_scale:
                    plt.yscale('log')
            else:
                plt.plot(range(len(val_loss)), val_loss, color='green')
                if log_scale:
                    plt.yscale('log')

        # plt.legend([task])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
            
        plt.legend()
        plt.title(os.path.basename('Val_Loss'))
        plt.savefig(f'output_folder/{SavedPath}/Val_Loss.png')

plt.clf()

plt.figure(dpi=300)
for folder in subfolders:
    txt_files = glob.glob(os.path.join(folder, "*.txt"))
    task = folder.split('/')[-1]
    # print(task)
    print(f"Files in {folder}:")
    counter_train = 0
    for file_path in txt_files:
        print(os.path.basename(file_path))
        
        test_loss = read_loss_from_txt(file_path, 2)
        if task.endswith('TwoSteps'):
            if counter_train % 5 == 0:
                counter_train += 1
                plt.plot(range(len(test_loss)), test_loss, color='blue', label=task)
                if log_scale:
                    plt.yscale('log')
            else:
                plt.plot(range(len(test_loss)), test_loss, color='blue')
                if log_scale:
                    plt.yscale('log')
            
        elif task.endswith('PromptTuning'):
            if counter_train % 5 == 0:
                counter_train += 1
                plt.plot(range(len(test_loss)), test_loss, color='red', label=task)
                if log_scale:
                    plt.yscale('log')
            else:
                plt.plot(range(len(test_loss)), test_loss, color='red')
                if log_scale:
                    plt.yscale('log')


        else:
            if counter_train % 5 == 0:
                counter_train += 1
                plt.plot(range(len(test_loss)), test_loss, color='green', label=task)
                if log_scale:
                    plt.yscale('log')
            else:
                plt.plot(range(len(test_loss)), test_loss, color='green')
                if log_scale:
                    plt.yscale('log')

        # plt.legend([task])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
            
        plt.legend()
        plt.title(os.path.basename('Test_Loss'))
        plt.savefig(f'output_folder/{SavedPath}/Test_Loss.png')

plt.clf()
