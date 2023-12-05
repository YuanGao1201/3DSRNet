import os
import re


def random_split():
    dir = "/media/gy/Data/VerSe/ctpro_RF128+spine"
    # dir = "/media/gy/Data/VerSe/ctpro_RF128"
    # dir = "/media/gy/Data/VerSe/ctpro_RF_crop"
    # dir = "/media/gy/Data/VerSe/ctpro"
    # dir = "/media/gy/Data/VerSe20/ctpro"
    outfile_train = "verse_RF128+spine_train.txt"
    outfile_test = "verse_RF128+spine_test.txt"
    txt_train = open(outfile_train, "a")
    txt_test = open(outfile_test, "a")
    for root, dirs, files in os.walk(dir):
        for index, dir in enumerate(dirs):
            if index % 10 == 1:
                txt_test.write(dir + "\n")
            else:
                txt_train.write(dir + "\n")
    txt_train.close()
    txt_test.close()


def data_rename():
    root_dir = "/media/gy/Data/VerSe/ctpro"
    for root, dirs, files in os.walk(root_dir):
        for dir in dirs:
            files = os.listdir(os.path.join(root, dir))
            for file_name in files:
                if re.findall('_CT-sag', file_name):
                    file_name_new = file_name.replace('_CT-sag', '')
                    os.rename(os.path.join(root, dir, file_name), os.path.join(root, dir, file_name_new))
                if re.findall('_CT-iso', file_name):
                    file_name_new = file_name.replace('_CT-iso', '')
                    os.rename(os.path.join(root, dir, file_name), os.path.join(root, dir, file_name_new))
                if re.findall('_CT-ax', file_name):
                    file_name_new = file_name.replace('_CT-ax', '')
                    os.rename(os.path.join(root, dir, file_name), os.path.join(root, dir, file_name_new))
                if re.findall('_CT_ax', file_name):
                    file_name_new = file_name.replace('_CT_ax', '')
                    os.rename(os.path.join(root, dir, file_name), os.path.join(root, dir, file_name_new))
                if re.findall('_CT-cor', file_name):
                    file_name_new = file_name.replace('_CT-cor', '')
                    os.rename(os.path.join(root, dir, file_name), os.path.join(root, dir, file_name_new))


def data_addname():
    root_dir = "/media/gy/Data/VerSe/drr"
    for root, dirs, files in os.walk(root_dir):
        for dir in dirs:
            dir_new = dir + '_19'
            os.rename(os.path.join(root, dir), os.path.join(root, dir_new))
            # files = os.listdir(os.path.join(root, dir))
            # for file_name in files:
            #     file_name_new = file_name.replace('_new.nii.gz', '_19_new.nii.gz')
            #     os.rename(os.path.join(root, dir, file_name), os.path.join(root, dir, file_name_new))



if __name__ == '__main__':
    random_split()
    # data_rename()
    # data_addname()
