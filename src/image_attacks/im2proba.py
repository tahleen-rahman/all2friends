# Created by rahman at 15:20 2020-03-05 using PyCharm


def slice_files(mediaFile, DATAPATH, cores):
    """
    slices our big file target_media into many slices for parallel processing of the image urls later
    :param mediaFile: big files containing all the media samples
    :param DATAPATH:
    :param cores: number of cores in the compute server that I want to use
    :return:
    """

    with open(mediaFile, "r") as big_file:

        lines = big_file.readlines()
        slice_size = int(len(lines) / cores)

        for slice in range(0, cores):

            file_slice = lines[slice * slice_size:(slice + 1) * slice_size]
            print (slice * slice_size, (slice + 1) * slice_size)

            newfile = DATAPATH + str(slice) + "media_cleaned.csv"

            with open(newfile, 'w') as small_file:
                small_file.writelines(file_slice)

        rem_idx =  (slice + 1) * slice_size  # first index of the remaining lines

    # leftover remaining samples in cores + "media_cleaned.csv"
    with open(mediaFile, "r") as big_file:

        lines = big_file.readlines()

        with open(DATAPATH + cores + "media_cleaned.csv", 'w') as small_file:

            small_file.writelines(lines[rem_idx:]) #8309400


def combine_files(DATAPATH, cores):
    """
    cobmines the image embeddings created by the parallel shell scripts back into 1 large file
    :param DATAPATH:
    :param cores:
    :return:
    """

    combi_file, i = DATAPATH + "bigger_combi_probfile.csv", 0

    with open(combi_file, 'w') as big_file:

        for slice in range(0, cores):

            newfile = DATAPATH + str(slice) + "probability_dist.csv"

            with open(newfile, 'r') as small_file:

                file_slice = small_file.readlines()
                print (len(file_slice))

                # write the header
                if i == 0:
                    big_file.write(file_slice[0])

                big_file.writelines(file_slice[1:])
                i+=1



if __name__ == '__main__':

    city = 'la' # 'ny', sys.argv[1]
    cores = 120

    DATAPATH = '../../data/' + city
    mediaFile = "target_media"


    slice_files(mediaFile, DATAPATH, cores)

    import subprocess

    subprocess.call(['./parallelize.sh', cores])

    combine_files(DATAPATH, cores)
