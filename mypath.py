class Path(object):
    @staticmethod
    def db_dir(database):
        if database == 'ucf101':
            # folder that contains class labels
            root_dir = '/data/Sadjad/Datasets/ucf101/UCF-101'

            # Save preprocess data into output_dir
            output_dir = '/data/Sadjad/Datasets/ucf101/pytorch-vdieo-recognition-output/'

            return root_dir, output_dir
        elif database == 'hmdb51':
            # folder that contains class labels
            root_dir = '/data/Sadjad/Datasets/hmdb-51'

            output_dir = '/data/Sadjad/Datasets/hmdb51/pytorch-video-recognition-output/'

            return root_dir, output_dir
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def model_dir():
        return 'models/c3d-pretrained.pth'
