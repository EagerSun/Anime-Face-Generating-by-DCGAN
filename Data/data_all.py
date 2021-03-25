from Data.data_original import data_produce_original
from Data.data_prepared import data_produce_prepared
import numpy as np
import h5py

def data_intoh5(ori_address, ori_address_2, data_store_add):
    '''
    ori_address(string) is the address of file which includes with raw images(anime faces included in but not pure.).
    ori_address_2(string) is the address of file which includes with well-preared images(anime faces included in are well-prepared.).
    data_store_add(string) is the address of file(.h5) which hold data.
    This function store anime faces from ori_address and ori_address_2 to address as data_store_add.
    '''
    print("Now, the prepared-datav is gonna processing")
    data_p1 = data_produce_prepared(ori_address = ori_address)
    print("\nNow, the raw-data is gonna processing")
    data_p2 = data_produce_original(ori_address = ori_address_2)
    data = np.concatenate((data_p1, data_p2), axis = 0)
    print("\n- The dataset is storing into target address...")
    save_add = os.path.join(os.getcwd(), data_store_add)
    with h5py.File(save_add, 'w') as hy:
        hy.create_dataset('data', data = data)
    print("- Congratulations, the dataset have been successfully stored!!!")

def read_data(data_store_add):
     '''
     data_store_add(string) is the address of file(.h5) you store the data(should be similar with data_store_add in data_intoh5()).
     Return value: data(np.array()) is the concatenated array for anime faces from .h5 file in address: data_store_add
     '''
    with h5py.File(data_store_add, 'r') as hy:
        data = np.array(hy.get('data'))

    return data