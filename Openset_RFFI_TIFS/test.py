import numpy as np
import pandas as pd

def abc(data_path,
    dev_range = np.arange(0,5, dtype = int),
    pkt_range = np.arange(0,5, dtype = int)
        ):
    df = pd.read_excel(data_path, header=None)
    file_data = df.values
    label = file_data[:,640].astype(int)
    label = np.transpose(label)
    label = label - 1

    label_start = int(label[0]) + 1
    label_end = int(label[-1]) + 1
    num_dev = label_end - label_start + 1

    num_pkt = len(label)

    num_pkt_per_dev = int(num_pkt / num_dev)
    print(label[0])
    print('Dataset information: Dev ' + str(label_start) + ' to Dev ' +
          str(label_end) + ', ' + str(num_pkt_per_dev) + ' packets per device.')

    sample_index_list = []

    for dev_idx in dev_range:
        sample_index_dev = np.where(label == dev_idx)[0][pkt_range].tolist()
        sample_index_list.extend(sample_index_dev)
    data_list = []
    for num in range(len(sample_index_list)):
        train_data1 = file_data[sample_index_list[num], :-1]
        # train_data1 = train_data1[,:-1]
        iq_i = train_data1[::2]
        iq_q = train_data1[1::2]
        train_data = iq_i +iq_q*1j
        data_list.extend(train_data)

        # label_list.extend(label[sample_index_list[num]])
    label_list = label[sample_index_list]
    data = np.array(data_list).astype(np.complex128).reshape(-1, 320)
    label = np.array(label_list).astype(int).reshape(-1, 1)

    return data, label
    print(df)
    print("123")
if __name__ == "__main__":
    # data_path = '/home/prince/桌面/wifi/devices_data/Final_result/test.xlsx'
    data_path = '/home/data/hw/prince/Datasets/device_identify_data/test.xlsx'
    # a,b = abc(data_path)
    # data = np.load("/home/data/hw/prince/Datasets/device_identify_data/data_train.npy")
    # label = np.load("/home/data/hw/prince/Datasets/device_identify_data/label_train.npy")
    arr =np.array(range(12)).reshape(3,4)
    np.random.shuffle(arr)
    print("123")
