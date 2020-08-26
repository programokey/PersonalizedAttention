portion = 0.05
with open('data_format1/user_log_format1.csv', 'r') as f_in:
    data = list(f_in.readlines())
    print(len(data))
    with open('data_format1/user_log_sample.csv', 'w') as f_out:
        print(int(len(data)*portion))
        f_out.writelines(data[:int(len(data)*portion)])

