from datetime import datetime

if __name__ == '__main__':
    fn = r'C:\Users\Rudy\PycharmProjects\visual-odometry\2011_09_26\2011_09_26\2011_09_26_drive_0001_sync\image_00' \
         r'\timestamps.txt '

    def read_timestamps_from_file(file_path: str) -> list:
        with open(file_path, 'r') as f:
            ts = []
            for line in f.readlines():
                d = datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')
                ts.append(d)

        return ts