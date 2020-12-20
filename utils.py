import cv2
from datetime import datetime
import os
import time


class ToolBox:

    @staticmethod
    def play_video_from_seq(frames: list, title: str = None):

        for frame in frames:
            cv2.imshow(title if title else '', frame)
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

    @staticmethod
    def read_timestamps_from_file(file_path: str) -> list:
        with open(file_path, 'r') as f:
            ts = []
            for line in f.readlines():
                d = datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')
                ts.append(d)
        return ts

    @staticmethod
    def play_video_from_seq_with_ts(frames: list, timestamps: list, title: str = None):
        assert len(frames) == len(timestamps), "Amount of frames must be equals to amount of timestamps."

        start = time.time()
        for idx, frame in enumerate(frames):

            actual_ts = timestamps[idx]
            wait = int((timestamps[idx + 1] - actual_ts).total_seconds() * 10 ** 3) if idx + 1 < len(timestamps) else 30
            frame = cv2.putText(frame, str(actual_ts), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.imshow(title if title else '', frame)

            if cv2.waitKey(wait) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        print(f"{len(frames)} frames played on {time.time()-start} seconds")


if __name__ == '__main__':
    folder_path = r'2011_09_26\2011_09_26_drive_0001_sync\image_03/data'
    frames = [cv2.imread(os.path.join(folder_path, rf'0000000{str(i).zfill(3)}.png')) for i in range(108)]
    timestamps = ToolBox.read_timestamps_from_file(r'C:\Users\Rudy\PycharmProjects\visual-odometry\2011_09_26'
                                                   r'\2011_09_26\2011_09_26_drive_0001_sync\image_03\timestamps.txt')
    ToolBox.play_video_from_seq_with_ts(frames, timestamps)

