import cv2
from tqdm import tqdm
from time import time
from mrcnn import visualize

class VideoEditor:
    def __init__(self, name, class_names, model):
        self.cap = cv2.VideoCapture(name)
        self.class_names = class_names
        self.model = model
    @staticmethod
    def frame_iter(capture, description):
        def tqdm_iterator():
            while capture.grab():
                yield capture.retrieve()[1]

        return tqdm(
            tqdm_iterator(),
            desc=description,
            total=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        )

    @staticmethod
    def put_FPS_on_image(image, dt):
        font = cv2.FONT_HERSHEY_SIMPLEX
        height, width, channels = image.shape
        org = (width - 150, height - 50)
        # fontScale
        fontScale = 1
        # Blue color in BGR
        color = (255, 0, 0)
        # Line thickness of 2 px
        thickness = 2
        # Using cv2.putText() method
        FPS = int(1 / dt)
        image = cv2.putText(image, 'FPS: %d' % FPS, org, font,
                            fontScale, color, thickness, cv2.LINE_AA)
        return image

    def display_video(self):
        for frame in self.frame_iter(self.cap, 'Pedestrian Detection'):
            try:
                frame = self.__Detect(frame)
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except :
                break

        self.cap.release()
        cv2.destroyAllWindows()
    def write_video(self, filename):

        frame_width = int(self.cap.get(3))
        frame_height = int(self.cap.get(4))
        video_writer = cv2.VideoWriter("result_mask/%s.avi"%filename, cv2.VideoWriter_fourcc(*'XVID'), 10,
                                         (frame_width, frame_height))
        count = 0
        for frame in self.frame_iter(self.cap, 'Pedestrian Detection'):
            try:
                frame = self.__Detect(frame)
                video_writer.write(frame)
                count += 1
                # print("image {} took {:.3f}s".format(count, dt))
            except :
                # print("Error ", count)
                break
        self.cap.release()
        video_writer.release()
        print("video is written successfully !")
    def __Detect(self, frame):
        start = time()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model.detect([frame], verbose=1)
        r = results[0]
        dt = time() - start
        frame = visualize.cv2_display_instances(frame, r['rois'], r['masks'], r['class_ids'],
                                                self.class_names, r['scores'])

        frame = self.put_FPS_on_image(frame, dt)
        return frame