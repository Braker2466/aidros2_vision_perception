import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import aidcv as cv2

import numpy as np
from blazeface import *
import aidlite
import subprocess

# 摄像头设备目录
root_dir = "/sys/class/video4linux/"
 
# 图像预处理函数，用于TFLite模型输入
def preprocess_image_for_tflite32(image, model_image_size=192):
    """
    对图像进行预处理，使其适合TFLite模型输入
    1. 将BGR格式转换为RGB格式
    2. 调整图像大小为模型输入尺寸
    3. 添加批次维度
    4. 归一化像素值到[-1, 1]范围
    5. 转换为float32类型
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (model_image_size, model_image_size))
    image = np.expand_dims(image, axis=0)
    image = (2.0 / 255.0) * image - 1.0
    image = image.astype('float32')
 
    return image
 
# 图像填充和预处理函数
def preprocess_img_pad(img, image_size=128):
    """
    对图像进行填充和预处理
    1. 将图像填充为正方形
    2. 保存原始填充图像用于后续显示
    3. 调整填充后图像大小为模型输入尺寸
    4. 归一化并添加批次维度
    """
    # fit the image into a 128x128 square
    shape = np.r_[img.shape]
    pad_all = (shape.max() - shape[:2]).astype('uint32')
    pad = pad_all // 2
    img_pad_ori = np.pad(
        img,
        ((pad[0], pad_all[0] - pad[0]), (pad[1], pad_all[1] - pad[1]), (0, 0)),
        mode='constant')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pad = np.pad(
        img,
        ((pad[0], pad_all[0] - pad[0]), (pad[1], pad_all[1] - pad[1]), (0, 0)),
        mode='constant')
    img_small = cv2.resize(img_pad, (image_size, image_size))
    img_small = np.expand_dims(img_small, axis=0)
    img_small = (2.0 / 255.0) * img_small - 1.0
    img_small = img_small.astype('float32')
 
    return img_pad_ori, img_small, pad
 
# 在图像上绘制检测框
def plot_detections(img, detections, with_keypoints=True):
    """
    在图像上绘制人脸检测框
    1. 根据检测结果计算人脸区域
    2. 调整区域大小，确保包含完整人脸
    3. 在图像上绘制矩形框
    """
    output_img = img
    print(img.shape)
    x_min = 0
    x_max = 0
    y_min = 0
    y_max = 0
    print("Found %d faces" % len(detections))
    for i in range(len(detections)):
        ymin = detections[i][0] * img.shape[0]
        xmin = detections[i][1] * img.shape[1]
        ymax = detections[i][2] * img.shape[0]
        xmax = detections[i][3] * img.shape[1]
        w = int(xmax - xmin)
        h = int(ymax - ymin)
        h = max(w, h)
        h = h * 1.5
 
        x = (xmin + xmax) / 2.
        y = (ymin + ymax) / 2.
 
        xmin = x - h / 2.
        xmax = x + h / 2.
        ymin = y - h / 2. - 0.08 * h
        ymax = y + h / 2. - 0.08 * h
 
        x_min = int(xmin)
        y_min = int(ymin)
        x_max = int(xmax)
        y_max = int(ymax)
        p1 = (int(xmin), int(ymin))
        p2 = (int(xmax), int(ymax))
        cv2.rectangle(output_img, p1, p2, (0, 255, 255), 2, 1)
 
    return x_min, y_min, x_max, y_max
 
# 在图像上绘制面部网格
def draw_mesh(image, mesh, mark_size=2, line_width=1):
    """
    在图像上绘制面部网格
    1. 将归一化的关键点坐标转换为图像坐标
    2. 在每个关键点位置绘制圆点
    3. 连接关键点形成面部轮廓（眼睛等）
    """
    # The mesh are normalized which means we need to convert it back to fit
    # the image size.
    image_size = image.shape[0]
    mesh = mesh * image_size
    for point in mesh:
        cv2.circle(image, (point[0], point[1]),
                   mark_size, (0, 255, 128), -1)
 
    # Draw the contours.
    # Eyes
    left_eye_contour = np.array([mesh[33][0:2],
                                 mesh[7][0:2],
                                 mesh[163][0:2],
                                 mesh[144][0:2],
                                 mesh[145][0:2],
                                 mesh[153][0:2],
                                 mesh[154][0:2],
                                 mesh[155][0:2],
                                 mesh[133][0:2],
                                 mesh[173][0:2],
                                 mesh[157][0:2],
                                 mesh[158][0:2],
                                 mesh[159][0:2],
                                 mesh[160][0:2],
                                 mesh[161][0:2],
                                 mesh[246][0:2], ]).astype(np.int32)
    right_eye_contour = np.array([mesh[263][0:2],
                                  mesh[249][0:2],
                                  mesh[390][0:2],
                                  mesh[373][0:2],
                                  mesh[374][0:2],
                                  mesh[380][0:2],
                                  mesh[381][0:2],
                                  mesh[382][0:2],
                                  mesh[362][0:2],
                                  mesh[398][0:2],
                                  mesh[384][0:2],
                                  mesh[385][0:2],
                                  mesh[386][0:2],
                                  mesh[387][0:2],
                                  mesh[388][0:2],
                                  mesh[466][0:2]]).astype(np.int32)
    # Lips
    cv2.polylines(image, [left_eye_contour, right_eye_contour], False,
                  (255, 255, 255), line_width, cv2.LINE_AA)

def draw_landmarks(image, mesh):
    """
    在图像上绘制面部关键点和连接线
    1. 将归一化的关键点坐标转换为图像坐标
    2. 在每个关键点位置绘制圆点
    3. 连接特定关键点形成面部特征线（眉毛、眼睛、嘴巴等）
    """
    image_size = image.shape[0]
    mesh = mesh * image_size
    landmark_point = []
    for point in mesh:
        landmark_point.append((int(point[0]), int(point[1])))
        cv2.circle(image, (int(point[0]), int(point[1])), 2, (255, 255, 0), -1)
 
    if len(landmark_point) > 0:
 
 
        # 左眉毛(55：内側、46：外側)
        cv2.line(image, landmark_point[55], landmark_point[65], (0, 0, 255), 2, -3)
        cv2.line(image, landmark_point[65], landmark_point[52], (0, 0, 255), 2, -3)
        cv2.line(image, landmark_point[52], landmark_point[53], (0, 0, 255), 2, -3)
        cv2.line(image, landmark_point[53], landmark_point[46], (0, 0, 255), 2, -3)
 
        # 右眉毛(285：内側、276：外側)
        cv2.line(image, landmark_point[285], landmark_point[295], (0, 0, 255),
                 2)
        cv2.line(image, landmark_point[295], landmark_point[282], (0, 0, 255),
                 2)
        cv2.line(image, landmark_point[282], landmark_point[283], (0, 0, 255),
                 2)
        cv2.line(image, landmark_point[283], landmark_point[276], (0, 0, 255),
                 2)
 
        # 左目 (133：目頭、246：目尻)
        cv2.line(image, landmark_point[133], landmark_point[173], (0, 0, 255),
                 2)
        cv2.line(image, landmark_point[173], landmark_point[157], (0, 0, 255),
                 2)
        cv2.line(image, landmark_point[157], landmark_point[158], (0, 0, 255),
                 2)
        cv2.line(image, landmark_point[158], landmark_point[159], (0, 0, 255),
                 2)
        cv2.line(image, landmark_point[159], landmark_point[160], (0, 0, 255),
                 2)
        cv2.line(image, landmark_point[160], landmark_point[161], (0, 0, 255),
                 2)
        cv2.line(image, landmark_point[161], landmark_point[246], (0, 0, 255),
                 2)
 
        cv2.line(image, landmark_point[246], landmark_point[163], (0, 0, 255),
                 2)
        cv2.line(image, landmark_point[163], landmark_point[144], (0, 0, 255),
                 2)
        cv2.line(image, landmark_point[144], landmark_point[145], (0, 0, 255),
                 2)
        cv2.line(image, landmark_point[145], landmark_point[153], (0, 0, 255),
                 2)
        cv2.line(image, landmark_point[153], landmark_point[154], (0, 0, 255),
                 2)
        cv2.line(image, landmark_point[154], landmark_point[155], (0, 0, 255),
                 2)
        cv2.line(image, landmark_point[155], landmark_point[133], (0, 0, 255),
                 2)
 
        # 右目 (362：目頭、466：目尻)
        cv2.line(image, landmark_point[362], landmark_point[398], (0, 0, 255),
                 2)
        cv2.line(image, landmark_point[398], landmark_point[384], (0, 0, 255),
                 2)
        cv2.line(image, landmark_point[384], landmark_point[385], (0, 0, 255),
                 2)
        cv2.line(image, landmark_point[385], landmark_point[386], (0, 0, 255),
                 2)
        cv2.line(image, landmark_point[386], landmark_point[387], (0, 0, 255),
                 2)
        cv2.line(image, landmark_point[387], landmark_point[388], (0, 0, 255),
                 2)
        cv2.line(image, landmark_point[388], landmark_point[466], (0, 0, 255),
                 2)
 
        cv2.line(image, landmark_point[466], landmark_point[390], (0, 0, 255),
                 2)
        cv2.line(image, landmark_point[390], landmark_point[373], (0, 0, 255),
                 2)
        cv2.line(image, landmark_point[373], landmark_point[374], (0, 0, 255),
                 2)
        cv2.line(image, landmark_point[374], landmark_point[380], (0, 0, 255),
                 2)
        cv2.line(image, landmark_point[380], landmark_point[381], (0, 0, 255),
                 2)
        cv2.line(image, landmark_point[381], landmark_point[382], (0, 0, 255),
                 2)
        cv2.line(image, landmark_point[382], landmark_point[362], (0, 0, 255),
                 2)
 
        # 口 (308：右端、78：左端)
        cv2.line(image, landmark_point[308], landmark_point[415], (0, 0, 255),
                 2)
        cv2.line(image, landmark_point[415], landmark_point[310], (0, 0, 255),
                 2)
        cv2.line(image, landmark_point[310], landmark_point[311], (0, 0, 255),
                 2)
        cv2.line(image, landmark_point[311], landmark_point[312], (0, 0, 255),
                 2)
        cv2.line(image, landmark_point[312], landmark_point[13], (0, 0, 255), 2)
        cv2.line(image, landmark_point[13], landmark_point[82], (0, 0, 255), 2)
        cv2.line(image, landmark_point[82], landmark_point[81], (0, 0, 255), 2)
        cv2.line(image, landmark_point[81], landmark_point[80], (0, 0, 255), 2)
        cv2.line(image, landmark_point[80], landmark_point[191], (0, 0, 255), 2)
        cv2.line(image, landmark_point[191], landmark_point[78], (0, 0, 255), 2)
 
        cv2.line(image, landmark_point[78], landmark_point[95], (0, 0, 255), 2)
        cv2.line(image, landmark_point[95], landmark_point[88], (0, 0, 255), 2)
        cv2.line(image, landmark_point[88], landmark_point[178], (0, 0, 255), 2)
        cv2.line(image, landmark_point[178], landmark_point[87], (0, 0, 255), 2)
        cv2.line(image, landmark_point[87], landmark_point[14], (0, 0, 255), 2)
        cv2.line(image, landmark_point[14], landmark_point[317], (0, 0, 255), 2)
        cv2.line(image, landmark_point[317], landmark_point[402], (0, 0, 255),
                 2)
        cv2.line(image, landmark_point[402], landmark_point[318], (0, 0, 255),
                 2)
        cv2.line(image, landmark_point[318], landmark_point[324], (0, 0, 255),
                 2)
        cv2.line(image, landmark_point[324], landmark_point[308], (0, 0, 255),
                 2)
 
    return image
 
# 获取摄像头ID
def get_cap_id():
    """
    获取可用的USB摄像头ID
    1. 通过系统命令查询视频设备
    2. 筛选出USB摄像头
    3. 返回最小的可用摄像头ID
    """
    try:
        # 构造命令，使用awk处理输出
        cmd = "ls -l /sys/class/video4linux | awk -F ' -> ' '/usb/{sub(/.*video/, \"\", $2); print $2}'"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        output = result.stdout.strip().split()
 
        # 转换所有捕获的编号为整数，找出最小值
        video_numbers = list(map(int, output))
        if video_numbers:
            return min(video_numbers)
        else:
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
 
# 初始化人脸检测模型参数
inShape =[[1 , 128 , 128 ,3]]  # 输入张量形状
outShape= [[1 , 896,16],[1,896,1]]  # 输出张量形状
model_path="models/face_detection_front.tflite"  # 人脸检测模型路径
model_path2="models/face_landmark.tflite"  # 面部关键点识别模型路径
inShape2 =[[1 , 192 , 192 ,3]]  # 第二个模型的输入张量形状
outShape2= [[1,1404],[1]]  # 第二个模型的输出张量形状
 
# 初始化人脸检测模型
# 创建Model实例对象，并设置模型相关参数
model = aidlite.Model.create_instance(model_path)
if model is None:
    print("Create face_detection_front model failed !")
# 设置模型属性
model.set_model_properties(inShape, aidlite.DataType.TYPE_FLOAT32, outShape,aidlite.DataType.TYPE_FLOAT32)
# 创建Config实例对象，并设置配置信息
config = aidlite.Config.create_instance()
config.implement_type = aidlite.ImplementType.TYPE_FAST
config.framework_type = aidlite.FrameworkType.TYPE_TFLITE
config.accelerate_type = aidlite.AccelerateType.TYPE_CPU  # 使用CPU加速
config.number_of_threads = 4  # 使用4个线程
 
# 创建推理解释器对象
fast_interpreter = aidlite.InterpreterBuilder.build_interpretper_from_model_and_config(model, config)
if fast_interpreter is None:
    print("face_detection_front model build_interpretper_from_model_and_config failed !")
# 完成解释器初始化
result = fast_interpreter.init()
if result != 0:
    print("face_detection_front model interpreter init failed !")
# 加载模型
result = fast_interpreter.load_model()
if result != 0:
    print("face_detection_front model interpreter load face_detection_front model failed !")
print("face_detection_front model model load success!")
 
# 初始化面部关键点识别模型
# 创建Model实例对象，并设置模型相关参数
model2 = aidlite.Model.create_instance(model_path2)
if model2 is None:
    print("Create face_landmark model failed !")
# 设置模型参数
model2.set_model_properties(inShape2, aidlite.DataType.TYPE_FLOAT32, outShape2,aidlite.DataType.TYPE_FLOAT32)
# 创建Config实例对象，并设置配置信息
config2 = aidlite.Config.create_instance()
config2.implement_type = aidlite.ImplementType.TYPE_FAST
config2.framework_type = aidlite.FrameworkType.TYPE_TFLITE
config2.accelerate_type = aidlite.AccelerateType.TYPE_GPU  # 使用GPU加速
config2.number_of_threads = 4  # 使用4个线程
 
# 创建推理解释器对象
fast_interpreter2 = aidlite.InterpreterBuilder.build_interpretper_from_model_and_config(model2, config2)
if fast_interpreter2 is None:
    print("face_landmark model build_interpretper_from_model_and_config failed !")
# 完成解释器初始化
result = fast_interpreter2.init()
if result != 0:
    print("face_landmark model interpreter init failed !")
# 加载模型
result = fast_interpreter2.load_model()
if result != 0:
    print("face_landmark model interpreter load model failed !")
print("face_landmark model load success!")
 
# 加载人脸检测模型的锚点数据
anchors = np.load('models/anchors.npy').astype(np.float32)
aidlux_type="root"  # Aidlux平台类型
# 0-后置，1-前置
camId = 1  # 默认使用前置摄像头
opened = False



class FaceRecognitionNode(Node):
    def __init__(self):
        super().__init__('face_recognition_node')
        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.publisher = self.create_publisher(
            Bool, '/face_recognition', 10)
        self.bridge = CvBridge()
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.get_logger().info('Face Recognition Node has started')

        

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        found_face = len(faces) > 0
        self.publisher.publish(Bool(data=found_face))
        if found_face:
            self.get_logger().info(f'Face detected: {len(faces)} face(s)')

def main(args=None):
    rclpy.init(args=args)
    node = FaceRecognitionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()