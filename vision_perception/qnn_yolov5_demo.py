# import sys
import os
import sys
import time
from ament_index_python.packages import get_package_share_directory
import aidlite
import aidcv as cv2
import numpy as np

OBJ_CLASS_NUM = 80
NMS_THRESH = 0.45
BOX_THRESH = 0.5
MODEL_SIZE = 640

OBJ_NUMB_MAX_SIZE = 64
PROP_BOX_SIZE = 5 + OBJ_CLASS_NUM
STRIDE8_SIZE = MODEL_SIZE / 8
STRIDE16_SIZE = MODEL_SIZE / 16
STRIDE32_SIZE = MODEL_SIZE / 32

anchors = [
    [10, 13, 16, 30, 33, 23],
    [30, 61, 62, 45, 59, 119],
    [116, 90, 156, 198, 373, 326],
]

coco_class = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

# coco_class = [
#     "person",
# ]


def eqprocess(image, size1, size2):
    h, w, _ = image.shape
    mask = np.zeros((size1, size2, 3), dtype=np.float32)
    scale1 = h / size1
    scale2 = w / size2
    if scale1 > scale2:
        scale = scale1
    else:
        scale = scale2
    img = cv2.resize(image, (int(w / scale), int(h / scale)))
    mask[: int(h / scale), : int(w / scale), :] = img
    return mask, scale


def xywh2xyxy(x):
    """
    Box (center x, center y, width, height) to (x1, y1, x2, y2)
    """
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def xyxy2xywh(box):
    """
    Box (left_top x, left_top y, right_bottom x, right_bottom y) to (left_top x, left_top y, width, height)
    """
    box[:, 2:] = box[:, 2:] - box[:, :2]
    return box


def NMS(dets, scores, thresh):
    """
    单类NMS算法
    dets.shape = (N, 5), (left_top x, left_top y, right_bottom x, right_bottom y, Scores)
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    keep = []
    index = scores.argsort()[::-1]
    while index.size > 0:
        i = index[0]  # every time the first is the biggst, and add it directly
        keep.append(i)
        x11 = np.maximum(x1[i], x1[index[1:]])  # calculate the points of overlap
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])
        w = np.maximum(0, x22 - x11 + 1)  # the weights of overlap
        h = np.maximum(0, y22 - y11 + 1)  # the height of overlap
        overlaps = w * h
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        idx = np.where(ious <= thresh)[0]
        index = index[idx + 1]  # because index start from 1

    return keep


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clip(0, img_shape[1], out=boxes[:, 0])  # x1
    boxes[:, 1].clip(0, img_shape[0], out=boxes[:, 1])  # y1
    boxes[:, 2].clip(0, img_shape[1], out=boxes[:, 2])  # x2
    boxes[:, 3].clip(0, img_shape[0], out=boxes[:, 3])  # y2


def detect_postprocess(
    prediction, img0shape, img1shape, conf_thres=0.25, iou_thres=0.45
):
    """
    检测输出后处理
    prediction: aidlite模型预测输出
    img0shape: 原始图片shape
    img1shape: 输入图片shape
    conf_thres: 置信度阈值
    iou_thres: IOU阈值
    return: list[np.ndarray(N, 5)], 对应类别的坐标框信息, xywh、conf
    """
    h, w, _ = img1shape
    valid_condidates = prediction[prediction[..., 4] > conf_thres]
    valid_condidates[:, 5:] *= valid_condidates[:, 4:5]
    valid_condidates[:, :4] = xywh2xyxy(valid_condidates[:, :4])

    max_det = 300
    max_wh = 7680
    max_nms = 30000
    valid_condidates[:, 4] = valid_condidates[:, 5:].max(1)
    valid_condidates[:, 5] = valid_condidates[:, 5:].argmax(1)
    sort_id = np.argsort(valid_condidates[:, 4])[::-1]
    valid_condidates = valid_condidates[sort_id[:max_nms]]
    boxes, scores = (
        valid_condidates[:, :4] + valid_condidates[:, 5:6] * max_wh,
        valid_condidates[:, 4],
    )
    index = NMS(boxes, scores, iou_thres)[:max_det]
    out_boxes = valid_condidates[index]
    clip_coords(out_boxes[:, :4], img0shape)
    out_boxes[:, :4] = xyxy2xywh(out_boxes[:, :4])
    # print("检测到{}个区域".format(len(out_boxes)))
    return out_boxes


def draw_detect_res(img, det_pred):
    """
    检测结果绘制
    """
    img = img.astype(np.uint8)
    color_step = int(255 / len(coco_class))
    for i in range(len(det_pred)):
        x1, y1, x2, y2 = [int(t) for t in det_pred[i][:4]]
        score = det_pred[i][4]
        cls_id = int(det_pred[i][5])

        print(i + 1, [x1, y1, x2 + x1, y2 + y1], score, coco_class[cls_id])

        cv2.putText(
            img,
            f"{coco_class[cls_id]}",
            (x1, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv2.rectangle(
            img,
            (x1, y1),
            (x2 + x1, y2 + y1),
            (0, int(cls_id * color_step), int(255 - cls_id * color_step)),
            thickness=2,
        )

    return img


class Detect:
    # YOLOv5 Detect head for detection models
    def __init__(self, nc=80, anchors=(), stride=[], image_size=640):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.stride = stride
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid, self.anchor_grid = [0] * self.nl, [0] * self.nl
        self.anchors = np.array(anchors, dtype=np.float32).reshape(self.nl, -1, 2)

        base_scale = image_size // 8
        for i in range(self.nl):
            self.grid[i], self.anchor_grid[i] = self._make_grid(
                base_scale // (2**i), base_scale // (2**i), i
            )

    def _make_grid(self, nx=20, ny=20, i=0):
        y, x = np.arange(ny, dtype=np.float32), np.arange(nx, dtype=np.float32)
        yv, xv = np.meshgrid(y, x)
        yv, xv = yv.T, xv.T
        # add grid offset, i.e. y = 2.0 * x - 0.5
        grid = np.stack((xv, yv), 2)
        grid = grid[np.newaxis, np.newaxis, ...]
        grid = np.repeat(grid, self.na, axis=1) - 0.5
        anchor_grid = self.anchors[i].reshape((1, self.na, 1, 1, 2))
        anchor_grid = np.repeat(anchor_grid, repeats=ny, axis=2)
        anchor_grid = np.repeat(anchor_grid, repeats=nx, axis=3)
        return grid, anchor_grid

    def sigmoid(self, arr):
        return 1 / (1 + np.exp(-arr))

    def __call__(self, x):
        z = []  # inference output
        for i in range(self.nl):
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].reshape(bs, self.na, self.no, ny, nx).transpose(0, 1, 3, 4, 2)
            y = self.sigmoid(x[i])
            y[..., 0:2] = (y[..., 0:2] * 2.0 + self.grid[i]) * self.stride[i]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
            z.append(y.reshape(bs, self.na * nx * ny, self.no))

        return np.concatenate(z, 1)


def main():
    acc_type = 3
    resource_dir = get_package_share_directory('vision_perception')
    config = aidlite.Config.create_instance()
    if config is None:
        print("Create config failed !")
        return False
    model_name = "cutoff_yolov5s_sigmoid_w8a8.qnn229.ctx.bin"
    if acc_type == 1:
        config.accelerate_type = aidlite.AccelerateType.TYPE_CPU
    elif acc_type == 2:
        config.accelerate_type = aidlite.AccelerateType.TYPE_GPU
        config.is_quantify_model = 0
    elif acc_type == 3:
        config.accelerate_type = aidlite.AccelerateType.TYPE_DSP
        config.is_quantify_model = 1
    else:
        return False

    config.implement_type = aidlite.ImplementType.TYPE_LOCAL
    config.framework_type = aidlite.FrameworkType.TYPE_QNN229
    model_path = os.path.join(resource_dir, model_name)
    model = aidlite.Model.create_instance(model_path)
    if model is None:
        print("Create model failed !")
        return False
    input_shapes = [[1, MODEL_SIZE, MODEL_SIZE, 3]]
    output_shapes = [[1, 20, 20, 255], [1, 40, 40, 255], [1, 80, 80, 255]]
    model.set_model_properties(
        input_shapes,
        aidlite.DataType.TYPE_FLOAT32,
        output_shapes,
        aidlite.DataType.TYPE_FLOAT32,
    )

    interpreter = aidlite.InterpreterBuilder.build_interpretper_from_model_and_config(
        model, config
    )
    if interpreter is None:
        print("build_interpretper_from_model_and_config failed !")
        return None
    result = interpreter.init()
    if result != 0:
        print("interpreter init failed !")
        return False
    result = interpreter.load_model()
    if result != 0:
        print("interpreter load model failed !")
        return False
    # print("detect model load success!")

    input_tensor_info = interpreter.get_input_tensor_info()
    if len(input_tensor_info) == 0 :
        print("interpreter get_input_tensor_info() failed !\n")
        return False
    output_tensor_info = interpreter.get_output_tensor_info()
    if len(output_tensor_info) == 0 :
        print("interpreter get_output_tensor_info() failed !\n")
        return False
    stride8 = stride16 = stride32 = None

    image_path = os.path.join(resource_dir, "test1.jpg")
    frame = cv2.imread(image_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_input, scale = eqprocess(frame, MODEL_SIZE, MODEL_SIZE)
    img_input = img_input / 255
    img_input = img_input.astype(np.float32)

    sum_time_0 = 0.0
    sum_time_1 = 0.0
    sum_time_2 = 0.0
    _counter = 10
    for idx in range(_counter):
        st0 = time.time()
        input_tensor_data = img_input.data
        # result = interpreter.set_input_tensor(0, input_tensor_data)
        result = interpreter.set_input_tensor("images", input_tensor_data)
        if result != 0:
            print("interpreter set_input_tensor() failed")
            return False
        et0 = time.time()
        dur0 = et0 - st0
        sum_time_0 += dur0
        # print(f"current [{idx}] set_input_tensor cost time :{dur0} ms")

        st1 = time.time()
        result = interpreter.invoke()
        if result != 0:
            # print("interpreter set_input_tensor() failed")
            return False
        et1 = time.time()
        dur1 = et1 - st1
        sum_time_1 += dur1
        # print(f"current [{idx}] invoke cost time :{dur1} ms")

        st2 = time.time()

        # stride8 = interpreter.get_output_tensor(0)
        stride8 = interpreter.get_output_tensor("_326")
        if stride8 is None:
            # print("sample : interpreter->get_output_tensor() 0 failed !")
            return False
        # print(f"len(stride8 {len(stride8)}")

        # stride16 = interpreter.get_output_tensor(1)
        stride16 = interpreter.get_output_tensor("_364")
        if stride16 is None:
            # print("sample : interpreter->get_output_tensor() 1 failed !")
            return False
        # print(f"len(stride16 {len(stride16)}")

        # stride32 = interpreter.get_output_tensor(2)
        stride32 = interpreter.get_output_tensor("_402")
        if stride32 is None:
            # print("sample : interpreter->get_output_tensor() 2 failed !")
            return False
        # print(f"len(stride32 {len(stride32)}")
        et2 = time.time()
        dur2 = et2 - st2
        sum_time_2 += dur2
        # print(f"current [{idx}] get_output_tensor cost time :{dur2} ms")

    print(
        f"repeat [{_counter}] times , input[{sum_time_0 * 1000}]ms --- invoke[{sum_time_1 * 1000}]ms --- output[{sum_time_2 * 1000}]ms --- sum[{(sum_time_0 + sum_time_1 + sum_time_2) * 1000}]ms"
    )

    stride = [8, 16, 32]
    yolo_head = Detect(OBJ_CLASS_NUM, anchors, stride, MODEL_SIZE)
    # 后处理部分reshape需要知道模型的output_shapes
    validCount0 = stride8.reshape(*output_shapes[2]).transpose(0, 3, 1, 2)
    validCount1 = stride16.reshape(*output_shapes[1]).transpose(0, 3, 1, 2)
    validCount2 = stride32.reshape(*output_shapes[0]).transpose(0, 3, 1, 2)
    pred = yolo_head([validCount0, validCount1, validCount2])
    det_pred = detect_postprocess(
        pred, frame.shape, [MODEL_SIZE, MODEL_SIZE, 3], conf_thres=0.5, iou_thres=0.45
    )


    # 取出所有person
    person_det = det_pred[det_pred[:, 5] == 0]
    det_pred = person_det
    # 输出置信度最高的中心像素坐标（改为：面积最大的框）
    if len(person_det) > 0:
        # 计算每个框的面积
        x1, y1, x2, y2 = person_det[:, 0], person_det[:, 1], person_det[:, 2], person_det[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        # 找到面积最大的框的索引
        best_index = np.argmax(areas)
        # 取出面积最大的person框
        best_person = person_det[[best_index]] # 使用 [[index]] 保持二维数组结构
        det_pred = best_person
        # 提取坐标并计算中心点
        x1, y1, x2, y2 = best_person[0, :4].astype(int)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        print(f"Person 中心坐标: ({center_x}, {center_y})")
    else:
         print("未检测到 person")



        
    det_pred[np.isnan(det_pred)] = 0.0
    det_pred[:, :4] = det_pred[:, :4] * scale
    res_img = draw_detect_res(frame, det_pred)

    result = interpreter.destory()
    if result != 0:
        print("interpreter set_input_tensor() failed")
        return False
    frame_bgr = cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR)
    # result_img_path = f"{os.path.splitext(os.path.abspath(__file__))[0]}.jpg"
    result_img_path = f"/home/aidlux/aidcode/result.jpg"
    cv2.imwrite(result_img_path, frame_bgr)
    print(f"The result image has been saved to : {result_img_path}")
    return True

def detect_API(frame):
    acc_type = 3
    # acc_type 1 cpu 2 gpu 3 npu
    aidlite.set_log_level(aidlite.LogLevel.INFO)
    aidlite.log_to_stderr()

    resource_dir = "../"

    config = aidlite.Config.create_instance()
    if config is None:
        print("Create config failed !")
        return False

    model_name = "cutoff_yolov5s_sigmoid_w8a8.qnn229.ctx.bin"
    if acc_type == 1:
        config.accelerate_type = aidlite.AccelerateType.TYPE_CPU
    elif acc_type == 2:
        config.accelerate_type = aidlite.AccelerateType.TYPE_GPU
        config.is_quantify_model = 0
    elif acc_type == 3:
        config.accelerate_type = aidlite.AccelerateType.TYPE_DSP
        config.is_quantify_model = 1
    else:
        return False

    config.implement_type = aidlite.ImplementType.TYPE_LOCAL
    config.framework_type = aidlite.FrameworkType.TYPE_QNN229
    model_path = os.path.join(resource_dir, model_name)
    model = aidlite.Model.create_instance(model_path)
    if model is None:
        print("Create model failed !")
        return False
    input_shapes = [[1, MODEL_SIZE, MODEL_SIZE, 3]]
    output_shapes = [[1, 20, 20, 255], [1, 40, 40, 255], [1, 80, 80, 255]]
    model.set_model_properties(
        input_shapes,
        aidlite.DataType.TYPE_FLOAT32,
        output_shapes,
        aidlite.DataType.TYPE_FLOAT32,
    )

    interpreter = aidlite.InterpreterBuilder.build_interpretper_from_model_and_config(
        model, config
    )
    if interpreter is None:
        print("build_interpretper_from_model_and_config failed !")
        return None
    result = interpreter.init()
    if result != 0:
        print("interpreter init failed !")
        return False
    result = interpreter.load_model()
    if result != 0:
        print("interpreter load model failed !")
        return False
    print("detect model load success!")

    input_tensor_info = interpreter.get_input_tensor_info()
    if len(input_tensor_info) == 0 :
        print("interpreter get_input_tensor_info() failed !\n")
        return False
    for gi, graph_tensor_info in enumerate(input_tensor_info):
        for ti, tensor_info in enumerate(graph_tensor_info):
            print(  f"Input  tensor : Graph[{gi}]-Tensor[{ti}]-name[{tensor_info.name}]"
                    f"-element_count[{tensor_info.element_count}]-element_type[{tensor_info.element_type}]"
                    f"-dimensions[{tensor_info.dimensions}]-shape{tensor_info.shape}")

    output_tensor_info = interpreter.get_output_tensor_info()
    if len(output_tensor_info) == 0 :
        print("interpreter get_output_tensor_info() failed !\n")
        return False
    for gi, graph_tensor_info in enumerate(output_tensor_info):
        for ti, tensor_info in enumerate(graph_tensor_info):
            print(  f"Output tensor : Graph[{gi}]-Tensor[{ti}]-name[{tensor_info.name}]"
                    f"-element_count[{tensor_info.element_count}]-element_type[{tensor_info.element_type}]"
                    f"-dimensions[{tensor_info.dimensions}]-shape{tensor_info.shape}")

    stride8 = stride16 = stride32 = None

    image_path = os.path.join(resource_dir, "test1.jpg")
    frame = cv2.imread(image_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_input, scale = eqprocess(frame, MODEL_SIZE, MODEL_SIZE)
    img_input = img_input / 255
    img_input = img_input.astype(np.float32)

    sum_time_0 = 0.0
    sum_time_1 = 0.0
    sum_time_2 = 0.0
    _counter = 10
    for idx in range(_counter):
        st0 = time.time()
        input_tensor_data = img_input.data
        # result = interpreter.set_input_tensor(0, input_tensor_data)
        result = interpreter.set_input_tensor("images", input_tensor_data)
        if result != 0:
            print("interpreter set_input_tensor() failed")
            return False
        et0 = time.time()
        dur0 = et0 - st0
        sum_time_0 += dur0
        print(f"current [{idx}] set_input_tensor cost time :{dur0} ms")

        st1 = time.time()
        result = interpreter.invoke()
        if result != 0:
            print("interpreter set_input_tensor() failed")
            return False
        et1 = time.time()
        dur1 = et1 - st1
        sum_time_1 += dur1
        print(f"current [{idx}] invoke cost time :{dur1} ms")

        st2 = time.time()

        # stride8 = interpreter.get_output_tensor(0)
        stride8 = interpreter.get_output_tensor("_326")
        if stride8 is None:
            # print("sample : interpreter->get_output_tensor() 0 failed !")
            return False
        # print(f"len(stride8 {len(stride8)}")

        # stride16 = interpreter.get_output_tensor(1)
        stride16 = interpreter.get_output_tensor("_364")
        if stride16 is None:
            # print("sample : interpreter->get_output_tensor() 1 failed !")
            return False
        # print(f"len(stride16 {len(stride16)}")

        # stride32 = interpreter.get_output_tensor(2)
        stride32 = interpreter.get_output_tensor("_402")
        if stride32 is None:
            # print("sample : interpreter->get_output_tensor() 2 failed !")
            return False
        # print(f"len(stride32 {len(stride32)}")
        et2 = time.time()
        dur2 = et2 - st2
        sum_time_2 += dur2
        # print(f"current [{idx}] get_output_tensor cost time :{dur2} ms")

    # print(
    #     f"repeat [{_counter}] times , input[{sum_time_0 * 1000}]ms --- invoke[{sum_time_1 * 1000}]ms --- output[{sum_time_2 * 1000}]ms --- sum[{(sum_time_0 + sum_time_1 + sum_time_2) * 1000}]ms"
    # )

    stride = [8, 16, 32]
    yolo_head = Detect(OBJ_CLASS_NUM, anchors, stride, MODEL_SIZE)
    # 后处理部分reshape需要知道模型的output_shapes
    validCount0 = stride8.reshape(*output_shapes[2]).transpose(0, 3, 1, 2)
    validCount1 = stride16.reshape(*output_shapes[1]).transpose(0, 3, 1, 2)
    validCount2 = stride32.reshape(*output_shapes[0]).transpose(0, 3, 1, 2)
    pred = yolo_head([validCount0, validCount1, validCount2])
    det_pred = detect_postprocess(
        pred, frame.shape, [MODEL_SIZE, MODEL_SIZE, 3], conf_thres=0.5, iou_thres=0.45
    )
    det_pred[np.isnan(det_pred)] = 0.0
    det_pred[:, :4] = det_pred[:, :4] * scale
    res_img = draw_detect_res(frame, det_pred)
    result = interpreter.destory()
    if result != 0:
        print("interpreter set_input_tensor() failed")
        return False
    # frame_bgr = cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR)
    # result_img_path = f"/home/aidlux/aidcode/result.jpg"
    # cv2.imwrite(result_img_path, frame_bgr)
    # print(f"The result image has been saved to : {result_img_path}")
    return True


if __name__ == "__main__":
    main()
