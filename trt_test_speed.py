import ctypes
import time
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np
import tensorrt as trt
import torch
import torch.nn.functional as F


def load_engine(engine_path):
    logger = trt.Logger(trt.Logger.ERROR)
    with open(engine_path, "rb") as f:
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError("Failed to load engine")
    return engine


# Load CUDA Runtime
cudart = ctypes.CDLL("libcudart.so")

# -------------------------
# CUDA error codes
# -------------------------
cudaSuccess = 0

# -------------------------
# cudaMemcpyKind enum (MANUAL!)
# -------------------------
cudaMemcpyHostToHost = 0
cudaMemcpyHostToDevice = 1
cudaMemcpyDeviceToHost = 2
cudaMemcpyDeviceToDevice = 3
cudaMemcpyDefault = 4

# -------------------------
# Function signatures
# -------------------------

# cudaMalloc(void **devPtr, size_t size)
cudart.cudaMalloc.restype = ctypes.c_int
cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]

# cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind)
cudart.cudaMemcpy.restype = ctypes.c_int
cudart.cudaMemcpy.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
]


# ============================================================
#                  UPDATED CONFIG WITH CONSTANTS
# ============================================================
@dataclass
class Config:
    # Image size, to be set after engine loads
    train_height: Optional[int] = None
    train_width: Optional[int] = None

    # Original image size
    original_image_height: int = 1080
    original_image_width: int = 1920

    n: int = 800
    mean: np.ndarray = field(
        default_factory=lambda: np.array([0.485, 0.456, 0.406], dtype=np.float32)
    )
    std: np.ndarray = field(
        default_factory=lambda: np.array([0.229, 0.224, 0.225], dtype=np.float32)
    )

    last_n_rows: int = 750
    black_bar_ratio: float = 0.5

    # Output shapes
    num_grid_row: Optional[int] = None
    num_cls_row: Optional[int] = None
    num_lane_row: Optional[int] = None
    num_grid_col: Optional[int] = None
    num_cls_col: Optional[int] = None
    num_lane_col: Optional[int] = None

    # Lane indices
    row_lane_idx: List[int] = field(default_factory=lambda: [1, 2])
    col_lane_idx: List[int] = field(default_factory=lambda: [0, 3])

    # ------------------------------
    # ADDED CONSTANTS HERE
    # ------------------------------
    K: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                1.595670753182030012e03,
                0,
                9.549628573763222903e02,
                0,
                1.602230802012541062e03,
                5.384237247698257534e02,
                0,
                0,
                1,
            ],
            dtype=np.float32,
        ).reshape(3, 3)
    )

    D: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                -3.623766942436448812e-01,
                1.749287771660668622e-01,
                -1.563359095306956397e-03,
                6.625874294289102549e-04,
                -5.333212038737372013e-02,
            ],
            dtype=np.float32,
        )
    )

    H: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                -9.699118924366116890e-01,
                -6.692142227856339609e00,
                2.394451484153518322e03,
                -1.127449670934883574e-01,
                -1.488161138959835261e00,
                -1.236693230365772251e03,
                2.404776845372572024e-05,
                -6.943593727037860111e-03,
                1.300326227112413413e00,
            ],
            dtype=np.float32,
        ).reshape(3, 3)
    )

    H_inv: Optional[np.ndarray] = None  # will be computed in __post_init__

    # VIDEO CONSTANTS
    N_frames_median: int = 15
    N_pixels_mean: int = 100
    Center_threshold_pixels: int = 50

    def __post_init__(self):
        self.H_inv = np.linalg.inv(self.H)

    def set_output_shapes(
        self, loc_row_shape, loc_col_shape, exist_row_shape, exist_col_shape
    ):
        # Set loc output shapes
        self.num_grid_row, self.num_cls_row, self.num_lane_row = loc_row_shape[1:]
        self.num_grid_col, self.num_cls_col, self.num_lane_col = loc_col_shape[1:]

        # Validate lane dimensions match
        assert self.num_lane_row == self.num_lane_col, (
            "num_lane_row must equal num_lane_col"
        )

        # Validate exist outputs
        assert exist_row_shape[2] == self.num_cls_row, "exist_row num_cls mismatch"
        assert exist_row_shape[3] == self.num_lane_row, "exist_row num_lane mismatch"
        assert exist_col_shape[2] == self.num_cls_col, "exist_col num_cls mismatch"
        assert exist_col_shape[3] == self.num_lane_col, "exist_col num_lane mismatch"

        # Optional: validate batch and 2 channels dimension
        assert exist_row_shape[1] == 2, "exist_row second dimension must be 2"
        assert exist_col_shape[1] == 2, "exist_col second dimension must be 2"


class TRTModel:
    def __init__(self, engine_path, cfg: Config):

        self.engine = load_engine(engine_path)
        self.context = self.engine.create_execution_context()

        # -------------------------------------------
        # Enumerate I/O tensors (TRT 10 API)
        # -------------------------------------------
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.bindings_dict = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.inputs.append(name)
            else:
                self.outputs.append(name)

        self.input_name = self.inputs[0]
        self.output_names = self.outputs

        # ----------------------
        # Input buffer allocation
        # ----------------------
        self.input_dtype = trt.nptype(self.engine.get_tensor_dtype(self.input_name))
        self.input_shape = self.engine.get_tensor_shape(self.input_name)
        self.input_nbytes = (
            np.prod(self.input_shape) * np.dtype(self.input_dtype).itemsize
        )

        # Allocate GPU input buffer
        self.d_input = ctypes.c_void_p()
        err = cudart.cudaMalloc(
            ctypes.byref(self.d_input), ctypes.c_size_t(self.input_nbytes)
        )
        self.bindings.append(self.d_input.value)
        self.bindings_dict[self.input_name] = self.d_input.value
        if err != 0:
            raise RuntimeError(f"cudaMalloc(input) failed with error code {err}")

        # --------------------------
        # Output buffer allocations
        # --------------------------
        self.output_dptrs = {}
        self.output_hbufs = {}

        for out_name in self.output_names:
            shape = self.engine.get_tensor_shape(out_name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(out_name))
            nbytes = np.prod(shape) * np.dtype(dtype).itemsize

            # Host buffer
            self.output_hbufs[out_name] = np.empty(shape, dtype=dtype)

            # Device buffer
            dptr = ctypes.c_void_p()
            err = cudart.cudaMalloc(ctypes.byref(dptr), ctypes.c_size_t(nbytes))
            if err != 0:
                raise RuntimeError(
                    f"cudaMalloc(output {out_name}) failed with error code {err}"
                )

            self.output_dptrs[out_name] = (dptr, nbytes)
            self.bindings.append(dptr.value)
            self.bindings_dict[out_name] = dptr.value
        # Create CUDA stream
        self.stream = ctypes.c_void_p()
        cudart.cudaStreamCreate(ctypes.byref(self.stream))
        # -----------------------------
        # Fill config from tensor shapes
        # -----------------------------
        # Input H/W
        cfg.train_height = self.input_shape[2]  # H
        cfg.train_width = self.input_shape[3]  # W

        # Output shapes
        loc_row_shape = self.engine.get_tensor_shape("loc_row")
        loc_col_shape = self.engine.get_tensor_shape("loc_col")
        exist_row_shape = self.engine.get_tensor_shape("exist_row")
        exist_col_shape = self.engine.get_tensor_shape("exist_col")
        cfg.set_output_shapes(
            loc_row_shape, loc_col_shape, exist_row_shape, exist_col_shape
        )

    def infer(self, inp):

        # -------------------------------
        # Copy input host → device
        # -------------------------------
        res = cudart.cudaMemcpy(
            ctypes.c_void_p(self.d_input.value),  # dst
            ctypes.c_void_p(inp.ctypes.data),  # src
            ctypes.c_size_t(self.input_nbytes),
            1,  # cudaMemcpyHostToDevice = 1
        )
        if res != 0:
            raise RuntimeError(f"cudaMemcpy H2D failed with code {res}")

        # -------------------------------
        # Set input tensor address
        # -------------------------------
        self.context.set_tensor_address(self.input_name, self.d_input.value)

        # -------------------------------
        # Set output tensor device pointers
        # -------------------------------
        for name, (dptr, _) in self.output_dptrs.items():
            self.context.set_tensor_address(name, dptr.value)

        # -------------------------------
        # Run the inference
        # -------------------------------
        # Inference
        start_time = time.perf_counter()
        self.context.execute_v2(self.bindings)
        # self.context.execute_async_v3(self.stream.value)

        # Wait for inference to complete
        res = cudart.cudaStreamSynchronize(self.stream)
        if res != 0:
            raise RuntimeError(f"cudaStreamSynchronize failed with code {res}")
        end_time = time.perf_counter()

        # Compute and print FPS
        latency = end_time - start_time
        fps = 1.0 / latency if latency > 0 else 0
        # print(f"Inference time: {latency*1000:.2f} ms | FPS: {fps:.2f}")

        # -------------------------------
        # Copy device → host for outputs
        # -------------------------------
        for name, (dptr, nbytes) in self.output_dptrs.items():
            hbuf = self.output_hbufs[name]

            res = cudart.cudaMemcpy(
                ctypes.c_void_p(hbuf.ctypes.data),  # dst
                ctypes.c_void_p(dptr.value),  # src
                ctypes.c_size_t(nbytes),
                2,  # cudaMemcpyDeviceToHost = 2
            )
            if res != 0:
                raise RuntimeError(f"cudaMemcpy D2H failed for {name} with code {res}")

        return self.output_hbufs

    def infer_torch(self, inp: torch.Tensor):
        """
        inp: 1xCxHxW torch.Tensor on CUDA
        """
        # Ensure contiguous
        inp = inp.contiguous()
        # Get device pointer
        ptr = inp.data_ptr()

        # Copy tensor memory to TRT input
        res = cudart.cudaMemcpy(
            ctypes.c_void_p(self.d_input.value),
            ctypes.c_void_p(ptr),
            ctypes.c_size_t(self.input_nbytes),
            cudaMemcpyDeviceToDevice,  # 3
        )
        if res != 0:
            raise RuntimeError(f"cudaMemcpy D2D failed with code {res}")

        # Execute as before
        self.context.execute_v2(self.bindings)
        cudart.cudaStreamSynchronize(self.stream)

        # Copy outputs back to CPU as needed
        outputs = {}
        for name, (dptr, nbytes) in self.output_dptrs.items():
            hbuf = self.output_hbufs[name]
            res = cudart.cudaMemcpy(
                ctypes.c_void_p(hbuf.ctypes.data),
                ctypes.c_void_p(dptr.value),
                ctypes.c_size_t(nbytes),
                cudaMemcpyDeviceToHost,
            )
            outputs[name] = hbuf
        return outputs


def preprocess_torch(frame: np.ndarray, cfg):
    """
    GPU-accelerated preprocessing using PyTorch.
    frame: HxWxC uint8 numpy array
    Returns: 1xCxHxW float32/float16 tensor on GPU
    """

    device = torch.device("cuda")

    frame = frame[-cfg.last_n_rows :, :, :]  # (H_crop, W, C)
    frame_torch = torch.from_numpy(frame).pin_memory()  # pinned CPU memory
    # Convert to float tensor and move to GPU
    img = frame_torch.to(device, non_blocking=True, dtype=torch.float32)  # H x W x C
    img = img / 255.0  # Normalize 0-1

    # Crop last N rows

    # Compute resize dimensions
    black_bar_size = int(cfg.train_width * cfg.black_bar_ratio)
    resize_width = cfg.train_width - black_bar_size
    resize_height = cfg.train_height

    # Permute to CxHxW for interpolate
    img = img.permute(2, 0, 1).unsqueeze(0)  # 1 x C x H x W

    # Resize with bilinear interpolation
    img = F.interpolate(
        img, size=(resize_height, resize_width), mode="bilinear", align_corners=False
    )

    # Pad black bars (left/right)
    pad_left = pad_right = black_bar_size // 2
    img = F.pad(img, (pad_left, pad_right, 0, 0), mode="constant", value=0.0)

    # Normalize
    mean = torch.from_numpy(cfg.mean).to(device).view(1, -1, 1, 1)
    std = torch.from_numpy(cfg.std).to(device).view(1, -1, 1, 1)
    img = (img - mean) / std

    # Already CxHxW and contiguous
    return img


def preprocess(frame, cfg: Config):
    a = frame[-cfg.last_n_rows :, :, :]
    black_bar_size = int(cfg.train_width * cfg.black_bar_ratio)
    b = cv2.resize(a, (cfg.train_width - black_bar_size, cfg.original_image_height))
    c = cv2.copyMakeBorder(
        b,
        0,
        0,
        black_bar_size // 2,
        black_bar_size // 2,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )

    img = c.astype(np.float32) / 255.0  # If still 0–255

    img = cv2.subtract(img, cfg.mean)
    img = cv2.divide(img, cfg.std)
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    return np.expand_dims(img, 0)


def distort_points(pts, K, D):
    """
    pts: Nx2 undistorted pixel coords (after inverse homography)
    K: camera matrix
    D: distortion [k1, k2, p1, p2, k3]
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    k1, k2, p1, p2, k3 = D.ravel()

    # Convert to normalized
    x = (pts[:, 0] - cx) / fx
    y = (pts[:, 1] - cy) / fy

    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2

    radial = 1 + k1 * r2 + k2 * r4 + k3 * r6

    x_t = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    y_t = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y

    x_d = x * radial + x_t
    y_d = y * radial + y_t

    u = fx * x_d + cx
    v = fy * y_d + cy

    return np.vstack([u, v]).T


def pred2coords(pred, cfg: Config):
    delta = cfg.last_n_rows / cfg.num_cls_row

    coords = []
    logits = pred["loc_row"][..., cfg.row_lane_idx]
    valid = pred["exist_row"][..., cfg.row_lane_idx].argmax(0).astype(np.bool_)
    indecies = np.arange(cfg.num_grid_row)[:, np.newaxis, np.newaxis]
    exp_logits = np.exp(logits)
    x = np.sum(exp_logits / np.sum(exp_logits, 0) * indecies, 0)
    x = (
        (x - cfg.num_grid_row * cfg.black_bar_ratio / 2)
        / ((cfg.num_grid_row - 1) * (1 - cfg.black_bar_ratio))
        * cfg.original_image_width
    )
    yi = np.linspace(
        cfg.original_image_height - cfg.last_n_rows + delta / 2,
        cfg.original_image_height - delta / 2,
        cfg.num_cls_row,
    )

    coords = [
        list(np.stack([xi[v], yi[v]], axis=1).astype(np.int32))
        for v, xi in zip(valid.T, x.T)
    ]

    return coords


def main():
    cfg = Config()

    engine_path = "resources/culane_res34.engine"
    print("Loading TensorRT engine:", engine_path)
    model = TRTModel(engine_path, cfg)

    image_center = cv2.perspectiveTransform(
        np.array(
            [[(cfg.original_image_width / 2, cfg.original_image_height / 2)]],
            dtype=np.float32,
        ),
        cfg.H_inv,
    ).reshape(-1, 2)

    image_center = distort_points(image_center, cfg.K, cfg.D)[0]

    cap = cv2.VideoCapture("centar_grada_kraci.mp4")

    i = 0

    center_left_buffer = []
    center_right_buffer = []

    while True:
        ret, frame = cap.read()
        start_time = time.perf_counter()
        if not ret:
            break

        frame = cv2.resize(frame, (cfg.original_image_width, cfg.original_image_height))
        # inp = preprocess(frame, cfg)
        inp = preprocess_torch(frame, cfg)
        # outputs = model.infer(inp)
        outputs = model.infer_torch(inp)

        pred = {
            "loc_row": outputs[model.output_names[0]][0],
            "loc_col": outputs[model.output_names[1]][0],
            "exist_row": outputs[model.output_names[2]][0],
            "exist_col": outputs[model.output_names[3]][0],
        }

        coords = pred2coords(pred, cfg)

        coords_transformed = []
        for coord in coords:
            if len(coord) == 0:
                coords_transformed.append(coord)
                continue
            coord = np.reshape(coord, (-1, 1, 2)).astype(np.float32)
            undist = cv2.undistortPoints(coord, cfg.K, cfg.D, P=cfg.K)
            coord_transformed = cv2.perspectiveTransform(undist, cfg.H).reshape(-1, 2)
            coords_transformed.append(coord_transformed)

        # img_undistort = cv2.undistort(frame, cfg.K, cfg.D, newCameraMatrix=cfg.K)
        # bev_img = cv2.warpPerspective(img_undistort, cfg.H, (cfg.original_image_width, cfg.original_image_height))
        # for lane in coords_transformed:
        #     for x, y in lane:
        #         cv2.circle(bev_img, (int(x), int(y)), 3, (0, 255, 0), -1)

        cv2.polylines(
            frame,
            [np.array(cord).astype(np.int32) for cord in coords],
            isClosed=False,  # True = closes the shape
            color=(0, 0, 255),  # Green
            thickness=3,
            lineType=cv2.LINE_AA,  # Smooth line
        )
        for lane in coords:
            for x, y in lane:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
        left_lane, right_lane = coords_transformed
        warning = False
        center = None

        if len(left_lane) != 0:
            left_lane = left_lane[
                left_lane[:, 1] < (cfg.original_image_height / 2 + cfg.N_pixels_mean)
            ]
        if len(right_lane) != 0:
            right_lane = right_lane[
                right_lane[:, 1] < (cfg.original_image_height / 2 + cfg.N_pixels_mean)
            ]
        if len(left_lane) != 0:
            left_center = left_lane.mean(axis=0)
            center_left_buffer.append(left_center)
        else:
            center_left_buffer.append([np.nan, np.nan])
        center_left_buffer = center_left_buffer[-cfg.N_frames_median :]
        left_center = np.nanmedian(center_left_buffer, axis=0)
        if len(right_lane) != 0:
            right_center = right_lane.mean(axis=0)
            center_right_buffer.append(right_center)
        else:
            center_right_buffer.append([np.nan, np.nan])
        center_right_buffer = center_right_buffer[-cfg.N_frames_median :]
        right_center = np.nanmedian(center_right_buffer, axis=0)
        if not np.isnan(left_center).any() and not np.isnan(right_center).any():
            center = (left_center + right_center) / 2
        # cv2.circle(bev_img, (int(cfg.original_image_width/2), int(cfg.original_image_height/2)), 3, (255, 0, 0), -1)
        if center is not None:
            # cv2.circle(bev_img, (int(center[0]), int(center[1])), 3, (0, 0, 255), -1)
            if (
                np.abs(cfg.original_image_width / 2 - center[0])
                > cfg.Center_threshold_pixels
            ):
                warning = True
            if warning:
                # cv2.putText(bev_img, "WARNING: Lane Departure!", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255),3)
                cv2.putText(
                    frame,
                    "WARNING: Lane Departure!",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 0, 255),
                    3,
                )
            center = np.array(center, dtype=np.float32).reshape(-1, 1, 2)
            center_inv = cv2.perspectiveTransform(center, cfg.H_inv).reshape(-1, 2)
            center_inv_dist = distort_points(center_inv, cfg.K, cfg.D)[0]
            cv2.circle(
                frame,
                (int(center_inv_dist[0]), int(center_inv_dist[1])),
                11,
                (0, 0, 255),
                -1,
            )
        cv2.circle(
            frame, (int(image_center[0]), int(image_center[1])), 11, (255, 255, 0), -1
        )

        i += 1
        end_time = time.perf_counter()
        # Compute and print FPS
        latency = end_time - start_time
        fps = 1.0 / latency if latency > 0 else 0
        print(f"Inference time all: {latency * 1000:.2f} ms | FPS: {fps:.2f}")
        # cv2.imwrite("output_trt/frame_%04d.png"%(i%10), frame)
        # break
        cv2.imshow(
            "CULane UFLDv2 TensorRT",
            cv2.resize(frame, (int(960 * 1.5), int(540 * 1.5))),
        )
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
