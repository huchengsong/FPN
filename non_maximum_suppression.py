import numpy as np
import cupy as cp
import torch


def non_maximum_suppression_rpn(bbox, thresh, score=None, limit=None):
    """
    apply non maximun suppression
    :param bbox: (N, 4), ndarray or pytorch tensor
    :param thresh: threshold of IOU for suppression
    :param score: (N, ), ndarray or pytorch tensor, score of each bounding box
    :param limit: number of bounding boxes to output
    :return: (K, 4), darray or pytorch tensor, bounding boxes after nms. Sorting from highest score to lowest score
    """
    # cp.asarray is able to read the buffer object
    bbox_cp = cp.asarray(bbox)
    if score is not None:
        score = cp.asarray(score)
    ind_bbox = _non_maximum_suppression_gpu(bbox_cp, thresh, score, limit)
    # ind_bbox is cupy array
    # the time to transfer data from GPU to CPU and again back to GPU is 0.004s
    ind_bbox = torch.from_numpy(ind_bbox.get()).long().cuda()
    selected_bbox = bbox[ind_bbox]
    return ind_bbox, selected_bbox


def non_maximum_suppression_roi(box_scores, bboxes, class_list, score_thresh, iou_thresh):
    """
    using non maximum suppression to reduce bbox number
    :param box_scores: (N, class_num) pytorch tensor
    :param bboxes: (N, 4 * class_num) pytorch tensor
    :param class_list: list of class ID that NMS apply to
    :param score_thresh: score threshold for box selection
    :param iou_thresh: iou threshold
    :return: ndarray: label (K, ), score (K, ), box (K, 4)
    """
    bbox_result = []
    score_result = []
    label_result = []

    for class_id in class_list:
        score_candidate = box_scores[:, class_id]
        mask = torch.nonzero(score_candidate > score_thresh).squeeze()
        if len(mask) == 0:
            continue
        score_candidate = score_candidate[mask]
        box_candidate = bboxes[:, class_id * 4:(class_id + 1) * 4][mask, :]

        ind_bbox, selected_bbox = non_maximum_suppression_rpn(box_candidate, iou_thresh, score_candidate)

        selected_score = score_candidate[ind_bbox]
        selected_bbox = list(selected_bbox.cpu().numpy())
        selected_score = list(selected_score.cpu().numpy())
        selected_label = [class_id] * len(selected_score)
        bbox_result += selected_bbox
        score_result += selected_score
        label_result += selected_label

    return np.array(bbox_result).astype(np.float32), \
           np.array(score_result).astype(np.float32), \
           np.array(label_result).astype(np.int32)


@cp.util.memoize(for_each_device=True)
def _load_kernel(kernel_name, code, options=()):
    cp.cuda.runtime.free(0)
    assert isinstance(options, tuple)
    kernel_code = cp.cuda.compile_with_cache(code, options=options)
    return kernel_code.get_function(kernel_name)


def _non_maximum_suppression_gpu(bbox, thresh, score=None, limit=None):
    if len(bbox) == 0:
        return cp.zeros((0,), dtype=np.int32)

    n_bbox = bbox.shape[0]

    if score is not None:
        order = score.argsort()[::-1].astype(np.int32)
    else:
        order = cp.arange(n_bbox, dtype=np.int32)

    sorted_bbox = bbox[order, :]
    selec, n_selec = _call_nms_kernel(
        sorted_bbox, thresh)
    selec = selec[:n_selec]
    selec = order[selec]
    if limit is not None:
        selec = selec[:limit]
    return selec


_nms_gpu_code = '''
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
int const threadsPerBlock = sizeof(unsigned long long) * 8;

__device__
inline float devIoU(float const *const bbox_a, float const *const bbox_b) {
  float top = max(bbox_a[0], bbox_b[0]);
  float bottom = min(bbox_a[2], bbox_b[2]);
  float left = max(bbox_a[1], bbox_b[1]);
  float right = min(bbox_a[3], bbox_b[3]);
  float height = max(bottom - top, 0.f);
  float width = max(right - left, 0.f);
  float area_i = height * width;
  float area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1]);
  float area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1]);
  return area_i / (area_a + area_b - area_i);
}

extern "C"
__global__
void nms_kernel(const int n_bbox, const float thresh,
                const float *dev_bbox,
                unsigned long long *dev_mask) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  const int row_size =
        min(n_bbox - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
        min(n_bbox - col_start * threadsPerBlock, threadsPerBlock);

  __shared__ float block_bbox[threadsPerBlock * 4];
  if (threadIdx.x < col_size) {
    block_bbox[threadIdx.x * 4 + 0] =
        dev_bbox[(threadsPerBlock * col_start + threadIdx.x) * 4 + 0];
    block_bbox[threadIdx.x * 4 + 1] =
        dev_bbox[(threadsPerBlock * col_start + threadIdx.x) * 4 + 1];
    block_bbox[threadIdx.x * 4 + 2] =
        dev_bbox[(threadsPerBlock * col_start + threadIdx.x) * 4 + 2];
    block_bbox[threadIdx.x * 4 + 3] =
        dev_bbox[(threadsPerBlock * col_start + threadIdx.x) * 4 + 3];
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const float *cur_box = dev_bbox + cur_box_idx * 4;
    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      if (devIoU(cur_box, block_bbox + i * 4) >= thresh) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = DIVUP(n_bbox, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}
'''


def _call_nms_kernel(bbox, thresh):
    n_bbox = bbox.shape[0]
    threads_per_block = 64
    col_blocks = np.ceil(n_bbox / threads_per_block).astype(np.int32)
    blocks = (col_blocks, col_blocks, 1)
    threads = (threads_per_block, 1, 1)

    mask_dev = cp.zeros((n_bbox * col_blocks,), dtype=np.uint64)
    bbox = cp.ascontiguousarray(bbox, dtype=np.float32)
    kern = _load_kernel('nms_kernel', _nms_gpu_code)
    kern(blocks, threads, args=(cp.int32(n_bbox), cp.float32(thresh),
                                bbox, mask_dev))

    mask_host = mask_dev.get()
    selection, n_selec = _nms_gpu_post(
        mask_host, n_bbox, threads_per_block, col_blocks)
    return selection, n_selec


def _nms_gpu_post(mask, n_bbox, threads_per_block, col_blocks):
    n_selection = 0
    one_ull = np.array([1], dtype=np.uint64)
    selection = np.zeros((n_bbox,), dtype=np.int32)
    remv = np.zeros((col_blocks,), dtype=np.uint64)

    for i in range(n_bbox):
        nblock = i // threads_per_block
        inblock = i % threads_per_block

        if not (remv[nblock] & one_ull << inblock):
            selection[n_selection] = i
            n_selection += 1

            index = i * col_blocks
            for j in range(nblock, col_blocks):
                remv[j] |= mask[index + j]
    return selection, n_selection
