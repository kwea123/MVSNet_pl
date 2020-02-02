def abs_error(depth_pred, depth_gt, mask):
    depth_pred, depth_gt = depth_pred[mask], depth_gt[mask]
    return (depth_pred - depth_gt).abs()

def error_threshold(depth_pred, depth_gt, mask, threshold):
    """
    computes the percentage of pixels whose depth error is not more than @threshold
    """
    errors = abs_error(depth_pred, depth_gt, mask)
    err_mask = errors > threshold
    return err_mask.float()