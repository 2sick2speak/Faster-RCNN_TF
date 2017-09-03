# coding=utf-8
import pandas as pd

def union(au, bu):
    x = min(au[0], bu[0])
    y = min(au[1], bu[1])
    w = max(au[2], bu[2]) - x
    h = max(au[3], bu[3]) - y
    return x, y, w, h


def intersection(ai, bi):
    x = max(ai[0], bi[0])
    y = max(ai[1], bi[1])
    w = min(ai[2], bi[2]) - x
    h = min(ai[3], bi[3]) - y
    if w < 0 or h < 0:
        return 0, 0, 0, 0
    return x, y, w, h


def iou(a, b):
    # a and b should be (x1,y1,x2,y2)
    assert a[0] < a[2]
    assert a[1] < a[3]
    assert b[0] < b[2]
    assert b[1] < b[3]

    i = intersection(a, b)
    u = union(a, b)
    return float(i[2] / u[2]) * float(i[3] / u[3])


def collect_metrics(ground_truth, pred_boxes_dict,
                    validation_files, iou_threshold=0.5):
    """
    Parameters
    ----------
    ground_truth : dict
        Dict with ground truth boxes in format {validation_name: boxes_list}.
        Validation name = img name without extension
    pred_boxes_dict : dict
        Dict with predicted boxes in format {validation_name: boxes_list}.
        Validation name = img name without extension
    validation_files : list
        List of validation filenames.
        Validation name = img name without extension
    iou_threshold : float, optional (default=0.5)
        Threshhold for iou to assign certain pred box to certain gt box
    """
#    print(ground_truth)
#    print(pred_boxes_dict)
#    print(validation_files)

    metrics = []
    for img_name in validation_files:
        metric = dict()
        metric['img_name'] = img_name

        # Pairwise iou for bboxes
        bbox_iou = []

        all_gt_boxes = ground_truth[img_name]
        all_pred_boxes = pred_boxes_dict.get(img_name, [])
        if len(all_gt_boxes) == 0 and len(all_pred_boxes) == 0:
            # No boxes for image. All good
            metric['tp'] = 0
            metric['fp'] = 0
            metric['fn'] = 0
            metric['aiou'] = 1
            metric['gtp'] = 0
            metric['precision'] = 1
            metric['recall'] = 1
            metric['accuracy'] = 1
            metric['f'] = 1
            metric['num_pred_boxes'] = 0
            metric['num_gt_boxes'] = 0
            metrics.append(metric)
            continue

        elif len(all_gt_boxes) == 0 and len(all_pred_boxes) > 0:
            # Only false positivies
            metric['tp'] = 0
            metric['fp'] = len(all_pred_boxes)
            metric['fn'] = 0
            metric['aiou'] = 0
            metric['gtp'] = 0
            metric['precision'] = 0
            metric['recall'] = 0
            metric['accuracy'] = 0
            metric['f'] = 0
            metric['num_pred_boxes'] = len(all_pred_boxes)
            metric['num_gt_boxes'] = 0
            metrics.append(metric)
            continue

        # First - predicted, second - gt
        for i, gt_box in enumerate(all_gt_boxes):
            for j, pred_box in enumerate(all_pred_boxes):
                iou_value = iou(pred_box, gt_box)
                if iou_value > 0:
                    bbox_iou.append(tuple([[j, i], iou_value]))
        bbox_iou  = sorted(
            bbox_iou,
            key=lambda x: x[1],
            reverse=True)

        pred_set = set(range(len(all_pred_boxes)))
        gt_set = set(range(len(all_gt_boxes)))
        iou_sum, metric['tp'], metric['fp'], metric['fn'] = 0, 0, 0, 0

        for [i, j], iou_value in bbox_iou:
            # Filter intersections with only one allowed per gt
            # i - predicted, j - ground true
            if i in pred_set and j in gt_set:
                iou_sum += iou_value
                pred_set.remove(i)
                gt_set.remove(j)
                if iou_value > iou_threshold:
                    metric['tp'] += 1
                else:
                    metric['fp'] += 1
                    metric['fn'] += 1
            if len(pred_set) == 0 or len(gt_set) == 0:
                # Have no available match place
                break

        metric['fp'] += len(pred_set)
        metric['fn'] += len(gt_set)
        all_boxes_len = len(all_gt_boxes) + len(all_pred_boxes)

        metric['aiou'] = iou_sum / all_boxes_len if all_boxes_len > 0 else 0
        metric['gtp'] = metric['tp'] + metric['fn']
        metric['precision'] = metric['tp'] / (metric['tp'] + metric['fp']) \
            if len(all_pred_boxes) > 0 else 0
        metric['recall'] = metric['tp'] / metric['gtp']
        metric['accuracy'] = metric['tp'] / (metric['tp'] + metric['fp'] + metric['fn'])
        metric['f'] = 0 if metric['precision'] + metric['recall'] == 0 else \
            2 * (metric['precision'] * metric['recall']) / \
            (metric['precision'] + metric['recall'])
        metric['num_pred_boxes'] = len(all_pred_boxes)
        metric['num_gt_boxes'] = len(all_gt_boxes)

        # Check metric sanity
        assert metric['fp'] + metric['tp'] == len(all_pred_boxes)
        metrics.append(metric)
    metrics = pd.DataFrame(metrics)

    # Check correct boxes amount
    total_boxes_predicted = sum(
        [len(pred_boxes_dict[key]) for key in pred_boxes_dict.keys()])
    assert metrics['num_pred_boxes'].sum() == total_boxes_predicted

#    print(metrics)
    
    return metrics


def get_from_files(dataset_name):
    from datasets.factory import get_imdb
    get_imdb(dataset_name)
    pass
