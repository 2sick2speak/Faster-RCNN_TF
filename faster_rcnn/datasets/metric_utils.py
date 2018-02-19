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


def collect_metrics(gt_df, pred_df, img_names, classes, iou_threshold):

    gt_df['id'] = list(range(len(gt_df)))
    pred_df['id'] = list(range(len(pred_df)))
    # print(gt_df)
    # print(pred_df)
    # print(img_names)
    print('Classes: {}'.format(classes))
    print('IOU threshold: {}'.format(iou_threshold))

    img_metrics = list()
    for img_name in img_names:
        img_metric = dict()
        img_metric['img_name'] = img_name

        metrics = list()
        # class_id == 0: total metrics for all classes
        for class_id in range(len(classes) if len(classes) > 2 else 1):
            metric = dict()

            if class_id:
                gt_boxes = gt_df.query('img_name == @img_name and class_id == @class_id')
                pred_boxes = pred_df.query('img_name == @img_name and \
                                                class_id == @class_id')
            else:
                gt_boxes = gt_df.query('img_name == @img_name')
                pred_boxes = pred_df.query('img_name == @img_name')

            num_gt_boxes = len(gt_boxes)
            num_pred_boxes = len(pred_boxes)

            if num_gt_boxes == 0 and num_pred_boxes == 0:
                # No boxes for image
                metric['tp'] = 0
                metric['fp'] = 0
                metric['fn'] = 0
                # print(img_name, class_id, metric['tp'], metric['fp'], metric['fn'])
                metrics.append(metric)

            elif num_gt_boxes == 0 and num_pred_boxes > 0:
                # Only false positivies
                metric['tp'] = 0
                metric['fp'] = len(pred_boxes)
                metric['fn'] = 0
                # print(img_name, class_id, metric['tp'], metric['fp'], metric['fn'])
                metrics.append(metric)

            elif num_gt_boxes > 0 and num_pred_boxes == 0:
                # Only false negatives
                metric['tp'] = 0
                metric['fp'] = 0
                metric['fn'] = len(gt_boxes)
                # print(img_name, class_id, metric['tp'], metric['fp'], metric['fn'])
                metrics.append(metric)

            else:
                # Calculate pairwise iou for intersecting bboxes
                union_df = pd.merge(pred_boxes, gt_boxes, how='outer', on='class_id',
                                    suffixes=('_pred', '_gt'), copy=False)
                for u in union_df.itertuples():
                    if u.x1_pred == u.x1_pred and u.x1_gt == u.x1_gt:
                        union_df.loc[u.Index, 'iou'] = iou((u.x1_pred, u.y1_pred,
                                                            u.x2_pred, u.y2_pred),
                                                           (u.x1_gt, u.y1_gt,
                                                            u.x2_gt, u.y2_gt))
                    else:
                        union_df.loc[u.Index, 'iou'] = 0
                union_df.sort_values('iou', ascending=False, inplace=True)

                pred_set = set(union_df.id_pred)
                gt_set = set(union_df.id_gt)
                metric['tp'], metric['fp'], metric['fn'] = 0, 0, 0
                # iou_sum = 0

                for u in union_df.itertuples():
                    if u.id_pred in pred_set and u.id_gt in gt_set and u.iou > 0:
                        # Filter intersections with only one allowed per gt
                        # iou_sum += u.iou
                        pred_set.remove(u.id_pred)
                        gt_set.remove(u.id_gt)
                        if u.iou > iou_threshold:
                            metric['tp'] += 1
                        else:
                            metric['fp'] += 1
                            metric['fn'] += 1
                        # print(img_name, class_id, metric['tp'], metric['fp'], metric['fn'],
                        #       pred_set, gt_set)
                    elif min(len(pred_set), len(gt_set)) == 0 or u.iou == 0:
                        # Have no available match place or no intersecting bboxes left
                        break

                metric['fp'] += len([pred for pred in pred_set if pred == pred])
                metric['fn'] += len([gt for gt in gt_set if gt == gt])

            # calculate_metrics()
            if not class_id:
                for m in metric:
                    img_metric[m] = metric[m]
            else:
                for m in metric:
                    img_metric['zz_{}_{}'.format(classes[class_id], m)] = metric[m]
            
        img_metrics.append(img_metric)
            
    img_metrics_df = pd.DataFrame(img_metrics).fillna(0)

    for class_id in range(len(classes) if len(classes) > 2 else 1):
        if not class_id:
            pfx = ''
        else:
            pfx = 'zz_{}_'.format(classes[class_id])

        num_tp = img_metrics_df[pfx + 'tp'].mean()
        num_fp = img_metrics_df[pfx + 'fp'].mean()
        num_fn = img_metrics_df[pfx + 'fn'].mean()
        num_unique_boxes = num_tp + num_fp + num_fn
        img_metrics_df[pfx + 'num_gt_boxes'] = num_gt_boxes = num_tp + num_fn
        img_metrics_df[pfx + 'num_pred_boxes'] = num_pred_boxes = num_tp + num_fp

        img_metrics_df[pfx + 'precision'] = precision = num_tp / num_pred_boxes \
            if num_pred_boxes > 0 else 1
        img_metrics_df[pfx + 'recall'] = recall = num_tp / num_gt_boxes \
            if num_gt_boxes > 0 else 1
        img_metrics_df[pfx + 'f'] = 2 * (precision * recall) / (precision + recall) \
            if precision + recall > 0 else 0
        img_metrics_df[pfx + 'f2'] = 5 * (precision * recall) / (4 * precision + recall) \
            if precision + recall > 0 else 0

        if not class_id:
            img_metrics_df[pfx + 'gtp'] = num_tp + num_fn
            img_metrics_df[pfx + 'accuracy'] = num_tp / num_unique_boxes \
                if num_unique_boxes > 0 else 1

    # print(img_metrics)
    
    return img_metrics_df


def get_from_files(dataset_name):
    from datasets.factory import get_imdb
    get_imdb(dataset_name)
    pass
