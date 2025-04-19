import os
import json
import numpy as np
from pyquaternion import Quaternion
from nuscenes.eval.detection.evaluate import DetectionEval
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.detection.config import config_factory
from nuscenes.nuscenes import NuScenes

# Mapping from class ID to nuScenes detection_name
CLASS_ID_TO_NAME = {
    1: "car",
    2: "truck",
    3: "bus",
    4: "trailer",
    5: "construction_vehicle",
    6: "pedestrian",
    7: "motorcycle",
    8: "bicycle",
    9: "traffic_cone",
    10: "barrier"
}

# Attribute prediction defaults based on class
ATTRIBUTE_NAME = {
    "car": "vehicle.moving",
    "truck": "vehicle.moving",
    "bus": "vehicle.moving",
    "trailer": "vehicle.moving",
    "construction_vehicle": "vehicle.moving",
    "pedestrian": "pedestrian.moving",
    "motorcycle": "cycle.with_rider",
    "bicycle": "cycle.with_rider",
    "traffic_cone": "none",
    "barrier": "none"
}


def convert_preds_to_nuscenes_format(preds_by_sample, output_path):
    """
    Convert your model predictions into the nuScenes JSON format for evaluation.
    """
    results = {}
    meta = {
        "use_camera": True,
        "use_lidar": False,
        "use_radar": True,
        "use_map": False,
        "use_external": False
    }

    for sample_token, predictions in preds_by_sample.items():
        results[sample_token] = []
        for pred in predictions:
            if pred is None or len(pred) < 9:
                continue
            # Ensure Python-native types for JSON
            translation = [float(x) for x in pred[:3]]
            size = [float(x) for x in pred[3:6]]
            rotation = (
                [float(x) for x in pred[6]] 
                if isinstance(pred[6], (list, np.ndarray)) and len(pred[6]) == 4 
                else [0.0, 0.0, 0.0, 1.0]
            )
            class_id = int(pred[7]) if isinstance(pred[7], (int, np.integer)) else 0
            name = CLASS_ID_TO_NAME.get(class_id, "car")
            score = float(pred[8]) if len(pred) > 8 else 0.0

            box = DetectionBox(
                sample_token=sample_token,
                translation=translation,
                size=size,
                rotation=rotation,
                velocity=[0.0, 0.0],
                detection_name=name,
                detection_score=score,
                attribute_name=ATTRIBUTE_NAME.get(name, "none")
            )
            results[sample_token].append(box.serialize())

    final_dict = {"results": results, "meta": meta}
    os.makedirs(output_path, exist_ok=True)
    json_path = os.path.join(output_path, "preds.json")
    with open(json_path, "w") as f:
        json.dump(final_dict, f)
    return json_path


def run_nuscenes_eval(result_path, nusc_root, version, eval_set, output_dir):
    """
    Launch nuScenes detection evaluation.
    """
    cfg = config_factory("detection_cvpr_2019")
    # Load the correct metadata version
    nusc = NuScenes(version=version, dataroot=nusc_root, verbose=False)
    evaluator = DetectionEval(
        nusc, config=cfg, result_path=result_path,
        eval_set=eval_set, output_dir=output_dir
    )
    metrics = evaluator.main()
    return metrics


def compute_3d_detection_metrics(preds_by_sample, nusc_root, version, eval_set):
    """
    Convert preds, run eval, and print key metrics.
    """
    output_dir = os.path.join("eval_output", eval_set)
    result_path = convert_preds_to_nuscenes_format(preds_by_sample, output_dir)
    metrics = run_nuscenes_eval(
        result_path, nusc_root,
        version=version,
        eval_set=eval_set,
        output_dir=output_dir
    )
    print("\n📊 Evaluation Metrics:")
    print(f"✔️  mAP: {metrics['mean_ap']:.4f},  NDS: {metrics['nd_score']:.4f}")
    return metrics
