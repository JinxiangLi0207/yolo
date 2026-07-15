"""Diagnose QualityDetect calibration against matched foreground IoU targets."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import torch

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.nn.modules import QualityDetect
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import make_anchors
from ultralytics.utils.torch_utils import select_device


class QualityDiagnosticValidator(DetectionValidator):
    """Collect per-positive quality calibration data during standard detection validation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = None
        self.class_names = {}
        self._raw_quality_batch = None
        self.positive = defaultdict(list)
        self.negative = defaultdict(lambda: {"count": 0, "sum": 0.0, "gt_01": 0, "gt_03": 0, "gt_05": 0})

    def postprocess(self, preds):
        """Save raw QualityDetect outputs before running normal NMS."""
        self._raw_quality_batch = preds[1] if isinstance(preds, (list, tuple)) and len(preds) > 1 else None
        return super().postprocess(preds)

    def update_metrics(self, preds, batch):
        """Collect quality statistics and then update the standard detection metrics."""
        self._collect_quality_statistics(self._raw_quality_batch, batch)
        return super().update_metrics(preds, batch)

    @torch.no_grad()
    def _collect_quality_statistics(self, raw, batch):
        if not isinstance(raw, dict) or "det" not in raw or "quality" not in raw:
            raise RuntimeError("Quality diagnostics require raw outputs from a QualityDetect head.")
        if self.criterion is None:
            raise RuntimeError("Quality diagnostics criterion was not initialized.")

        feats, quality_preds = raw["det"], raw["quality"]
        batch_size = feats[0].shape[0]
        pred_distri, pred_scores = torch.cat(
            [feat.view(batch_size, self.criterion.no, -1) for feat in feats], 2
        ).split((self.criterion.reg_max * 4, self.criterion.nc), 1)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_quality = torch.cat(
            [quality.view(batch_size, 1, -1) for quality in quality_preds], 2
        ).permute(0, 2, 1)

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.criterion.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.criterion.stride, 0.5)

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.criterion.preprocess(
            targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]]
        )
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.criterion.bbox_decode(anchor_points, pred_distri)
        target_labels, target_bboxes, target_scores, fg_mask, _ = self.criterion.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        quality_score = pred_quality.sigmoid().squeeze(-1)
        level_ids = torch.cat(
            [
                torch.full(
                    (feat.shape[2] * feat.shape[3],), level, device=self.device, dtype=torch.long
                )
                for level, feat in enumerate(feats)
            ]
        ).unsqueeze(0).expand(batch_size, -1)

        for level in range(len(feats)):
            negative_values = quality_score[(level_ids == level) & ~fg_mask].float()
            if negative_values.numel():
                stats = self.negative[level]
                stats["count"] += negative_values.numel()
                stats["sum"] += negative_values.sum().item()
                stats["gt_01"] += (negative_values > 0.1).sum().item()
                stats["gt_03"] += (negative_values > 0.3).sum().item()
                stats["gt_05"] += (negative_values > 0.5).sum().item()

        if not fg_mask.any():
            return

        pred_boxes_px = pred_bboxes.detach() * stride_tensor
        quality_target = bbox_iou(
            pred_boxes_px[fg_mask], target_bboxes[fg_mask], xywh=False
        ).detach().clamp_(0, 1).view(-1)
        matched_boxes = target_bboxes[fg_mask]
        wh = (matched_boxes[:, 2:4] - matched_boxes[:, 0:2]).clamp_min(0)
        area_ratio = wh.prod(-1) / imgsz.prod().clamp_min(1)

        values = {
            "quality": quality_score[fg_mask].float(),
            "iou": quality_target.float(),
            "level": level_ids[fg_mask],
            "class": target_labels[fg_mask].long(),
            "task_weight": target_scores.sum(-1)[fg_mask].float(),
            "area_ratio": area_ratio.float(),
        }
        for key, value in values.items():
            self.positive[key].append(value.cpu())

    @staticmethod
    def _pearson(x, y):
        if x.numel() < 2:
            return float("nan")
        x, y = x.float(), y.float()
        x_centered, y_centered = x - x.mean(), y - y.mean()
        denominator = x_centered.square().sum().sqrt() * y_centered.square().sum().sqrt()
        return (x_centered * y_centered).sum().div(denominator).item() if denominator > 0 else float("nan")

    def _summarize(self, data, group_type, group, mask):
        quality, iou = data["quality"][mask], data["iou"][mask]
        if not quality.numel():
            return None
        error = quality - iou
        return {
            "group_type": group_type,
            "group": group,
            "positives": quality.numel(),
            "quality_mean": quality.mean().item(),
            "iou_mean": iou.mean().item(),
            "bias_q_minus_iou": error.mean().item(),
            "mae": error.abs().mean().item(),
            "pearson": self._pearson(quality, iou),
            "task_weight_mean": data["task_weight"][mask].mean().item(),
            "area_ratio_mean": data["area_ratio"][mask].mean().item(),
            "q_p10": quality.quantile(0.10).item(),
            "q_p50": quality.quantile(0.50).item(),
            "q_p90": quality.quantile(0.90).item(),
            "high_iou_low_q_rate": ((iou >= 0.7) & (quality < 0.5)).float().mean().item(),
        }

    def build_reports(self):
        """Build positive calibration and negative quality summary rows."""
        if not self.positive:
            raise RuntimeError("No positive quality samples were collected.")
        data = {key: torch.cat(value) for key, value in self.positive.items()}
        all_mask = torch.ones_like(data["level"], dtype=torch.bool)
        rows = [self._summarize(data, "overall", "all", all_mask)]

        for level in sorted(data["level"].unique().tolist()):
            rows.append(self._summarize(data, "level", f"P{int(level) + 3}", data["level"] == level))
        for class_id in sorted(data["class"].unique().tolist()):
            class_name = self.class_names.get(int(class_id), str(int(class_id)))
            rows.append(self._summarize(data, "class", class_name, data["class"] == class_id))
        for level in sorted(data["level"].unique().tolist()):
            for class_id in sorted(data["class"].unique().tolist()):
                class_name = self.class_names.get(int(class_id), str(int(class_id)))
                mask = (data["level"] == level) & (data["class"] == class_id)
                rows.append(self._summarize(data, "level_class", f"P{int(level) + 3}/{class_name}", mask))

        size_groups = (
            ("small", data["area_ratio"] < 0.0025),
            ("medium", (data["area_ratio"] >= 0.0025) & (data["area_ratio"] < 0.0225)),
            ("large", data["area_ratio"] >= 0.0225),
        )
        rows.extend(self._summarize(data, "size", name, mask) for name, mask in size_groups)
        rows = [row for row in rows if row is not None]

        negative_rows = []
        for level, stats in sorted(self.negative.items()):
            count = max(stats["count"], 1)
            negative_rows.append(
                {
                    "level": f"P{level + 3}",
                    "negatives": stats["count"],
                    "quality_mean": stats["sum"] / count,
                    "rate_q_gt_0.1": stats["gt_01"] / count,
                    "rate_q_gt_0.3": stats["gt_03"] / count,
                    "rate_q_gt_0.5": stats["gt_05"] / count,
                }
            )
        return rows, negative_rows

    def write_reports(self, output_dir):
        """Write CSV reports and print the headline groups for easy log sharing."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        rows, negative_rows = self.build_reports()

        positive_path = output_dir / "quality_positive_calibration.csv"
        negative_path = output_dir / "quality_negative_summary.csv"
        with positive_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        with negative_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=negative_rows[0].keys())
            writer.writeheader()
            writer.writerows(negative_rows)

        print("\nQUALITY_DIAGNOSTIC_SUMMARY")
        print("group_type,group,positives,quality_mean,iou_mean,bias,mae,pearson,task_weight,area_ratio")
        for row in rows:
            if row["group_type"] in {"overall", "level", "class", "size"}:
                print(
                    f'{row["group_type"]},{row["group"]},{row["positives"]},'
                    f'{row["quality_mean"]:.6f},{row["iou_mean"]:.6f},'
                    f'{row["bias_q_minus_iou"]:.6f},{row["mae"]:.6f},{row["pearson"]:.6f},'
                    f'{row["task_weight_mean"]:.6f},{row["area_ratio_mean"]:.8f}'
                )
        print(f"Positive report: {positive_path}")
        print(f"Negative report: {negative_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Path to an N4 QualityDetect best.pt checkpoint.")
    parser.add_argument("--data", default="ultralytics/cfg/datasets/DUO.yaml", help="Dataset YAML path.")
    parser.add_argument("--batch", type=int, default=96)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", required=True, help="Output name under runs/quality_diagnostics.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = select_device(args.device, batch=args.batch)
    yolo = YOLO(args.weights)
    head = yolo.model.model[-1]
    if not isinstance(head, QualityDetect):
        raise TypeError(f"Expected QualityDetect, but found {type(head).__name__}.")
    head.quality_power = 3.0
    head.quality_mix = 1.0
    yolo.model.to(device)

    validator = QualityDiagnosticValidator(
        args={
            "data": args.data,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "workers": args.workers,
            "device": args.device,
            "plots": False,
            "save_json": False,
            "verbose": False,
            "half": False,
            "rect": True,
            "project": "runs/quality_diagnostics",
            "name": args.name,
            "exist_ok": True,
        }
    )
    validator.criterion = yolo.model.init_criterion()
    validator.class_names = yolo.model.names
    validator(model=yolo.model)
    validator.write_reports(validator.save_dir)


if __name__ == "__main__":
    main()
