"""
Оценка качества поиска рамок: предсказания бейзлайна vs ручная разметка (эталон).

Эталон:       разметка из CVAT, выгруженная в YOLO 1.1 (папка или zip),
              либо yolo_annotations.zip из auto_markup.py.
Предсказания: JSON-массив записей (по одной на страницу), как в predictions_virtum.json:
              {"page": 1, "image_size": [w, h], "dpi": 200,
               "reg_number_box": [x1, y1, x2, y2] | null,
               "signature_box":  [x1, y1, x2, y2] | null, ...}
              Имя файла берётся из первого найденного поля:
              image / image_name / file / filename / path / name / pdf.

Запуск:
    python evaluate.py --gt yolo_annotations.zip --pred predictions.json
    python evaluate.py --gt gt_folder --pred predictions.json --iou-reg 0.3 --iou-sig 0.5

Что считает (отдельно для регистрационного номера и электронной подписи):
    TP  - рамка найдена и IoU с эталоном >= порога
    FP  - рамка найдена, но её нет в эталоне ИЛИ она не совпала с эталоном
    FN  - в эталоне рамка есть, а бейзлайн её не нашёл (или нашёл не там)
    precision = TP/(TP+FP), recall = TP/(TP+FN), средний IoU по TP
Подробный разбор ошибок по страницам пишется в errors.csv.
"""
import argparse
import csv
import json
import re
import zipfile
from pathlib import Path

# запасной порядок классов, если в эталоне нет obj.names (как в нашем CVAT-проекте)
DEFAULT_NAMES = ["регистрационный номер", "электронная подпись", "шапка",
                 "получатель", "фио", "должность"]
NAME_KEYS = ("image", "image_name", "file", "filename", "path", "name", "pdf")
SKIP_TXT = {"train.txt", "val.txt", "test.txt", "valid.txt"}


# ---------- чтение эталона ----------
def read_gt(path):
    """Возвращает (names, {stem: [(class_id, xc, yc, w, h), ...]})."""
    path = Path(path)
    files = {}
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            for n in z.namelist():
                files[n] = z.read(n).decode("utf-8", errors="ignore")
    else:
        for p in path.rglob("*"):
            if p.is_file() and p.suffix.lower() in (".txt", ".names"):
                files[str(p.relative_to(path))] = p.read_text(encoding="utf-8", errors="ignore")

    names = DEFAULT_NAMES
    for n, txt in files.items():
        if Path(n).name == "obj.names":
            names = [l.strip() for l in txt.splitlines() if l.strip()]

    gt = {}
    for n, txt in files.items():
        base = Path(n).name
        if not base.endswith(".txt") or base in SKIP_TXT:
            continue
        boxes = []
        for line in txt.splitlines():
            parts = line.split()
            if len(parts) == 5:
                boxes.append((int(parts[0]), *map(float, parts[1:])))
        gt[Path(n).stem] = boxes
    return names, gt


# ---------- геометрия ----------
def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def yolo_to_px(box, w, h):
    _, xc, yc, bw, bh = box
    return [(xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h]


def candidate_keys(rec):
    for k in NAME_KEYS:
        if rec.get(k):
            stem = Path(str(rec[k])).stem
            page = rec.get("page")
            keys = [stem]
            if page is not None:
                keys.append(f"{stem}_p{page}")
            return keys
    return []


# ---------- основная оценка ----------
def evaluate(names, gt, preds, thresholds):
    classes = {"reg_number_box": names.index("регистрационный номер"),
               "signature_box": names.index("электронная подпись")}
    stats = {k: {"TP": 0, "FP": 0, "FN": 0, "iou": []} for k in classes}
    errors, unmatched = [], []

    for rec in preds:
        key = next((k for k in candidate_keys(rec) if k in gt), None)
        if key is None:
            unmatched.append(candidate_keys(rec) or ["<нет имени>"])
            continue
        w, h = rec["image_size"]
        for field, cid in classes.items():
            gt_boxes = [yolo_to_px(b, w, h) for b in gt[key] if b[0] == cid]
            gt_box = gt_boxes[0] if gt_boxes else None
            pr = rec.get(field)
            thr = thresholds[field]
            s = stats[field]
            if pr and gt_box:
                v = iou(pr, gt_box)
                if v >= thr:
                    s["TP"] += 1
                    s["iou"].append(v)
                    status = "TP"
                else:
                    s["FP"] += 1
                    s["FN"] += 1
                    status = "не совпала (FP+FN)"
                errors.append((key, field, status, round(v, 3), pr, gt_box))
            elif pr and not gt_box:
                s["FP"] += 1
                errors.append((key, field, "FP (лишняя рамка)", 0, pr, None))
            elif gt_box and not pr:
                s["FN"] += 1
                errors.append((key, field, "FN (не найдена)", 0, None, gt_box))
    return stats, errors, unmatched


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True, help="эталон: папка или zip с YOLO-разметкой")
    ap.add_argument("--pred", required=True, help="JSON с предсказаниями")
    ap.add_argument("--iou-reg", type=float, default=0.3,
                    help="порог IoU для номера (по обсуждению с заказчиком точность не критична)")
    ap.add_argument("--iou-sig", type=float, default=0.5, help="порог IoU для подписи")
    ap.add_argument("--errors", default="errors.csv", help="куда писать разбор ошибок")
    args = ap.parse_args()

    names, gt = read_gt(args.gt)
    preds = json.loads(Path(args.pred).read_text(encoding="utf-8"))
    if isinstance(preds, dict):
        preds = [preds]

    thresholds = {"reg_number_box": args.iou_reg, "signature_box": args.iou_sig}
    stats, errors, unmatched = evaluate(names, gt, preds, thresholds)

    print(f"Страниц в эталоне: {len(gt)} | записей в предсказаниях: {len(preds)} "
          f"| не сопоставлено по имени: {len(unmatched)}")
    print(f"\n{'класс':<22}{'порог':>7}{'TP':>5}{'FP':>5}{'FN':>5}{'precision':>11}{'recall':>8}{'mIoU':>8}")
    for field, s in stats.items():
        tp, fp, fn = s["TP"], s["FP"], s["FN"]
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        miou = sum(s["iou"]) / len(s["iou"]) if s["iou"] else 0.0
        print(f"{field:<22}{thresholds[field]:>7.2f}{tp:>5}{fp:>5}{fn:>5}{p:>11.2f}{r:>8.2f}{miou:>8.2f}")

    if unmatched:
        print("\nНе нашлось пары в эталоне (проверь имена файлов):")
        for u in unmatched[:10]:
            print("  ", u)

    with open(args.errors, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["страница", "класс", "результат", "IoU", "предсказание", "эталон"])
        for e in errors:
            wr.writerow(e)
    print(f"\nРазбор по страницам: {args.errors}")


if __name__ == "__main__":
    main()
