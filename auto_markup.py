"""
PDF -> картинки (full / first / signature) + черновая разметка для CVAT.

Что делает:
  1. Берёт все PDF из папки pdfs (по умолчанию ~/Desktop/pdfs).
  2. Находит страницу с шапкой/номером и страницу с подписью.
  3. Сохраняет страницы как JPG в папки full / first / signature.
  4. Находит рамки: шапка, регистрационный номер, электронная подпись,
     должность, фио (получатель пока не ищется).
  5. Пишет разметку для загрузки в CVAT:
       cvat_annotations.zip  -> Upload annotations -> "CVAT 1.1"
       yolo_annotations.zip  -> Upload annotations -> "YOLO 1.1" (запасной вариант)
       predictions.json      -> те же рамки в формате для evaluate.py (оценка качества)
       report.txt            -> что нашлось, а что надо проверить глазами

Запуск:   python auto_markup.py
Другие папки:  python auto_markup.py --src /путь/к/pdfs --out /путь/для/результата

Нужен pymupdf:  pip install pymupdf
"""
import argparse
import json
import re
import zipfile
from pathlib import Path

try:
    import pymupdf as fitz
except ImportError:  # старые версии
    import fitz

DPI = 150
SCALE = DPI / 72.0

# Порядок и цвета меток как в JSON из CVAT (порядок = номера классов в YOLO)
LABELS = [
    ("регистрационный номер", "#FF6B6B"),
    ("электронная подпись", "#51CF66"),
    ("шапка", "#339AF0"),
    ("получатель", "#FCC419"),
    ("фио", "#CC5DE8"),
    ("должность", "#FF922B"),
]
CLASS_ID = {name: i for i, (name, _) in enumerate(LABELS)}

TITLES = {"ПРИКАЗ", "РАСПОРЯЖЕНИЕ", "ПОСТАНОВЛЕНИЕ", "РЕШЕНИЕ", "УКАЗ",
          "ПИСЬМО", "ПРОТОКОЛ", "ПОЛОЖЕНИЕ", "ОПРЕДЕЛЕНИЕ"}
POSITION_WORDS = {"заместитель", "первый", "министр", "руководитель", "директор",
                  "начальник", "председатель", "глава", "врио", "исполняющий",
                  "мэр", "префект", "ректор", "генеральный", "управляющий",
                  "помощник", "секретарь"}
INITIALS = re.compile(r"^[А-ЯЁ]\.([А-ЯЁ]\.)?$")
SURNAME = re.compile(r"^[А-ЯЁ][а-яё\-]{2,}$")

PAD = 1.5  # небольшой запас вокруг текста, pt


# ---------- геометрия ----------
def union(rects):
    rects = [r for r in rects if r is not None]
    if not rects:
        return None
    return fitz.Rect(min(r.x0 for r in rects), min(r.y0 for r in rects),
                     max(r.x1 for r in rects), max(r.y1 for r in rects))


def pad(r, p=PAD):
    return fitz.Rect(r.x0 - p, r.y0 - p, r.x1 + p, r.y1 + p)


# ---------- чтение страницы ----------
class Page:
    def __init__(self, page):
        self.page = page
        self.W, self.H = page.rect.width, page.rect.height
        self.words = []
        for w in page.get_text("words"):
            self.words.append({"r": fitz.Rect(w[:4]), "t": w[4], "b": w[5], "l": w[6]})
        self.images = []
        for im in page.get_images(full=True):
            for r in page.get_image_rects(im[0]):
                self.images.append(fitz.Rect(r))
        self.drawings = page.get_drawings()

    def lines(self):
        d = {}
        for w in self.words:
            d.setdefault((w["b"], w["l"]), []).append(w)
        out = []
        for ws in d.values():
            ws.sort(key=lambda w: w["r"].x0)
            out.append({"r": union([w["r"] for w in ws]),
                        "text": " ".join(w["t"] for w in ws),
                        "words": ws})
        out.sort(key=lambda l: l["r"].y0)
        return out


# ---------- поиск элементов ----------
def find_title(pg):
    for ln in pg.lines():
        if ln["r"].y0 > pg.H * 0.45:
            break
        norm = re.sub(r"\s+", "", ln["text"]).upper()
        if norm in TITLES:
            return ln["r"]
    return None


def find_reg(pg, title):
    y_min = title.y1 - 2 if title else 0
    y_max = y_min + 160 if title else pg.H * 0.35
    cands = [w for w in pg.words
             if w["t"].startswith("№") and y_min <= w["r"].y0 <= y_max]
    if not cands:
        return None
    anchor = min(cands, key=lambda w: w["r"].y0)["r"]
    cy = (anchor.y0 + anchor.y1) / 2
    parts = [w["r"] for w in pg.words if abs((w["r"].y0 + w["r"].y1) / 2 - cy) < 7]
    # картинки-вставки с датой и номером в той же строке
    for im in pg.images:
        if im.height < 30 and abs((im.y0 + im.y1) / 2 - cy) < 12 and im.width > 40:
            parts.append(im)
    box = union(parts)
    # линии-подчёркивания рядом со строкой
    for d in pg.drawings:
        r = d["rect"]
        if r.height < 2.5 and r.width > 15 and box.y0 - 6 <= r.y0 <= box.y1 + 9 \
                and r.x1 > box.x0 - 5 and r.x0 < box.x1 + 80:
            box = union([box, r])
    return box


def find_header(pg, title, reg):
    bottom_ref = title.y0 if title else (reg.y0 - 2 if reg else None)
    if bottom_ref is None:
        return None
    parts = [w["r"] for w in pg.words if w["r"].y1 <= bottom_ref + 1]
    parts += [im for im in pg.images if im.y1 <= bottom_ref + 1 and im.y0 < pg.H * 0.35]
    return union(parts)


def find_stamp(pg):
    # 1) векторный штамп (скруглённая рамка)
    for d in pg.drawings:
        r = d["rect"]
        if 90 <= r.width <= 220 and 35 <= r.height <= 110 and len(d["items"]) >= 4 \
                and r.y0 > pg.H * 0.15:
            return r
    # 2) штамп-картинка
    best = None
    for im in pg.images:
        asp = im.width / max(im.height, 1)
        if 90 <= im.width <= 220 and 35 <= im.height <= 110 and 1.4 <= asp <= 3.6 \
                and im.y0 > pg.H * 0.15 and im.y1 < pg.H * 0.97:
            best = im if best is None or im.y0 > best.y0 else best
    if best:
        return best
    # 3) по тексту штампа: подряд идут слова "СВЕДЕНИЯ О СЕРТИФИКАТЕ"
    ws = pg.words
    for i in range(len(ws) - 2):
        if (ws[i]["t"].upper() == "СВЕДЕНИЯ" and ws[i + 1]["t"].upper() == "О"
                and ws[i + 2]["t"].upper().startswith("СЕРТИФИКАТЕ")):
            a = union([ws[i]["r"], ws[i + 2]["r"]])
            return fitz.Rect(a.x0 - 8, a.y0 - 18, a.x1 + 8, a.y1 + 40)
    return None


def find_fio(pg):
    cands = []
    ws = pg.words
    for i, w in enumerate(ws):
        if not INITIALS.match(w["t"]):
            continue
        j = i + 1
        rect = fitz.Rect(w["r"])
        while j < len(ws) and INITIALS.match(ws[j]["t"]) and abs(ws[j]["r"].y0 - w["r"].y0) < 3:
            rect = union([rect, ws[j]["r"]])
            j += 1
        if j < len(ws) and SURNAME.match(ws[j]["t"]) and abs(ws[j]["r"].y0 - w["r"].y0) < 3:
            rect = union([rect, ws[j]["r"]])
            if rect.x0 > pg.W * 0.5 and rect.y1 < pg.H * 0.93 and rect.y0 > pg.H * 0.2:
                cands.append(rect)
    return max(cands, key=lambda r: r.y0) if cands else None


def find_position(pg, fio, stamp):
    if fio is None:
        return None
    zone = fitz.Rect(0, fio.y0 - 110, fio.x0 - 4, fio.y1 + 3)
    words = []
    for w in pg.words:
        r = w["r"]
        c = fitz.Point((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2)
        if stamp and fitz.Rect(stamp.x0 - 2, stamp.y0 - 2, stamp.x1 + 2, stamp.y1 + 2).contains(c):
            continue
        if r.x1 <= zone.x1 + 1 and zone.y0 <= r.y0 and r.y1 <= zone.y1 + 6:
            words.append(w)
    if not words:
        return None
    # строки сверху вниз
    words.sort(key=lambda w: (w["r"].y0 + w["r"].y1) / 2)
    rows, cur = [], [words[0]]
    for w in words[1:]:
        cy = (w["r"].y0 + w["r"].y1) / 2
        ref = sum((x["r"].y0 + x["r"].y1) / 2 for x in cur) / len(cur)
        if abs(cy - ref) < 4:
            cur.append(w)
        else:
            rows.append(cur)
            cur = [w]
    rows.append(cur)
    # идём вверх от строки с ФИО, пока строки идут плотно
    fio_cy = (fio.y0 + fio.y1) / 2
    idx = min(range(len(rows)),
              key=lambda k: abs(sum((x["r"].y0 + x["r"].y1) / 2 for x in rows[k]) / len(rows[k]) - fio_cy))
    keep = [rows[idx]]
    for k in range(idx - 1, -1, -1):
        gap = min(x["r"].y0 for x in keep[0]) - max(x["r"].y1 for x in rows[k])
        if gap > 14:
            break
        keep.insert(0, rows[k])
    # обрезаем сверху до строки, где первое слово - должность
    start = None
    for k, row in enumerate(keep):
        first = min(row, key=lambda w: w["r"].x0)["t"].lower().strip(".,«»")
        if first in POSITION_WORDS:
            start = k
            break
    if start is None:
        return None
    return union([w["r"] for row in keep[start:] for w in row])


def detect(pg, is_first_page):
    found = {}
    title = find_title(pg) if is_first_page else None
    reg = find_reg(pg, title) if is_first_page else None
    header = find_header(pg, title, reg) if is_first_page else None
    stamp = find_stamp(pg)
    fio = find_fio(pg)
    pos = find_position(pg, fio, stamp)
    if header is not None:
        found["шапка"] = pad(header)
    if reg is not None:
        found["регистрационный номер"] = pad(reg)
    if stamp is not None:
        found["электронная подпись"] = fitz.Rect(stamp)
    if fio is not None:
        found["фио"] = pad(fio)
    if pos is not None:
        found["должность"] = pad(pos)
    return found


# ---------- вывод разметки ----------
def to_px(r, W, H):
    return (max(0, r.x0 * SCALE), max(0, r.y0 * SCALE),
            min(W, r.x1 * SCALE), min(H, r.y1 * SCALE))


def write_cvat_xml(items, path):
    out = ['<?xml version="1.0" encoding="utf-8"?>', "<annotations>", "  <version>1.1</version>",
           "  <meta><task><mode>annotation</mode><labels>"]
    for name, color in LABELS:
        out.append(f"<label><name>{name}</name><color>{color}</color><type>rectangle</type>"
                   f"<attributes></attributes></label>")
    out.append("</labels></task></meta>")
    for i, it in enumerate(items):
        out.append(f'  <image id="{i}" name="{it["name"]}" width="{it["w"]}" height="{it["h"]}">')
        for label, (x0, y0, x1, y1) in it["boxes"].items():
            out.append(f'    <box label="{label}" source="manual" occluded="0" '
                       f'xtl="{x0:.2f}" ytl="{y0:.2f}" xbr="{x1:.2f}" ybr="{y1:.2f}" z_order="0"></box>')
        out.append("  </image>")
    out.append("</annotations>")
    xml = "\n".join(out)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("annotations.xml", xml)


def write_yolo_zip(items, path):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("obj.names", "\n".join(n for n, _ in LABELS) + "\n")
        z.writestr("obj.data", f"classes = {len(LABELS)}\nnames = obj.names\n"
                               f"train = train.txt\nbackup = backup/\n")
        z.writestr("train.txt", "".join(f"data/obj_train_data/{it['name']}\n" for it in items))
        for it in items:
            lines = []
            for label, (x0, y0, x1, y1) in it["boxes"].items():
                xc, yc = (x0 + x1) / 2 / it["w"], (y0 + y1) / 2 / it["h"]
                w, h = (x1 - x0) / it["w"], (y1 - y0) / it["h"]
                lines.append(f"{CLASS_ID[label]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            z.writestr(f"obj_train_data/{Path(it['name']).stem}.txt", "\n".join(lines) + ("\n" if lines else ""))


def write_predictions(items, path):
    """JSON в формате для evaluate.py: по записи на картинку, рамки в пикселях."""
    recs = []
    for it in items:
        b = it["boxes"]
        recs.append({
            "image": it["name"],
            "page": int(re.search(r"_p(\d+)\.jpg$", it["name"]).group(1)),
            "image_size": [it["w"], it["h"]],
            "dpi": DPI,
            "reg_number_box": [round(v, 1) for v in b["регистрационный номер"]] if "регистрационный номер" in b else None,
            "signature_box": [round(v, 1) for v in b["электронная подпись"]] if "электронная подпись" in b else None,
        })
    Path(path).write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------- основной цикл ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(Path.home() / "Desktop" / "pdfs"))
    ap.add_argument("--out", default=str(Path.home() / "Desktop" / "images"))
    ap.add_argument("--debug", action="store_true", help="рисовать найденные рамки на картинках")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    for sub in ("full", "first", "signature"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    pdfs = sorted(list(src.glob("*.pdf")) + list(src.glob("*.PDF")))
    items, report = [], []

    for n, pdf in enumerate(pdfs, 1):
        try:
            doc = fitz.open(pdf)
        except Exception as e:
            report.append(f"[ОШИБКА] {pdf.name}: не открылся ({e})")
            continue

        total = len(doc)
        pages = [Page(doc[k]) for k in range(total)]
        dets = [detect(p, k == 0) for k, p in enumerate(pages)]
        has_text = any(p.words for p in pages)

        first_p = next((k for k, d in enumerate(dets) if "шапка" in d or "регистрационный номер" in d), None)
        sig_p = next((k for k, d in enumerate(dets) if "электронная подпись" in d or "фио" in d), None)

        notes = []
        if not has_text:
            notes.append("нет текстового слоя (скан) - размечать вручную")
        if first_p is None:
            first_p = 0
            notes.append("шапка/номер не найдены, взята 1-я страница")
        if sig_p is None:
            notes.append("подпись/ФИО не найдены - проверь вручную")

        jobs = []  # (страница, папка)
        if sig_p is not None and sig_p == first_p:
            jobs.append((first_p, "full"))
        else:
            jobs.append((first_p, "first"))
            if sig_p is not None:
                jobs.append((sig_p, "signature"))

        for k, folder in jobs:
            name = f"doc{n:03d}_p{k + 1}.jpg"
            pix = doc[k].get_pixmap(dpi=DPI)
            boxes = {}
            for label, r in dets[k].items():
                # на "first" не берём подпись, на "signature" не берём шапку/номер,
                # чтобы разметка соответствовала папке
                boxes[label] = to_px(r, pix.width, pix.height)
            if args.debug:
                from PIL import Image, ImageDraw
                im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                dr = ImageDraw.Draw(im)
                cols = dict(LABELS)
                for label, b in boxes.items():
                    dr.rectangle(b, outline=cols[label], width=5)
                    dr.text((b[0], max(0, b[1] - 14)), label, fill=cols[label])
                im.save(out / folder / name, quality=90)
            else:
                pix.save(str(out / folder / name))
            items.append({"name": name, "w": pix.width, "h": pix.height, "boxes": boxes, "folder": folder})
            report.append(f"{pdf.name} [{total} стр.] -> {folder}/{name}: "
                          + (", ".join(boxes) or "ничего не найдено"))
        for note in notes:
            report.append(f"   ВНИМАНИЕ: {note}")

    write_cvat_xml(items, out / "cvat_annotations.zip")
    write_yolo_zip(items, out / "yolo_annotations.zip")
    write_predictions(items, out / "predictions.json")
    (out / "report.txt").write_text("\n".join(report), encoding="utf-8")

    print("\n".join(report))
    print(f"\nPDF: {len(pdfs)} | картинок: {len(items)}")
    for sub in ("full", "first", "signature"):
        print(f"  {sub}: {len(list((out / sub).glob('*.jpg')))}")
    print(f"\nГотово. Разметка: {out / 'cvat_annotations.zip'}")


if __name__ == "__main__":
    main()
