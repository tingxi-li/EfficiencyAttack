# Pipeline Features: Batching, Confidence Filter, Buffer/Drop

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add batching, OD confidence filtering, and queue buffer/drop to the traffic monitoring pipeline.

**Architecture:** Each feature is opt-in via `set_config()` parameters with backwards-compatible defaults. Batching greedily drains up to N items per loop iteration. Confidence filter applies after existing post-processing. Buffer/drop uses tail-drop when queue exceeds threshold.

**Tech Stack:** Python multiprocessing, PyTorch, existing pipeline components.

---

## File Map

All files under `/home/lxt230026/EfficiencyAttack/traffic/`:

| File | Changes |
|------|---------|
| `cmpnt/c2_det.py` | Batching, confidence filter, buffer/drop on 3 output queues |
| `cmpnt/c3_fr.py` | Batching, buffer/drop on output queue |
| `cmpnt/c4_lpr.py` | Batching, buffer/drop on output queue |
| `cmpnt/c5_cap.py` | Batching, buffer/drop on output queue |
| `cmpnt/c6_kr.py` | Batching, buffer/drop on output queue |
| `cmpnt/c7_udp.py` | Batching (loop over batch) |
| `cmpnt/c1_img.py` | Buffer/drop on output queue |
| `pipe.py` | Pass new config params to `set_config()` and add CLI args |

---

### Task 1: Add batching + confidence filter + buffer/drop to c2_det.py (Object Detection)

**Files:**
- Modify: `cmpnt/c2_det.py` — `set_config()` (lines 42-44), `_run()` (lines 114-209)

- [ ] **Step 1: Update set_config() to accept new parameters**

In `cmpnt/c2_det.py`, replace the `set_config` method (lines 42-44):

```python
def set_config(self, model_id, profile_save_path=None, batch_size=1, conf_filter=None, buffer_limit=None):
    self.model_id = model_id
    self.profile_save_path = profile_save_path + f"/{self.__class__.__name__}"
    self.batch_size = batch_size
    self.conf_filter = conf_filter
    self.buffer_limit = buffer_limit
    self.drop_count = 0
```

- [ ] **Step 2: Add helper methods for batched queue drain and buffered put**

Insert these methods after `set_config()` (after line 44):

```python
def _drain_queue(self, queue, max_items):
    """Greedily drain up to max_items from queue. Returns list of items. Stops on None sentinel."""
    items = []
    sentinel = False
    for _ in range(max_items):
        try:
            data = queue.get(block=False)
            if data is None:
                sentinel = True
                break
            items.append(data)
        except Empty:
            break
    return items, sentinel

def _buffered_put(self, queue, item):
    """Put item into queue. If buffer_limit set and queue exceeds it, drop the item."""
    if self.buffer_limit is not None and queue.qsize() >= self.buffer_limit:
        self.drop_count += 1
        return
    while queue.full():
        time.sleep(0.01)
    queue.put(item)
```

- [ ] **Step 3: Rewrite the _run() processing loop to use batching, confidence filter, and buffered put**

Replace the main loop inside `_run()` (lines 128-190) with:

```python
        while not self.stop_event.is_set():
            items, sentinel = self._drain_queue(self.img2od_queue, self.batch_size)

            if sentinel and len(items) == 0:
                break

            for data in items:
                if isinstance(data, np.ndarray):
                    data_tensor = torch.from_numpy(data).to(self.device)
                elif isinstance(data, torch.Tensor):
                    data_tensor = data.to(self.device)

                with torch.no_grad():
                    preds = self.model(data_tensor)
                    output = self.processor.post_process_object_detection(
                        preds,
                        threshold=0.25,
                        target_sizes=[data_tensor.shape[2:]]
                    )[0]
                    self.count += 1

                scores, labels, boxes = output["scores"], output["labels"], output["boxes"]

                # Apply secondary confidence filter
                if self.conf_filter is not None:
                    mask = scores >= self.conf_filter
                    scores = scores[mask]
                    labels = labels[mask]
                    boxes = boxes[mask]

                face_indices = (labels == 0).nonzero(as_tuple=True)[0]
                plate_indices = (labels == 2).nonzero(as_tuple=True)[0]

                if len(face_indices) > 0:
                    for idx in face_indices:
                        if self.valid_bbox(boxes[idx]):
                            cropped_np = self.crop_box(data_tensor, boxes[idx])
                            self._buffered_put(self.od2fr_queue, cropped_np)
                            del cropped_np

                if len(plate_indices) > 0:
                    for idx in plate_indices:
                        if self.valid_bbox(boxes[idx]):
                            cropped_np = self.crop_box(data_tensor, boxes[idx])
                            self._buffered_put(self.od2lpr_queue, cropped_np)
                            del cropped_np

                if len(face_indices) > 0 or len(plate_indices) > 0:
                    all_indices = torch.cat([face_indices, plate_indices]) if len(face_indices) > 0 and len(plate_indices) > 0 else face_indices if len(face_indices) > 0 else plate_indices
                    merged_box = self.merge_bbox(all_indices, boxes)
                    if self.valid_bbox(merged_box):
                        cropped_np = self.crop_box(data_tensor, merged_box)
                        self._buffered_put(self.od2cap_queue, cropped_np)
                        del cropped_np
                    del all_indices, merged_box

                torch.cuda.empty_cache()
                gc.collect()
                del data_tensor, preds, output, labels, boxes, face_indices, plate_indices

            if sentinel:
                break
```

- [ ] **Step 4: Add drop_count to profiling JSON output**

In the `finally` block of `_run()` (around line 198), update the `content` dict:

```python
            content = {
                "count" : self.count,
                "time" : self.time_elapsed,
                "energy" : self.energy,
                "drop_count" : self.drop_count
            }
```

- [ ] **Step 5: Commit**

```bash
git add cmpnt/c2_det.py
git commit -m "feat(od): add batching, confidence filter, and buffer/drop to object detection"
```

---

### Task 2: Add batching + buffer/drop to c3_fr.py (Face Recognition)

**Files:**
- Modify: `cmpnt/c3_fr.py` — `set_config()` (lines 36-38), `_run()` (lines 71-125)

- [ ] **Step 1: Update set_config()**

Replace `set_config` (lines 36-38):

```python
def set_config(self, model_id="vggface2", profile_save_path=None, batch_size=1, buffer_limit=None):
    self.model_id = model_id
    self.profile_save_path = profile_save_path + f"/{self.__class__.__name__}"
    self.batch_size = batch_size
    self.buffer_limit = buffer_limit
    self.drop_count = 0
```

- [ ] **Step 2: Add _drain_queue and _buffered_put helper methods**

Insert after `set_config()`:

```python
def _drain_queue(self, queue, max_items):
    items = []
    sentinel = False
    for _ in range(max_items):
        try:
            data = queue.get(block=False)
            if data is None:
                sentinel = True
                break
            items.append(data)
        except Empty:
            break
    return items, sentinel

def _buffered_put(self, queue, item):
    if self.buffer_limit is not None and queue.qsize() >= self.buffer_limit:
        self.drop_count += 1
        return
    while queue.full():
        time.sleep(0.01)
    queue.put(item)
```

- [ ] **Step 3: Rewrite the _run() processing loop**

Replace the main loop inside `_run()` (lines 81-106) with:

```python
        while not self.stop_event.is_set():
            items, sentinel = self._drain_queue(self.od2fr_queue, self.batch_size)

            if sentinel and len(items) == 0:
                break

            for data in items:
                data_tensor = torch.from_numpy(data).to(self.device)

                with torch.no_grad():
                    if data_tensor.dim() == 3:
                        data_tensor = data_tensor.unsqueeze(0)

                    padded_image = torch.nn.functional.interpolate(
                        data_tensor, size=(160, 160), mode='bilinear', align_corners=False
                    )
                    face_embedding = self.facenet(padded_image).cpu().numpy()

                self._buffered_put(self.fr2kr_queue, face_embedding)
                self.count += 1

                del data, data_tensor, padded_image, face_embedding
                torch.cuda.empty_cache()
                gc.collect()

            if sentinel:
                break
```

- [ ] **Step 4: Add drop_count to profiling JSON**

In the `finally` block of `_run()`, update:

```python
            content = {
                "count" : self.count,
                "time" : self.time_elapsed,
                "energy" : self.energy,
                "drop_count" : self.drop_count
            }
```

- [ ] **Step 5: Commit**

```bash
git add cmpnt/c3_fr.py
git commit -m "feat(fr): add batching and buffer/drop to face recognition"
```

---

### Task 3: Add batching + buffer/drop to c4_lpr.py (License Plate Recognition)

**Files:**
- Modify: `cmpnt/c4_lpr.py` — `set_config()` (lines 43-45), `_run()` (lines 108-179)

- [ ] **Step 1: Update set_config()**

Replace `set_config` (lines 43-45):

```python
def set_config(self, model_id="model_v2.pth", profile_save_path=None, batch_size=1, buffer_limit=None):
    self.model_id = model_id
    self.profile_save_path = profile_save_path + f"/{self.__class__.__name__}"
    self.batch_size = batch_size
    self.buffer_limit = buffer_limit
    self.drop_count = 0
```

- [ ] **Step 2: Add _drain_queue and _buffered_put helper methods**

Insert after `set_config()`:

```python
def _drain_queue(self, queue, max_items):
    items = []
    sentinel = False
    for _ in range(max_items):
        try:
            data = queue.get(block=False)
            if data is None:
                sentinel = True
                break
            items.append(data)
        except Empty:
            break
    return items, sentinel

def _buffered_put(self, queue, item):
    if self.buffer_limit is not None and queue.qsize() >= self.buffer_limit:
        self.drop_count += 1
        return
    while queue.full():
        time.sleep(0.01)
    queue.put(item)
```

- [ ] **Step 3: Rewrite the _run() processing loop**

Replace the main loop inside `_run()` (lines 125-158) with:

```python
        while not self.stop_event.is_set():
            items, sentinel = self._drain_queue(self.od2lpr_queue, self.batch_size)

            if sentinel and len(items) == 0:
                break

            for data in items:
                try:
                    data_tensor = torch.from_numpy(data).to(self.device)

                    with torch.no_grad():
                        pred = self.segmentation(data_tensor, self.deeplabv3)
                        plate_tensor = self.post_process(pred, data_tensor.detach().clone())
                        plate_result = self.ocr(plate_tensor)[0]
                        plate_text = plate_result.plate if hasattr(plate_result, 'plate') else str(plate_result)

                    self._buffered_put(self.lpr2kr_queue, plate_text)
                    self.count += 1
                    torch.cuda.empty_cache()

                except Exception as e:
                    logger.warning(f"{self.__class__.__name__:<12} : error processing item: {str(e)}")
                    self.count += 1
                    continue

                del data, data_tensor, pred, plate_tensor, plate_text

            if sentinel:
                break
```

- [ ] **Step 4: Add drop_count to profiling JSON**

In the `finally` block of `_run()`, update:

```python
            content = {
                "count" : self.count,
                "time" : self.time_elapsed,
                "energy" : self.energy,
                "drop_count" : self.drop_count
            }
```

- [ ] **Step 5: Commit**

```bash
git add cmpnt/c4_lpr.py
git commit -m "feat(lpr): add batching and buffer/drop to license plate recognition"
```

---

### Task 4: Add batching + buffer/drop to c5_cap.py (Image Captioning)

**Files:**
- Modify: `cmpnt/c5_cap.py` — `set_config()` (lines 40-42), `_run()` (lines 75-134)

- [ ] **Step 1: Update set_config()**

Replace `set_config` (lines 40-42):

```python
def set_config(self, model_id="microsoft/git-base", profile_save_path=None, batch_size=1, buffer_limit=None):
    self.model_id = model_id
    self.profile_save_path = profile_save_path + f"/{self.__class__.__name__}"
    self.batch_size = batch_size
    self.buffer_limit = buffer_limit
    self.drop_count = 0
```

- [ ] **Step 2: Add _drain_queue and _buffered_put helper methods**

Insert after `set_config()`:

```python
def _drain_queue(self, queue, max_items):
    items = []
    sentinel = False
    for _ in range(max_items):
        try:
            data = queue.get(block=False)
            if data is None:
                sentinel = True
                break
            items.append(data)
        except Empty:
            break
    return items, sentinel

def _buffered_put(self, queue, item):
    if self.buffer_limit is not None and queue.qsize() >= self.buffer_limit:
        self.drop_count += 1
        return
    while queue.full():
        time.sleep(0.01)
    queue.put(item)
```

- [ ] **Step 3: Rewrite the _run() processing loop**

Replace the main loop inside `_run()` (lines 93-115) with:

```python
        while not self.stop_event.is_set():
            items, sentinel = self._drain_queue(self.od2cap_queue, self.batch_size)

            if sentinel and len(items) == 0:
                break

            for data in items:
                if isinstance(data, np.ndarray):
                    data_tensor = torch.from_numpy(data).to(self.device)
                elif isinstance(data, torch.Tensor):
                    data_tensor = data.to(self.device)

                with torch.no_grad():
                    caption = self.inference(data_tensor)

                self._buffered_put(self.cap2lm_queue, caption)
                self.count += 1

                del data, data_tensor, caption
                torch.cuda.empty_cache()
                gc.collect()

            if sentinel:
                break
```

- [ ] **Step 4: Add drop_count to profiling JSON**

In the `finally` block of `_run()`, update:

```python
            content = {
                "count" : self.count,
                "time" : self.time_elapsed,
                "energy" : self.energy,
                "drop_count" : self.drop_count
            }
```

- [ ] **Step 5: Commit**

```bash
git add cmpnt/c5_cap.py
git commit -m "feat(cap): add batching and buffer/drop to image captioning"
```

---

### Task 5: Add batching + buffer/drop to c6_kr.py (Knowledge Retrieval)

**Files:**
- Modify: `cmpnt/c6_kr.py` — `set_config()` (lines 35-37), `_run()` (lines 94-163)

- [ ] **Step 1: Update set_config()**

Replace `set_config` (lines 35-37):

```python
def set_config(self, embedding_path, profile_save_path=None, batch_size=1, buffer_limit=None):
    self.embedding_path = embedding_path
    self.profile_save_path = profile_save_path + f"/{self.__class__.__name__}"
    self.batch_size = batch_size
    self.buffer_limit = buffer_limit
    self.drop_count = 0
```

- [ ] **Step 2: Add _drain_queue and _buffered_put helper methods**

Insert after `set_config()`:

```python
def _drain_queue(self, queue, max_items):
    items = []
    sentinel = False
    for _ in range(max_items):
        try:
            data = queue.get(block=False)
            if data is None:
                sentinel = True
                break
            items.append(data)
        except Empty:
            break
    return items, sentinel

def _buffered_put(self, queue, item):
    if self.buffer_limit is not None and queue.qsize() >= self.buffer_limit:
        self.drop_count += 1
        return
    while queue.full():
        time.sleep(0.01)
    queue.put(item)
```

- [ ] **Step 3: Rewrite the _run() processing loop**

KR has dual input queues. Replace the main loop (lines 109-142) with:

```python
        while not self.stop_event.is_set():
            if fr_end_received and lpr_end_received:
                break

            if not fr_end_received:
                fr_items, fr_sentinel = self._drain_queue(self.fr2kr_queue, self.batch_size)
                if fr_sentinel:
                    fr_end_received = True
                    logger.info(f"{self.__class__.__name__:<12} : Received end signal from FR")
                for data in fr_items:
                    self.process_fr_data(data)
                    self.count += 1
                    del data

            if not lpr_end_received:
                lpr_items, lpr_sentinel = self._drain_queue(self.lpr2kr_queue, self.batch_size)
                if lpr_sentinel:
                    lpr_end_received = True
                    logger.info(f"{self.__class__.__name__:<12} : Received end signal from LPR")
                for data in lpr_items:
                    self.process_lpr_data(data)
                    self.count += 1
                    del data

            torch.cuda.empty_cache()
            gc.collect()
```

- [ ] **Step 4: Update process_fr_data and process_lpr_data to use _buffered_put**

In `process_fr_data` (around line 188), replace:

```python
            while self.kr2lm_queue.full():
                time.sleep(0.01)
            self.kr2lm_queue.put(result_string)
```

with:

```python
            self._buffered_put(self.kr2lm_queue, result_string)
```

In `process_lpr_data` (around line 203), replace:

```python
            while self.kr2lm_queue.full():
                time.sleep(0.01)
            self.kr2lm_queue.put(query_result)
```

with:

```python
            self._buffered_put(self.kr2lm_queue, query_result)
```

- [ ] **Step 5: Add drop_count to profiling JSON**

In the `finally` block of `_run()`, update:

```python
            content = {
                "count" : self.count,
                "time" : self.time_elapsed,
                "energy" : self.energy,
                "drop_count" : self.drop_count
            }
```

- [ ] **Step 6: Commit**

```bash
git add cmpnt/c6_kr.py
git commit -m "feat(kr): add batching and buffer/drop to knowledge retrieval"
```

---

### Task 6: Add batching to c7_udp.py (UDP/LM Output)

**Files:**
- Modify: `cmpnt/c7_udp.py` — `set_config()` (lines 36-38), `_run()` (lines 98-172)

- [ ] **Step 1: Update set_config()**

Replace `set_config` (lines 36-38):

```python
def set_config(self, model_id="gpt2", profile_save_path=None, batch_size=1):
    self.model_id = model_id
    self.profile_save_path = profile_save_path + f"/{self.__class__.__name__}"
    self.batch_size = batch_size
```

- [ ] **Step 2: Add _drain_queue helper method**

Insert after `set_config()`:

```python
def _drain_queue(self, queue, max_items):
    items = []
    sentinel = False
    for _ in range(max_items):
        try:
            data = queue.get(block=False)
            if data is None:
                sentinel = True
                break
            items.append(data)
        except Empty:
            break
    return items, sentinel
```

- [ ] **Step 3: Rewrite the _run() processing loop**

UDP has dual input queues. Replace the main loop (lines 115-153) with:

```python
        while not self.stop_event.is_set():
            if cap_end_received and kr_end_received:
                break

            if not cap_end_received:
                cap_items, cap_sentinel = self._drain_queue(self.cap2lm_queue, self.batch_size)
                if cap_sentinel:
                    cap_end_received = True
                    logger.info(f"{self.__class__.__name__:<12} : Received end signal from CAP")
                for data in cap_items:
                    self.send_message_udp(data)
                    self.count += 1
                    del data

            if not kr_end_received:
                kr_items, kr_sentinel = self._drain_queue(self.kr2lm_queue, self.batch_size)
                if kr_sentinel:
                    kr_end_received = True
                    logger.info(f"{self.__class__.__name__:<12} : Received end signal from KR")
                for data in kr_items:
                    self.send_message_udp(data)
                    self.count += 1
                    del data

            torch.cuda.empty_cache()
            gc.collect()
```

- [ ] **Step 4: Commit**

```bash
git add cmpnt/c7_udp.py
git commit -m "feat(udp): add batching to UDP/LM output"
```

---

### Task 7: Add buffer/drop to c1_img.py (Image Stream)

**Files:**
- Modify: `cmpnt/c1_img.py` — `set_config()` (lines 43-47), `_run()` (lines 80-136)

- [ ] **Step 1: Update set_config()**

Replace `set_config` (lines 43-47):

```python
def set_config(self, src_folder_path=None, fps=30, profile_save_path=None, eval_size=50, buffer_limit=None):
    self.src_folder_path = src_folder_path
    self.fps = fps
    self.profile_save_path = profile_save_path + f"/{self.__class__.__name__}"
    self.eval_size = eval_size
    self.buffer_limit = buffer_limit
    self.drop_count = 0
```

- [ ] **Step 2: Add _buffered_put helper method**

Insert after `set_config()`:

```python
def _buffered_put(self, queue, item):
    if self.buffer_limit is not None and queue.qsize() >= self.buffer_limit:
        self.drop_count += 1
        return
    while queue.full():
        time.sleep(0.01)
    queue.put(item)
```

- [ ] **Step 3: Replace queue put calls with _buffered_put**

In the `_run()` processing loop, replace:

```python
                while self.img2od_queue.full():
                    time.sleep(0.01)
                self.img2od_queue.put(input_tensor)
```

with:

```python
                self._buffered_put(self.img2od_queue, input_tensor)
```

- [ ] **Step 4: Add drop_count to profiling JSON**

In the `finally` block of `_run()`, update the content dict to include:

```python
            content = {
                "count" : self.count,
                "time" : self.time_elapsed,
                "energy" : self.energy,
                "drop_count" : self.drop_count
            }
```

- [ ] **Step 5: Commit**

```bash
git add cmpnt/c1_img.py
git commit -m "feat(img): add buffer/drop to image stream"
```

---

### Task 8: Wire new parameters through pipe.py

**Files:**
- Modify: `pipe.py` — CLI args (lines 18-32), `set_config()` calls (lines 126-133)

- [ ] **Step 1: Add CLI arguments**

After the existing `parser.add_argument` calls (after line 31), add:

```python
parser.add_argument("--batch_size", type=int, default=1, help="batch size for each module")
parser.add_argument("--conf_filter", type=float, default=None, help="secondary confidence filter for OD output")
parser.add_argument("--buffer_limit", type=int, default=None, help="max queue size before tail-drop")
```

- [ ] **Step 2: Update set_config() calls**

Replace the `set_config()` calls (lines 126-133) with:

```python
    img_stream.set_config(src_folder_path=input_dir, fps=30, profile_save_path=ps_path, eval_size=eval_size, buffer_limit=args.buffer_limit)
    od_stream.set_config(model_id=0, profile_save_path=ps_path, batch_size=args.batch_size, conf_filter=args.conf_filter, buffer_limit=args.buffer_limit)
    fr_stream.set_config(profile_save_path=ps_path, batch_size=args.batch_size, buffer_limit=args.buffer_limit)
    lpr_stream.set_config(profile_save_path=ps_path, batch_size=args.batch_size, buffer_limit=args.buffer_limit)
    cap_stream.set_config(profile_save_path=ps_path, batch_size=args.batch_size, buffer_limit=args.buffer_limit)
    kr_stream.set_config(embedding_path="./face_embeddings", profile_save_path=ps_path, batch_size=args.batch_size, buffer_limit=args.buffer_limit)
    udp_Stream.set_config(profile_save_path=ps_path, batch_size=args.batch_size)
```

- [ ] **Step 3: Commit**

```bash
git add pipe.py
git commit -m "feat(pipe): wire batching, conf_filter, and buffer_limit CLI args"
```

---

### Task 9: Smoke test

- [ ] **Step 1: Run pipeline with all defaults (backwards compatibility)**

```bash
cd /home/lxt230026/EfficiencyAttack/traffic
python pipe.py --model_id 0 --algorithm teaspoon --target_idx 2 --eval_size 1
```

Expected: Pipeline completes as before, no errors. Profile JSONs should include `"drop_count": 0`.

- [ ] **Step 2: Run with new features enabled**

```bash
python pipe.py --model_id 0 --algorithm teaspoon --target_idx 2 --eval_size 1 --batch_size 8 --conf_filter 0.5 --buffer_limit 100
```

Expected: Pipeline completes faster. Fewer items in downstream queues due to confidence filter. Drop counts > 0 if queues exceed 100.

- [ ] **Step 3: Commit any fixes**

```bash
git add -A
git commit -m "fix: smoke test corrections for pipeline features"
```
