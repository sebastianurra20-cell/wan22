import runpod
from runpod.serverless.utils import rp_upload
import os, websocket, base64, json, uuid, logging, urllib.request, urllib.parse, time, subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

server_address = os.getenv('SERVER_ADDRESS', '127.0.0.1')
client_id = str(uuid.uuid4())

def queue_prompt(prompt):
    url = f"http://{server_address}:8188/prompt"
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request(url, data=data)
    return json.loads(urllib.request.urlopen(req).read())

def get_history(prompt_id):
    url = f"http://{server_address}:8188/history/{prompt_id}"
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read())

def get_videos(ws, prompt):
    prompt_id = queue_prompt(prompt)['prompt_id']
    while True:
        out = ws.recv()
        if isinstance(out, str):
            msg = json.loads(out)
            if msg['type'] == 'executing' and msg['data']['node'] is None and msg['data']['prompt_id'] == prompt_id:
                break
    hist = get_history(prompt_id)[prompt_id]
    outputs = {}
    for nid, outnode in hist['outputs'].items():
        vids = []
        if 'gifs' in outnode:
            for v in outnode['gifs']:
                with open(v['fullpath'], 'rb') as f:
                    vids.append(base64.b64encode(f.read()).decode('utf-8'))
        outputs[nid] = vids
    return outputs

def load_workflow(path):
    with open(path, 'r') as f:
        return json.load(f)

def handler(job):
    job_input = job.get("input", {})
    logger.info(f"🎬 Received T2V job input: {job_input}")

    workflow_file = "/new_Wan22_t2v_api.json"
    prompt = load_workflow(workflow_file)

    # Parámetros del usuario
    prompt_text = job_input.get("prompt", "a futuristic city at sunset, cinematic lighting, ultra detailed")
    seed = job_input.get("seed", 123456789)
    cfg = job_input.get("cfg", 2.0)
    steps = job_input.get("steps", 10)
    length = job_input.get("length", 121)
    width = job_input.get("width", 832)
    height = job_input.get("height", 480)
    context_overlap = job_input.get("context_overlap", 48)
    lora_pairs = job_input.get("lora_pairs", [])

    # Aplicar parámetros
    prompt["135"]["inputs"]["positive_prompt"] = prompt_text
    prompt["220"]["inputs"]["seed"] = seed
    prompt["540"]["inputs"]["seed"] = seed
    prompt["540"]["inputs"]["cfg"] = cfg
    prompt["235"]["inputs"]["value"] = height
    prompt["236"]["inputs"]["value"] = width
    prompt["541"]["inputs"]["num_frames"] = length
    prompt["498"]["inputs"]["context_overlap"] = context_overlap

    if "834" in prompt:
        prompt["834"]["inputs"]["steps"] = steps
    if "569" in prompt:
        prompt["569"]["inputs"]["value"] = steps

    # LoRA aplicación
    if lora_pairs:
        logger.info(f"Applying {len(lora_pairs)} LoRA pairs.")
        high_node = "279"
        low_node = "553"
        for i, pair in enumerate(lora_pairs[:4]):
            h, l = pair.get("high"), pair.get("low")
            hw, lw = pair.get("high_weight", 1.0), pair.get("low_weight", 1.0)
            if h:
                prompt[high_node]["inputs"][f"lora_{i+1}"] = h
                prompt[high_node]["inputs"][f"strength_{i+1}"] = hw
            if l:
                prompt[low_node]["inputs"][f"lora_{i+1}"] = l
                prompt[low_node]["inputs"][f"strength_{i+1}"] = lw

    # Conectar al servidor
    ws_url = f"ws://{server_address}:8188/ws?clientId={client_id}"
    http_url = f"http://{server_address}:8188/"
    for _ in range(60):
        try:
            urllib.request.urlopen(http_url, timeout=5)
            break
        except Exception:
            time.sleep(1)

    ws = websocket.WebSocket()
    for _ in range(36):
        try:
            ws.connect(ws_url)
            break
        except Exception:
            time.sleep(5)

    videos = get_videos(ws, prompt)
    ws.close()

    for nid, vids in videos.items():
        if vids:
            return {"video": vids[0]}
    return {"error": "No video generated"}

runpod.serverless.start({"handler": handler})
