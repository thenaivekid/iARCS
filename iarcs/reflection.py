"""Agentic reward reflection loop using Gemini.

The LLM generates and iteratively refines task-specific reward programs
using evolution memory: the history of past reward functions and their
training statistics. Each accepted revision is written to
iarcs/rewards/custom_rewards/<task>_v{n}.py.
"""

import json
import os
import time

import torch

CUSTOM_REWARDS_DIR = os.path.join(
    os.path.dirname(__file__), "rewards", "custom_rewards",
)

SCENE_API_SPEC = """\
You are writing a Python TASK-reward function for a 3D bedroom-layout diffusion model.

Define exactly one function:
    def compute_reward(parsed_scene, **kwargs):
        # returns a torch.FloatTensor of shape (B,), one scalar reward per scene

`parsed_scene` is a dict with (B = batch size, N = max objects per scene):
    parsed_scene["device"]          -> torch device
    parsed_scene["object_indices"]  -> LongTensor (B, N), class id per slot
    parsed_scene["positions"]       -> FloatTensor (B, N, 3): (x, y, z); floor plane is XZ
    parsed_scene["sizes"]           -> FloatTensor (B, N, 3): HALF-extents (hx, hy, hz)
    parsed_scene["orientations"]    -> FloatTensor (B, N, 2): (cos, sin) facing dir in XZ
    parsed_scene["is_empty"]        -> BoolTensor (B, N): True = empty slot (ignore it)

Bedroom class ids:
0 armchair,1 bookshelf,2 cabinet,3 ceiling_lamp,4 chair,5 children_cabinet,
6 coffee_table,7 desk,8 double_bed,9 dressing_chair,10 dressing_table,11 kids_bed,
12 nightstand,13 pendant_lamp,14 shelf,15 single_bed,16 sofa,17 stool,18 table,
19 tv_stand,20 wardrobe. ("bed" = any of 8/11/15.)

IMPORTANT:
- Object-object COLLISIONS and OUT-OF-BOUND placement are ALREADY handled by
  separate universal rewards. Do NOT add collision / overlap / boundary terms.
- The reward should be shaped and dense, not sparse 0/1. Keep magnitudes in [-5, 5].
- The model CANNOT control which object classes appear — that is determined by
  the floor plan conditioning. Many scenes will NOT contain the task-relevant
  objects. For scenes missing any required object, return reward 0.0 (neutral),
  NOT a penalty. Only assign positive/negative rewards to scenes that actually
  contain the relevant objects. This is critical — penalizing missing objects
  destroys the learning signal because most scenes won't have them.
- DO NOT use orientations to infer semantic direction (e.g. "foot-to-head" or
  "headboard direction" for beds, or "screen facing direction" for TV stands).
  The orientation (cos, sin) is just a facing angle in XZ — it does NOT reliably
  encode semantic directions like foot-to-head. Any reward based on orientation
  semantics will be unreliable and may hurt learning.
- ONLY reward the constraints the user EXPLICITLY asked for. If the user says
  "a bedroom with a tv_stand and a bed present", reward PRESENCE and basic
  SPATIAL RELATIONSHIP (e.g. proximity, reasonable distance). Do NOT add
  nice-to-have features the user did not ask for (orientation alignment, lateral
  centering, viewing angle, etc.). Extra features that are not grounded in
  reliable data will introduce noise and prevent learning.
- SIMPLICITY PRINCIPLE: if the evolution memory shows that reward scores are
  plateauing or not improving, make the reward SIMPLER, not more complex.
  Remove components that are not clearly helping. A simple reward that captures
  the core user intent will outperform a complex one with unreliable signals.
- CURRICULUM: start with the simplest possible reward (e.g. just presence or
  proximity). Only add additional terms in later reflections if the simple
  version is already learning well (reward mean is clearly increasing across
  iterations in the evolution memory). Build complexity gradually — each
  reflection should add at most one new component. If adding a component causes
  reward to stop improving or regress, remove it in the next reflection.
- Pure function, no file/network access, no prints.
- Output ONLY a JSON object with keys:
    {"reasoning": str, "decision": str, "reward_code": str}
"""

_DUMMY_B, _DUMMY_N = 3, 6


def _dummy_scene():
    g = torch.Generator().manual_seed(0)
    idx = torch.randint(0, 21, (_DUMMY_B, _DUMMY_N), generator=g)
    pos = torch.randn(_DUMMY_B, _DUMMY_N, 3, generator=g)
    size = torch.rand(_DUMMY_B, _DUMMY_N, 3, generator=g) * 0.5 + 0.2
    ori = torch.randn(_DUMMY_B, _DUMMY_N, 2, generator=g)
    ori = ori / (ori.norm(dim=-1, keepdim=True) + 1e-6)
    empty = torch.zeros(_DUMMY_B, _DUMMY_N, dtype=torch.bool)
    empty[:, -1] = True
    return {"device": "cpu", "object_indices": idx, "positions": pos,
            "sizes": size, "orientations": ori, "is_empty": empty}


def _validate_reward_source(source):
    ns = {}
    try:
        exec(compile(source, "<reward>", "exec"), ns)
    except Exception as e:
        return False, f"compile failed: {e}"
    fn = ns.get("compute_reward")
    if not callable(fn):
        return False, "no compute_reward defined"
    try:
        out = fn(_dummy_scene())
    except Exception as e:
        return False, f"runtime error: {e}"
    if not isinstance(out, torch.Tensor) or out.shape[0] != _DUMMY_B:
        return False, f"bad output shape: {getattr(out, 'shape', None)}"
    if not torch.isfinite(out).all():
        return False, "output contains NaN/Inf"
    return True, None


class RewardReflector:
    def __init__(self, task_name, task_prompt, history_dir,
                 model="gemini-2.5-flash", api_key=None):
        self.task_name = task_name
        self.task_prompt = task_prompt
        self.history_dir = history_dir
        self.model = model
        self.iteration = 0
        os.makedirs(history_dir, exist_ok=True)
        self._evo_path = os.path.join(history_dir, f"{task_name}_evolution.json")
        self.evolution = self._load_evolution()

        from dotenv import load_dotenv
        from google import genai
        load_dotenv()
        self._client = genai.Client(
            api_key=api_key or os.environ.get("GEMINI_API_KEY"),
        )

    def _load_evolution(self):
        if os.path.exists(self._evo_path):
            try:
                with open(self._evo_path) as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def _save_evolution(self):
        with open(self._evo_path, "w") as f:
            json.dump(self.evolution, f, indent=2)

    def _record(self, iteration, module, decision, code, rationale, stats):
        self.evolution.append({
            "ts": time.time(), "iteration": iteration, "module": module,
            "decision": decision, "rationale": rationale, "code": code,
            "stats": stats,
        })
        self._save_evolution()

    def _evolution_block(self):
        if not self.evolution:
            return "(no previous versions)"
        parts = []
        for e in self.evolution:
            stats = e.get("stats")
            stats_str = ("none" if not stats else
                         ", ".join(f"{k}={v}" for k, v in stats.items() if v is not None))
            parts.append(
                f"--- version {e['iteration']} ({e['module']}, {e['decision']}) ---\n"
                f"stats: {stats_str}\n```python\n{e['code']}\n```\n"
            )
        return "\n".join(parts)

    def _ask(self, text):
        return self._client.models.generate_content(
            model=self.model, contents=text,
        ).text

    @staticmethod
    def _extract_json(text):
        t = text.strip()
        if t.startswith("```"):
            t = t.split("```", 2)[1]
            if t.startswith("json"):
                t = t[4:]
        s, e = t.find("{"), t.rfind("}")
        if s == -1 or e == -1:
            raise ValueError("No JSON object in response")
        return json.loads(t[s:e + 1])

    def _propose(self, prompt, attempts=4):
        last = "no attempts"
        for i in range(attempts):
            try:
                obj = self._extract_json(self._ask(prompt))
                source = obj["reward_code"]
                decision = obj.get("decision", "redefine")
                rationale = obj.get("reasoning", "")
            except Exception as e:
                last = f"call/parse failed: {e}"
                time.sleep(min(2 ** i * 5, 40))
                continue
            ok, err = _validate_reward_source(source)
            if not ok:
                last = f"validation: {err}"
                continue
            return source, decision, rationale
        return None, "failed", last

    def _write_module(self, source):
        os.makedirs(CUSTOM_REWARDS_DIR, exist_ok=True)
        name = f"{self.task_name}_v{self.iteration}"
        path = os.path.join(CUSTOM_REWARDS_DIR, f"{name}.py")
        with open(path, "w") as f:
            f.write(source.strip() + "\n")
        return name

    def sync_iteration_from_disk(self):
        import glob, re
        mx = 0
        for p in glob.glob(os.path.join(CUSTOM_REWARDS_DIR, f"{self.task_name}_v*.py")):
            m = re.search(rf"{re.escape(self.task_name)}_v(\d+)\.py$", os.path.basename(p))
            if m:
                mx = max(mx, int(m.group(1)))
        self.iteration = mx
        return mx

    def generate_initial(self):
        prompt = (SCENE_API_SPEC + f"\n\nTASK: {self.task_prompt!r}\n"
                  "Write the initial reward program.\n")
        source, decision, rationale = self._propose(prompt)
        if source is None:
            raise RuntimeError(f"generate_initial failed: {rationale}")
        self.iteration = 0
        name = self._write_module(source)
        self._record(0, name, "initial", source, rationale, None)
        return name

    def reflect(self, current_source, stats):
        """One refinement round. Returns (module_name, accepted: bool)."""
        if not self.evolution:
            cur_module = f"{self.task_name}_v{self.iteration}"
            self._record(self.iteration, cur_module, "seed", current_source, "", stats)
        else:
            self.evolution[-1]["stats"] = stats
            self._save_evolution()

        prompt = (
            SCENE_API_SPEC
            + f"\n\nTASK: {self.task_prompt!r}\n\n"
            + "EVOLUTION MEMORY:\n" + self._evolution_block() + "\n\n"
            + "Write an IMPROVED task reward. Keep what worked, fix what didn't.\n"
        )
        source, decision, rationale = self._propose(prompt)
        self.iteration += 1
        if source is None:
            self._record(self.iteration, "(rejected)", decision, "", rationale, None)
            return None, False
        name = self._write_module(source)
        self._record(self.iteration, name, decision, source, rationale, None)
        return name, True
