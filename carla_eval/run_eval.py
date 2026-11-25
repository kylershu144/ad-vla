"""Closed-loop evaluation of AutoVLA in CARLA with leaderboard-style metrics.

The policy is queried at 2 Hz with nuPlan-like cameras (front / front-left / front-right,
4 frames of history). A PID / pure-pursuit controller tracks the predicted trajectory
between queries. Metrics follow the CARLA Leaderboard definitions:
  Driving Score = Route Completion x prod(infraction penalties).

Usage:
  python carla_eval/run_eval.py --routes 5 --out results/clean
"""
import argparse
import json
import math
import queue
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

import carla

# CARLA's agents package (GlobalRoutePlanner) ships with the PythonAPI.
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.local_planner import RoadOption

SIM_DT = 0.05            # 20 Hz simulation
POLICY_EVERY = 10        # policy at 2 Hz (matches AutoVLA's 2 Hz frames / 0.5 s actions)
IMG_W, IMG_H, FOV = 960, 540, 64   # nuPlan CAM_F0 HFOV ~ 64 deg (16:9)
SIDE_YAW = 55            # nuPlan CAM_L0 / CAM_R0 yaw offset (deg)

PENALTY = {               # CARLA Leaderboard 2.0 coefficients
    "collision_pedestrian": 0.50,
    "collision_vehicle": 0.60,
    "collision_static": 0.65,
    "red_light": 0.70,
    "stop_sign": 0.80,
    "scenario_timeout": 0.70,
    "yield_emergency": 0.70,
    "min_speed": 0.70,
}


# ----------------------------------------------------------------------------- helpers
def world_to_ego(points_xy, ego_tf):
    """CARLA world (left-handed, y right) -> ego frame (x fwd, y left)."""
    yaw = math.radians(ego_tf.rotation.yaw)
    d = points_xy - np.array([ego_tf.location.x, ego_tf.location.y])
    fwd = d[:, 0] * math.cos(yaw) + d[:, 1] * math.sin(yaw)
    right = -d[:, 0] * math.sin(yaw) + d[:, 1] * math.cos(yaw)
    return np.stack([fwd, -right], 1)


def ego_to_world(traj_xy, ego_tf):
    """ego frame (x fwd, y left) -> CARLA world xy."""
    yaw = math.radians(ego_tf.rotation.yaw)
    fwd, right = traj_xy[:, 0], -traj_xy[:, 1]
    x = ego_tf.location.x + fwd * math.cos(yaw) - right * math.sin(yaw)
    y = ego_tf.location.y + fwd * math.sin(yaw) + right * math.cos(yaw)
    return np.stack([x, y], 1)


class PID:
    def __init__(self, kp, ki, kd, n=20):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.win = deque([0.0] * n, maxlen=n)

    def step(self, err):
        self.win.append(err)
        integral = float(np.mean(self.win))
        deriv = self.win[-1] - self.win[-2]
        return self.kp * err + self.ki * integral + self.kd * deriv


class TrajectoryController:
    """Tracks a world-frame trajectory (0.5 s spacing) with pure pursuit + speed PID."""

    def __init__(self):
        self.lon = PID(0.6, 0.05, 0.2)
        self.world_traj = None
        self.plan_time = 0.0

    def set_plan(self, traj_ego, ego_tf, t):
        pts = np.concatenate([np.zeros((1, 2)), traj_ego[:, :2]], 0)
        self.world_traj = ego_to_world(pts, ego_tf)
        self.plan_time = t

    def control(self, ego_tf, speed, t):
        ctrl = carla.VehicleControl()
        if self.world_traj is None:
            ctrl.brake = 1.0
            return ctrl, 0.0
        local = world_to_ego(self.world_traj, ego_tf)
        # Desired speed: trajectory speed at the current point in the plan (t - plan_time).
        seg = np.linalg.norm(np.diff(self.world_traj, axis=0), axis=1) / 0.5
        k = min(int((t - self.plan_time) / 0.5) + 1, len(seg) - 1)
        target_speed = float(np.mean(seg[max(k - 1, 0):k + 2]))
        if target_speed < 0.3:
            target_speed = 0.0
        # Pure pursuit on a speed-dependent look-ahead point.
        lookahead = float(np.clip(1.5 + 0.6 * speed, 3.0, 12.0))
        dists = np.linalg.norm(local, axis=1)
        ahead = np.where((local[:, 0] > 0.5) & (dists >= lookahead))[0]
        aim = local[ahead[0]] if len(ahead) else local[-1]
        wheelbase = 2.9
        if np.linalg.norm(aim) > 0.5 and aim[0] > 0:
            curvature = 2 * aim[1] / (np.dot(aim, aim))
            steer_angle = math.atan(wheelbase * curvature)
            ctrl.steer = float(np.clip(-steer_angle / math.radians(70), -1, 1))  # CARLA: +steer = right
        delta = target_speed - speed
        u = self.lon.step(delta)
        if target_speed == 0.0 or u < -0.15:
            ctrl.throttle = 0.0
            ctrl.brake = float(np.clip(-u, 0.3 if target_speed == 0.0 else 0.0, 1.0))
        else:
            ctrl.throttle = float(np.clip(u, 0.0, 0.75))
        return ctrl, target_speed


def route_command(route, idx, ego_loc, horizon=30.0):
    """High-level nuPlan-style command from the upcoming route (turn left/right, keep forward)."""
    travelled = 0.0
    prev = ego_loc
    for wp, opt in route[idx: idx + 200]:
        travelled += wp.transform.location.distance(prev)
        prev = wp.transform.location
        if travelled > horizon:
            break
        if opt == RoadOption.LEFT:
            return "turn left"
        if opt == RoadOption.RIGHT:
            return "turn right"
    return "keep forward"


# ----------------------------------------------------------------------------- episode
class Episode:
    def __init__(self, world, ego, route, args, out_dir):
        self.world, self.ego, self.route, self.args = world, ego, route, args
        self.map = world.get_map()
        self.out_dir = out_dir
        self.sensors = []
        self.cam_q = {}
        self.events = []
        self.route_idx = 0
        self.route_len = sum(
            route[i][0].transform.location.distance(route[i + 1][0].transform.location)
            for i in range(len(route) - 1)
        )
        self.cum_len = np.cumsum([0.0] + [
            route[i][0].transform.location.distance(route[i + 1][0].transform.location)
            for i in range(len(route) - 1)
        ])
        self._last_collision = {}
        self._red_lights_run = set()
        self._setup_sensors()

    def _setup_sensors(self):
        bl = self.world.get_blueprint_library()
        cam_bp = bl.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(IMG_W))
        cam_bp.set_attribute("image_size_y", str(IMG_H))
        cam_bp.set_attribute("fov", str(FOV))
        mounts = {
            "front": carla.Transform(carla.Location(x=0.8, z=1.7), carla.Rotation()),
            "front_left": carla.Transform(carla.Location(x=0.6, y=-0.3, z=1.7), carla.Rotation(yaw=-SIDE_YAW)),
            "front_right": carla.Transform(carla.Location(x=0.6, y=0.3, z=1.7), carla.Rotation(yaw=SIDE_YAW)),
        }
        for name, tf in mounts.items():
            cam = self.world.spawn_actor(cam_bp, tf, attach_to=self.ego)
            q = queue.Queue()
            cam.listen(q.put)
            self.cam_q[name] = q
            self.sensors.append(cam)
        col = self.world.spawn_actor(bl.find("sensor.other.collision"), carla.Transform(), attach_to=self.ego)
        col.listen(self._on_collision)
        self.sensors.append(col)

    def _on_collision(self, ev):
        other = ev.other_actor
        tid = other.type_id if other is not None else "static"
        if tid.startswith("walker"):
            kind = "collision_pedestrian"
        elif tid.startswith("vehicle"):
            kind = "collision_vehicle"
        else:
            kind = "collision_static"
        key = other.id if other is not None else -1
        t = ev.timestamp
        # Leaderboard counts repeated contact with the same actor once (debounce ~ 2 s).
        if key in self._last_collision and t - self._last_collision[key] < 2.0:
            return
        self._last_collision[key] = t
        self.events.append({"type": kind, "t": t, "other": tid})

    def grab_images(self, frame):
        imgs = {}
        for name, q in self.cam_q.items():
            while True:
                im = q.get(timeout=10.0)
                if im.frame >= frame:
                    break
            arr = np.frombuffer(im.raw_data, dtype=np.uint8).reshape(IMG_H, IMG_W, 4)[:, :, 2::-1]
            imgs[name] = Image.fromarray(arr.copy())
        return imgs

    def update_route_progress(self, loc):
        # Advance along the route to the closest waypoint within a forward window.
        win = self.route[self.route_idx: self.route_idx + 60]
        d = [wp.transform.location.distance(loc) for wp, _ in win]
        j = int(np.argmin(d))
        if d[j] < 6.0:
            self.route_idx += j
        return d[j]

    def check_red_light(self, loc):
        tl = self.ego.get_traffic_light()
        if tl is None or tl.get_state() != carla.TrafficLightState.Red:
            return
        if tl.id in self._red_lights_run:
            return
        wp = self.map.get_waypoint(loc)
        # Entering the junction while the controlling light is red counts as running it.
        if wp.is_junction and self.ego.get_velocity().length() > 1.0:
            self._red_lights_run.add(tl.id)
            self.events.append({"type": "red_light", "t": self.world.get_snapshot().timestamp.elapsed_seconds,
                                "light": tl.id})

    def destroy(self):
        for s in self.sensors:
            try:
                s.stop()
                s.destroy()
            except RuntimeError:
                pass


def run_episode(client, world, policy, route, args, ep_dir, tm):
    bl = world.get_blueprint_library()
    ego_bp = bl.find("vehicle.lincoln.mkz") if bl.filter("vehicle.lincoln.mkz") else bl.filter("vehicle.*")[0]
    ego_bp.set_attribute("role_name", "hero")
    start_tf = route[0][0].transform
    start_tf.location.z += 0.5
    ego = world.spawn_actor(ego_bp, start_tf)
    world.tick()
    ep = Episode(world, ego, route, args, ep_dir)
    ctl = TrajectoryController()
    hist = {k: deque(maxlen=4) for k in ("front", "front_left", "front_right")}
    log = []
    prev_speed, t0 = 0.0, None
    status, tick = "running", 0
    last_progress_t, last_route_idx = 0.0, 0
    offroad_dist = 0.0
    max_ticks = int(args.timeout_factor * (ep.route_len / 4.0 + 20) / SIM_DT)
    try:
        while status == "running":
            frame = world.tick()
            snap = world.get_snapshot()
            t = snap.timestamp.elapsed_seconds
            t0 = t if t0 is None else t0
            t -= t0
            tf = ego.get_transform()
            speed = ego.get_velocity().length()
            dev = ep.update_route_progress(tf.location)
            ep.check_red_light(tf.location)
            wp = world.get_map().get_waypoint(tf.location, project_to_road=False,
                                              lane_type=carla.LaneType.Driving)
            if wp is None:
                offroad_dist += speed * SIM_DT

            if tick % POLICY_EVERY == 0:
                imgs = ep.grab_images(frame)
                for k in hist:
                    hist[k].append(imgs[k])
                    while len(hist[k]) < 4:   # pad history at episode start
                        hist[k].appendleft(imgs[k])
                accel = (speed - prev_speed) / (SIM_DT * POLICY_EVERY)
                prev_speed = speed
                cmd = route_command(route, ep.route_idx, tf.location)
                tq = time.time()
                out = policy(list(hist["front"]), list(hist["front_left"]), list(hist["front_right"]),
                             speed, abs(accel), cmd)
                latency = time.time() - tq
                traj = out["trajectory"]
                if len(traj) >= 2:
                    ctl.set_plan(traj, tf, t)
                step = {
                    "t": round(t, 2), "frame": frame, "speed": speed, "accel": accel, "cmd": cmd,
                    "x": tf.location.x, "y": tf.location.y, "yaw": tf.rotation.yaw,
                    "route_idx": ep.route_idx, "route_dev": dev, "latency": latency,
                    "traj": np.asarray(traj).tolist(), "text": out["text"],
                    "action_logprobs": out["action_logprobs"].tolist(),
                }
                log.append(step)
                if args.save_images:
                    sd = ep_dir / "frames"
                    sd.mkdir(exist_ok=True)
                    imgs["front"].resize((480, 270)).save(sd / f"{tick:06d}_front.jpg", quality=85)
                print(f"  t={t:6.1f}s v={speed:4.1f} cmd={cmd:12s} RC={100*ep.cum_len[ep.route_idx]/ep.route_len:5.1f}% "
                      f"traj_end=({traj[-1][0]:.1f},{traj[-1][1]:.1f}) {latency:.1f}s "
                      f"{'CoT' if 'not needed' not in out['text'] else ''}", flush=True)

            ctrl, _ = ctl.control(tf, speed, t)
            ego.apply_control(ctrl)

            # termination
            if ep.route_idx > last_route_idx:
                last_route_idx, last_progress_t = ep.route_idx, t
            if ep.route_idx >= len(route) - 3:
                status = "completed"
            elif dev > 30.0:
                status = "route_deviation"
                ep.events.append({"type": "route_deviation", "t": t})
            elif t - last_progress_t > args.blocked_s:
                status = "agent_blocked"
                ep.events.append({"type": "agent_blocked", "t": t})
            elif tick >= max_ticks:
                status = "route_timeout"
                ep.events.append({"type": "route_timeout", "t": t})
            elif args.stop_on_collision and any(e["type"].startswith("collision") for e in ep.events):
                status = "collision"
            tick += 1
    finally:
        ep.destroy()
        ego.destroy()
        world.tick()

    rc = 100.0 * min(ep.cum_len[ep.route_idx] / ep.route_len, 1.0)
    if status == "completed":
        rc = 100.0
    # Leaderboard: off-road driving reduces route completion proportionally.
    rc = max(rc - 100.0 * offroad_dist / ep.route_len, 0.0)
    pen = 1.0
    for e in ep.events:
        pen *= PENALTY.get(e["type"], 1.0)
    result = {
        "status": status, "route_length_m": ep.route_len, "route_completion": rc,
        "infraction_penalty": pen, "driving_score": rc * pen, "events": ep.events,
        "offroad_m": offroad_dist, "sim_time_s": tick * SIM_DT, "n_policy_calls": len(log),
    }
    (ep_dir / "log.json").write_text(json.dumps(log))
    (ep_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result


def sample_route(world, grp, rng, min_len, max_len):
    sps = world.get_map().get_spawn_points()
    for _ in range(500):
        a, b = rng.sample(sps, 2)
        route = grp.trace_route(a.location, b.location)
        if len(route) < 10:
            continue
        L = sum(route[i][0].transform.location.distance(route[i + 1][0].transform.location)
                for i in range(len(route) - 1))
        turns = sum(opt in (RoadOption.LEFT, RoadOption.RIGHT) for _, opt in route)
        if min_len <= L <= max_len and turns > 0:
            return route
    raise RuntimeError("no route found")


def spawn_traffic(client, world, tm, n_vehicles, n_walkers, rng):
    actors = []
    bl = world.get_blueprint_library()
    sps = world.get_map().get_spawn_points()
    rng.shuffle(sps)
    vbps = [b for b in bl.filter("vehicle.*") if int(b.get_attribute("number_of_wheels")) == 4] \
        if bl.filter("vehicle.*") and bl.filter("vehicle.*")[0].has_attribute("number_of_wheels") else list(bl.filter("vehicle.*"))
    for sp in sps[:n_vehicles]:
        v = world.try_spawn_actor(rng.choice(vbps), sp)
        if v is not None:
            v.set_autopilot(True, tm.get_port())
            actors.append(v)
    # Pedestrians (optional)
    walkers, ctrls = [], []
    wbps = bl.filter("walker.pedestrian.*")
    for _ in range(n_walkers):
        loc = world.get_random_location_from_navigation()
        if loc is None or not wbps:
            continue
        w = world.try_spawn_actor(rng.choice(wbps), carla.Transform(loc))
        if w is not None:
            walkers.append(w)
    world.tick()
    cbp = bl.find("controller.ai.walker") if n_walkers else None
    for w in walkers:
        c = world.spawn_actor(cbp, carla.Transform(), attach_to=w)
        ctrls.append(c)
    world.tick()
    for c in ctrls:
        c.start()
        c.go_to_location(world.get_random_location_from_navigation())
        c.set_max_speed(1.4)
    return actors, walkers, ctrls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--tm-port", type=int, default=8000)
    ap.add_argument("--town", default=None)
    ap.add_argument("--routes", type=int, default=5)
    ap.add_argument("--min-len", type=float, default=150)
    ap.add_argument("--max-len", type=float, default=400)
    ap.add_argument("--vehicles", type=int, default=30)
    ap.add_argument("--walkers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout-factor", type=float, default=2.0)
    ap.add_argument("--blocked-s", type=float, default=60.0)
    ap.add_argument("--stop-on-collision", action="store_true")
    ap.add_argument("--save-images", action="store_true")
    ap.add_argument("--no-cot", action="store_true")
    ap.add_argument("--dry-policy", action="store_true", help="straight-line dummy policy (harness test)")
    ap.add_argument("--out", default="results/clean")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))

    if args.dry_policy:
        def policy(f, fl, fr, v, a, cmd):
            tr = np.stack([np.arange(1, 11) * 0.5 * 5.0, np.zeros(10), np.zeros(10)], 1)
            return {"trajectory": tr, "text": "dummy", "action_logprobs": np.zeros(10)}
    else:
        sys.path.insert(0, str(Path(__file__).parent))
        from autovla_policy import AutoVLAPolicy
        policy = AutoVLAPolicy(use_cot=not args.no_cot)

    client = carla.Client(args.host, args.port)
    client.set_timeout(120.0)
    world = client.get_world()
    if args.town and args.town not in world.get_map().name:
        world = client.load_world(args.town)
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = SIM_DT
    world.apply_settings(settings)
    tm = client.get_trafficmanager(args.tm_port)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(args.seed)

    rng = random.Random(args.seed)
    grp = GlobalRoutePlanner(world.get_map(), 2.0)
    results = []
    try:
        for r in range(args.routes):
            route = sample_route(world, grp, rng, args.min_len, args.max_len)
            ep_dir = out / f"route_{r:02d}"
            ep_dir.mkdir(exist_ok=True)
            vehicles, walkers, wctrls = spawn_traffic(client, world, tm, args.vehicles, args.walkers, rng)
            print(f"[route {r}] {len(route)} wps, start={route[0][0].transform.location}, "
                  f"{len(vehicles)} vehicles, {len(walkers)} walkers", flush=True)
            try:
                res = run_episode(client, world, policy, route, args, ep_dir, tm)
            finally:
                for c in wctrls:
                    c.stop()
                client.apply_batch_sync([carla.command.DestroyActor(a) for a in vehicles + walkers + wctrls], True)
            res["route"] = r
            results.append(res)
            print(f"[route {r}] {res['status']} RC={res['route_completion']:.1f} "
                  f"IP={res['infraction_penalty']:.2f} DS={res['driving_score']:.1f}", flush=True)
            (out / "summary.json").write_text(json.dumps(summarize(results), indent=2))
    finally:
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)
        tm.set_synchronous_mode(False)
    print(json.dumps(summarize(results), indent=2))


def summarize(results):
    n = len(results)
    ev = {}
    km = sum(r["route_completion"] / 100 * r["route_length_m"] for r in results) / 1000
    for r in results:
        for e in r["events"]:
            ev[e["type"]] = ev.get(e["type"], 0) + 1
    return {
        "n_routes": n,
        "driving_score": float(np.mean([r["driving_score"] for r in results])) if n else 0,
        "route_completion": float(np.mean([r["route_completion"] for r in results])) if n else 0,
        "infraction_penalty": float(np.mean([r["infraction_penalty"] for r in results])) if n else 0,
        "success_rate": float(np.mean([r["status"] == "completed" and not any(
            e["type"].startswith("collision") or e["type"] == "red_light" for e in r["events"])
            for r in results])) if n else 0,
        "km_driven": km,
        "infractions_per_km": {k: v / max(km, 1e-6) for k, v in ev.items()},
        "infraction_counts": ev,
        "per_route": [{k: r[k] for k in ("route", "status", "route_completion", "infraction_penalty",
                                          "driving_score")} for r in results],
    }


if __name__ == "__main__":
    main()
