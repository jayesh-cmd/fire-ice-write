import cv2
import mediapipe as mp
import numpy as np
import math
import random

PARTICLE_COUNT  = 5000       
HAND_RADIUS     = 180      
IDEAL_ORBIT_R   = 80         
ORBIT_FORCE     = 0.75    
REPEL_FORCE     = 0.015      
DRIFT_FORCE     = 0.0        
NOISE_F         = 1.0       
DAMPING         = 0.90       
HAND_DAMPING    = 0.90       
MAX_SPEED       = 6.0
HAND_SPEED      = 15.0
SMOOTH          = 0.18       
TRAIL_FADE      = 0.85       
CANVAS_FADE     = 0.99
CAM_BRIGHTNESS  = 0.35    

FIRE_PALETTE = [
    (100, 180, 255),
    (80,  120, 255),
    (40,  60,  255),
    (20,  40,  220),
    (200, 240, 255),
]

ICE_PALETTE = [
    (255, 255, 255),
    (255, 240, 240),
    (255, 220, 150),
    (255, 120, 120),
    (250, 200, 100),
]

mp_hands = mp.solutions.hands
PALM_IDS = [0, 5, 9, 13, 17]


def is_index_pointing(lm):
    index_up = lm[8].y < lm[6].y
    mid_down = lm[12].y > lm[10].y
    ring_down = lm[16].y > lm[14].y
    pinky_down = lm[20].y > lm[18].y
    
    return index_up and mid_down and ring_down and pinky_down


class Particle:
    def __init__(self, W, H):
        self.W = W
        self.H = H
        self.element = None
        self.color     = (255, 255, 255)
        self._spawn()

    def _spawn(self):
        self.x  = random.uniform(0, self.W)
        self.y  = random.uniform(0, self.H)
        self.vx = random.gauss(0, 0.6)
        self.vy = random.gauss(0, 0.6)

        self.size      = random.uniform(0.8, 1.8)
        self.original_alpha_base = random.uniform(0.6, 1.0)
        self.alpha_base= self.original_alpha_base

        self.drift_ang = random.uniform(0, math.pi * 2)
        self.drift_rot = random.uniform(0.005, 0.020) * random.choice([1, -1])

        self.twinkle    = random.uniform(0, math.pi * 2)
        self.twinkle_sp = random.uniform(0.04, 0.10)

    def update(self, hands_info):
        self.twinkle    += self.twinkle_sp
        self.drift_ang  += self.drift_rot

        fx, fy    = 0.0, 0.0
        near_hand = False
        best_d    = float('inf')
        best_hand = None

        if not hands_info:
            self.alpha_base = 0.0
            self.vx *= 0.5
            self.vy *= 0.5
            return False, 0.0
        else:
            self.alpha_base = self.original_alpha_base

        for h in hands_info:
            target_x, target_y = h['tip'] if h['is_writing'] else h['pos']
            dx = self.x - target_x
            dy = self.y - target_y
            d  = math.hypot(dx, dy) or 0.001
            if d < best_d:
                best_d    = d
                best_hand = (target_x, target_y, dx, dy, d, h)

        target_x, target_y, dx, dy, best_d, h_info = best_hand

        if self.element != h_info['element']:
            self.element = h_info['element']
            self.color = random.choice(FIRE_PALETTE if self.element == 'Fire' else ICE_PALETTE)

        is_writing = h_info['is_writing']
        snap_radius = HAND_RADIUS * 0.4 if is_writing else HAND_RADIUS * 1.2
        orbit_radius = IDEAL_ORBIT_R * 0.1 if is_writing else IDEAL_ORBIT_R
        hand_radius_logic = HAND_RADIUS * 0.5 if is_writing else HAND_RADIUS * 1.5

        if best_d > snap_radius:
            ang = random.uniform(0, math.pi * 2)
            r_new = random.uniform(0, snap_radius * 0.8)
            self.x = target_x + math.cos(ang) * r_new
            self.y = target_y + math.sin(ang) * r_new
            dx = self.x - target_x
            dy = self.y - target_y
            best_d = math.hypot(dx, dy) or 0.001

        if best_d < hand_radius_logic:
            near_hand = True

            nx = dx / best_d
            ny = dy / best_d
            tx = -ny
            ty =  nx

            t_strength = ORBIT_FORCE * (1.0 - best_d / hand_radius_logic)
            if is_writing: t_strength *= 3.0
            fx += tx * t_strength * 9.0
            fy += ty * t_strength * 9.0

            shell_err = orbit_radius - best_d
            repel = REPEL_FORCE * 5.0 if is_writing else REPEL_FORCE
            fx += nx * shell_err * repel
            fy += ny * shell_err * repel
            
            if self.element == 'Fire':
                fy -= 0.6
            elif self.element == 'Ice':
                fy += 0.3

        fx += random.gauss(0, NOISE_F)
        fy += random.gauss(0, NOISE_F)

        self.vx += fx
        self.vy += fy

        damp = HAND_DAMPING if near_hand else DAMPING
        if is_writing: damp *= 0.80
        self.vx *= damp
        self.vy *= damp

        spd = math.hypot(self.vx, self.vy)
        cap = HAND_SPEED if near_hand else MAX_SPEED
        if spd > cap:
            self.vx = self.vx / spd * cap
            self.vy = self.vy / spd * cap

        self.x += self.vx
        self.y += self.vy

        return near_hand, spd

    def draw(self, layer, near_hand, spd):
        if self.alpha_base <= 0.01:
            return
            
        ix = int(self.x)
        iy = int(self.y)
        if not (0 <= ix < self.W and 0 <= iy < self.H):
            return

        twinkle  = 0.6 + math.sin(self.twinkle) * 0.4
        spd_frac = min(spd / HAND_SPEED, 1.0)
        a        = self.alpha_base * twinkle
        r        = max(1, int(self.size * (1.0 + spd_frac * 0.5)))

        outer = tuple(int(c * a * 0.20) for c in self.color)
        mid   = tuple(int(c * a * 0.50) for c in self.color)
        core  = tuple(int(c * a)        for c in self.color)

        cv2.circle(layer, (ix, iy), r * 3, outer, -1, cv2.LINE_AA)
        cv2.circle(layer, (ix, iy), r * 2, mid,   -1, cv2.LINE_AA)
        cv2.circle(layer, (ix, iy), r,     core,  -1, cv2.LINE_AA)


def main():
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    ret, probe = cap.read()
    if not ret:
        print("Camera not found. Check connection.")
        return

    H, W = probe.shape[:2]
    print(f"Camera: {W}x{H}")
    print("✨ Fire & Ice Magic Loaded! ✨")
    print("Show your Left Hand for FIRE, Right Hand for ICE.")
    print("Point your index finger to enter WRITING MODE.")
    print("Q = quit  |  R = reset particles")

    particles     = [Particle(W, H) for _ in range(PARTICLE_COUNT)]
    particle_layer = np.zeros((H, W, 3), dtype=np.uint8)
    canvas_layer   = np.zeros((H, W, 3), dtype=np.uint8)
    smooth_hands  = []

    hands_model = mp_hands.Hands(
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.65,
        min_tracking_confidence=0.60,
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_model.process(rgb)

        raw_hands = []
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label
                element = 'Fire' if label == 'Left' else 'Ice'

                lm = hand_landmarks.landmark
                sx = sum(lm[i].x * W for i in PALM_IDS) / len(PALM_IDS)
                sy = sum(lm[i].y * H for i in PALM_IDS) / len(PALM_IDS)
                
                tx = lm[8].x * W
                ty = lm[8].y * H

                writing = is_index_pointing(lm)

                raw_hands.append({
                    'pos': [sx, sy],
                    'tip': [tx, ty],
                    'is_writing': writing,
                    'element': element
                })

        new_smooth = []
        for raw in raw_hands:
            best_s = None
            best_d = float('inf')
            for s in smooth_hands:
                if s['element'] == raw['element']:
                    d = math.hypot(s['pos'][0]-raw['pos'][0], s['pos'][1]-raw['pos'][1])
                    if d < best_d:
                        best_d = d
                        best_s = s
            
            if best_s and best_d < 300:
                sx = best_s['pos'][0] + (raw['pos'][0] - best_s['pos'][0]) * SMOOTH
                sy = best_s['pos'][1] + (raw['pos'][1] - best_s['pos'][1]) * SMOOTH
                tx = best_s['tip'][0] + (raw['tip'][0] - best_s['tip'][0]) * SMOOTH
                ty = best_s['tip'][1] + (raw['tip'][1] - best_s['tip'][1]) * SMOOTH
                new_smooth.append({
                    'pos': [sx, sy],
                    'tip': [tx, ty],
                    'is_writing': raw['is_writing'],
                    'element': raw['element']
                })
                
                if raw['is_writing'] and best_s['is_writing']:
                    c_glow = FIRE_PALETTE[2] if raw['element'] == 'Fire' else ICE_PALETTE[2]
                    pt1 = (int(best_s['tip'][0]), int(best_s['tip'][1]))
                    pt2 = (int(tx), int(ty))
                    cv2.line(canvas_layer, pt1, pt2, c_glow, 16, cv2.LINE_AA)
                    cv2.line(canvas_layer, pt1, pt2, (255,255,255), 6, cv2.LINE_AA)
            else:
                new_smooth.append(raw)
                
        smooth_hands = new_smooth

        canvas_layer = (canvas_layer * CANVAS_FADE).astype(np.uint8)
        particle_layer = (particle_layer * TRAIL_FADE).astype(np.uint8)

        n_hands = len(smooth_hands)
        for i, p in enumerate(particles):
            assigned_hands = [smooth_hands[i % n_hands]] if n_hands > 0 else []
            near, spd = p.update(assigned_hands)
            p.draw(particle_layer, near, spd)

        dark_cam = cv2.convertScaleAbs(frame, alpha=CAM_BRIGHTNESS, beta=0)
        combo = cv2.add(dark_cam, canvas_layer)
        output = cv2.add(combo, particle_layer)

        n = len(smooth_hands)
        label = "No hands" if n == 0 else ("1 hand" if n == 1 else "Both hands")
        y_hud = 36
        cv2.putText(output, label, (16, y_hud),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (80, 220, 80) if n else (80, 80, 80), 1, cv2.LINE_AA)
        
        cv2.putText(output, "Left: Fire | Right: Ice | Point: Draw", (16, H - 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(output, "Q quit  R reset", (16, H - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1, cv2.LINE_AA)

        cv2.imshow("Space Particles", output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            particles      = [Particle(W, H) for _ in range(PARTICLE_COUNT)]
            particle_layer = np.zeros((H, W, 3), dtype=np.uint8)
            canvas_layer   = np.zeros((H, W, 3), dtype=np.uint8)

    cap.release()
    hands_model.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()