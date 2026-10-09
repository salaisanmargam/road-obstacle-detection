import cv2
import time
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path("D:/road obstacle detection/road_damage_research")))
from phase3.integration.optimized_pipeline import OptimizedVideoPipeline
from phase3.security.security_gateway import SecurityGateway

RDD_TEST = Path("D:/road obstacle detection/rdd2022/RDD_SPLIT/test/images")
PHASE1_CKPT = Path("D:/road obstacle detection/road_damage_research/experiments/phase1_final_baseline/final/weights/best.pt")

def draw_overlay(img, res, fps, latency, condition, method, security_status):
    # Draw YOLO boxes
    if res and res[0].boxes:
        for box in res[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = res[0].names[cls_id]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{cls_name} {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
    # HUD overlay
    hud = [
        f"SECURITY: {security_status}",
        f"CONDITION: {condition}",
        f"ENHANCEMENT: {'ACTIVE ('+method+')' if method != 'bypass' else 'INACTIVE'}",
        f"FPS: {fps:.1f}",
        f"LATENCY: {latency:.1f} ms",
    ]
    
    y0, dy = 30, 30
    for i, text in enumerate(hud):
        y = y0 + i * dy
        # Background for text
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        # Highlight colors
        color = (255, 255, 255)
        if "SECURITY: SUSPICIOUS" in text: color = (0, 0, 255)
        elif "ENHANCEMENT: ACTIVE" in text: color = (0, 255, 255)
        elif "FPS:" in text and fps >= 20: color = (0, 255, 0)
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
    return img

def main():
    print("Initializing Phase 3 Real-Time Demo...")
    try:
        gateway = SecurityGateway.load(log_to_file=False)
    except Exception as e:
        print(f"Gateway load error: {e}. Running without security.")
        gateway = None

    pipe = OptimizedVideoPipeline(
        yolo_ckpt=str(PHASE1_CKPT),
        gateway=gateway,
        condition_interval=5,
        security_interval=30
    )
    
    # Use RDD2022 test images as a synthetic video stream
    img_paths = sorted(RDD_TEST.glob("*.jpg"))
    if not img_paths:
        print("No images found in RDD2022 test directory.")
        return
        
    print("Starting continuous processing... (Press 'q' to exit)")
    
    # Dummy flow data (normal)
    dummy_flow = np.zeros(77, dtype=np.float32)
    
    fps_history = []
    
    for i, p in enumerate(img_paths):
        img = cv2.imread(str(p))
        if img is None: continue
        
        # Simulate an attack every 100 frames for 5 frames
        current_flow = dummy_flow.copy()
        if 100 <= (i % 200) <= 105:
            # Force high values to trigger Random Forest 'suspicious'
            current_flow += 10.0
            
        t0 = time.perf_counter()
        out = pipe.process_frame(img, flow_features=current_flow if gateway else None)
        t_total = (time.perf_counter() - t0) * 1000
        
        fps = 1000.0 / t_total if t_total > 0 else 0
        fps_history.append(fps)
        if len(fps_history) > 10: fps_history.pop(0)
        avg_fps = sum(fps_history) / len(fps_history)
        
        if out["blocked"]:
            display_img = img.copy()
            cv2.putText(display_img, "NETWORK ATTACK DETECTED - FRAME BLOCKED", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
            display_img = draw_overlay(display_img, None, avg_fps, t_total, "UNKNOWN", "bypass", "SUSPICIOUS (BLOCKED)")
        else:
            display_img = draw_overlay(img.copy(), out["result"], avg_fps, t_total, out["condition"], out["method"], "TRUSTED")
            
        cv2.imshow("Phase 3 - Secure Road Damage Detection", display_img)
        
        # Wait 1ms for display refresh and key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Clean exit requested.")
            break

    cv2.destroyAllWindows()
    print("\nDemo finished.")
    summ = pipe.summary()
    print(f"Final Session Avg FPS: {summ['fps']}")
    print(f"Final Session Avg Latency: {summ['avg_total_ms']} ms")

if __name__ == "__main__":
    main()
