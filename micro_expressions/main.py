import argparse
import sys
import time
import cv2
from pathlib import Path
from engine import StressDetectionEngine

# Ensure modules are importable
sys.path.insert(0, str(Path(__file__).parent))

def parse_arguments():
    parser = argparse.ArgumentParser(description="Multi-Modal Stress Detection System")
    parser.add_argument('--modules', nargs='+', 
                       choices=['face', 'rppg', 'facs', 'stress'],
                       default=['face', 'rppg', 'facs', 'stress'],
                       help='Active pipeline modules')
    parser.add_argument('--camera', type=int, default=0, help='Input device ID')
    parser.add_argument('--debug', action='store_true', help='Overlay debug info')
    parser.add_argument('--validate', action='store_true', 
                       help='Enable heart rate validation mode')
    parser.add_argument('--minimal', '--ar-mode', action='store_true',
                       help='Enable minimal AR-like UI with rotatable button')
    return parser.parse_args()

def load_module_instance(name, config):
    """Factory method for module instantiation."""
    if name == 'face':
        from face_detection import FaceDetectionModule
        return FaceDetectionModule()
    elif name == 'rppg':
        from rppg_heart_rate import RPPGModule
        enable_validation = config.get('enable_validation', False)
        return RPPGModule(enable_validation=enable_validation)
    elif name == 'facs':
        from facs_action_units import FACSModule
        return FACSModule()
    elif name == 'stress':
        from stress_detector import StressDetectorModule
        return StressDetectorModule()
    return None

def main():
    args = parse_arguments()
    
    # Engine Configuration
    config = {
        'camera_id': args.camera,
        'debug_mode': args.debug,
        'enable_validation': args.validate,
        'is_video_file': False,
        'minimal_mode': args.minimal
    }

    print(f"Initializing Stress Detection Engine on Camera {args.camera}...")
    if args.minimal:
        print("Minimal AR mode enabled - clean interface with rotatable button")
    engine = StressDetectionEngine(config)
    
    # Initialize AR UI Controller and Hand Gesture Detector if minimal mode
    ar_controller = None
    hand_detector = None
    if args.minimal:
        from ar_ui_controller import ARUIController
        from hand_gesture_detector import HandGestureDetector
        ar_controller = ARUIController(button_position='top-right')
        try:
            hand_detector = HandGestureDetector()
            if hand_detector.detector is None:
                print("[WARN] Warning: Hand detector not initialized - hand gestures will not work")
            else:
                print("[OK] Hand gesture detection enabled (MediaPipe Hands)")
        except Exception as e:
            print(f"[WARN] Error initializing hand detector: {e}")
            hand_detector = None

    # Dependency Injection: Face detection is the foundation
    if 'face' not in args.modules:
        args.modules.insert(0, 'face')

    # Load and attach modules strictly in order
    pipeline_order = ['face', 'rppg', 'facs', 'stress']
    rppg_module = None
    stress_module = None
    
    for module_name in pipeline_order:
        if module_name in args.modules:
            try:
                module = load_module_instance(module_name, config)
                engine.attach_module(module_name, module)
                print(f"[OK] Module loaded: {module_name.upper()}")
                
                # Store rPPG module reference for validation
                if module_name == 'rppg':
                    rppg_module = module
                elif module_name == 'stress':
                    stress_module = module
            except Exception as e:
                print(f"[ERROR] Critical Error loading {module_name}: {e}")
                sys.exit(1)
    
    # Validation instructions
    if args.validate and rppg_module:
        print("\n" + "="*60)
        print("VALIDATION MODE ENABLED")
        print("="*60)
        print("Instructions:")
        print("1. Start recording: Press 'v' key")
        print("2. Enter Apple Watch readings: Press 'a' key, then enter BPM")
        print("3. Stop recording: Press 's' key")
        print("4. View metrics: Press 'm' key")
        print("5. Save results: Press 'w' key")
        print("="*60 + "\n")
        rppg_module.start_validation()

    # Execution
    try:
        # Camera processing
        cap = cv2.VideoCapture(config['camera_id'])
        if not cap.isOpened():
            raise IOError(f"Cannot open camera {config['camera_id']}")
        
        window_name = 'Multi-Modal Stress Detection'
        if args.minimal and ar_controller:
            print("\n" + "="*60)
            print("PURE AR MODE ENABLED - Hand Gesture Control")
            print("="*60)
            print("Hand Gestures:")
            print("  - TWO-HAND PINCH & TWIST: Pinch with both hands, rotate to unlock")
            print("  - SINGLE-HAND ROTATION: Pinch with one hand, rotate wrist")
            print("  - Button color changes as you rotate (Green -> Yellow -> Orange -> Red -> Purple)")
            print("  - More analytics unlock as rotation increases")
            print("  - Auto-resets to minimal after 3 seconds of no interaction")
            print("="*60 + "\n")
        
        print("Engine started. Press 'q' to exit, 'r' to reset baseline.")
        if args.validate:
            print("Validation mode: 'v'=start, 'a'=add AW reading, 's'=stop, 'm'=metrics, 'w'=save")
        
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            # Update shared state
            h, w = frame.shape[:2]
            engine.shared_state.update({
                'frame': frame,
                'frame_dimensions': (h, w),
                'timestamp': time.time(),
                'is_video_file': False,
                'minimal_mode': args.minimal
            })
            
            # Process modules
            for name, module in engine.modules.items():
                module.process(engine.shared_state)
            
            # Process hand gestures if in minimal mode
            if args.minimal and hand_detector:
                # Process hand detection BEFORE rendering
                hand_detector.process(frame, engine.shared_state)
            
            # Render
            display_frame = frame.copy()
            
            if args.minimal and ar_controller:
                # Minimal mode: Only render AR UI, hide other overlays
                display_frame = ar_controller.render(display_frame, engine.shared_state)
                
                # Render hand landmarks for debugging (AFTER AR UI so landmarks are on top)
                if hand_detector:
                    gestures = engine.shared_state.get('hand_gestures', {})
                    if gestures:  # Only render if gestures exist
                        display_frame = hand_detector.render_hands(display_frame, gestures, engine.shared_state)
                    else:
                        # Show debug message if no gestures
                        cv2.putText(display_frame, "Waiting for hand detection...", (10, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            else:
                # Normal mode: Render all modules
                for name, module in engine.modules.items():
                    display_frame = module.render(display_frame, engine.shared_state)
            
            cv2.imshow(window_name, display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif args.validate and rppg_module:
                if key == ord('v'):  # Start validation
                    rppg_module.start_validation()
                    print("[OK] Validation recording started")
                elif key == ord('s'):  # Stop validation
                    rppg_module.stop_validation()
                elif key == ord('m'):  # Show metrics
                    metrics = rppg_module.get_validation_metrics()
                    if metrics:
                        rppg_module.validator.print_metrics()
                    else:
                        print("[WARN] Not enough data for metrics")
                elif key == ord('w'):  # Save results
                    filepath = rppg_module.save_validation_results()
                    if filepath:
                        print(f"[OK] Results saved to: {filepath}")
                elif key == ord('a'):  # Add Apple Watch reading
                    try:
                        aw_bpm = input("Enter Apple Watch BPM: ").strip()
                        aw_bpm = float(aw_bpm)
                        if rppg_module.add_validation_reading(aw_bpm):
                            print(f"[OK] Added: AW={aw_bpm} BPM, rPPG={rppg_module.current_bpm} BPM")
                        else:
                            print("[WARN] Validation not active or invalid reading")
                    except (ValueError, KeyboardInterrupt):
                        print("[WARN] Invalid input")
            elif key == ord('r') and stress_module:
                stress_module.reset_baseline()
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Final validation summary
        if args.validate and rppg_module and rppg_module.validator:
            if rppg_module.validator.is_recording:
                rppg_module.stop_validation()
            if len(rppg_module.validator.rppg_readings) > 0:
                save = input("\nSave validation results? (y/n): ").strip().lower()
                if save == 'y':
                    rppg_module.save_validation_results()
        
        # Cleanup
        for module in engine.modules.values():
            if hasattr(module, 'cleanup'):
                module.cleanup()
        
        if hand_detector:
            hand_detector.cleanup()
                
    except KeyboardInterrupt:
        print("\nSystem shutdown requested.")

if __name__ == "__main__":
    main()