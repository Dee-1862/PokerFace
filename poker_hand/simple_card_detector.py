"""
Simple Card Detector - Detect MULTIPLE cards with stability filtering
Reduces flickering by only showing cards detected consistently
"""

import cv2
from ultralytics import YOLO
from collections import defaultdict

def main():
    print("🎴 Simple Card Detector (Multiple Cards - Stable)")
    print("=" * 50)
    
    # Load model
    print("Loading model...")
    model = YOLO('poker_v1.pt')
    print(f"✓ Model loaded. Classes: {len(model.names)}")
    
    # Open camera
    print("Opening camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return
    
    print("✓ Camera opened")
    print("\nPress 'q' to quit")
    print("=" * 50)
    
    # Stability tracking - two-stage system to eliminate flickering
    card_history = defaultdict(int)  # card_label -> frames_seen_count
    finalized_cards = {}  # card_label -> {card_data, last_seen_frame}
    STABILITY_THRESHOLD = 5  # Card must be seen in 5 frames to become finalized
    FINALIZE_THRESHOLD_MAX = 15  # Maximum frames to finalize (5-15 range)
    FADE_OUT_FRAMES = 30  # Keep finalized card visible for 30 frames even if not detected
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run YOLO detection - lower confidence to catch more cards
        results = model(frame, verbose=False, conf=0.5, iou=0.15)
        
        # Get ALL detected cards in this frame
        current_frame_cards = set()  # Use set for fast lookup
        detected_cards = []
        
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                
                current_frame_cards.add(label)
                
                # Add all detected cards to the list
                detected_cards.append({
                    'label': label,
                    'confidence': conf,
                    'box': box.xyxy[0].tolist()
                })
        
        # Update stability tracking
        # Increment count for cards seen in this frame
        for label in current_frame_cards:
            card_history[label] += 1
        
        # Decrement count for cards NOT seen in this frame (decay) - but only if not finalized
        for label in list(card_history.keys()):
            if label not in current_frame_cards and label not in finalized_cards:
                card_history[label] -= 1
                if card_history[label] <= 0:
                    del card_history[label]
        
        # Check for cards ready to be finalized (seen in 5-15 frames)
        for card in detected_cards:
            label = card['label']
            if label not in finalized_cards and STABILITY_THRESHOLD <= card_history[label] <= FINALIZE_THRESHOLD_MAX:
                # Finalize this card - lock it in permanently
                finalized_cards[label] = {
                    'label': label,
                    'confidence': card['confidence'],
                    'box': card['box'],
                    'last_seen_frame': frame_count
                }
                print(f"✓ Finalized card: {label} (seen in {card_history[label]} frames)")
        
        # Update finalized cards that are still being detected
        for card in detected_cards:
            label = card['label']
            if label in finalized_cards:
                # Update with latest detection (better position/confidence)
                finalized_cards[label].update({
                    'confidence': max(card['confidence'], finalized_cards[label]['confidence']),
                    'box': card['box'],
                    'last_seen_frame': frame_count
                })
        
        # Remove finalized cards that haven't been seen for too long
        for label in list(finalized_cards.keys()):
            frames_since_seen = frame_count - finalized_cards[label]['last_seen_frame']
            if frames_since_seen > FADE_OUT_FRAMES:
                del finalized_cards[label]
                print(f"✗ Removed finalized card: {label} (not seen for {frames_since_seen} frames)")
        
        # Build display list: finalized cards + stable non-finalized cards
        display_cards = {}
        
        # Add all finalized cards (these are locked in, no flickering)
        for label, card_data in finalized_cards.items():
            display_cards[label] = card_data
        
        # Add stable cards that aren't finalized yet
        for card in detected_cards:
            label = card['label']
            if label not in finalized_cards and card_history[label] >= STABILITY_THRESHOLD:
                # Keep highest confidence detection
                if label not in display_cards or card['confidence'] > display_cards[label]['confidence']:
                    display_cards[label] = card
        
        # Convert to list for display
        stable_cards_list = list(display_cards.values())
        
        # Draw stable detections (finalized cards won't flicker)
        if stable_cards_list:
            finalized_count = len([c for c in stable_cards_list if c['label'] in finalized_cards])
            # Show count in top-left
            cv2.putText(frame, f"Cards: {len(stable_cards_list)} (Finalized: {finalized_count})", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Draw each stable card
            for i, card in enumerate(stable_cards_list):
                x1, y1, x2, y2 = map(int, card['box'])
                label = card['label']
                conf = card['confidence']
                is_finalized = label in finalized_cards
                
                # Use green for finalized cards (locked in), blue for stable but not finalized
                if is_finalized:
                    color = (0, 255, 0)  # Green - finalized, no flickering
                    thickness = 3  # Thicker border for finalized cards
                else:
                    color = (255, 0, 0)  # Blue - stable but not yet finalized
                    thickness = 2
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                
                # Draw label on the box
                status = "✓" if is_finalized else ""
                text = f"{label} {status} ({conf:.2f})"
                cv2.putText(frame, text, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Show all stable card labels
            labels_text = ", ".join([card['label'] for card in stable_cards_list])
            # Split long text into multiple lines if needed
            y_offset = 60
            max_chars = 40
            for i in range(0, len(labels_text), max_chars):
                line = labels_text[i:i+max_chars]
                cv2.putText(frame, line, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20
        else:
            cv2.putText(frame, "No card detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Show frame
        cv2.imshow('Simple Card Detector', frame)
        
        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Done")

if __name__ == '__main__':
    main()