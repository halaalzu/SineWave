import pygame
import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
FPS = 30

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 120, 255)
RED = (255, 0, 0)

# Initialize screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("FlowState - Hand Tracking")
clock = pygame.time.Clock()

# Hand landmark connections for drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (5, 9), (9, 13), (13, 17)  # Palm
]

def draw_landmarks_on_surface(surface, detection_result, width, height):
    """Draw hand landmarks on the pygame surface."""
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            # Draw connections
            for connection in HAND_CONNECTIONS:
                start_idx, end_idx = connection
                start_landmark = hand_landmarks[start_idx]
                end_landmark = hand_landmarks[end_idx]
                
                start_point = (int(start_landmark.x * width), 
                             int(start_landmark.y * height))
                end_point = (int(end_landmark.x * width), 
                           int(end_landmark.y * height))
                
                pygame.draw.line(surface, GREEN, start_point, end_point, 3)
            
            # Draw landmarks
            for landmark in hand_landmarks:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                pygame.draw.circle(surface, BLUE, (x, y), 6)
                pygame.draw.circle(surface, WHITE, (x, y), 3)

def draw_ui_overlay(surface):
    """Draw UI overlay with information."""
    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 24)
    
    # Title
    title_text = font.render("FlowState Hand Tracking", True, WHITE)
    title_shadow = font.render("FlowState Hand Tracking", True, BLACK)
    surface.blit(title_shadow, (22, 22))
    surface.blit(title_text, (20, 20))
    
    # Instructions
    instructions = [
        "ESC - Exit",
        "SPACE - Toggle Info"
    ]
    
    y_offset = SCREEN_HEIGHT - 80
    for instruction in instructions:
        inst_text = small_font.render(instruction, True, WHITE)
        inst_shadow = small_font.render(instruction, True, BLACK)
        surface.blit(inst_shadow, (22, y_offset + 2))
        surface.blit(inst_text, (20, y_offset))
        y_offset += 30

def main():
    # Setup MediaPipe
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'hand_landmarker.task')
    
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    frame_timestamp_ms = 0
    running = True
    show_info = True
    
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        show_info = not show_info
            
            # Capture frame
            success, frame = cap.read()
            if not success:
                print("Failed to capture frame")
                continue
            
            # Flip frame horizontally for selfie view
            frame = cv2.flip(frame, 1)
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Detect hand landmarks
            detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            frame_timestamp_ms += int(1000 / FPS)
            
            # Convert frame to pygame surface
            frame = cv2.resize(frame, (SCREEN_WIDTH, SCREEN_HEIGHT))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.rot90(frame)
            frame = np.flipud(frame)
            frame_surface = pygame.surfarray.make_surface(frame)
            
            # Draw everything
            screen.blit(frame_surface, (0, 0))
            
            # Draw hand landmarks
            draw_landmarks_on_surface(screen, detection_result, SCREEN_WIDTH, SCREEN_HEIGHT)
            
            # Draw UI overlay
            if show_info:
                draw_ui_overlay(screen)
            
            # Display FPS
            fps_text = pygame.font.Font(None, 24).render(f"FPS: {int(clock.get_fps())}", True, GREEN)
            screen.blit(fps_text, (SCREEN_WIDTH - 100, 20))
            
            # Update display
            pygame.display.flip()
            clock.tick(FPS)
    
    # Cleanup
    cap.release()
    pygame.quit()

if __name__ == "__main__":
    main()
