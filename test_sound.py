import pygame
import numpy as np
import math
import time

pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)

def generate_piano_tone(frequency, duration=0.5, volume=0.6):
    """Generate a piano-like tone with harmonics and ADSR envelope"""
    sample_rate = 22050
    num_samples = int(sample_rate * duration)
    
    # Piano harmonics (fundamental + overtones with decreasing amplitude)
    harmonics = [
        (1.0, 1.0),      # Fundamental
        (2.0, 0.5),      # 2nd harmonic
        (3.0, 0.25),     # 3rd harmonic
        (4.0, 0.125),    # 4th harmonic
        (5.0, 0.0625),   # 5th harmonic
    ]
    
    samples = []
    for i in range(num_samples):
        t = float(i) / sample_rate
        
        # ADSR envelope with release for piano sound
        attack_time = 0.01
        decay_time = 0.05
        sustain_level = 0.6
        release_start = duration - 0.1
        
        if t < attack_time:
            envelope = t / attack_time
        elif t < attack_time + decay_time:
            envelope = 1.0 - (1.0 - sustain_level) * ((t - attack_time) / decay_time)
        elif t < release_start:
            envelope = sustain_level
        else:
            # Release (fade out)
            envelope = sustain_level * (1.0 - (t - release_start) / (duration - release_start))
        
        # Sum all harmonics
        value = 0
        for harmonic_mult, harmonic_amp in harmonics:
            value += harmonic_amp * math.sin(2 * math.pi * frequency * harmonic_mult * t)
        
        # Apply envelope and volume
        value = int(volume * envelope * 32767 * value / len(harmonics))
        samples.append(value)
    
    # Create 1D array for mono sound
    sound_array = np.array(samples, dtype=np.int16)
    sound = pygame.sndarray.make_sound(sound_array)
    return sound

print('🎹 Testing Hot Cross Buns notes (E-D-C)...\n')

# Hot Cross Buns melody: E-D-C, E-D-C, C-C-C-C-D-D-D-D, E-D-C
notes = [
    (329.63, 'E'), (293.66, 'D'), (261.63, 'C'),  # Hot cross buns
    (329.63, 'E'), (293.66, 'D'), (261.63, 'C'),  # Hot cross buns
]

print('Playing: E D C (Hot cross buns)')
for freq, name in notes[:3]:
    print(f'🎵 {name} ({freq} Hz)')
    sound = generate_piano_tone(freq, duration=0.5, volume=0.6)
    sound.play()
    time.sleep(0.6)

time.sleep(0.3)

print('\nPlaying: E D C (Hot cross buns)')
for freq, name in notes[3:6]:
    print(f'🎵 {name} ({freq} Hz)')
    sound = generate_piano_tone(freq, duration=0.5, volume=0.6)
    sound.play()
    time.sleep(0.6)

print('\n✅ Audio test complete!')
print('💡 In the app:')
print('   ☝️  Pose 1 → E (highest)')
print('   ✌️  Pose 2 → D (middle)')
print('   🤟 Pose 3 → C (lowest)')
print('\n🎶 Play Hot Cross Buns: 1-2-3, 1-2-3')
pygame.quit()
