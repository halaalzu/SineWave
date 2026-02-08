import { useState, useEffect, useRef } from "react";
import CassetteLayout from "@/components/CassetteLayout";
import { CameraFeed } from "@/components/CameraFeed";
import { useToast } from "@/hooks/use-toast";

// Hot Cross Buns note mapping
const POSE_NOTES: Record<string, { freq: number; name: string }> = {
  '1': { freq: 329.63, name: 'E' },  // E4 - highest note
  '2': { freq: 293.66, name: 'D' },  // D4 - middle note
  '3': { freq: 261.63, name: 'C' }   // C4 - lowest note
};

const Freestyle = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [geminiAnalysis, setGeminiAnalysis] = useState<string>("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [lastPose, setLastPose] = useState<string | null>(null);
  const [lastNote, setLastNote] = useState<string>("");
  const [audioEnabled, setAudioEnabled] = useState(false);
  const { toast } = useToast();
  const audioContextRef = useRef<AudioContext | null>(null);
  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const lastPlayTimeRef = useRef<number>(0);
  const poseStartTimeRef = useRef<number>(0);
  const activeOscillatorsRef = useRef<OscillatorNode[]>([]);
  const activeMasterGainRef = useRef<GainNode | null>(null);

  // Initialize audio context on user interaction
  const initAudio = () => {
    if (!audioContextRef.current) {
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
      setAudioEnabled(true);
      toast({
        title: "🎵 Audio Enabled",
        description: "Piano sounds ready for pose detection!",
      });
    }
  };

  // Generate piano-like tone with harmonics
  const generatePianoTone = (frequency: number, duration: number = 0.5) => {
    if (!audioContextRef.current) return;

    // Stop any currently playing notes to prevent overlap
    if (activeMasterGainRef.current) {
      const now = audioContextRef.current.currentTime;
      activeMasterGainRef.current.gain.cancelScheduledValues(now);
      activeMasterGainRef.current.gain.setValueAtTime(activeMasterGainRef.current.gain.value, now);
      activeMasterGainRef.current.gain.linearRampToValueAtTime(0, now + 0.01); // Quick fade out
    }
    
    activeOscillatorsRef.current.forEach(osc => {
      try {
        osc.stop();
      } catch (e) {
        // Oscillator might already be stopped
      }
    });
    activeOscillatorsRef.current = [];

    const audioContext = audioContextRef.current;
    const now = audioContext.currentTime;
    
    // Create master gain for envelope
    const masterGain = audioContext.createGain();
    masterGain.connect(audioContext.destination);
    masterGain.gain.setValueAtTime(0, now);
    activeMasterGainRef.current = masterGain;
    
    // Attack, Decay, Sustain, Release envelope
    const attack = 0.01;
    const decay = 0.05;
    const sustain = 0.6;
    const release = 0.1;
    
    masterGain.gain.linearRampToValueAtTime(0.3, now + attack);
    masterGain.gain.linearRampToValueAtTime(0.3 * sustain, now + attack + decay);
    masterGain.gain.setValueAtTime(0.3 * sustain, now + duration - release);
    masterGain.gain.linearRampToValueAtTime(0, now + duration);
    
    // Add harmonics for piano-like sound
    const harmonics = [
      { mult: 1.0, amp: 1.0 },    // Fundamental
      { mult: 2.0, amp: 0.5 },    // First overtone
      { mult: 3.0, amp: 0.25 },   // Second overtone
      { mult: 4.0, amp: 0.125 },  // Third overtone
      { mult: 5.0, amp: 0.0625 }  // Fourth overtone
    ];
    
    harmonics.forEach(({ mult, amp }) => {
      const oscillator = audioContext.createOscillator();
      const gainNode = audioContext.createGain();
      
      oscillator.frequency.value = frequency * mult;
      oscillator.type = 'sine';
      gainNode.gain.value = amp;
      
      oscillator.connect(gainNode);
      gainNode.connect(masterGain);
      
      oscillator.start(now);
      oscillator.stop(now + duration);
      
      activeOscillatorsRef.current.push(oscillator);
    });
    
    // Clean up after note finishes
    setTimeout(() => {
      activeOscillatorsRef.current = [];
      activeMasterGainRef.current = null;
    }, duration * 1000 + 100);
  };

  // Play note for detected pose
  const playNote = (pose: string) => {
    const noteInfo = POSE_NOTES[pose];
    if (noteInfo && audioContextRef.current) {
      generatePianoTone(noteInfo.freq);
      setLastNote(noteInfo.name);
    }
  };

  // Poll for pose changes
  const checkForPoseChange = async () => {
    try {
      const response = await fetch('http://localhost:5001/api/current_pose');
      if (response.ok) {
        const data = await response.json();
        const currentPose = data.pose;
        const now = Date.now();
        
        // Play if it's a musical pose (1, 2, or 3)
        if (POSE_NOTES[currentPose]) {
          // If pose changed, play immediately
          if (currentPose !== lastPose) {
            playNote(currentPose);
            setLastPose(currentPose);
            lastPlayTimeRef.current = now;
            poseStartTimeRef.current = now;
          } 
          // If same pose held for 300ms, allow re-trigger (for repeated notes like E-E or D-D)
          else if (now - lastPlayTimeRef.current > 300) {
            playNote(currentPose);
            lastPlayTimeRef.current = now;
          }
        } else {
          // Reset when no musical pose detected
          setLastPose(currentPose);
        }
      }
    } catch (error) {
      console.error('Error checking pose:', error);
    }
  };

  // Start/stop pose polling
  useEffect(() => {
    if (audioEnabled) {
      pollingIntervalRef.current = setInterval(checkForPoseChange, 50); // 50ms for faster response
    }
    
    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
      }
    };
  }, [audioEnabled, lastPose]);

  const startRecording = async () => {
    try {
      const response = await fetch('http://localhost:5001/api/session/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: 'default_user',
          session_type: 'freestyle'
        })
      });
      
      if (!response.ok) throw new Error('Failed to start recording');
      
      const data = await response.json();
      setSessionId(data.session_id);
      setIsRecording(true);
      
      // Start fetching live feedback
      fetchLiveFeedback();
      
      toast({
        title: "Recording Started",
        description: "Practice your hand movements freely!",
      });
    } catch (error) {
      console.error('Error starting recording:', error);
      toast({
        title: "Error",
        description: "Failed to start recording. Make sure Flask server is running.",
        variant: "destructive"
      });
    }
  };

  const fetchLiveFeedback = async () => {
    setIsAnalyzing(true);
    try {
      const response = await fetch('http://localhost:5001/api/gemini/live-session/default_user', {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (response.ok) {
        const data = await response.json();
        setGeminiAnalysis(data.analysis || "Keep moving! I'm analyzing your movements...");
      }
    } catch (error) {
      console.error('Error fetching feedback:', error);
    }
    setIsAnalyzing(false);
  };

  const stopRecording = async () => {
    if (!sessionId) return;
    
    try {
      const response = await fetch('http://localhost:5001/api/session/stop', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId })
      });
      
      if (!response.ok) throw new Error('Failed to stop recording');
      
      const data = await response.json();
      setIsRecording(false);
      setSessionId(null);
      
      // Get final analysis
      setTimeout(() => fetchLiveFeedback(), 1000);
      
      toast({
        title: "Recording Saved",
        description: `Session completed with ${data.frames} frames recorded.`,
      });
      
      // Refresh feedback after save
      setTimeout(() => fetchLiveFeedback(), 500);
    } catch (error) {
      console.error('Error stopping recording:', error);
      toast({
        title: "Error",
        description: "Failed to stop recording.",
        variant: "destructive"
      });
    }
  };

  return (
    <CassetteLayout
      verticalText="PLAY"
      showBackButton
      showRedPanel={false}
      rightVerticalText="FREE STYLE"
    >
      <div className="mt-8 pr-0 md:pr-12">
        {/* Title */}
        <div className="mb-6 flex items-center justify-between">
          <h1 className="font-display text-4xl md:text-5xl tracking-[0.25em] text-foreground leading-[0.9]">
            FREE STYLE
          </h1>
          {isRecording && (
            <div className="flex items-center gap-2 bg-foreground px-3 py-1 border-2 border-primary">
              <div className="w-3 h-3 rounded-full bg-primary animate-blink" />
              <span className="font-display text-primary tracking-widest text-sm">REC</span>
            </div>
          )}
        </div>

        {/* Camera Feed */}
        <div className="mb-6 border-[5px] border-foreground shadow-card-retro overflow-hidden">
          <CameraFeed className="w-full aspect-video" />
        </div>

        {/* Instructions */}
        <div className="bg-card border-4 border-foreground p-4 mb-6 shadow-card-retro">
          <div className="font-display text-sm tracking-widest text-foreground/70 mb-2">
            📋 INSTRUCTIONS
          </div>
          <ul className="text-xs font-bold tracking-wider text-foreground/80 space-y-1">
            <li>• Move your hands naturally in front of the camera</li>
            <li>• Practice different movements at your own pace</li>
            <li>• Press RECORD to track your session</li>
            <li>• Your movements will be analyzed for rehabilitation progress</li>
            <li className="text-primary">• Make poses ☝️ (E), ✌️ (D), 🤟 (C) to play Hot Cross Buns!</li>
          </ul>
        </div>

        {/* Audio Control */}
        {!audioEnabled && (
          <button
            onClick={initAudio}
            className="w-full bg-primary border-[5px] border-foreground px-4 py-3 font-display text-lg tracking-widest text-primary-foreground shadow-card-retro hover:-translate-x-[2px] hover:-translate-y-[2px] hover:shadow-card-hover active:translate-x-[2px] active:translate-y-[2px] active:shadow-card-active transition-all mb-6"
          >
            🎹 ENABLE PIANO SOUNDS
          </button>
        )}

        {/* Pose & Note Display */}
        {audioEnabled && lastNote && (
          <div className="bg-primary/10 border-4 border-primary p-3 mb-6 shadow-card-retro">
            <div className="flex items-center justify-between">
              <div className="font-display text-sm tracking-wider text-foreground/70">
                🎵 LAST NOTE
              </div>
              <div className="font-display text-3xl tracking-widest text-primary">
                {lastNote}
              </div>
            </div>
          </div>
        )}

        {/* Instructions */}
        <div className="bg-card border-4 border-foreground p-4 mb-6 shadow-card-retro">
          <div className="font-display text-sm tracking-widest text-foreground/70 mb-2">
            📋 INSTRUCTIONS
          </div>
          <ul className="text-xs font-bold tracking-wider text-foreground/80 space-y-1">
            <li>• Move your hands naturally in front of the camera</li>
            <li>• Practice different movements at your own pace</li>
            <li>• Press RECORD to track your session</li>
            <li>• Your movements will be analyzed for rehabilitation progress</li>
          </ul>
        </div>

        {/* Controls */}
        <div className="flex gap-3 mb-6">
          {!isRecording ? (
            <button
              onClick={startRecording}
              className="flex-1 bg-primary border-[5px] border-foreground px-4 py-4 font-display text-xl tracking-widest text-primary-foreground shadow-card-retro hover:-translate-x-[2px] hover:-translate-y-[2px] hover:shadow-card-hover active:translate-x-[2px] active:translate-y-[2px] active:shadow-card-active transition-all"
            >
              ⏺ RECORD
            </button>
          ) : (
            <button
              onClick={stopRecording}
              className="flex-1 bg-secondary border-[5px] border-foreground px-4 py-4 font-display text-xl tracking-widest text-secondary-foreground shadow-card-retro hover:-translate-x-[2px] hover:-translate-y-[2px] hover:shadow-card-hover active:translate-x-[2px] active:translate-y-[2px] active:shadow-card-active transition-all"
            >
              ⏹ STOP
            </button>
          )}
        </div>

        {/* Live AI Feedback */}
        {(isRecording || geminiAnalysis) && (
          <div className="bg-card border-4 border-foreground p-4 shadow-card-retro">
            <div className="flex items-center justify-between mb-3">
              <div className="font-display text-lg tracking-[0.2em] text-foreground">
                🤖 AI COACH FEEDBACK
              </div>
              {isAnalyzing && (
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-primary animate-pulse" />
                  <span className="text-xs font-bold text-foreground/50 tracking-wider">ANALYZING...</span>
                </div>
              )}
            </div>
            <div className="text-sm font-bold text-foreground/80 tracking-wider leading-relaxed">
              {geminiAnalysis || "Start recording to get live feedback on your movements..."}
            </div>
            {isRecording && (
              <button
                onClick={fetchLiveFeedback}
                disabled={isAnalyzing}
                className="mt-3 bg-primary border-3 border-foreground px-4 py-2 font-display text-sm tracking-widest text-primary-foreground shadow-btn hover:shadow-btn-hover active:shadow-btn-active transition-all disabled:opacity-40"
              >
                🔄 REFRESH FEEDBACK
              </button>
            )}
          </div>
        )}
      </div>
    </CassetteLayout>
  );
};

export default Freestyle;
