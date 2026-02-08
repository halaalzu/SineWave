import { useState, useEffect } from "react";
import { useNavigate, useParams } from "react-router-dom";
import CassetteLayout from "@/components/CassetteLayout";
import { CameraFeed } from "@/components/CameraFeed";
import { useToast } from "@/hooks/use-toast";

interface Exercise {
  name: string;
  description: string;
  icon: string;
  duration: number; // seconds
}

const levelExercises: Record<string, Exercise[]> = {
  "1": [
    { name: "Hot Cross Buns", description: "Follow the hand motion rhythm", icon: "🎵", duration: 90 },
  ],
  "2": [
    { name: "Grip Squeeze", description: "Squeeze and release a ball", icon: "✊", duration: 45 },
    { name: "Finger Spread", description: "Max finger spread, hold 3s", icon: "🖐️", duration: 45 },
    { name: "Thumb Circles", description: "Rotate thumb in circles", icon: "👍", duration: 45 },
  ],
  "3": [
    { name: "Rapid Taps", description: "Fast sequential finger taps", icon: "☝️", duration: 60 },
    { name: "Complex Grip", description: "Grip objects of varying sizes", icon: "🤏", duration: 60 },
    { name: "Full Range", description: "Complete range of motion test", icon: "💪", duration: 60 },
  ],
};

const LevelSession = () => {
  const { levelId } = useParams<{ levelId: string }>();
  const navigate = useNavigate();
  const { toast } = useToast();
  
  const [currentExerciseIndex, setCurrentExerciseIndex] = useState(0);
  const [isRecording, setIsRecording] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [timeRemaining, setTimeRemaining] = useState(0);
  
  const exercises = levelExercises[levelId || "1"] || levelExercises["1"];
  const currentExercise = exercises[currentExerciseIndex];

  useEffect(() => {
    if (timeRemaining > 0 && isRecording) {
      const timer = setTimeout(() => {
        setTimeRemaining(timeRemaining - 1);
      }, 1000);
      return () => clearTimeout(timer);
    } else if (timeRemaining === 0 && isRecording) {
      // Auto-advance to next exercise
      handleNextExercise();
    }
  }, [timeRemaining, isRecording]);

  const startRecording = async () => {
    try {
      const response = await fetch('http://localhost:5001/api/session/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: 'default_user',
          session_type: `level_${levelId}_exercise_${currentExerciseIndex}`
        })
      });
      
      if (!response.ok) throw new Error('Failed to start recording');
      
      const data = await response.json();
      setSessionId(data.session_id);
      setIsRecording(true);
      setTimeRemaining(currentExercise.duration);
      
      toast({
        title: "Exercise Started",
        description: currentExercise.name,
      });
    } catch (error) {
      console.error('Error starting recording:', error);
      toast({
        title: "Error",
        description: "Failed to start recording.",
        variant: "destructive"
      });
    }
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
      
      setIsRecording(false);
      setSessionId(null);
      setTimeRemaining(0);
    } catch (error) {
      console.error('Error stopping recording:', error);
    }
  };

  const handleNextExercise = async () => {
    await stopRecording();
    
    if (currentExerciseIndex < exercises.length - 1) {
      setCurrentExerciseIndex(currentExerciseIndex + 1);
      toast({
        title: "Exercise Complete!",
        description: "Moving to next exercise...",
      });
    } else {
      toast({
        title: "Level Complete!",
        description: "Great work! All exercises finished.",
      });
      setTimeout(() => navigate("/levels"), 2000);
    }
  };

  const handleSkip = () => {
    if (isRecording) {
      stopRecording();
    }
    if (currentExerciseIndex < exercises.length - 1) {
      setCurrentExerciseIndex(currentExerciseIndex + 1);
    } else {
      navigate("/levels");
    }
  };

  return (
    <CassetteLayout
      verticalText="THERAPY"
      showBackButton
      showRedPanel={false}
      rightVerticalText={`LEVEL ${levelId}`}
    >
      <div className="mt-8 pr-0 md:pr-12">
        {/* Header */}
        <div className="mb-6">
          <div className="flex items-center justify-between mb-2">
            <h1 className="font-display text-4xl tracking-[0.25em] text-foreground leading-[0.9]">
              LEVEL {levelId?.padStart(2, '0')}
            </h1>
            <div className="font-display text-lg tracking-widest text-foreground/70">
              {currentExerciseIndex + 1}/{exercises.length}
            </div>
          </div>
          
          {/* Progress bar */}
          <div className="w-full h-4 border-4 border-foreground bg-card shadow-card-retro">
            <div
              className="h-full bg-primary transition-all duration-300"
              style={{ width: `${((currentExerciseIndex + 1) / exercises.length) * 100}%` }}
            />
          </div>
        </div>

        {/* Camera Feed */}
        <div className="mb-6 border-[5px] border-foreground shadow-card-retro overflow-hidden relative">
          <CameraFeed className="w-full aspect-video" />
          
          {/* Exercise overlay */}
          <div className="absolute top-4 left-4 right-4 bg-foreground/90 border-4 border-primary p-4">
            <div className="flex items-center gap-3">
              <span className="text-4xl">{currentExercise.icon}</span>
              <div className="flex-1">
                <div className="font-display text-2xl tracking-[0.15em] text-primary leading-tight">
                  {currentExercise.name}
                </div>
                <div className="text-xs font-bold text-primary/80 tracking-wider mt-1">
                  {currentExercise.description}
                </div>
              </div>
              {isRecording && (
                <div className="text-right">
                  <div className="font-display text-3xl text-primary">
                    {timeRemaining}s
                  </div>
                  <div className="flex items-center gap-2 justify-end">
                    <div className="w-2 h-2 rounded-full bg-primary animate-blink" />
                    <span className="text-xs font-bold text-primary tracking-widest">REC</span>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Controls */}
        <div className="flex gap-3">
          {!isRecording ? (
            <>
              <button
                onClick={startRecording}
                className="flex-1 bg-primary border-[5px] border-foreground px-4 py-4 font-display text-xl tracking-widest text-primary-foreground shadow-card-retro hover:-translate-x-[2px] hover:-translate-y-[2px] hover:shadow-card-hover active:translate-x-[2px] active:translate-y-[2px] active:shadow-card-active transition-all"
              >
                ▶ START
              </button>
              <button
                onClick={handleSkip}
                className="bg-card border-[5px] border-foreground px-4 py-4 font-display text-xl tracking-widest text-foreground shadow-card-retro hover:-translate-x-[2px] hover:-translate-y-[2px] hover:shadow-card-hover active:translate-x-[2px] active:translate-y-[2px] active:shadow-card-active transition-all"
              >
                SKIP →
              </button>
            </>
          ) : (
            <>
              <button
                onClick={stopRecording}
                className="flex-1 bg-secondary border-[5px] border-foreground px-4 py-4 font-display text-xl tracking-widest text-secondary-foreground shadow-card-retro hover:-translate-x-[2px] hover:-translate-y-[2px] hover:shadow-card-hover active:translate-x-[2px] active:translate-y-[2px] active:shadow-card-active transition-all"
              >
                ⏹ STOP
              </button>
              <button
                onClick={handleNextExercise}
                className="bg-primary border-[5px] border-foreground px-4 py-4 font-display text-xl tracking-widest text-primary-foreground shadow-card-retro hover:-translate-x-[2px] hover:-translate-y-[2px] hover:shadow-card-hover active:translate-x-[2px] active:translate-y-[2px] active:shadow-card-active transition-all"
              >
                NEXT →
              </button>
            </>
          )}
        </div>
      </div>
    </CassetteLayout>
  );
};

export default LevelSession;
