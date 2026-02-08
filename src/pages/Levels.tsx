import { useState } from "react";
import { useNavigate } from "react-router-dom";
import CassetteLayout from "@/components/CassetteLayout";

interface Exercise {
  name: string;
  description: string;
  icon: string;
  completed: boolean;
}

interface Level {
  level: number;
  name: string;
  difficulty: string;
  exercises: Exercise[];
  unlocked: boolean;
}

const initialLevels: Level[] = [
  {
    level: 1,
    name: "HOT CROSS BUNS",
    difficulty: "RHYTHM GAME",
    unlocked: true,
    exercises: [
      { name: "Hot Cross Buns", description: "Follow the hand motion rhythm", icon: "🎵", completed: false },
    ],
  },
  {
    level: 2,
    name: "BUILD UP",
    difficulty: "MEDIUM",
    unlocked: true,
    exercises: [
      { name: "Grip Squeeze", description: "Squeeze and release a ball", icon: "✊", completed: false },
      { name: "Finger Spread", description: "Max finger spread, hold 3s", icon: "🖐️", completed: false },
      { name: "Thumb Circles", description: "Rotate thumb in circles", icon: "👍", completed: false },
    ],
  },
  {
    level: 3,
    name: "CHALLENGE",
    difficulty: "HARD",
    unlocked: false,
    exercises: [
      { name: "Rapid Taps", description: "Fast sequential finger taps", icon: "☝️", completed: false },
      { name: "Complex Grip", description: "Grip objects of varying sizes", icon: "🤏", completed: false },
      { name: "Full Range", description: "Complete range of motion test", icon: "💪", completed: false },
    ],
  },
];

const Levels = () => {
  const [levels] = useState<Level[]>(initialLevels);
  const [expandedLevel, setExpandedLevel] = useState<number | null>(null);
  const navigate = useNavigate();

  return (
    <CassetteLayout
      verticalText="LEVELS"
      showBackButton
      showRedPanel={false}
      rightVerticalText="RETROFIT THERAPY"
    >
      <div className="mb-8 mt-8">
        <h1 className="font-display text-5xl md:text-6xl tracking-[0.25em] text-foreground leading-[0.9]">
          THERAPY<br />LEVELS
        </h1>
        <p className="mt-4 text-sm font-bold tracking-wider text-muted-foreground">
          COMPLETE EACH LEVEL TO UNLOCK THE NEXT
        </p>
      </div>

      <div className="flex flex-col gap-6 pr-0 md:pr-12">
        {levels.map((level) => {
          const completedCount = level.exercises.filter((e) => e.completed).length;
          const totalCount = level.exercises.length;
          const isExpanded = expandedLevel === level.level;

          return (
            <div key={level.level} className="relative">
              {/* Level card */}
              <div
                onClick={() => {
                  if (level.unlocked) {
                    setExpandedLevel(isExpanded ? null : level.level);
                  }
                }}
                className={`
                  border-[6px] border-foreground p-6 relative shadow-card-retro transition-all duration-300 cursor-pointer
                  ${level.unlocked
                    ? "bg-primary hover:-translate-x-[2px] hover:-translate-y-[2px] hover:shadow-card-hover active:translate-x-[2px] active:translate-y-[2px] active:shadow-card-active"
                    : "bg-muted opacity-60 cursor-not-allowed"
                  }
                `}
              >
                {/* Inner border */}
                <div className="absolute top-2 left-2 right-2 bottom-2 border-2 border-foreground/20 pointer-events-none" />

                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-display text-lg tracking-[0.3em] text-primary-foreground/70">
                      LEVEL {String(level.level).padStart(2, "0")}
                    </div>
                    <div className="font-display text-3xl md:text-4xl tracking-[0.2em] text-primary-foreground drop-shadow-[2px_2px_0_rgba(0,0,0,0.3)]">
                      {level.name}
                    </div>
                    <div className="text-primary-foreground/80 text-xs font-bold tracking-wider mt-1">
                      DIFFICULTY: {level.difficulty}
                    </div>
                  </div>

                  <div className="text-right">
                    {level.unlocked ? (
                      <>
                        <div className="font-display text-3xl tracking-widest text-primary-foreground">
                          {completedCount}/{totalCount}
                        </div>
                        <div className="text-primary-foreground/70 text-xs font-bold tracking-wider">
                          COMPLETE
                        </div>
                        {/* Progress bar */}
                        <div className="w-24 h-3 border-2 border-foreground/40 mt-2 bg-foreground/20">
                          <div
                            className="h-full bg-card transition-all duration-500"
                            style={{ width: `${(completedCount / totalCount) * 100}%` }}
                          />
                        </div>
                      </>
                    ) : (
                      <div className="font-display text-4xl">🔒</div>
                    )}
                  </div>
                </div>
              </div>

              {/* Expanded exercises */}
              {isExpanded && level.unlocked && (
                <div className="border-x-[6px] border-b-[6px] border-foreground bg-card shadow-card-retro">
                  {level.exercises.map((exercise, idx) => (
                    <div
                      key={idx}
                      className={`
                        p-4 flex items-center gap-4
                        ${idx < level.exercises.length - 1 ? "border-b-2 border-foreground/20" : ""}
                      `}
                    >
                      <span className="text-2xl">{exercise.icon}</span>
                      <div className="flex-1">
                        <div className="font-display text-xl tracking-[0.15em] text-foreground">
                          {exercise.name}
                        </div>
                        <div className="text-xs font-bold text-foreground/60 tracking-wider">
                          {exercise.description}
                        </div>
                      </div>
                      <div className="font-display text-2xl">
                        {exercise.completed ? (
                          <span className="text-success">✓</span>
                        ) : (
                          <span className="text-foreground/30">○</span>
                        )}
                      </div>
                    </div>
                  ))}

                  {/* Start level button */}
                  <div className="p-4 border-t-2 border-foreground/20">
                    <button
                      onClick={() => navigate(`/level/${level.level}`)}
                      className="w-full bg-primary border-4 border-foreground p-3 font-display text-xl tracking-widest text-primary-foreground shadow-btn hover:-translate-x-0.5 hover:-translate-y-0.5 hover:shadow-btn-hover active:translate-x-0.5 active:translate-y-0.5 active:shadow-btn-active transition-all"
                    >
                      ▶ START LEVEL
                    </button>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </CassetteLayout>
  );
};

export default Levels;
