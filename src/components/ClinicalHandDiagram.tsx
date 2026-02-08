import { useState } from "react";

export interface JointData {
  id: string;
  label: string;
  smoothness: number; // 0-100 SPARC-based
  rom: number; // Range of motion 0-100% of healthy baseline
  trajectoryAccuracy: number; // 0-100
  consistency: number; // 0-100
  movementDuration: number; // ms average
  baselineDuration: number; // healthy baseline ms
  cx: number; // SVG x position
  cy: number; // SVG y position
}

const getStatusColor = (score: number) => {
  if (score >= 75) return "hsl(var(--success))";
  if (score >= 50) return "hsl(40 80% 50%)";
  if (score >= 30) return "hsl(25 80% 50%)";
  return "hsl(var(--cassette-red))";
};

const getStatusLabel = (score: number) => {
  if (score >= 75) return "STRONG";
  if (score >= 50) return "MODERATE";
  if (score >= 30) return "AT RISK";
  return "WEAK";
};

const getPulseClass = (score: number) => {
  if (score < 30) return "animate-blink";
  return "";
};

interface ClinicalHandDiagramProps {
  joints: JointData[];
  onJointSelect?: (joint: JointData) => void;
  selectedJointId?: string | null;
  handSide?: 'left' | 'right';
  hasData?: boolean;
}

const ClinicalHandDiagram = ({ 
  joints, 
  onJointSelect, 
  selectedJointId, 
  handSide = 'right',
  hasData = true 
}: ClinicalHandDiagramProps) => {
  const [hoveredJoint, setHoveredJoint] = useState<string | null>(null);

  const getOverallScore = (j: JointData) =>
    Math.round((j.smoothness + j.rom + j.trajectoryAccuracy + j.consistency) / 4);

  if (!hasData) {
    return (
      <div className="relative">
        <div className="flex items-center gap-3 mb-4">
          <span className="text-4xl">{handSide === 'left' ? '🤚' : '✋'}</span>
          <div>
            <h2 className="font-display text-xl tracking-[0.2em] text-foreground">
              {handSide === 'left' ? 'LEFT' : 'RIGHT'} HAND
            </h2>
            <p className="text-xs font-bold tracking-wider text-muted-foreground">NO DATA RECORDED</p>
          </div>
        </div>
        <div className="text-center py-12 text-muted-foreground border-2 border-dashed border-foreground/20 rounded">
          <div className="text-5xl mb-4 opacity-30">📊</div>
          <p className="font-display text-sm tracking-widest opacity-50">RECORD SESSIONS TO VIEW</p>
        </div>
      </div>
    );
  }

  return (
    <div className="relative">
      <div className="bg-card border-4 border-foreground p-6 shadow-card-retro mb-4">
        <div className="flex items-center gap-3 mb-2">
          <span className="text-4xl">{handSide === 'left' ? '🤚' : '✋'}</span>
          <div>
            <h2 className="font-display text-2xl tracking-[0.2em] text-foreground">
              {handSide === 'left' ? 'LEFT' : 'RIGHT'} HAND
            </h2>
            <p className="text-xs font-bold tracking-wider text-muted-foreground">TAP A JOINT TO VIEW DETAILED METRICS</p>
          </div>
        </div>
      </div>
      <svg viewBox="0 0 300 420" className="w-full max-w-[300px] mx-auto" style={{ transform: handSide === 'left' ? 'scaleX(-1)' : 'none' }}>
        {/* Hand outline - right hand, palm facing viewer */}
        <g stroke="hsl(var(--foreground))" strokeWidth="2.5" fill="none">
          {/* Palm */}
          <path d="M 100 320 Q 80 280 85 240 Q 88 220 95 200 L 95 195 Q 90 180 92 165 L 105 150 L 115 135 Q 120 125 125 120 L 130 135 L 135 155 Q 138 165 135 180 L 130 200 Q 128 210 130 220" />
          <path d="M 200 320 Q 220 280 215 240 Q 212 220 205 200 L 205 195 Q 210 180 208 165 L 195 148 L 185 130 Q 180 120 175 115 L 170 130 L 165 155 Q 162 165 165 180 L 170 200 Q 172 210 170 220" />
          
          {/* Thumb */}
          <path d="M 95 200 Q 70 195 55 175 Q 45 160 40 140 Q 38 125 45 115 Q 55 105 65 110 Q 75 115 80 130 Q 85 145 92 165" />
          {/* Thumb tip marker */}
          <circle cx="52" cy="93" r="3" fill="none" stroke="hsl(var(--foreground))" strokeWidth="1.5" opacity="0.5" />
          
          {/* Index finger */}
          <path d="M 105 150 Q 100 120 95 90 Q 92 65 95 45 Q 98 30 108 28 Q 118 30 120 45 Q 122 60 120 80 Q 118 100 115 135" />
          {/* Index tip marker */}
          <circle cx="100" cy="23" r="3" fill="none" stroke="hsl(var(--foreground))" strokeWidth="1.5" opacity="0.5" />
          
          {/* Middle finger */}
          <path d="M 125 120 Q 130 85 135 55 Q 138 30 140 15 Q 143 5 150 5 Q 157 5 160 15 Q 162 30 165 55 Q 168 85 175 115" />
          {/* Middle tip marker */}
          <circle cx="153" cy="5" r="3" fill="none" stroke="hsl(var(--foreground))" strokeWidth="1.5" opacity="0.5" />
          
          {/* Ring finger */}
          <path d="M 135 155 Q 150 115 155 80 Q 158 55 160 35 Q 162 22 168 22 Q 174 22 176 35 Q 178 55 175 80 Q 172 115 165 155" />
          {/* Ring tip marker */}
          <circle cx="170" cy="20" r="3" fill="none" stroke="hsl(var(--foreground))" strokeWidth="1.5" opacity="0.5" />
          
          {/* Pinky */}
          <path d="M 130 200 Q 155 165 165 130 Q 170 105 172 85 Q 174 70 180 68 Q 186 70 188 85 Q 190 105 185 130 Q 178 165 170 200" />
          {/* Pinky tip marker */}
          <circle cx="186" cy="62" r="3" fill="none" stroke="hsl(var(--foreground))" strokeWidth="1.5" opacity="0.5" />
          
          {/* Wrist */}
          <path d="M 100 320 Q 110 340 150 345 Q 190 340 200 320" />
          
          {/* Palm center lines */}
          <path d="M 105 240 Q 130 225 170 235" strokeWidth="1" opacity="0.3" />
          <path d="M 110 265 Q 140 255 185 260" strokeWidth="1" opacity="0.3" />
          <path d="M 115 290 Q 145 280 180 285" strokeWidth="1" opacity="0.3" />
        </g>

        {/* Joint indicators */}
        {joints.map((joint) => {
          const overall = getOverallScore(joint);
          const color = getStatusColor(overall);
          const isSelected = selectedJointId === joint.id;
          const isHovered = hoveredJoint === joint.id;
          const radius = isSelected || isHovered ? 12 : 9;

          return (
            <g
              key={joint.id}
              onClick={() => onJointSelect?.(joint)}
              onMouseEnter={() => setHoveredJoint(joint.id)}
              onMouseLeave={() => setHoveredJoint(null)}
              className="cursor-pointer"
            >
              {/* Pulse ring for weak joints */}
              {overall < 50 && (
                <circle
                  cx={joint.cx}
                  cy={joint.cy}
                  r={radius + 6}
                  fill="none"
                  stroke={color}
                  strokeWidth="2"
                  opacity="0.4"
                  className={getPulseClass(overall)}
                />
              )}
              
              {/* Selection ring */}
              {isSelected && (
                <circle
                  cx={joint.cx}
                  cy={joint.cy}
                  r={radius + 4}
                  fill="none"
                  stroke="hsl(var(--foreground))"
                  strokeWidth="3"
                />
              )}

              {/* Main joint circle */}
              <circle
                cx={joint.cx}
                cy={joint.cy}
                r={radius}
                fill={color}
                stroke="hsl(var(--foreground))"
                strokeWidth="2.5"
              />

              {/* Score text */}
              <text
                x={joint.cx}
                y={joint.cy + 1}
                textAnchor="middle"
                dominantBaseline="middle"
                fontFamily="Bebas Neue, sans-serif"
                fontSize="10"
                fill="hsl(var(--foreground))"
                fontWeight="bold"
                style={{ transform: handSide === 'left' ? 'scaleX(-1)' : 'none', transformOrigin: `${joint.cx}px ${joint.cy}px` }}
              >
                {overall}
              </text>

              {/* Label on hover/select */}
              {(isHovered || isSelected) && (
                <g style={{ transform: handSide === 'left' ? 'scaleX(-1)' : 'none', transformOrigin: `${joint.cx}px ${joint.cy}px` }}>
                  <rect
                    x={joint.cx + 14}
                    y={joint.cy - 10}
                    width={joint.label.length * 7 + 12}
                    height={18}
                    fill="hsl(var(--foreground))"
                    stroke="none"
                  />
                  <text
                    x={joint.cx + 20}
                    y={joint.cy + 1}
                    fontFamily="Bebas Neue, sans-serif"
                    fontSize="11"
                    fill="hsl(var(--label-cream))"
                    letterSpacing="1"
                  >
                    {joint.label}
                  </text>
                </g>
              )}
            </g>
          );
        })}

        {/* Legend */}
        <g transform="translate(5, 370)">
          {[
            { color: "hsl(var(--success))", label: "75-100 STRONG" },
            { color: "hsl(40 80% 50%)", label: "50-74 MODERATE" },
            { color: "hsl(25 80% 50%)", label: "30-49 AT RISK" },
            { color: "hsl(var(--cassette-red))", label: "0-29 WEAK" },
          ].map((item, i) => (
            <g key={i} transform={`translate(${i * 72}, 0)`}>
              <circle cx="6" cy="6" r="5" fill={item.color} stroke="hsl(var(--foreground))" strokeWidth="1.5" />
              <text x="15" y="10" fontFamily="Courier Prime, monospace" fontSize="6" fontWeight="700" fill="hsl(var(--foreground))">
                {item.label}
              </text>
            </g>
          ))}
        </g>
      </svg>
    </div>
  );
};

export { getStatusColor, getStatusLabel };
export default ClinicalHandDiagram;
