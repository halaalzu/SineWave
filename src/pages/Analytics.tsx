import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import CassetteLayout from "@/components/CassetteLayout";
import ClinicalHandDiagram, { getStatusColor, getStatusLabel } from "@/components/ClinicalHandDiagram";
import type { JointData } from "@/components/ClinicalHandDiagram";
import { GeminiChat } from "@/components/GeminiChat";
import { useToast } from "@/hooks/use-toast";
import { Button } from "@/components/ui/button";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis,
  PolarRadiusAxis, Radar, BarChart, Bar,
} from "recharts";

const Analytics = () => {
  const navigate = useNavigate();
  const [selectedJoint, setSelectedJoint] = useState<JointData | null>(null);
  const [handSide, setHandSide] = useState<'left' | 'right'>('right');
  const [jointData, setJointData] = useState<JointData[]>([]);
  const [leftHandData, setLeftHandData] = useState<JointData[]>([]);
  const [rightHandData, setRightHandData] = useState<JointData[]>([]);
  const [weeklyProgress, setWeeklyProgress] = useState<any[]>([]);
  const [overallScores, setOverallScores] = useState<any>({});
  const [isLoading, setIsLoading] = useState(true);
  const { toast } = useToast();

  useEffect(() => {
    fetchAnalytics();
  }, []);

  useEffect(() => {
    // Update displayed data when hand side changes
    if (handSide === 'left') {
      setJointData(leftHandData);
    } else {
      setJointData(rightHandData);
    }
  }, [handSide, leftHandData, rightHandData]);

  const fetchAnalytics = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('http://localhost:5001/api/user/default_user/analytics');
      
      if (!response.ok) {
        throw new Error('No data available');
      }

      const data = await response.json();
      
      // Set positions for joints (simplified mapping)
      const positionedJoints = data.jointData.map((joint: any) => ({
        ...joint,
        cx: getJointPosition(joint.id).cx,
        cy: getJointPosition(joint.id).cy
      }));

      // For now, use same data for both hands (TODO: separate by hand in recording)
      setRightHandData(positionedJoints);
      setLeftHandData([]); // No left hand data yet
      setJointData(positionedJoints);
      setWeeklyProgress(data.weeklyProgress);
      setOverallScores(data.overallScores);
    } catch (error) {
      console.error('Error fetching analytics:', error);
      toast({
        title: "No Data Available",
        description: "Record some sessions in Free Style or Levels to see your analytics!",
        variant: "destructive"
      });
      // Use empty data as fallback
      setJointData([]);
      setLeftHandData([]);
      setRightHandData([]);
      setWeeklyProgress([]);
      setOverallScores({ smoothness: 0, rom: 0, trajectory: 0, consistency: 0 });
    } finally {
      setIsLoading(false);
    }
  };

  const getJointPosition = (jointId: string): { cx: number, cy: number } => {
    // All 21 MediaPipe landmarks mapped to hand diagram positions
    const positions: Record<string, { cx: number, cy: number }> = {
      // Thumb (4 joints)
      'Thumb_CMC': { cx: 62, cy: 140 },
      'Thumb_MCP': { cx: 50, cy: 118 },
      'Thumb_IP': { cx: 48, cy: 102 },
      'Thumb_TIP': { cx: 52, cy: 93 },
      // Index finger (4 joints)
      'Index_MCP': { cx: 107, cy: 130 },
      'Index_PIP': { cx: 102, cy: 78 },
      'Index_DIP': { cx: 100, cy: 42 },
      'Index_TIP': { cx: 100, cy: 23 },
      // Middle finger (4 joints)
      'Middle_MCP': { cx: 140, cy: 118 },
      'Middle_PIP': { cx: 143, cy: 60 },
      'Middle_DIP': { cx: 150, cy: 14 },
      'Middle_TIP': { cx: 153, cy: 5 },
      // Ring finger (4 joints)
      'Ring_MCP': { cx: 165, cy: 130 },
      'Ring_PIP': { cx: 168, cy: 72 },
      'Ring_DIP': { cx: 168, cy: 32 },
      'Ring_TIP': { cx: 170, cy: 20 },
      // Pinky finger (4 joints)
      'Pinky_MCP': { cx: 178, cy: 158 },
      'Pinky_PIP': { cx: 183, cy: 108 },
      'Pinky_DIP': { cx: 184, cy: 78 },
      'Pinky_TIP': { cx: 186, cy: 62 },
      // Wrist
      'Wrist_BASE': { cx: 150, cy: 330 }
    };
    
    return positions[jointId] || { cx: 150, cy: 200 };
  };

  const fmaProxy = overallScores.smoothness && overallScores.rom 
    ? Math.round((overallScores.smoothness + overallScores.rom + overallScores.trajectory + overallScores.consistency) / 4)
    : 0;

  const radarData = overallScores.smoothness ? [
    { metric: "Smoothness", value: overallScores.smoothness || 0, fullMark: 100 },
    { metric: "ROM", value: overallScores.rom || 0, fullMark: 100 },
    { metric: "Accuracy", value: overallScores.trajectory || 0, fullMark: 100 },
    { metric: "Consistency", value: overallScores.consistency || 0, fullMark: 100 },
    { metric: "Speed", value: overallScores.rom || 0, fullMark: 100 },
  ] : [];

  // Find weakest areas - analyze all 20 joints (5 fingers × 4 joints each)
  const fingerGroups = jointData.length > 0 ? [
    { name: "THUMB", joints: jointData.filter(j => j.id.startsWith("Thumb")) },
    { name: "INDEX", joints: jointData.filter(j => j.id.startsWith("Index")) },
    { name: "MIDDLE", joints: jointData.filter(j => j.id.startsWith("Middle")) },
    { name: "RING", joints: jointData.filter(j => j.id.startsWith("Ring")) },
    { name: "PINKY", joints: jointData.filter(j => j.id.startsWith("Pinky")) },
  ].map(g => ({
    ...g,
    avgScore: g.joints.length > 0 
      ? Math.round(g.joints.reduce((a, j) => a + (j.smoothness + j.rom + j.trajectoryAccuracy + j.consistency) / 4, 0) / g.joints.length)
      : 0,
  })).sort((a, b) => a.avgScore - b.avgScore) : [];

  const weakestAreas = fingerGroups.filter(g => g.avgScore < 50 && g.avgScore > 0);

  // ROM by finger
  const romByFinger = fingerGroups.length > 0 ? fingerGroups.map(group => ({
    finger: group.name,
    current: group.avgScore,
    baseline: 100
  })) : [];

  return (
    <CassetteLayout
      verticalText="STATS"
      showBackButton
      showRedPanel={false}
      rightVerticalText="ANALYTICS"
    >
      <div className="mb-6 mt-8">
        <div className="flex items-center justify-between mb-4">
          <h1 className="font-display text-4xl md:text-5xl tracking-[0.25em] text-foreground leading-[0.9]">
            CLINICAL<br />ANALYTICS
          </h1>
          <div className="flex gap-2">
            <button
              onClick={() => navigate('/hand-comparison')}
              className="px-4 py-2 bg-gradient-to-r from-purple-500 to-blue-500 hover:from-purple-600 hover:to-blue-600 text-white rounded-lg font-semibold transition-all"
            >
              🤚 Compare Hands
            </button>
            <button
              onClick={fetchAnalytics}
              disabled={isLoading}
              className="bg-secondary border-3 border-foreground px-4 py-2 font-display text-sm tracking-wider text-secondary-foreground hover:shadow-btn transition-all disabled:opacity-40"
            >
              🔄 REFRESH
            </button>
            <button
              onClick={() => setHandSide('left')}
              className={`font-display text-sm tracking-wider px-4 py-2 border-3 border-foreground transition-all ${
                handSide === 'left'
                  ? 'bg-primary text-primary-foreground shadow-btn'
                  : 'bg-card text-foreground hover:bg-muted'
              }`}
            >
              ← LEFT
            </button>
            <button
              onClick={() => setHandSide('right')}
              className={`font-display text-sm tracking-wider px-4 py-2 border-3 border-foreground transition-all ${
                handSide === 'right'
                  ? 'bg-primary text-primary-foreground shadow-btn'
                  : 'bg-card text-foreground hover:bg-muted'
              }`}
            >
              RIGHT →
            </button>
          </div>
        </div>
        <p className="text-sm font-bold tracking-wider text-muted-foreground">
          MOVEMENT QUALITY METRICS • BASED ON CLINICAL REHABILITATION STANDARDS
        </p>
      </div>

      {isLoading ? (
        <div className="bg-card border-4 border-foreground p-12 text-center shadow-card-retro">
          <div className="font-display text-2xl tracking-widest text-foreground/50 animate-pulse">
            LOADING DATA...
          </div>
        </div>
      ) : jointData.length === 0 ? (
        <div className="bg-card border-4 border-foreground p-12 text-center shadow-card-retro">
          <div className="text-5xl mb-4">🎯</div>
          <div className="font-display text-2xl tracking-widest text-foreground mb-2">
            NO DATA YET
          </div>
          <p className="text-sm font-bold tracking-wider text-foreground/60">
            Complete some sessions in Free Style or Levels mode to see your analytics!
          </p>
        </div>
      ) : (
        <>
      {/* ── FMA Proxy Score ── */}
      <div className="bg-foreground border-[6px] border-foreground p-5 mb-6 shadow-card-retro mr-0 md:mr-12">
        <div className="flex items-center justify-between">
          <div>
            <div className="font-display text-sm tracking-[0.3em] text-card/60">
              FUGL-MEYER PROXY SCORE
            </div>
            <div className="font-display text-6xl tracking-widest text-card">
              {fmaProxy}<span className="text-2xl text-card/50">/100</span>
            </div>
            <div className="text-[10px] font-bold tracking-wider text-card/50 mt-1">
              APPROXIMATED FROM SMOOTHNESS • ROM • ACCURACY • CONSISTENCY
            </div>
          </div>
          <div className="text-5xl">🩺</div>
        </div>
      </div>

      {/* ── Core Metrics Row ── */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6 mr-0 md:mr-12">
        {[
          { label: "SMOOTHNESS", sublabel: "SPARC-BASED", value: overallScores.smoothness, icon: "〰️" },
          { label: "ROM", sublabel: "VS BASELINE", value: overallScores.rom, icon: "📐" },
          { label: "TRAJECTORY", sublabel: "PATH ACCURACY", value: overallScores.trajectory, icon: "🎯" },
          { label: "CONSISTENCY", sublabel: "TRIAL VARIANCE", value: overallScores.consistency, icon: "🔄" },
        ].map((m) => (
          <div key={m.label} className="bg-card border-4 border-foreground p-3 shadow-card-retro relative">
            <div className="absolute top-1 left-1 right-1 bottom-1 border border-foreground/15 pointer-events-none" />
            <div className="text-xl mb-1">{m.icon}</div>
            <div className="font-display text-3xl tracking-widest" style={{ color: getStatusColor(m.value) }}>
              {m.value}
            </div>
            <div className="font-display text-[10px] tracking-widest text-foreground">{m.label}</div>
            <div className="text-[8px] font-bold tracking-wider text-foreground/40">{m.sublabel}</div>
          </div>
        ))}
      </div>

      {/* ── Hand Diagram + Joint Detail ── */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6 mr-0 md:mr-12">
        {/* Hand */}
        <div className="bg-card border-[6px] border-foreground p-4 shadow-card-retro">
          <ClinicalHandDiagram
            joints={jointData}
            onJointSelect={setSelectedJoint}
            selectedJointId={selectedJoint?.id}
            handSide={handSide}
            hasData={jointData.length > 0}
          />
        </div>

        {/* Joint detail panel */}
        <div className="bg-card border-[6px] border-foreground p-4 shadow-card-retro">
          {selectedJoint ? (
            <>
              <div className="font-display text-lg tracking-[0.2em] text-foreground mb-1">
                📍 {selectedJoint.label}
              </div>
              <div className="font-display text-sm tracking-widest mb-4" style={{ color: getStatusColor(Math.round((selectedJoint.smoothness + selectedJoint.rom + selectedJoint.trajectoryAccuracy + selectedJoint.consistency) / 4)) }}>
                {getStatusLabel(Math.round((selectedJoint.smoothness + selectedJoint.rom + selectedJoint.trajectoryAccuracy + selectedJoint.consistency) / 4))}
              </div>

              <div className="space-y-3">
                {[
                  { label: "SMOOTHNESS (SPARC)", value: selectedJoint.smoothness, desc: "Motor control quality" },
                  { label: "RANGE OF MOTION", value: selectedJoint.rom, desc: "% of healthy baseline" },
                  { label: "TRAJECTORY ACCURACY", value: selectedJoint.trajectoryAccuracy, desc: "Path deviation from ideal" },
                  { label: "CONSISTENCY", value: selectedJoint.consistency, desc: "Trial-to-trial variability" },
                ].map((metric) => (
                  <div key={metric.label}>
                    <div className="flex justify-between items-baseline mb-1">
                      <span className="font-display text-[11px] tracking-widest text-foreground">{metric.label}</span>
                      <span className="font-display text-lg tracking-widest" style={{ color: getStatusColor(metric.value) }}>
                        {metric.value}
                      </span>
                    </div>
                    <div className="h-4 border-2 border-foreground/20 bg-foreground/5 relative">
                      <div
                        className="h-full transition-all duration-500"
                        style={{ width: `${metric.value}%`, backgroundColor: getStatusColor(metric.value) }}
                      />
                    </div>
                    <div className="text-[8px] font-bold tracking-wider text-foreground/40 mt-0.5">{metric.desc}</div>
                  </div>
                ))}

                {/* Movement duration */}
                <div className="border-t-2 border-foreground/15 pt-3 mt-3">
                  <div className="flex justify-between">
                    <span className="font-display text-[11px] tracking-widest text-foreground">MOVEMENT DURATION</span>
                    <span className="font-display text-lg tracking-widest text-foreground">
                      {selectedJoint.movementDuration}<span className="text-[10px] text-foreground/50">ms</span>
                    </span>
                  </div>
                  <div className="text-[8px] font-bold tracking-wider text-foreground/40">
                    HEALTHY BASELINE: {selectedJoint.baselineDuration}ms • 
                    {selectedJoint.movementDuration > selectedJoint.baselineDuration
                      ? ` ${Math.round(((selectedJoint.movementDuration - selectedJoint.baselineDuration) / selectedJoint.baselineDuration) * 100)}% SLOWER`
                      : " ON TARGET"}
                  </div>
                </div>
              </div>
            </>
          ) : (
            <div className="h-full flex flex-col items-center justify-center text-center py-8">
              <div className="text-5xl mb-4">👈</div>
              <div className="font-display text-xl tracking-widest text-foreground/50">
                SELECT A JOINT
              </div>
              <p className="text-[10px] font-bold tracking-wider text-foreground/30 mt-2 max-w-[200px]">
                TAP ANY JOINT ON THE HAND DIAGRAM TO VIEW CLINICAL METRICS
              </p>
            </div>
          )}
        </div>
      </div>

      {/* ── Movement Quality Radar ── */}
      <div className="bg-card border-[6px] border-foreground p-4 mb-6 shadow-card-retro mr-0 md:mr-12">
        <div className="font-display text-lg tracking-[0.2em] text-foreground mb-3">
          🧠 MOVEMENT QUALITY PROFILE
        </div>
        <div className="h-56">
          <ResponsiveContainer width="100%" height="100%">
            <RadarChart cx="50%" cy="50%" outerRadius="70%" data={radarData}>
              <PolarGrid stroke="hsl(0 0% 10% / 0.15)" />
              <PolarAngleAxis
                dataKey="metric"
                tick={{ fontFamily: "Courier Prime", fontSize: 10, fontWeight: 700, fill: "hsl(0 0% 10% / 0.7)" }}
              />
              <PolarRadiusAxis
                angle={90}
                domain={[0, 100]}
                tick={{ fontFamily: "Courier Prime", fontSize: 8, fill: "hsl(0 0% 10% / 0.4)" }}
              />
              <Radar
                name="Current"
                dataKey="value"
                stroke="hsl(0 52% 54%)"
                fill="hsl(0 52% 54%)"
                fillOpacity={0.25}
                strokeWidth={2}
              />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* ── Progress Over Time ── */}
      <div className="bg-card border-[6px] border-foreground p-4 mb-6 shadow-card-retro mr-0 md:mr-12">
        <div className="font-display text-lg tracking-[0.2em] text-foreground mb-3">
          📈 REHABILITATION PROGRESS (7 WEEKS)
        </div>
        <div className="h-52">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={weeklyProgress}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(0 0% 10% / 0.1)" />
              <XAxis
                dataKey="week"
                tick={{ fontFamily: "Courier Prime", fontSize: 10, fontWeight: 700 }}
                stroke="hsl(0 0% 10% / 0.3)"
              />
              <YAxis
                domain={[0, 100]}
                tick={{ fontFamily: "Courier Prime", fontSize: 10, fontWeight: 700 }}
                stroke="hsl(0 0% 10% / 0.3)"
              />
              <Tooltip
                contentStyle={{
                  fontFamily: "Courier Prime", fontSize: 11, fontWeight: 700,
                  background: "hsl(31 52% 92%)", border: "3px solid hsl(0 0% 10%)",
                  borderRadius: 0, boxShadow: "4px 4px 0 rgba(0,0,0,0.3)",
                }}
              />
              <Line type="monotone" dataKey="smoothness" stroke="hsl(0 52% 54%)" strokeWidth={2} dot={{ r: 3 }} name="Smoothness" />
              <Line type="monotone" dataKey="rom" stroke="hsl(120 40% 40%)" strokeWidth={2} dot={{ r: 3 }} name="ROM" />
              <Line type="monotone" dataKey="accuracy" stroke="hsl(40 80% 50%)" strokeWidth={2} dot={{ r: 3 }} name="Accuracy" />
              <Line type="monotone" dataKey="consistency" stroke="hsl(27 14% 25%)" strokeWidth={2} dot={{ r: 3 }} name="Consistency" />
              <Line type="monotone" dataKey="fmaProxy" stroke="hsl(0 0% 10%)" strokeWidth={3} strokeDasharray="5 5" dot={{ r: 4 }} name="FMA Proxy" />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div className="flex flex-wrap gap-3 mt-3">
          {[
            { color: "hsl(0 52% 54%)", label: "SMOOTHNESS" },
            { color: "hsl(120 40% 40%)", label: "ROM" },
            { color: "hsl(40 80% 50%)", label: "ACCURACY" },
            { color: "hsl(27 14% 25%)", label: "CONSISTENCY" },
            { color: "hsl(0 0% 10%)", label: "FMA PROXY", dashed: true },
          ].map((l) => (
            <div key={l.label} className="flex items-center gap-1">
              <div className="w-4 h-[3px]" style={{ backgroundColor: l.color, borderTop: l.dashed ? "2px dashed" : undefined }} />
              <span className="text-[8px] font-bold tracking-wider text-foreground/60">{l.label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* ── Clinical Recommendations ── */}
      <div className="bg-foreground border-[6px] border-foreground p-5 mb-6 shadow-card-retro mr-0 md:mr-12">
        <div className="font-display text-lg tracking-[0.2em] text-card mb-3">
          ⚠️ CLINICAL RECOMMENDATIONS
        </div>

        {weakestAreas.length > 0 && (
          <div className="mb-4">
            <div className="font-display text-sm tracking-widest text-primary mb-2">
              PRIORITY WEAKNESS AREAS
            </div>
            <div className="flex flex-wrap gap-2">
              {weakestAreas.map((area) => (
                <span key={area.name} className="bg-primary text-primary-foreground font-display text-sm tracking-widest px-3 py-1 border-2 border-primary">
                  {area.name} — {area.avgScore}%
                </span>
              ))}
            </div>
          </div>
        )}

        <div className="space-y-3">
          <div className="border-l-4 border-primary pl-3">
            <div className="font-display text-sm tracking-widest text-card">SMOOTHNESS TRAINING</div>
            <p className="text-[10px] font-bold text-card/60 tracking-wider">
              Ring and pinky fingers show jerky movement patterns (low SPARC scores). Focus on slow, controlled repetitions to improve motor planning before increasing speed.
            </p>
          </div>
          <div className="border-l-4 border-warning pl-3">
            <div className="font-display text-sm tracking-widest text-card">RANGE OF MOTION</div>
            <p className="text-[10px] font-bold text-card/60 tracking-wider">
              Ulnar-side ROM is significantly below baseline. Gentle stretching exercises targeting ring/pinky flexion and extension, with gradual progression.
            </p>
          </div>
          <div className="border-l-4 border-card pl-3">
            <div className="font-display text-sm tracking-widest text-card">TRAJECTORY & CONSISTENCY</div>
            <p className="text-[10px] font-bold text-card/60 tracking-wider">
              Path deviations suggest compensatory movements. Practice reaching tasks with visual targets to improve direct motor pathways. Aim for consistent patterns across trials.
            </p>
          </div>
        </div>
      </div>

      {/* ── AI Chat Assistant ── */}
      <div className="mr-0 md:mr-12 mb-8">
        <GeminiChat userId="default_user" />
      </div>
      </>
      )}
    </CassetteLayout>
  );
};

export default Analytics;
