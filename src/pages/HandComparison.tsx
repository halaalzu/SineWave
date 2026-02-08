import { useEffect, useState } from 'react';
import { Card } from '@/components/ui/card';
import { ArrowLeft } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

interface HandMetrics {
  hand: string;
  avg_tremor: number;
  avg_smoothness: number;
  avg_speed: number;
  message: string;
}

export default function HandComparison() {
  const navigate = useNavigate();
  const [leftHand, setLeftHand] = useState<HandMetrics | null>(null);
  const [rightHand, setRightHand] = useState<HandMetrics | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchHandData = async () => {
      try {
        const [leftRes, rightRes] = await Promise.all([
          fetch('http://localhost:5001/api/user/default_user/analytics/hand/Left'),
          fetch('http://localhost:5001/api/user/default_user/analytics/hand/Right')
        ]);

        if (leftRes.ok) {
          const leftData = await leftRes.json();
          setLeftHand(leftData);
        }

        if (rightRes.ok) {
          const rightData = await rightRes.json();
          setRightHand(rightData);
        }
      } catch (error) {
        console.error('Error fetching hand comparison:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchHandData();
  }, []);

  const getScoreColor = (value: number, isInverted: boolean = false) => {
    // For tremor (lower is better)
    if (isInverted) {
      if (value < 0.001) return 'text-green-500';
      if (value < 0.005) return 'text-yellow-500';
      return 'text-red-500';
    }
    // For smoothness/speed (higher is better)
    if (value > 0.7) return 'text-green-500';
    if (value > 0.4) return 'text-yellow-500';
    return 'text-red-500';
  };

  const getShakinessScore = (tremor: number) => {
    return Math.max(0, Math.min(100, 100 - (tremor * 8000)));
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-purple-50 to-blue-50 flex items-center justify-center">
        <div className="text-xl">Loading hand comparison...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 to-blue-50 p-8">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="flex items-center gap-4 mb-8">
          <button
            onClick={() => navigate('/analytics')}
            className="p-2 hover:bg-white rounded-lg transition-colors"
          >
            <ArrowLeft size={24} />
          </button>
          <div>
            <h1 className="text-4xl font-bold bg-gradient-to-r from-purple-600 to-blue-600 bg-clip-text text-transparent">
              Left vs Right Hand Comparison
            </h1>
            <p className="text-gray-600 mt-2">
              Compare shakiness and smoothness between your hands
            </p>
          </div>
        </div>

        {/* Comparison Grid */}
        <div className="grid md:grid-cols-2 gap-8">
          {/* Left Hand */}
          <Card className="p-6 bg-white shadow-lg">
            <div className="text-center mb-6">
              <div className="text-6xl mb-2">🤚</div>
              <h2 className="text-2xl font-bold text-purple-600">Left Hand</h2>
            </div>

            {leftHand ? (
              <div className="space-y-4">
                <div className="p-4 bg-purple-50 rounded-lg">
                  <div className="text-sm text-gray-600 mb-1">Shakiness Score</div>
                  <div className={`text-3xl font-bold ${getScoreColor(getShakinessScore(leftHand.avg_tremor) / 100)}`}>
                    {getShakinessScore(leftHand.avg_tremor).toFixed(0)}/100
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    {leftHand.avg_tremor < 0.001 ? '✅ Very Smooth' : 
                     leftHand.avg_tremor < 0.005 ? '👍 Good' : '⚠️ Shaky'}
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="p-3 bg-gray-50 rounded-lg">
                    <div className="text-xs text-gray-600">Tremor</div>
                    <div className={`text-lg font-semibold ${getScoreColor(leftHand.avg_tremor, true)}`}>
                      {(leftHand.avg_tremor * 1000).toFixed(3)}
                    </div>
                    <div className="text-xs text-gray-400">×10⁻³</div>
                  </div>

                  <div className="p-3 bg-gray-50 rounded-lg">
                    <div className="text-xs text-gray-600">Smoothness</div>
                    <div className={`text-lg font-semibold ${getScoreColor(leftHand.avg_smoothness / 1000)}`}>
                      {leftHand.avg_smoothness.toFixed(0)}
                    </div>
                    <div className="text-xs text-gray-400">jerk</div>
                  </div>
                </div>

                <div className="p-3 bg-gray-50 rounded-lg">
                  <div className="text-xs text-gray-600">Speed</div>
                  <div className={`text-lg font-semibold ${getScoreColor(leftHand.avg_speed)}`}>
                    {leftHand.avg_speed.toFixed(3)}
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center text-gray-500 py-8">
                No data available for left hand
              </div>
            )}
          </Card>

          {/* Right Hand */}
          <Card className="p-6 bg-white shadow-lg">
            <div className="text-center mb-6">
              <div className="text-6xl mb-2">👋</div>
              <h2 className="text-2xl font-bold text-blue-600">Right Hand</h2>
            </div>

            {rightHand ? (
              <div className="space-y-4">
                <div className="p-4 bg-blue-50 rounded-lg">
                  <div className="text-sm text-gray-600 mb-1">Shakiness Score</div>
                  <div className={`text-3xl font-bold ${getScoreColor(getShakinessScore(rightHand.avg_tremor) / 100)}`}>
                    {getShakinessScore(rightHand.avg_tremor).toFixed(0)}/100
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    {rightHand.avg_tremor < 0.001 ? '✅ Very Smooth' : 
                     rightHand.avg_tremor < 0.005 ? '👍 Good' : '⚠️ Shaky'}
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="p-3 bg-gray-50 rounded-lg">
                    <div className="text-xs text-gray-600">Tremor</div>
                    <div className={`text-lg font-semibold ${getScoreColor(rightHand.avg_tremor, true)}`}>
                      {(rightHand.avg_tremor * 1000).toFixed(3)}
                    </div>
                    <div className="text-xs text-gray-400">×10⁻³</div>
                  </div>

                  <div className="p-3 bg-gray-50 rounded-lg">
                    <div className="text-xs text-gray-600">Smoothness</div>
                    <div className={`text-lg font-semibold ${getScoreColor(rightHand.avg_smoothness / 1000)}`}>
                      {rightHand.avg_smoothness.toFixed(0)}
                    </div>
                    <div className="text-xs text-gray-400">jerk</div>
                  </div>
                </div>

                <div className="p-3 bg-gray-50 rounded-lg">
                  <div className="text-xs text-gray-600">Speed</div>
                  <div className={`text-lg font-semibold ${getScoreColor(rightHand.avg_speed)}`}>
                    {rightHand.avg_speed.toFixed(3)}
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center text-gray-500 py-8">
                No data available for right hand
              </div>
            )}
          </Card>
        </div>

        {/* Instructions */}
        <Card className="mt-8 p-6 bg-gradient-to-r from-purple-100 to-blue-100">
          <h3 className="text-lg font-semibold mb-3">📝 How to Use This Feature</h3>
          <div className="space-y-2 text-sm text-gray-700">
            <p>• Go to <strong>Freestyle</strong> page and record a session with your LEFT hand only</p>
            <p>• Then record another session with your RIGHT hand only</p>
            <p>• Return here to compare shakiness and smoothness between hands</p>
            <p>• Lower tremor scores = less shaky = better control</p>
            <p>• Try recording <strong>intentionally shaky</strong> vs <strong>smooth</strong> movements to see the difference!</p>
          </div>
        </Card>

        {/* Training Mode Link */}
        <Card className="mt-4 p-6 bg-gradient-to-r from-green-100 to-teal-100">
          <h3 className="text-lg font-semibold mb-3">🎯 Train the Model</h3>
          <div className="space-y-2 text-sm text-gray-700">
            <p>Want to improve shakiness detection accuracy?</p>
            <p className="font-mono bg-white px-3 py-2 rounded mt-2">
              cd FlowState && python training_mode.py
            </p>
            <p className="text-xs text-gray-600 mt-2">
              This will guide you through recording labeled training data (smooth vs shaky movements)
            </p>
          </div>
        </Card>
      </div>
    </div>
  );
}
