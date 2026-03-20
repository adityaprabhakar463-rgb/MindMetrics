import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Brain, Moon, BookOpen, Zap, Target, Calendar, TrendingUp, AlertTriangle, CheckCircle2 } from "lucide-react";
import { Slider } from "@/components/ui/slider";
import { Card, CardContent } from "@/components/ui/card";
import { getRecommendations } from "@/lib/predictions";
// Removed mock predictions import: predictProductivity, predictBurnout

interface InputParam {
  label: string;
  icon: React.ElementType;
  min: number;
  max: number;
  step: number;
  value: number;
  unit: string;
}

const Index = () => {
  const [studyHours, setStudyHours] = useState(6);
  const [sleepHours, setSleepHours] = useState(7);
  const [mood, setMood] = useState(4);
  const [distraction, setDistraction] = useState(2);
  const [difficulty, setDifficulty] = useState(3);
  const [examProximity, setExamProximity] = useState(15);

  // States for API predictions
  const [productivity, setProductivity] = useState(0);
  const [burnoutProb, setBurnoutProb] = useState(0);
  const [isBurnout, setIsBurnout] = useState(false);
  const [recommendations, setRecommendations] = useState<string[]>([]);

  // Function to fetch predictions from API
  const fetchPredictions = async () => {
    try {
      const payload = {
        "Study Hours": studyHours,
        "Sleep Hours": sleepHours,
        "Mood": mood,
        "Distraction": distraction,
        "Difficulty": difficulty,
        "Exam Proximity": examProximity
      };

      // Fetch Productivity
      const prodResponse = await fetch('http://localhost:5001/api/predict/productivity', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const prodData = await prodResponse.json();
      console.log("Productivity response:", prodData); // Debug log
      if (prodData.productivity_score !== undefined) {
        setProductivity(prodData.productivity_score);
      } else {
        console.error("No productivity score in response");
      }

      // Fetch Burnout
      const burnoutResponse = await fetch('http://localhost:5001/api/predict/burnout', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const burnoutData = await burnoutResponse.json();
      console.log("Burnout response:", burnoutData); // Debug log
      let currentIsBurnout = isBurnout;
      if (burnoutData.burnout_probability !== undefined) {
        const burnoutRisk = burnoutData.burnout_risk === 1;
        setBurnoutProb(burnoutData.burnout_probability);
        setIsBurnout(burnoutRisk);
        currentIsBurnout = burnoutRisk;
      }

      // Generate recommendations (local logic or could be API)
      const recs = getRecommendations(currentIsBurnout, studyHours, sleepHours, distraction, mood);
      setRecommendations(recs);

    } catch (error) {
      console.error("Error fetching predictions:", error);
    }
  };

  // Update predictions when inputs change
  useEffect(() => {
    const timer = setTimeout(() => {
      fetchPredictions();
    }, 500); // 500ms debounce to avoid too many API calls
    return () => clearTimeout(timer);
  }, [studyHours, sleepHours, mood, distraction, difficulty, examProximity]);

  const inputs: (InputParam & { setter: (v: number) => void })[] = [
    { label: "Study Hours", icon: BookOpen, min: 0, max: 12, step: 0.1, value: studyHours, unit: "hrs", setter: setStudyHours },
    { label: "Sleep Hours", icon: Moon, min: 0, max: 12, step: 0.1, value: sleepHours, unit: "hrs", setter: setSleepHours },
    { label: "Mood", icon: Zap, min: 1, max: 5, step: 1, value: mood, unit: "/5", setter: setMood },
    { label: "Distraction", icon: Target, min: 1, max: 5, step: 1, value: distraction, unit: "/5", setter: setDistraction },
    { label: "Difficulty", icon: TrendingUp, min: 1, max: 5, step: 1, value: difficulty, unit: "/5", setter: setDifficulty },
    { label: "Exam Proximity", icon: Calendar, min: 0, max: 30, step: 1, value: examProximity, unit: "days", setter: setExamProximity },
  ];

  const scoreColor = productivity >= 70 ? "text-success" : productivity >= 40 ? "text-accent" : "text-destructive";
  const scoreGlow = productivity >= 70 ? "glow-success" : productivity >= 40 ? "glow-warning" : "";

  return (
    <div className="min-h-screen gradient-mesh">
      {/* Header */}
      <header className="border-b border-border/50 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-5 flex items-center gap-3">
          <div className="p-2 rounded-lg bg-primary/10 border border-primary/20">
            <Brain className="h-6 w-6 text-primary" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight">AI Productivity Predictor</h1>
            <p className="text-xs text-muted-foreground">Study smarter, not harder</p>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8 max-w-6xl">
        <div className="grid lg:grid-cols-5 gap-8">
          {/* Left: Inputs */}
          <div className="lg:col-span-3 space-y-5">
            <h2 className="text-lg font-semibold text-muted-foreground tracking-wide uppercase text-sm">Parameters</h2>
            <div className="grid sm:grid-cols-2 gap-4">
              {inputs.map((input) => (
                <Card key={input.label} className="bg-card/80 border-border/50 backdrop-blur-sm hover:border-primary/30 transition-colors">
                  <CardContent className="p-5">
                    <div className="flex items-center justify-between mb-4">
                      <div className="flex items-center gap-2">
                        <input.icon className="h-4 w-4 text-primary" />
                        <span className="text-sm font-medium">{input.label}</span>
                      </div>
                      <span className="text-lg font-bold font-mono text-primary">
                        {input.step < 1 ? input.value.toFixed(1) : input.value}
                        <span className="text-xs text-muted-foreground ml-1">{input.unit}</span>
                      </span>
                    </div>
                    <Slider
                      value={[input.value]}
                      min={input.min}
                      max={input.max}
                      step={input.step}
                      onValueChange={([v]) => input.setter(v)}
                      className="cursor-pointer"
                    />
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>

          {/* Right: Results */}
          <div className="lg:col-span-2 space-y-5">
            <h2 className="text-lg font-semibold text-muted-foreground tracking-wide uppercase text-sm">Predictions</h2>

            {/* Productivity Score */}
            <motion.div layout>
              <Card className={`bg-card/80 border-border/50 backdrop-blur-sm ${scoreGlow}`}>
                <CardContent className="p-6 text-center">
                  <p className="text-sm text-muted-foreground mb-2 uppercase tracking-wider">Productivity Score</p>
                  <motion.p
                    key={productivity.toFixed(0)}
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    className={`text-6xl font-bold font-mono ${scoreColor}`}
                  >
                    {productivity.toFixed(1)}
                  </motion.p>
                  <p className="text-xs text-muted-foreground mt-1">out of 100</p>
                  {/* Progress bar */}
                  <div className="mt-4 h-2 bg-muted rounded-full overflow-hidden">
                    <motion.div
                      className="h-full rounded-full bg-primary"
                      initial={{ width: 0 }}
                      animate={{ width: `${productivity}%` }}
                      transition={{ type: "spring", stiffness: 100 }}
                    />
                  </div>
                </CardContent>
              </Card>
            </motion.div>

            {/* Burnout Risk */}
            <Card className={`bg-card/80 border-border/50 backdrop-blur-sm ${isBurnout ? "glow-warning border-accent/30" : ""}`}>
              <CardContent className="p-6">
                <div className="flex items-center justify-between mb-3">
                  <p className="text-sm text-muted-foreground uppercase tracking-wider">Burnout Risk</p>
                  {isBurnout ? (
                    <span className="px-2.5 py-1 rounded-full bg-accent/15 text-accent text-xs font-semibold flex items-center gap-1">
                      <AlertTriangle className="h-3 w-3" /> High
                    </span>
                  ) : (
                    <span className="px-2.5 py-1 rounded-full bg-success/15 text-success text-xs font-semibold flex items-center gap-1">
                      <CheckCircle2 className="h-3 w-3" /> Low
                    </span>
                  )}
                </div>
                <p className="text-3xl font-bold font-mono">
                  {(burnoutProb * 100).toFixed(0)}%
                </p>
                <div className="mt-3 h-2 bg-muted rounded-full overflow-hidden">
                  <motion.div
                    className={`h-full rounded-full ${isBurnout ? "bg-accent" : "bg-success"}`}
                    animate={{ width: `${burnoutProb * 100}%` }}
                    transition={{ type: "spring", stiffness: 100 }}
                  />
                </div>
              </CardContent>
            </Card>

            {/* Recommendations */}
            <Card className="bg-card/80 border-border/50 backdrop-blur-sm">
              <CardContent className="p-6">
                <p className="text-sm text-muted-foreground uppercase tracking-wider mb-4">Recommendations</p>
                <AnimatePresence mode="popLayout">
                  <ul className="space-y-2.5">
                    {recommendations.map((rec, i) => (
                      <motion.li
                        key={rec}
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: 10 }}
                        transition={{ delay: i * 0.05 }}
                        className="text-sm text-secondary-foreground leading-relaxed"
                      >
                        {rec}
                      </motion.li>
                    ))}
                  </ul>
                </AnimatePresence>
              </CardContent>
            </Card>
          </div>
        </div>
      </main>
    </div>
  );
};

export default Index;
