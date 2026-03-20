export function predictProductivity(
  studyHours: number,
  sleepHours: number,
  mood: number,
  distraction: number,
  difficulty: number,
): number {
  const score = studyHours * 8 + sleepHours * 5 + mood * 6 - distraction * 7 - difficulty * 3;
  return Math.max(0, Math.min(100, score));
}

export function predictBurnout(
  studyHours: number,
  sleepHours: number,
  examProximity: number,
): { isBurnout: boolean; probability: number } {
  const risk1 = studyHours > 8 && sleepHours < 6;
  const risk2 = examProximity < 5 && sleepHours < 5;
  const isBurnout = risk1 || risk2;

  // Simple probability estimation
  let prob = 0;
  if (studyHours > 8) prob += 0.3;
  if (sleepHours < 6) prob += 0.25;
  if (sleepHours < 5) prob += 0.15;
  if (examProximity < 5) prob += 0.2;
  if (examProximity < 10) prob += 0.1;
  prob = Math.min(1, prob);

  return { isBurnout, probability: prob };
}

export function getRecommendations(
  isBurnout: boolean,
  studyHours: number,
  sleepHours: number,
  distraction: number,
  mood: number,
): string[] {
  const recs: string[] = [];
  if (isBurnout) {
    recs.push("⚠️ High burnout risk detected — prioritize rest and recovery.");
  }
  if (sleepHours < 6) recs.push("💤 Increase sleep to at least 7 hours for better cognitive function.");
  if (studyHours > 8) recs.push("📚 Consider shorter, focused study sessions with breaks.");
  if (distraction > 3) recs.push("🎯 Minimize distractions — try app blockers or a quiet environment.");
  if (mood < 3) recs.push("🧘 Low mood affects focus — try exercise, meditation, or social time.");
  if (recs.length === 0) recs.push("✅ You're on track — keep up the balanced habits!");
  return recs;
}
