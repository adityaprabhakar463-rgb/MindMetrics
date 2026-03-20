import pandas as pd
import numpy as np

np.random.seed(42)

def generate_dataset(num_samples = 200):
    study_hours = np.random.uniform(1, 10, num_samples)
    sleep_hours = np.random.uniform(4, 9, num_samples)
    mood = np.random.randint(1,6,num_samples)
    distraction = np.random.randint(1,6,num_samples)
    difficulty = np.random.randint(1,6,num_samples)
    exam_proximity = np.random.randint(1,31,num_samples)

    productivity = (
        study_hours * 8 + 
        sleep_hours * 5 +
        mood * 6 -
        distraction * 7 -
        difficulty * 3
    )

    productivity = np.clip(productivity, 0, 100)

    burnout = (
        (study_hours > 8) &
        (sleep_hours < 6) |
        (exam_proximity < 5) & (sleep_hours < 5)
    ).astype(int)

    data = pd.DataFrame({
        'Study Hours': study_hours,
        'Sleep Hours': sleep_hours,
        'Mood': mood,
        'Distraction': distraction,
        'Difficulty': difficulty,
        'Exam Proximity': exam_proximity,
        'Productivity Score': productivity,
        'Burnout Risk': burnout
    })

    return data
if __name__ == "__main__":
    df = generate_dataset(300)
    df.to_csv("data/study_data.csv", index = False)