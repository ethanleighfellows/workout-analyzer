# Ethan's Workout Analytics

A client-side web application for analyzing weightlifting and fitness data exported from workout tracker logs (like Strong). It processes data locally in the browser to generate visualizations and analytics entirely without a backend database.

## Features

- **Local-First Architecture:** Workout data is parsed via PapaParse and persisted purely in the browser using IndexedDB. Client-side processing ensures complete data privacy.
- **Interactive Dashboard:** Visualizes volume over time, workout frequency, and muscle group distribution using Chart.js.
- **Biomechanics Analysis:** Calculates symmetry ratios (Push vs. Pull) and indentifies the proportion of unilateral vs. bilateral movements.
- **Estimated 1RM Predictions:** Tracks historical 1-Rep Max estimations across major compound lifts (Squat, Bench, Deadlift, Overhead Press) using the Brzycki and Epley formulas.
- **Historical Periodization:** Automatically identifies mesocycle training blocks based on rest periods and calculates Acute-to-Chronic Workload Ratios (ACWR).
- **Plateau Detection:** Flags lifts where progression has stalled (<2% improvement over a rolling 4-week window) versus ones currently progressing.

## Tech Stack

- **Framework:** React 18, Vite
- **Language:** TypeScript
- **State Management:** Zustand
- **Storage:** IndexedDB (`idb`)
- **Visualizations:** Chart.js (`react-chartjs-2`)
- **Styling:** Vanilla CSS & CSS Variables
- **Icons:** Lucide React

## Local Development

To run the application locally:

1. Clone the repository.
2. Navigate to the `webapp` directory: `cd webapp`
3. Install dependencies: `npm install`
4. Start the development server: `npm run dev`

## Deployment

This repository is configured with a GitHub Actions workflow (`deploy.yml`) that automatically builds and deploys the `webapp` directory to GitHub Pages whenever code is pushed to the `main` branch.

## Usage

1. Open the application.
2. Drag and drop your workout history CSV file (e.g., exported from the Strong app) into the upload zone.
3. The dataset will be parsed, persisted in your browser, and the analytics dashboard will immediately populate.
