
# 🏋️ Workout Analyzer

**Workout Analyzer** is a lightweight Python tool that analyzes and visualizes workout data exported from the **Strong App**.
It transforms raw CSV exports into clear, informative visual insights about your training history.

---

## 📊 What It Does

* Parses CSV workout exports from the **Strong App**
* Generates analytical summaries of training data
* Produces visualizations using industry-standard Python plotting libraries

This tool is designed for quick, local analysis—no accounts, APIs, or external services required.

---

## 📦 Requirements

Ensure you have **Python 3** installed, then install the required dependencies:

```bash
pip3 install seaborn
pip3 install matplotlib
```

---

## 🚀 Usage

Run the script from the command line with one of the following commands:

### Analyze Workout Data

```bash
python3 workout_analyzer_enhanced.py analyze their_workouts.csv
```

### Visualize Workout Data

```bash
python3 workout_analyzer_enhanced.py visualize their_workouts.csv
```

Replace `their_workouts.csv` with the path to your Strong App export file.

---

## 📁 Input Format

* CSV file exported directly from the **Strong App**
* No preprocessing required
* Works with default Strong export structure

---

## 📤 How to Export Data from Strong

1. Open the **Strong App**
2. Navigate to **Settings**
3. Select **Export Data**
4. Export workouts as a **CSV file**

Example export screen:

![Strong App Export](https://github.com/user-attachments/assets/b8d79c0d-e3dd-4314-8bf2-f4c1351e8f3a)

---

## 🛠️ Customization

You can easily extend the script to:

* Add new charts or metrics
* Filter by exercise, date range, or volume
* Export visualizations to image files

---

## 📜 License

This project is provided as-is for personal use.
Refer to the repository’s license file for full terms.
