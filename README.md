# Comprehensive Analysis of Demographic and Personality Data

This project focuses on analyzing demographic and personality data to uncover patterns and insights. It consists of two data sets and a Python script that processes and visualizes this data.

## Project Contents

### CSV Files

1. **demographics.csv**
   - **Description**: Contains fictional demographic data such as age, gender, CAO poins, and geographic location for a sample population of students in the course.
   - **Usage**: This file can be used to understand the distribution of various demographic factors in the population and to correlate these factors with other variables.

2. **personalities.csv**
   - **Description**: Includes data on different personality traits scored on a scale. It includes traits like openness, conscientiousness, extraversion, agreeableness, and neuroticism. Both files can be linked by the last four digits of the phone number column.
   - **Usage**: Analyze to assess the prevalence of certain personality traits and their correlations with demographic variables.

### Python Script

- **CA259-1.py**
  - **Purpose**: This script is designed to load, merge, and analyze the data from the CSV files. It includes functions for data cleaning, basic statistical analysis, and data visualization.
  - **Usage**: Run this script in a Python environment (Python 3.x recommended). Ensure you have the necessary libraries installed (pandas, matplotlib, etc.).

## Getting Started

To use this project, follow these steps:

1. **Set Up Your Environment**:
   - Ensure Python 3.x is installed on your system.
   - Install required Python libraries:
     ```bash
     pip install pandas matplotlib seaborn numpy
     ```

2. **Data Preparation**:
   - Download the `demographics.csv` and `personalities.csv` files.
   - Place them in a known directory.

3. **Running the Script**:
   - Navigate to the directory containing the script and CSV files in your terminal or command prompt.
   - Run the script:
     ```bash
     python CA259-1.py
     ```
   - Follow any prompts in the script to perform data analysis and view plots.

## Features

- Data cleaning and preprocessing.
- Merging datasets for comprehensive analysis.
- Generating descriptive statistics and visualizations to illustrate data trends.

## Contributing

Feel free to fork this repository or submit pull requests with your suggested changes. For major changes, please open an issue first to discuss what you would like to change.

## License

This project can be used without any restrictions.
